# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

from typing_extensions import override

from aphrodite.logger import init_logger
from aphrodite.platforms import current_platform
from aphrodite.utils.math_utils import round_up
from aphrodite.v1.kv_offload.base import (
    CanonicalKVCaches,
    OffloadingCounterMetadata,
    OffloadingGaugeMetadata,
    OffloadingHistogramMetadata,
    OffloadingManager,
    OffloadingMetricMetadata,
    OffloadingSpec,
    OffloadingWorker,
)
from aphrodite.v1.kv_offload.config import OffloadingConfig
from aphrodite.v1.kv_offload.cpu.common import CPUOffloadingMetrics
from aphrodite.v1.kv_offload.cpu.gpu_worker import CPUOffloadingWorker
from aphrodite.v1.kv_offload.cpu.manager import CPUOffloadingManager

logger = init_logger(__name__)


class CPUOffloadingSpec(OffloadingSpec):
    BLOCK_SIZE_ALIGNMENT = 1

    @classmethod
    def build_metric_definitions(cls, extra_config: dict[str, Any]) -> dict[str, OffloadingMetricMetadata]:
        definitions: dict[str, OffloadingMetricMetadata] = {
            CPUOffloadingMetrics.CPU_CACHE_USAGE_PERC: OffloadingGaugeMetadata(
                documentation=(
                    "Fraction of CPU KV-cache space currently pinned by active "
                    "transfers (0.0 = idle, 1.0 = saturated). Sustained high "
                    "values indicate transfers (stores or promotions) may be "
                    "dropped due to insufficient capacity."
                ),
            ),
            CPUOffloadingMetrics.CPU_CACHE_WRITE_USAGE_PERC: OffloadingGaugeMetadata(
                documentation=(
                    "Fraction of CPU KV-cache space currently pinned by "
                    "in-flight stores that have not yet "
                    "completed (0.0 = idle, 1.0 = saturated)."
                ),
            ),
            CPUOffloadingMetrics.CPU_CACHE_READ_USAGE_PERC: OffloadingGaugeMetadata(
                documentation=(
                    "Fraction of CPU KV-cache space currently pinned by "
                    "in-flight loads that have not yet "
                    "completed (0.0 = idle, 1.0 = saturated)."
                ),
            ),
            CPUOffloadingMetrics.CPU_ALLOCATION_SIZE: OffloadingHistogramMetadata(
                documentation=(
                    "Histogram of the number of CPU blocks requested by each KV offload prepare_store call."
                ),
                buckets=(
                    1,
                    4,
                    16,
                    64,
                    256,
                    1024,
                    4096,
                    16384,
                    65536,
                    262144,
                ),
            ),
        }
        store_threshold = int(extra_config.get("store_threshold", 0))
        if store_threshold >= 2:
            definitions[CPUOffloadingMetrics.STORES_SKIPPED] = OffloadingCounterMetadata(
                documentation=("Number of KV offload stores skipped because the reuse threshold was not reached."),
            )
        return definitions

    def __init__(self, config: OffloadingConfig):
        super().__init__(config)

        cpu_bytes_to_use = self.extra_config.get("cpu_bytes_to_use")
        if not cpu_bytes_to_use:
            raise Exception("cpu_bytes_to_use must be specified in kv_connector_extra_config")

        world_size = config.parallel.world_size
        self.num_blocks = 0
        self.kv_bytes_per_chunk = 0
        self.cpu_page_size_per_worker = 0
        if config.worker_kv_bytes_per_block > 0 and world_size > 0:
            kv_bytes_per_block = config.worker_kv_bytes_per_block * world_size
            kv_bytes_per_chunk = kv_bytes_per_block * self.blocks_per_chunk

            # calculate cpu_page_size_per_worker
            self.cpu_page_size_per_worker = kv_bytes_per_chunk // world_size

            # calculate num_blocks
            aligned_kv_bytes_per_chunk = round_up(kv_bytes_per_chunk, self.BLOCK_SIZE_ALIGNMENT)
            self.num_blocks = int(cpu_bytes_to_use) // aligned_kv_bytes_per_chunk

            # Expose aligned_kv_bytes_per_chunk as
            # kv_bytes_per_chunk. Note that this might contain
            # some padding. i.e. each offloaded block is of the form,
            # |--- W0-B0---|---- W1-B0---| ... |---- Wn-B0---| *** maybe-pad *** |
            self.kv_bytes_per_chunk = aligned_kv_bytes_per_chunk

        # scheduler-side
        self._manager: OffloadingManager | None = None

        # worker-side
        self._worker: CPUOffloadingWorker | None = None

        self.eviction_policy: str = self.extra_config.get("eviction_policy", "lru")

        self._log_capacity_summary(int(cpu_bytes_to_use))

    def _log_capacity_summary(self, cpu_bytes_to_use: int) -> None:
        """Startup summary of CPU offload pool size and nominal token capacity.

        Nominal density is a floor: hybrid models store full sliding-window
        history, so measured density is higher.
        """
        # Row granularity uses the smallest block size (global group).
        tokens_per_block = min(self.tokens_per_block) if self.tokens_per_block else 0
        tokens_per_chunk = tokens_per_block * self.blocks_per_chunk
        row_bytes = self.kv_bytes_per_chunk
        if self.num_blocks <= 0 or tokens_per_chunk <= 0 or row_bytes <= 0:
            logger.warning(
                "KV offload capacity summary unavailable: num_blocks=%d tokens_per_chunk=%d row_bytes=%d",
                self.num_blocks,
                tokens_per_chunk,
                row_bytes,
            )
            return
        mib = 1024**2
        token_capacity = self.num_blocks * tokens_per_chunk
        logger.info(
            "KV offload CPU pool sizing: budget=%.1f GiB num_blocks=%d "
            "row_bytes=%d (%.2f MiB/row) tokens_per_chunk=%d => nominal "
            "%.3f MiB/token, token_capacity~=%d tokens (floor).",
            cpu_bytes_to_use / 1024**3,
            self.num_blocks,
            row_bytes,
            row_bytes / mib,
            tokens_per_chunk,
            (row_bytes / tokens_per_chunk) / mib,
            token_capacity,
        )

    @override
    def get_manager(self) -> OffloadingManager:
        if not self._manager:
            # store_threshold: how many times a block must appear in lookup()
            # before it is eligible for CPU offloading.  Values < 2 disable
            # filtering (a threshold of 1 equals no filter; 0 is the default).
            store_threshold = int(self.extra_config.get("store_threshold", 0))

            # Maximum entries in the internal tracker's LRU table.
            max_tracker_size = int(self.extra_config.get("max_tracker_size", 64_000))

            if "gpu_residency_aware" in self.extra_config:
                logger.warning(
                    "kv_connector_extra_config: 'gpu_residency_aware' is deprecated and "
                    "ignored; the CPU tier now holds only GPU-evicted (offloaded) blocks."
                )

            self._manager = CPUOffloadingManager(
                num_blocks=self.num_blocks,
                cache_policy=self.eviction_policy,  # type: ignore[arg-type]
                enable_events=self.kv_events_config.enable_kv_cache_events,
                store_threshold=store_threshold,
                max_tracker_size=max_tracker_size,
            )
        return self._manager

    def create_worker(self, kv_caches: CanonicalKVCaches) -> CPUOffloadingWorker:
        return CPUOffloadingWorker(
            kv_caches=kv_caches,
            blocks_per_chunk=self.blocks_per_chunk,
            num_cpu_blocks=self.num_blocks,
        )

    @override
    def get_worker(self, kv_caches: CanonicalKVCaches) -> OffloadingWorker:
        if not self._worker:
            if not (current_platform.is_cuda_alike() or current_platform.is_xpu()):
                raise Exception("CPU Offloading is currently only supported on CUDA-alike and XPU GPUs")
            self._worker = self.create_worker(kv_caches)

        assert self._worker is not None
        return self._worker
