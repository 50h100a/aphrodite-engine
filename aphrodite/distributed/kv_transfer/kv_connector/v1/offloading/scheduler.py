# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from itertools import islice
from typing import Any, NamedTuple

from aphrodite.config import AphroditeConfig
from aphrodite.distributed.kv_events import KVCacheEvent
from aphrodite.distributed.kv_transfer.kv_connector.utils import yield_req_data
from aphrodite.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from aphrodite.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingConnectorMetadata,
    OffloadingWorkerMetadata,
    ReqId,
    TransferJob,
)
from aphrodite.distributed.kv_transfer.kv_connector.v1.offloading.events import (
    OffloadingEventGroupSpec,
    OffloadingEventsTracker,
    get_offloading_event_group_spec,
)
from aphrodite.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    OffloadingConnectorStats,
    _ConnectorMetricName,
    _TransferMetricName,
)
from aphrodite.logger import init_logger
from aphrodite.utils.math_utils import cdiv, round_down
from aphrodite.v1.core.block_pool import BlockPool
from aphrodite.v1.core.kv_cache_manager import KVCacheBlocks
from aphrodite.v1.core.kv_cache_utils import get_block_hash, get_group_id
from aphrodite.v1.core.sched.output import SchedulerOutput
from aphrodite.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
    MambaSpec,
    SlidingWindowSpec,
)
from aphrodite.v1.kv_offload.base import (
    GPULoadStoreSpec,
    LookupResult,
    OffloadingManager,
    OffloadingSpec,
    OffloadKey,
    OffloadPolicy,
    ReqContext,
    RequestOffloadingContext,
    ScheduleEndContext,
    make_offload_key,
)
from aphrodite.v1.outputs import KVConnectorOutput
from aphrodite.v1.request import Request

logger = init_logger(__name__)


@dataclass(slots=True)
class TransferJobStatus:
    """Tracks scheduler-side state for a single transfer job."""

    req_id: ReqId
    # Number of workers still pending. Starts at num_workers,
    # decremented as each worker reports completion. Job is done at 0.
    pending_count: int
    # Offload keys this job covers; passed to manager.complete_*().
    keys: set[OffloadKey]
    is_store: bool
    # Store src block IDs whose ref_cnt protects them while the request
    # runs. Only registered in _block_id_to_pending_jobs on request_finished.
    non_sliding_window_block_ids: list[int] | None = None
    # Store src block IDs that may be freed before the request finishes.
    # Registered in _block_id_to_pending_jobs at store creation time.
    sliding_window_block_ids: list[int] | None = None
    # True for request-decoupled offload-on-eviction (write-back) store jobs.
    # These are not tracked in _req_status; on completion they release their
    # reserved GPU blocks back to the pool instead.
    is_writeback: bool = False
    # For write-back jobs: the reserved GPU block IDs to release on completion.
    writeback_block_ids: list[int] | None = None


class GroupOffloadConfig(NamedTuple):
    group_idx: int
    tokens_per_block: int
    tokens_per_chunk: int
    hashes_per_chunk: int
    # KV cache spec metadata propagated onto emitted BlockStored events so
    # KV-aware consumers can classify and filter the group.
    kv_event_group_spec: OffloadingEventGroupSpec
    # None below means full attention
    sliding_window_size_in_chunks: int | None
    # Number of this group's offloaded blocks per full-attention alignment
    # segment. Used to skip storing SWA blocks that can never serve a load
    # hit (e.g. DeepSeek V4 where SWA groups have much smaller block sizes
    # than the MLA full-attention group).
    # None for full-attention groups or when the optimization doesn't apply.
    alignment_chunk_count: int | None = None
    # True when alignment_chunk_count comes from the opt-in
    # swa_store_alignment_chunks knob rather than a real full-attention
    # alignment constraint. Synthetic alignment widens what is stored
    # (segment tails) but does not constrain the load path, so the
    # prompt-tail store gate still applies on top of it.
    alignment_is_synthetic: bool = False
    # True for EAGLE/MTP draft-model attention groups. The trailing block
    # of these groups is volatile and lacks a stable hash, so it must
    # be excluded from store and load scheduling.
    is_eagle_group: bool = False
    # True for genuine sliding-window attention groups (excludes Mamba,
    # whose window of 1 makes every stored chunk a useful hit endpoint,
    # and eagle draft groups). Gates SWA store-skipping optimizations.
    is_true_swa: bool = False


def get_sliding_window_size_in_chunks(kv_cache_spec: KVCacheSpec, tokens_per_chunk: int) -> int | None:
    if isinstance(kv_cache_spec, SlidingWindowSpec):
        assert kv_cache_spec.sliding_window > 0
        return cdiv(kv_cache_spec.sliding_window, tokens_per_chunk)

    if isinstance(kv_cache_spec, MambaSpec):
        # Mamba depends on a single state
        return 1

    assert isinstance(kv_cache_spec, FullAttentionSpec)
    return None


def resolve_mamba_align_size(spec: "OffloadingSpec", kv_cache_config: KVCacheConfig) -> int | None:
    """Scan all KV cache groups in *spec* and return the single mamba alignment
    size, or None if no group requires mamba alignment.

    For MambaSpec groups in "align" cache mode the hit window must be rounded
    down to a multiple of the offloaded block size. Asserts that all such
    groups agree on the same value.
    """
    mamba_align_size: int | None = None
    for idx, tokens_per_block in enumerate(spec.tokens_per_block):
        kv_spec = kv_cache_config.kv_cache_groups[idx].kv_cache_spec
        if isinstance(kv_spec, MambaSpec) and kv_spec.mamba_cache_mode == "align":
            tokens_per_chunk = tokens_per_block * spec.blocks_per_chunk
            assert mamba_align_size is None or mamba_align_size == tokens_per_chunk
            mamba_align_size = tokens_per_chunk
    return mamba_align_size


class SchedulerOffloadConfig(NamedTuple):
    kv_group_configs: tuple[GroupOffloadConfig, ...]
    blocks_per_chunk: int
    num_workers: int
    offload_prompt_only: bool
    # Store only the trailing sliding-window worth of prompt chunks for
    # true-SWA groups without a real alignment constraint (see
    # OffloadingSpec.swa_store_prompt_tail_only).
    swa_prompt_tail_only: bool = True

    @classmethod
    def from_spec(
        cls,
        spec: OffloadingSpec,
        aphrodite_config: AphroditeConfig,
        kv_cache_config: KVCacheConfig,
    ) -> "SchedulerOffloadConfig":
        # EAGLE/MTP draft groups, annotated upstream in get_kv_cache_groups.
        # If annotation was unavailable (multi-process executor) mark all groups.
        eagle_groups = {idx for idx, g in enumerate(kv_cache_config.kv_cache_groups) if g.is_eagle_group}

        use_eagle = aphrodite_config.speculative_config is not None and aphrodite_config.speculative_config.use_eagle()
        if use_eagle and not eagle_groups:
            logger.warning(
                "KV offloading: EAGLE/MTP enabled but no draft group annotated; marking all %d groups volatile.",
                len(kv_cache_config.kv_cache_groups),
            )
            eagle_groups = set(range(len(kv_cache_config.kv_cache_groups)))

        # Alignment token count = tokens_per_chunk of the full-attention
        # group(s); load hits align to it so earlier SWA blocks can't serve a
        # hit (DeepSeek V4-style MLA + SWA). Eagle groups are excluded: a draft
        # full-attention chunk differs in size and would make this ambiguous.
        full_attn_tokens_per_chunk: set[int] = set()
        for idx, tokens_per_block in enumerate(spec.tokens_per_block):
            if idx in eagle_groups:
                continue
            kv_spec = kv_cache_config.kv_cache_groups[idx].kv_cache_spec
            sw = get_sliding_window_size_in_chunks(kv_spec, tokens_per_block * spec.blocks_per_chunk)
            if sw is None:
                full_attn_tokens_per_chunk.add(tokens_per_block * spec.blocks_per_chunk)

        # Only apply the optimization if there's a single consistent
        # full-attention alignment size.
        alignment_tokens: int | None = None
        if len(full_attn_tokens_per_chunk) == 1:
            alignment_tokens = full_attn_tokens_per_chunk.pop()

        def _alignment_chunk_count(
            tokens_per_chunk: int,
            sliding_window_size_in_chunks: int | None,
            is_true_swa: bool,
        ) -> tuple[int | None, bool]:
            """Return (alignment_chunk_count, is_synthetic)."""
            if sliding_window_size_in_chunks is None:
                return None, False
            if alignment_tokens is not None and alignment_tokens > tokens_per_chunk:
                per_segment = alignment_tokens // tokens_per_chunk
                if sliding_window_size_in_chunks < per_segment:
                    return per_segment, False
            # No real full-attention alignment (e.g. all groups share one
            # block size, as in Gemma-style hybrids). Fall back to the
            # opt-in synthetic segment size so mid-prompt SWA hits survive
            # at that granularity.
            if is_true_swa and spec.swa_store_alignment_chunks > sliding_window_size_in_chunks:
                return spec.swa_store_alignment_chunks, True
            return None, False

        if eagle_groups:
            logger.info(
                "KV offloading: EAGLE/MTP draft attention groups %s "
                "detected. The trailing block of these groups will be "
                "excluded from offloading due to volatility.",
                sorted(eagle_groups),
            )

        def _build_group_config(idx: int, tokens_per_block: int) -> GroupOffloadConfig:
            tokens_per_chunk = tokens_per_block * spec.blocks_per_chunk
            kv_spec = kv_cache_config.kv_cache_groups[idx].kv_cache_spec
            sw = get_sliding_window_size_in_chunks(kv_spec, tokens_per_chunk)
            is_true_swa = isinstance(kv_spec, SlidingWindowSpec) and idx not in eagle_groups
            alignment_chunk_count, alignment_is_synthetic = _alignment_chunk_count(tokens_per_chunk, sw, is_true_swa)
            return GroupOffloadConfig(
                group_idx=idx,
                tokens_per_block=tokens_per_block,
                tokens_per_chunk=tokens_per_chunk,
                hashes_per_chunk=tokens_per_chunk // spec.tokens_per_hash,
                sliding_window_size_in_chunks=sw,
                alignment_chunk_count=alignment_chunk_count,
                alignment_is_synthetic=alignment_is_synthetic,
                kv_event_group_spec=get_offloading_event_group_spec(kv_cache_config.kv_cache_groups[idx]),
                is_eagle_group=idx in eagle_groups,
                is_true_swa=is_true_swa,
            )

        return cls(
            num_workers=aphrodite_config.parallel_config.world_size,
            kv_group_configs=tuple(
                _build_group_config(idx, tokens_per_block)
                for idx, tokens_per_block in enumerate(spec.tokens_per_block)
            ),
            blocks_per_chunk=spec.blocks_per_chunk,
            offload_prompt_only=spec.offload_prompt_only,
            swa_prompt_tail_only=spec.swa_store_prompt_tail_only,
        )


@dataclass
class RequestGroupState:
    offload_keys: list[OffloadKey] = field(default_factory=list)
    block_ids: list[int] = field(default_factory=list)
    # index of next block (of size tokens_per_chunk) to offload
    next_stored_chunk_idx: int = 0
    # number of offloaded blocks hit (including GPU prefix cache)
    # when the request first started
    num_hit_chunks: int = 0


@dataclass(slots=True)
class RequestOffloadState:
    config: SchedulerOffloadConfig
    req: Request
    req_context: ReqContext
    offloading_context: RequestOffloadingContext
    group_states: tuple[RequestGroupState, ...] = field(init=False)
    # upper bound on tokens to offload for this request; None means no cap
    max_offload_tokens: int | None = None
    # number of hits in the GPU cache
    num_locally_computed_tokens: int = 0
    # In-flight job IDs. Per the connector's invariant, at any given time
    # this contains either a single load job, or one or more store jobs.
    transfer_jobs: set[int] = field(default_factory=set)
    # time.monotonic() of this request's first deferred offload lookup;
    # None once consumed (observed) or while no lookup is pending.
    deferred_lookup_start_time: float | None = None

    def __post_init__(self) -> None:
        self.group_states = tuple(RequestGroupState() for _ in self.config.kv_group_configs)
        params = self.req.kv_transfer_params

        # NOTE: This field is experimental and subject to change in the future.
        raw = params.get("max_offload_tokens") if params else None
        if type(raw) is int and raw >= 0:
            self.max_offload_tokens = raw
            logger.debug(
                "Request %s: max_offload_tokens set to %d",
                self.req.request_id,
                raw,
            )
        elif raw is not None:
            logger.warning("max_offload_tokens must be a non-negative int, got %r; ignoring", raw)

    def update_offload_keys(self) -> None:
        for group_config, group_state in zip(self.config.kv_group_configs, self.group_states):
            for req_block_hash in islice(
                self.req.block_hashes,
                group_config.hashes_per_chunk * len(group_state.offload_keys) + group_config.hashes_per_chunk - 1,
                None,
                group_config.hashes_per_chunk,
            ):
                group_state.offload_keys.append(make_offload_key(req_block_hash, group_config.group_idx))

    def update_block_id_groups(self, new_block_id_groups: tuple[list[int], ...] | None) -> None:
        if new_block_id_groups is None:
            return

        assert len(new_block_id_groups) == len(self.group_states)
        for group_state, new_blocks in zip(self.group_states, new_block_id_groups):
            group_state.block_ids.extend(new_blocks)

    def storable_chunks(self, group_config: "GroupOffloadConfig", num_offloadable_tokens: int) -> int:
        """Number of leading offloaded blocks eligible for store.

        For eagle/MTP groups the volatile trailing block of the offloadable
        range is excluded while decoding: the draft-layer KV of the last
        accepted position may be rewritten after spec-token rejection. During
        prefill the trailing block is stable (the draft input for a chunk's
        last position is the next prompt token), so it is stored immediately.
        The exclusion must be applied consistently everywhere
        ``next_stored_chunk_idx`` is derived: otherwise the trailing block of
        each step is skipped on collection but jumped over by
        ``next_stored_chunk_idx``, so it is never re-considered and a
        permanent hole breaks prefix-reuse lookup.
        """
        num_blocks = num_offloadable_tokens // group_config.tokens_per_chunk
        is_decoding = num_offloadable_tokens > self.req.num_prompt_tokens
        if group_config.is_eagle_group and is_decoding:
            num_blocks = max(0, num_blocks - 1)
        return num_blocks

    def advance_stored_idx(self, num_offloadable_tokens: int) -> None:
        # max(): at the prefill->decode transition of a block-aligned prompt,
        # storable_chunks drops by one (the eagle exclusion kicks in), and the
        # index must not move backwards past already-stored blocks.
        for group_config, group_state in zip(self.config.kv_group_configs, self.group_states):
            group_state.next_stored_chunk_idx = max(
                group_state.next_stored_chunk_idx,
                self.storable_chunks(group_config, num_offloadable_tokens),
            )

    def update_num_hit_chunks(self, num_cached_tokens: int) -> None:
        for group_config, group_state in zip(self.config.kv_group_configs, self.group_states):
            group_state.num_hit_chunks = num_cached_tokens // group_config.tokens_per_chunk


def _create_req_context(req: Request) -> ReqContext:
    return ReqContext(
        req_id=req.request_id,
        kv_transfer_params=req.kv_transfer_params,
    )


class OffloadingConnectorScheduler:
    """Implementation of Scheduler side methods"""

    def __init__(
        self,
        spec: OffloadingSpec,
        aphrodite_config: AphroditeConfig,
        kv_cache_config: KVCacheConfig,
    ):
        self.config = SchedulerOffloadConfig.from_spec(spec, aphrodite_config, kv_cache_config)
        self.manager: OffloadingManager = spec.get_manager()
        self._connector_stats = OffloadingConnectorStats()

        full_attention_groups: list[int] = []
        sliding_window_groups: list[int] = []
        for group_config in self.config.kv_group_configs:
            if group_config.sliding_window_size_in_chunks is None:
                full_attention_groups.append(group_config.group_idx)
            else:
                sliding_window_groups.append(group_config.group_idx)

        # sort sliding window groups by window size in decreasing order
        def _sliding_window_sort_key(i: int) -> int:
            val = self.config.kv_group_configs[i].sliding_window_size_in_chunks
            assert val is not None
            return val

        sliding_window_groups.sort(key=_sliding_window_sort_key, reverse=True)

        # used by _lookup
        self._sliding_window_groups: tuple[int, ...] = tuple(sliding_window_groups)
        self._lookup_groups = tuple(full_attention_groups) + self._sliding_window_groups
        self._mamba_align_size: int | None = resolve_mamba_align_size(spec, kv_cache_config)

        self._req_status: dict[ReqId, RequestOffloadState] = {}
        self._current_batch_load_jobs: dict[int, TransferJob] = {}
        self._current_batch_jobs_to_flush: set[int] = set()
        # GPU block IDs allocated in the current engine step
        self._current_batch_allocated_block_ids: set[int] = set()
        # if GPU prefix caching is enabled,
        # track loaded blocks to avoid redundant loads
        self._chunks_being_loaded: set[OffloadKey] | None = (
            set() if aphrodite_config.cache_config.enable_prefix_caching else None
        )

        # Job ID counter shared by loads and stores.
        self._job_counter: int = 0
        # Threshold value for stale jobs. All job ids >= _stale_job_threshold are
        # active jobs.
        self._stale_job_threshold: int = 0
        self._jobs: dict[int, TransferJobStatus] = {}

        # block_id -> pending store job_ids. Used to track jobs that needs
        # flushing in case a block is re-allocated by the KV cache manager.
        # Populated only for finished requests (running-request blocks are
        # protected by their ref_cnt) and for sliding window blocks (which can
        # be freed before a request finishes).
        self._block_id_to_pending_jobs: dict[int, set[int]] = {}

        self._events_tracker = OffloadingEventsTracker(spec.kv_events_config)

        # Offload-on-eviction (write-back victim buffer). When enabled, the CPU
        # tier is populated by spilling the GPU prefix cache's LRU tail under
        # pressure instead of mirroring prompt blocks at compute time, so CPU
        # capacity is additive to GPU rather than a subset of it.
        self._writeback_offload: bool = bool(spec.extra_config.get("writeback_offload", False))
        if self._writeback_offload and self.config.blocks_per_chunk != 1:
            raise ValueError(
                "writeback_offload requires blocks_per_chunk == 1 (one GPU block per "
                f"offload chunk); got {self.config.blocks_per_chunk}. Unset 'block_size'/"
                "'blocks_per_chunk' in kv_connector_extra_config."
            )
        # Max GPU blocks written back per engine step (bounds copy bandwidth and
        # how many blocks are pinned out of the free pool at once).
        self._writeback_max_per_step: int = int(spec.extra_config.get("writeback_max_per_step", 1024))
        # The spiller must stay ahead of the allocator: it runs after allocation
        # each step, so the free-block headroom below the watermark must exceed
        # what a single step can allocate plus the in-flight (pinned) spill
        # batch. Headroom is sized adaptively as
        #   max_per_step + factor * (peak blocks allocated in one step)
        # which self-tunes to the batch budget and the model's group structure
        # (via the observed per-step allocation) rather than a fixed fraction of
        # total capacity. An explicit block count overrides the adaptive value.
        self._writeback_headroom_factor: float = float(
            spec.extra_config.get("writeback_headroom_factor", 2.0)
        )
        self._writeback_headroom_blocks_override: int | None = (
            int(spec.extra_config["writeback_headroom_blocks"])
            if "writeback_headroom_blocks" in spec.extra_config
            else None
        )
        # Safeguards against adaptive headroom misbehaving under bursts:
        #  - floor the watermark so a burst can never vent more than a bounded
        #    fraction of the GPU cache (keeps a minimum working set resident);
        #  - decay the observed peak so a transient burst's inflated headroom
        #    fades instead of pinning the watermark low forever;
        #  - cap total in-flight (pinned, copy-pending) blocks so the spiller
        #    can never starve a live request's allocation.
        self._writeback_min_watermark_frac: float = float(
            spec.extra_config.get("writeback_min_watermark_frac", 0.5)
        )
        self._writeback_peak_decay: float = float(spec.extra_config.get("writeback_peak_decay", 0.98))
        self._writeback_max_inflight_blocks: int = int(
            spec.extra_config.get("writeback_max_inflight_blocks", 2 * self._writeback_max_per_step)
        )
        self._gpu_block_pool: BlockPool | None = None
        self._writeback_num_gpu_blocks: int = 0
        # Decaying peak of blocks allocated in a single engine step (drives adaptive headroom).
        self._writeback_peak_step_alloc: int = 0
        # Blocks currently reserved with a copy in flight (bounds pin pressure).
        self._writeback_inflight_blocks: int = 0
        # Diagnostics.
        self._writeback_stored_total: int = 0
        self._writeback_last_log: float = 0.0
        # Shared context for request-decoupled write-back stores.
        self._writeback_ctx: ReqContext = ReqContext(req_id="__writeback__", kv_transfer_params=None)

    def bind_gpu_block_pool(self, gpu_block_pool: BlockPool) -> None:
        # Retained for the offload-on-eviction store path, which reserves
        # eviction-bound cached blocks from the pool for write-back.
        self._gpu_block_pool = gpu_block_pool
        self._writeback_num_gpu_blocks = gpu_block_pool.num_gpu_blocks
        if self._writeback_offload:
            logger.info(
                "writeback_offload enabled: num_gpu_blocks=%d headroom=%s max_per_step=%d",
                gpu_block_pool.num_gpu_blocks,
                self._writeback_headroom_blocks_override
                if self._writeback_headroom_blocks_override is not None
                else f"adaptive(factor={self._writeback_headroom_factor})",
                self._writeback_max_per_step,
            )

    def _writeback_watermark(self) -> int:
        """Cached-block count above which the LRU tail is spilled to CPU.

        Leaves enough free headroom that the (post-allocation) spiller stays
        ahead of the allocator: one step's allocation plus the in-flight spill
        batch. Adaptive unless an explicit block count is configured.
        """
        if self._writeback_headroom_blocks_override is not None:
            headroom = self._writeback_headroom_blocks_override
        else:
            headroom = self._writeback_max_per_step + int(
                self._writeback_headroom_factor * self._writeback_peak_step_alloc
            )
        watermark = self._writeback_num_gpu_blocks - headroom
        # Floor: never spill the GPU cache below this working-set minimum.
        floor = int(self._writeback_num_gpu_blocks * self._writeback_min_watermark_frac)
        return max(floor, watermark, 0)

    def _maybe_observe_lookup_async_delay(self, req_status: RequestOffloadState) -> None:
        start_time = req_status.deferred_lookup_start_time
        if start_time is None:
            return
        req_status.deferred_lookup_start_time = None
        self._connector_stats.observe_histogram(
            _ConnectorMetricName.LOOKUP_ASYNC_DELAY,
            time.monotonic() - start_time,
        )

    def _generate_job_id(self) -> int:
        job_id = self._job_counter
        self._job_counter += 1
        return job_id

    def _remove_pending_job(self, job_id: int, block_ids: list[int] | None) -> None:
        for bid in block_ids or ():
            pending = self._block_id_to_pending_jobs[bid]
            pending.remove(job_id)
            if not pending:
                del self._block_id_to_pending_jobs[bid]

    def _maximal_prefix_lookup(self, keys: Iterable[OffloadKey], req_context: ReqContext) -> int | None:
        """Return the number of consecutive offloaded blocks from the start,
        or None if the backend deferred a lookup."""
        hit_count = 0
        defer_lookup = False
        for key in keys:
            match self.manager.lookup(key, req_context):
                case LookupResult.HIT:
                    hit_count += 1
                case LookupResult.HIT_PENDING:
                    defer_lookup = True
                    hit_count += 1
                case LookupResult.RETRY:
                    # Don't break: keep scanning to let manager kick off
                    # async lookups (until a miss is detected).
                    defer_lookup = True
                case LookupResult.MISS:
                    break
        return hit_count if not defer_lookup else None

    def _sliding_window_lookup(
        self,
        keys: Sequence[OffloadKey],
        sliding_window_size: int,
        req_context: ReqContext,
    ) -> int | None:
        """Return the end index (in `keys`) of the last run of
        `sliding_window_size` consecutive hits, scanning from the end.
        Returns 0 on miss, None if the backend deferred a lookup."""
        defer_lookup = False
        consecutive_hits = 0
        for idx in range(len(keys) - 1, -1, -1):
            match self.manager.lookup(keys[idx], req_context):
                case LookupResult.HIT:
                    consecutive_hits += 1
                case LookupResult.HIT_PENDING:
                    # Block is in cache, just not readable yet — counts
                    # as hit for the consecutive streak. Don't break:
                    # keep scanning to let manager kick off async lookups.
                    defer_lookup = True
                    consecutive_hits += 1
                case LookupResult.RETRY:
                    # Block location uncertain — does not count as hit.
                    # Don't break: keep scanning to let manager kick off
                    # async lookups.
                    defer_lookup = True
                    consecutive_hits = 0
                case LookupResult.MISS:
                    consecutive_hits = 0
            if consecutive_hits == sliding_window_size:
                return idx + sliding_window_size if not defer_lookup else None
        return consecutive_hits if not defer_lookup else None

    def _touch(self, req_status: RequestOffloadState):
        for group_config, group_state in zip(self.config.kv_group_configs, req_status.group_states):
            if group_config.sliding_window_size_in_chunks is None:
                self.manager.touch(group_state.offload_keys, req_status.req_context)
            else:
                # we aim to keep just blocks that are necessary to hit
                # the original request (+ decoded blocks)
                blocks_to_skip = max(
                    0,
                    group_state.num_hit_chunks - group_config.sliding_window_size_in_chunks,
                )
                self.manager.touch(
                    group_state.offload_keys[blocks_to_skip:],
                    req_status.req_context,
                )

    def _lookup(self, req_status: RequestOffloadState) -> int | None:
        """
        Find how many tokens beyond num_locally_computed_tokens can be loaded.

        Iterates full-attention groups first (prefix lookup), then sliding-window
        groups (suffix lookup). Each group may tighten max_hit_size_tokens, which
        can invalidate an earlier group's result, so the loop re-runs when that
        happens until num_hit_tokens converges.
        """
        num_computed_tokens = req_status.num_locally_computed_tokens
        max_hit_size_tokens: int = req_status.req.num_tokens
        if self._sliding_window_groups:
            # the last prompt token has to be recomputed to get the logprobs
            # for sliding window attention, we must reduce by 1 to make sure
            # we still have a hit after reduction
            max_hit_size_tokens -= 1
            if self._mamba_align_size is not None:
                # Constrain hit-window to the mamba block size.
                max_hit_size_tokens = round_down(max_hit_size_tokens, self._mamba_align_size)

        num_hit_tokens: int = 0
        defer_lookup = False
        lookup_groups = self._lookup_groups

        # Tracks which eagle groups have already popped their volatile trailing block
        # in the current convergence iteration. Reset when a non-eagle group
        # tightens the hit boundary, requiring a fresh pop.
        eagle_verified: set[int] = set()
        while lookup_groups:
            looked_up_sliding_window: bool = False
            groups_iter = iter(lookup_groups)
            lookup_groups = ()
            for group_idx in groups_iter:
                group_config: GroupOffloadConfig = self.config.kv_group_configs[group_idx]
                group_state: RequestGroupState = req_status.group_states[group_idx]
                tokens_per_chunk = group_config.tokens_per_chunk
                offload_keys = group_state.offload_keys

                assert len(offload_keys) >= req_status.req.num_tokens // tokens_per_chunk

                is_eagle_unverified = group_config.is_eagle_group and group_idx not in eagle_verified

                # Constrain to block-aligned boundary for this group
                max_hit_size_tokens = min(max_hit_size_tokens, len(offload_keys) * tokens_per_chunk)
                if max_hit_size_tokens - num_computed_tokens < tokens_per_chunk:
                    # we can only load less than a block, better skip
                    return 0

                sliding_window_size_in_chunks = group_config.sliding_window_size_in_chunks

                # For eagle groups, query one extra block that will be popped.
                # We only need to increase the query size for sliding window groups.
                query_max = max_hit_size_tokens
                if is_eagle_unverified and sliding_window_size_in_chunks is not None:
                    query_max = min(
                        max_hit_size_tokens + tokens_per_chunk,
                        len(offload_keys) * tokens_per_chunk,
                    )

                num_blocks = min(cdiv(query_max, tokens_per_chunk), len(offload_keys))
                start_block_idx = num_computed_tokens // tokens_per_chunk
                offload_keys = offload_keys[start_block_idx:num_blocks]

                # end index (in the sliced offload_keys) up to which we
                # have backend-confirmed hits
                num_hit_chunks: int | None
                if sliding_window_size_in_chunks is None:
                    num_hit_chunks = self._maximal_prefix_lookup(offload_keys, req_status.req_context)
                else:
                    required_window = sliding_window_size_in_chunks
                    if is_eagle_unverified:
                        required_window += 1
                    num_hit_chunks = self._sliding_window_lookup(
                        offload_keys,
                        required_window,
                        req_status.req_context,
                    )
                if num_hit_chunks == 0:
                    return 0

                if num_hit_chunks is None:
                    defer_lookup = True
                else:
                    if is_eagle_unverified:
                        num_hit_chunks -= 1
                        eagle_verified.add(group_idx)

                    max_hit_size_tokens = min(
                        max_hit_size_tokens,
                        tokens_per_chunk * (start_block_idx + num_hit_chunks),
                    )

                new_num_hit_tokens = max_hit_size_tokens - num_computed_tokens
                if new_num_hit_tokens < tokens_per_chunk:
                    # we can only load less than a block, better skip
                    return 0

                if new_num_hit_tokens < num_hit_tokens:
                    if not group_config.is_eagle_group:
                        eagle_verified.clear()
                    if defer_lookup:
                        # make another iteration on all groups to check
                        # if we still need to defer lookup
                        defer_lookup = False
                        lookup_groups = self._lookup_groups
                    elif looked_up_sliding_window and not lookup_groups:
                        # we need another iteration to confirm previously looked up
                        # sliding window works with the new_num_hit_tokens
                        lookup_groups = self._sliding_window_groups

                looked_up_sliding_window |= sliding_window_size_in_chunks is not None
                num_hit_tokens = new_num_hit_tokens

        if defer_lookup:
            logger.debug(
                "Offloading manager delayed request %s as backend requested",
                req_status.req.request_id,
            )
            return None

        # possibly delay request if any of the hit blocks is already being loaded
        if self._chunks_being_loaded:
            for group_config, group_state in zip(self.config.kv_group_configs, req_status.group_states):
                tokens_per_chunk = group_config.tokens_per_chunk
                sliding_window_size_in_chunks = group_config.sliding_window_size_in_chunks
                offload_keys = group_state.offload_keys
                num_blocks = cdiv(num_computed_tokens + num_hit_tokens, tokens_per_chunk)
                start_block_idx = num_computed_tokens // tokens_per_chunk
                offload_keys = offload_keys[start_block_idx:num_blocks]
                if sliding_window_size_in_chunks is not None:
                    offload_keys = offload_keys[-sliding_window_size_in_chunks:]
                if any(key in self._chunks_being_loaded for key in offload_keys):
                    # hit blocks are being loaded, delay request
                    logger.debug(
                        "Delaying request %s since some of its blocks are already being loaded",
                        req_status.req.request_id,
                    )
                    return None

        logger.debug(
            "Request %s hit %s offloaded tokens after %s GPU hit tokens",
            req_status.req.request_id,
            num_hit_tokens,
            num_computed_tokens,
        )

        return num_hit_tokens

    def on_new_request(self, request: Request) -> None:
        """Called when a new request is added to the scheduler."""
        req_context = _create_req_context(request)
        offloading_context = self.manager.on_new_request(req_context)
        req_status = RequestOffloadState(
            config=self.config,
            req=request,
            req_context=req_context,
            offloading_context=offloading_context,
        )
        self._req_status[request.request_id] = req_status

    def get_num_new_matched_tokens(self, request: Request, num_computed_tokens: int) -> tuple[int | None, bool]:
        """
        Get number of new tokens that can be loaded beyond the
        num_computed_tokens.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request

        Returns:
            A tuple with the following elements:
                - The number of tokens that can be loaded beyond what is
                  already computed.
                  If None, it means that the connector needs more time to
                  determine the number of matched tokens, and the scheduler
                  should query for this request again later.
                - `True` if tokens will be loaded asynchronously
                  (between scheduler steps).
        """
        req_status = self._req_status[request.request_id]
        for group_state in req_status.group_states:
            group_state.block_ids.clear()

        if req_status.transfer_jobs:
            logger.debug(
                "Delaying request %s since it still has in-flight transfers",
                request.request_id,
            )
            return None, False

        req_status.update_offload_keys()
        req_status.num_locally_computed_tokens = num_computed_tokens

        num_hit_tokens: int | None
        if request.skip_reading_prefix_cache:
            num_hit_tokens = 0
        else:
            lookup_start = time.monotonic()
            num_hit_tokens = self._lookup(req_status)
            self._connector_stats.observe_histogram(
                _ConnectorMetricName.LOOKUP_SYNC_DELAY,
                time.monotonic() - lookup_start,
            )
            if num_hit_tokens is None:
                if req_status.deferred_lookup_start_time is None:
                    req_status.deferred_lookup_start_time = lookup_start
            else:
                self._maybe_observe_lookup_async_delay(req_status)
        req_status.update_num_hit_chunks(num_computed_tokens + (num_hit_tokens or 0))

        self._touch(req_status)

        return num_hit_tokens, bool(num_hit_tokens)

    def update_state_after_alloc(self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int):
        if num_external_tokens == 0:
            return

        req_status = self._req_status[request.request_id]

        num_locally_computed_tokens = req_status.num_locally_computed_tokens
        num_cached_tokens = num_locally_computed_tokens + num_external_tokens

        keys_to_load: list[OffloadKey] = []
        dst_block_ids: list[int] = []
        # per group
        group_sizes: list[int] = []
        block_indices: list[int] = []
        for group_config, group_state, group_blocks in zip(
            self.config.kv_group_configs,
            req_status.group_states,
            blocks.blocks,
        ):
            self._current_batch_allocated_block_ids.update(
                block.block_id for block in group_blocks if block.block_id != 0
            )

            tokens_per_block = group_config.tokens_per_block
            tokens_per_chunk = group_config.tokens_per_chunk
            offload_keys = group_state.offload_keys
            num_gpu_blocks = cdiv(num_cached_tokens, tokens_per_block)

            assert len(group_blocks) >= num_gpu_blocks
            num_locally_computed_gpu_blocks = num_gpu_blocks
            # Skip null placeholder blocks (used for sliding window or mamba padding).
            for i, block in enumerate(group_blocks[:num_gpu_blocks]):
                if not block.is_null and block.block_hash is None:
                    num_locally_computed_gpu_blocks = i
                    break

            assert num_locally_computed_tokens <= num_locally_computed_gpu_blocks * tokens_per_block
            num_pending_gpu_blocks = num_gpu_blocks - num_locally_computed_gpu_blocks

            if group_config.sliding_window_size_in_chunks is not None:
                assert (
                    num_pending_gpu_blocks <= group_config.sliding_window_size_in_chunks * self.config.blocks_per_chunk
                )

            num_blocks = cdiv(num_cached_tokens, tokens_per_chunk)
            assert len(offload_keys) >= num_blocks
            if num_pending_gpu_blocks:
                start_block_idx = num_locally_computed_gpu_blocks // self.config.blocks_per_chunk
                keys_to_load.extend(offload_keys[start_block_idx:num_blocks])

            dst_block_ids.extend(
                block.block_id for block in group_blocks[num_locally_computed_gpu_blocks:num_gpu_blocks]
            )
            group_sizes.append(num_pending_gpu_blocks)
            block_indices.append(num_locally_computed_gpu_blocks)

            # Skip prefix-hit blocks for block-level policy; for
            # request-level, next_stored_chunk_idx stays at 0 so all
            # blocks (including hits) are offloaded.
            if req_status.offloading_context.policy == OffloadPolicy.BLOCK_LEVEL:
                group_state.next_stored_chunk_idx = num_blocks

        src_spec = self.manager.prepare_load(keys_to_load, req_status.req_context)
        dst_spec = GPULoadStoreSpec(dst_block_ids, group_sizes=group_sizes, block_indices=block_indices)

        load_job_id = self._generate_job_id()
        self._current_batch_load_jobs[load_job_id] = TransferJob(
            req_id=request.request_id,
            src_spec=src_spec,
            dst_spec=dst_spec,
        )
        # a load can only be issued when no other jobs are pending.
        assert not req_status.transfer_jobs
        req_status.transfer_jobs.add(load_job_id)
        self._jobs[load_job_id] = TransferJobStatus(
            req_id=request.request_id,
            pending_count=self.config.num_workers,
            keys=set(keys_to_load),
            is_store=False,
        )

        if self._chunks_being_loaded is not None:
            self._chunks_being_loaded.update(keys_to_load)

    def _update_req_states(self, scheduler_output: SchedulerOutput) -> None:
        """
        Update request states from the Scheduler's output.
        """

        # new_block_ids_end[req_id][i] = end of pre-existing block_ids for
        # the i-th sliding window group (before this step's extend).
        # Used to detect sliding window blocks that got re-allocated.
        new_block_ids_end: dict[str, tuple[int, ...]] = {}

        for req_id, new_block_id_groups, preempted in yield_req_data(scheduler_output):
            req_status = self._req_status[req_id]
            req_status.update_offload_keys()

            if preempted:
                for group_state in req_status.group_states:
                    group_state.block_ids.clear()

            if new_block_id_groups:
                if self._sliding_window_groups:
                    new_block_ids_end[req_id] = tuple(
                        len(req_status.group_states[grp_idx].block_ids) for grp_idx in self._sliding_window_groups
                    )
                req_status.update_block_id_groups(new_block_id_groups)
                for new_blocks in new_block_id_groups:
                    for bid in new_blocks:
                        if bid != 0:
                            self._current_batch_allocated_block_ids.add(bid)

        # Zero out stale block_ids in sliding window groups' pending-store
        # positions. Only sliding window groups can have stale entries (blocks
        # freed by remove_skipped_blocks then reallocated). Only positions in
        # [next_stored_chunk_idx * bsf, end) need checking where end is the
        # pre-extend length: earlier positions were already offloaded, later
        # ones are fresh allocations from this step.
        if self._sliding_window_groups and self._current_batch_allocated_block_ids:
            blocks_per_chunk = self.config.blocks_per_chunk
            for req_id, req_status in self._req_status.items():
                ends = new_block_ids_end.get(req_id)
                for i, grp_idx in enumerate(self._sliding_window_groups):
                    group_state = req_status.group_states[grp_idx]
                    start = group_state.next_stored_chunk_idx * blocks_per_chunk
                    end = ends[i] if ends is not None else len(group_state.block_ids)
                    for j in range(start, end):
                        if group_state.block_ids[j] in self._current_batch_allocated_block_ids:
                            group_state.block_ids[j] = 0

    def _build_writeback_store_jobs(self) -> dict[int, TransferJob]:
        """Offload-on-eviction: spill the GPU prefix cache's LRU tail to CPU.

        Only runs under cache pressure (cached blocks above the high
        watermark). Reserves the eviction-soonest cached blocks (pinning their
        KV data), copies them GPU->CPU, and on completion evicts them from the
        GPU cache so their content lives only on the CPU tier. This keeps the
        GPU holding the recent working set while the CPU accumulates the
        overflow, making capacity additive.
        """
        pool = self._gpu_block_pool
        if pool is None:
            return {}

        num_cached = pool.num_cached_blocks()
        watermark = self._writeback_watermark()
        over = num_cached - watermark
        now = time.monotonic()
        if now - self._writeback_last_log >= 2.0:
            self._writeback_last_log = now
            logger.info(
                "writeback: num_cached=%d watermark=%d (peak_step_alloc=%d) over=%d stored_total=%d",
                num_cached,
                watermark,
                self._writeback_peak_step_alloc,
                over,
                self._writeback_stored_total,
            )
        if over <= 0:
            return {}
        # Cap total pinned, copy-pending blocks so the spiller can never starve
        # a live request's allocation, regardless of copy latency.
        inflight_budget = self._writeback_max_inflight_blocks - self._writeback_inflight_blocks
        if inflight_budget <= 0:
            return {}
        num_to_reserve = min(over, self._writeback_max_per_step, inflight_budget)
        victims = pool.reserve_writeback_victims(num_to_reserve)
        if not victims:
            return {}

        num_groups = len(self.config.kv_group_configs)
        # (block_id, offload_key) per group, in group order.
        per_group: list[list[tuple[int, OffloadKey]]] = [[] for _ in range(num_groups)]
        for block_id, bhg in victims:
            gid = get_group_id(bhg)
            key = make_offload_key(get_block_hash(bhg), gid)
            per_group[gid].append((block_id, key))

        # Flatten in ascending group order (the order the worker expects).
        ordered: list[tuple[int, OffloadKey, int]] = [
            (block_id, key, gid) for gid in range(num_groups) for (block_id, key) in per_group[gid]
        ]

        store_output = self.manager.prepare_store([key for _, key, _ in ordered], self._writeback_ctx)
        if store_output is None:
            # CPU tier could not make room: return the victims unchanged; they
            # stay GPU-cached eviction candidates and are retried next step.
            for block_id, _ in victims:
                pool.release_writeback_victim(block_id, stored=False)
            return {}

        keys_to_store = set(store_output.keys_to_store)

        src_block_ids: list[int] = []
        group_sizes = [0] * num_groups
        job_block_ids: list[int] = []
        for block_id, key, gid in ordered:
            if key not in keys_to_store:
                # Already resident on the CPU tier (dedup): the content is safe,
                # so evict it from the GPU cache and free the block now.
                pool.release_writeback_victim(block_id, stored=True)
                continue
            src_block_ids.append(block_id)
            group_sizes[gid] += 1
            job_block_ids.append(block_id)

        if not job_block_ids:
            return {}

        self._writeback_stored_total += len(job_block_ids)
        self._writeback_inflight_blocks += len(job_block_ids)
        logger.info(
            "writeback: reserved=%d storing=%d (already_on_cpu=%d) num_cached=%d inflight=%d",
            len(victims),
            len(job_block_ids),
            len(victims) - len(job_block_ids),
            num_cached,
            self._writeback_inflight_blocks,
        )

        # blocks_per_chunk == 1 for the write-back path, so the worker's
        # partial-chunk skip (block_idx % blocks_per_chunk) is always 0.
        src_spec = GPULoadStoreSpec(src_block_ids, group_sizes=group_sizes, block_indices=[0] * num_groups)

        job_id = self._generate_job_id()
        self._jobs[job_id] = TransferJobStatus(
            req_id=self._writeback_ctx.req_id,
            pending_count=self.config.num_workers,
            keys=set(store_output.keys_to_store),
            is_store=True,
            is_writeback=True,
            writeback_block_ids=job_block_ids,
        )
        return {job_id: TransferJob(req_id=self._writeback_ctx.req_id, src_spec=src_spec, dst_spec=store_output.store_spec)}

    def _build_store_jobs(
        self,
        scheduler_output: SchedulerOutput,
    ) -> dict[int, TransferJob]:
        blocks_per_chunk = self.config.blocks_per_chunk
        store_jobs: dict[int, TransferJob] = {}
        for req_id in scheduler_output.num_scheduled_tokens:
            req_status = self._req_status.get(req_id)
            if req_status is None:
                continue
            req = req_status.req

            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
            num_tokens_after_batch = req.num_computed_tokens + num_scheduled_tokens
            # with async scheduling, some tokens may be missing
            num_offloadable_tokens = min(num_tokens_after_batch, req.num_tokens)
            max_offload_tokens = req_status.max_offload_tokens
            if max_offload_tokens is not None:
                num_offloadable_tokens = min(num_offloadable_tokens, max_offload_tokens)

            # Skip decode-phase blocks: clamp to the prompt length so only
            # prefill (prompt) blocks become eligible for store. next_stored_idx
            # never advances past this boundary, so decode blocks are never
            # queued in this or any later step.
            if self.config.offload_prompt_only:
                num_offloadable_tokens = min(num_offloadable_tokens, req.num_prompt_tokens)

            # Filter out blocks skipped due to sliding window attention / SSM
            # or unreachable by the load path's alignment constraints.
            new_offload_keys: list[OffloadKey] = []
            for group_config, group_state in zip(self.config.kv_group_configs, req_status.group_states):
                num_blocks = req_status.storable_chunks(group_config, num_offloadable_tokens)

                start_block_idx = group_state.next_stored_chunk_idx
                if num_blocks <= start_block_idx:
                    continue
                offload_keys = group_state.offload_keys[start_block_idx:num_blocks]
                # For each block to offload, take the last corresponding GPU block.
                # e.g. if block size factor is 3 and GPU block IDs are
                # 1 5 6 7 2 4 9 3 8 then we'll take blocks 6 4 8.
                # A block_id of 0 means either a sliding window / SSM skip
                # or a stale entry that was zeroed out — skip it either way.
                offload_block_ids = group_state.block_ids[
                    start_block_idx * blocks_per_chunk + blocks_per_chunk - 1 : num_blocks
                    * blocks_per_chunk : blocks_per_chunk
                ]
                assert len(offload_keys) == len(offload_block_ids)

                alignment_chunk_count = group_config.alignment_chunk_count
                tail = group_config.sliding_window_size_in_chunks

                # Skip SWA prompt blocks outside the trailing window:
                # _sliding_window_lookup only ever uses the last `tail`
                # chunks before a matched prefix end, so for the dominant
                # re-send/multi-turn hit pattern only the prompt tail can
                # serve a load. Applies to true-SWA groups without a real
                # (load-path-constraining) alignment; decode-region chunks
                # (abs_block_idx >= prompt_chunks) always pass this gate.
                prompt_tail_start: int | None = None
                if (
                    group_config.is_true_swa
                    and self.config.swa_prompt_tail_only
                    and (alignment_chunk_count is None or group_config.alignment_is_synthetic)
                ):
                    assert tail is not None
                    effective_prompt = req.num_prompt_tokens
                    if max_offload_tokens is not None:
                        effective_prompt = min(effective_prompt, max_offload_tokens)
                    prompt_chunks = effective_prompt // group_config.tokens_per_chunk
                    prompt_tail_start = max(0, prompt_chunks - tail)

                for key_idx, (offload_key, block_id) in enumerate(zip(offload_keys, offload_block_ids)):
                    if block_id == 0:
                        continue
                    # Skip SWA blocks that can never serve a load hit:
                    # within each full-attention alignment segment, only the
                    # trailing `tail` blocks are reachable by
                    # _sliding_window_lookup. For DeepSeek V4 with 100K
                    # tokens this reduces SWA stores by ~78%. A chunk is
                    # stored if it passes any active gate (segment tail or
                    # prompt tail).
                    if alignment_chunk_count is not None or prompt_tail_start is not None:
                        abs_block_idx = start_block_idx + key_idx
                        stored = False
                        if alignment_chunk_count is not None:
                            assert tail is not None
                            pos_in_segment = abs_block_idx % alignment_chunk_count
                            stored = pos_in_segment >= alignment_chunk_count - tail
                        if not stored and prompt_tail_start is not None:
                            stored = abs_block_idx >= prompt_tail_start
                        if not stored:
                            continue
                    new_offload_keys.append(offload_key)

            if not new_offload_keys:
                req_status.advance_stored_idx(num_offloadable_tokens)
                continue

            store_output = self.manager.prepare_store(new_offload_keys, req_status.req_context)
            if store_output is None:
                self._connector_stats.increase_counter(_ConnectorMetricName.ALLOCATION_FAILURE)
                logger.warning("Request %s: cannot store blocks", req_id)
                continue

            if not store_output.keys_to_store:
                req_status.advance_stored_idx(num_offloadable_tokens)
                continue

            self._touch(req_status)

            keys_to_store = set(store_output.keys_to_store)

            group_sizes: list[int] = []
            block_indices: list[int] = []
            src_block_ids: list[int] = []
            sliding_window_block_ids: list[int] = []
            non_sliding_window_block_ids: list[int] = []
            for group_config, group_state in zip(self.config.kv_group_configs, req_status.group_states):
                is_sliding_window = group_config.sliding_window_size_in_chunks is not None
                num_blocks = req_status.storable_chunks(group_config, num_offloadable_tokens)
                start_block_idx = group_state.next_stored_chunk_idx
                block_ids = group_state.block_ids
                num_group_blocks = 0
                start_gpu_block_idx: int | None = None
                for idx, offload_key in enumerate(group_state.offload_keys[start_block_idx:num_blocks]):
                    if offload_key not in keys_to_store:
                        continue

                    chunk_idx = start_block_idx + idx

                    self._events_tracker.record_store(req, group_config, chunk_idx, offload_key)

                    gpu_block_idx = chunk_idx * blocks_per_chunk
                    for i in range(blocks_per_chunk):
                        block_id = block_ids[gpu_block_idx + i]
                        if block_id == 0:
                            continue
                        if start_gpu_block_idx is None:
                            start_gpu_block_idx = gpu_block_idx + i
                        src_block_ids.append(block_id)
                        num_group_blocks += 1
                        if is_sliding_window:
                            sliding_window_block_ids.append(block_id)
                        else:
                            non_sliding_window_block_ids.append(block_id)

                group_sizes.append(num_group_blocks)
                block_indices.append(start_gpu_block_idx or 0)
                group_state.next_stored_chunk_idx = max(group_state.next_stored_chunk_idx, num_blocks)

            src_spec = GPULoadStoreSpec(src_block_ids, group_sizes=group_sizes, block_indices=block_indices)
            dst_spec = store_output.store_spec

            job_id = self._generate_job_id()
            # a store can only be issued when no load is pending.
            if req_status.transfer_jobs:
                any_jid = next(iter(req_status.transfer_jobs))
                assert self._jobs[any_jid].is_store
            req_status.transfer_jobs.add(job_id)

            # Watch sliding window blocks as they may get evicted
            # before the request finishes
            for bid in sliding_window_block_ids or ():
                self._block_id_to_pending_jobs.setdefault(bid, set()).add(job_id)

            # the non-sliding window blocks will be watched only
            # when the request finishes
            self._jobs[job_id] = TransferJobStatus(
                req_id=req_id,
                pending_count=self.config.num_workers,
                keys=set(keys_to_store),
                is_store=True,
                non_sliding_window_block_ids=non_sliding_window_block_ids,
                sliding_window_block_ids=sliding_window_block_ids or None,
            )

            store_jobs[job_id] = TransferJob(req_id=req_id, src_spec=src_spec, dst_spec=dst_spec)

            logger.debug(
                "Request %s offloading %s blocks upto %d tokens (job %d)",
                req_id,
                len(keys_to_store),
                num_offloadable_tokens,
                job_id,
            )

        return store_jobs

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        self._update_req_states(scheduler_output)
        schedule_end_context = ScheduleEndContext(
            new_req_ids=[req.req_id for req in scheduler_output.scheduled_new_reqs],
            preempted_req_ids=scheduler_output.preempted_req_ids or (),
        )
        self.manager.on_schedule_end(schedule_end_context)

        # Flush jobs for preempted requests.
        for req_id in scheduler_output.preempted_req_ids or ():
            req_status = self._req_status.get(req_id)
            if req_status is None or not req_status.transfer_jobs:
                continue
            any_jid = next(iter(req_status.transfer_jobs))
            assert self._jobs[any_jid].is_store
            self._current_batch_jobs_to_flush.update(req_status.transfer_jobs)

        # Flush jobs that contain re-allocated blocks.
        if self._block_id_to_pending_jobs and not self._block_id_to_pending_jobs.keys().isdisjoint(
            self._current_batch_allocated_block_ids
        ):
            self._current_batch_jobs_to_flush.update(
                jid
                for bid in self._current_batch_allocated_block_ids
                if bid in self._block_id_to_pending_jobs
                for jid in self._block_id_to_pending_jobs[bid]
            )

        if self._writeback_offload:
            # Decaying peak of per-step allocation drives the adaptive headroom:
            # a burst raises it immediately; it fades over ~seconds so headroom
            # (and GPU retention) recovers instead of staying inflated forever.
            self._writeback_peak_step_alloc = max(
                len(self._current_batch_allocated_block_ids),
                int(self._writeback_peak_step_alloc * self._writeback_peak_decay),
            )
            store_jobs = self._build_writeback_store_jobs()
        else:
            store_jobs = self._build_store_jobs(scheduler_output)
        meta = OffloadingConnectorMetadata(
            load_jobs=self._current_batch_load_jobs,
            store_jobs=store_jobs,
            jobs_to_flush=self._current_batch_jobs_to_flush,
        )
        self._current_batch_load_jobs = {}
        self._current_batch_jobs_to_flush = set()
        self._current_batch_allocated_block_ids = set()
        return meta

    def has_pending_push_work(self) -> bool:
        """Whether the engine must keep stepping.

        While True, build_connector_meta() and update_connector_output()
        continue to be called even when no requests are scheduled.
        """
        return bool(self._jobs) or self.manager.has_pending_work()

    def update_connector_output(self, connector_output: KVConnectorOutput):
        """
        Update KVConnector state from worker-side connectors output.

        Args:
            connector_output (KVConnectorOutput): the worker-side
                connectors output.
        """
        meta = connector_output.kv_connector_worker_meta
        if not isinstance(meta, OffloadingWorkerMetadata):
            assert meta is None
            meta = OffloadingWorkerMetadata()
        if not meta.transfer_stats.is_empty():
            transfer_stats = OffloadingConnectorStats()
            if not meta.transfer_stats.load.is_empty():
                transfer_stats.increase_counter(
                    _TransferMetricName.LOAD_BYTES,
                    meta.transfer_stats.load.bytes,
                )
                transfer_stats.increase_counter(
                    _TransferMetricName.LOAD_TIME,
                    meta.transfer_stats.load.time,
                )
                for size in meta.transfer_stats.load.sizes:
                    transfer_stats.observe_histogram(_TransferMetricName.LOAD_SIZE, size)
            if not meta.transfer_stats.store.is_empty():
                transfer_stats.increase_counter(
                    _TransferMetricName.STORE_BYTES,
                    meta.transfer_stats.store.bytes,
                )
                transfer_stats.increase_counter(
                    _TransferMetricName.STORE_TIME,
                    meta.transfer_stats.store.time,
                )
                for size in meta.transfer_stats.store.sizes:
                    transfer_stats.observe_histogram(_TransferMetricName.STORE_SIZE, size)
            self._connector_stats.aggregate(transfer_stats)

        for job_id, count in meta.completed_jobs.items():
            assert count > 0
            if job_id < self._stale_job_threshold:
                logger.debug(
                    "Skipping stale completed job %d (pre-reset counter: %d)",
                    job_id,
                    self._stale_job_threshold,
                )
                continue
            job_status = self._jobs[job_id]
            job_status.pending_count -= count
            if job_status.pending_count > 0:
                continue
            assert job_status.pending_count == 0

            if job_status.is_writeback:
                # Request-decoupled write-back store: mark the CPU blocks
                # readable, then evict the source blocks from the GPU cache and
                # return them to the pool (content now lives on the CPU tier).
                self.manager.complete_store(job_status.keys, self._writeback_ctx)
                assert self._gpu_block_pool is not None
                block_ids = job_status.writeback_block_ids or ()
                for block_id in block_ids:
                    self._gpu_block_pool.release_writeback_victim(block_id, stored=True)
                self._writeback_inflight_blocks -= len(block_ids)
                del self._jobs[job_id]
                continue

            req_status = self._req_status[job_status.req_id]
            if job_status.is_store:
                self.manager.complete_store(job_status.keys, req_status.req_context)
            else:
                self.manager.complete_load(job_status.keys, req_status.req_context)
                if self._chunks_being_loaded:
                    self._chunks_being_loaded.difference_update(job_status.keys)
            if self._block_id_to_pending_jobs:
                # Sliding window blocks are tracked from store creation
                # and must be cleaned up unconditionally.
                self._remove_pending_job(job_id, job_status.sliding_window_block_ids)
                # Non-sliding-window blocks are only tracked after
                # request_finished, so only clean up for finished requests.
                if req_status.req.is_finished():
                    self._remove_pending_job(job_id, job_status.non_sliding_window_block_ids)

            del self._jobs[job_id]
            req_status.transfer_jobs.remove(job_id)
            if not req_status.transfer_jobs and req_status.req.is_finished():
                del self._req_status[job_status.req_id]

    def get_stats(self) -> OffloadingConnectorStats | None:
        stats: OffloadingConnectorStats | None = None
        if not self._connector_stats.is_empty():
            stats = self._connector_stats
            self._connector_stats = OffloadingConnectorStats()

        manager_stats = self.manager.get_stats()
        if manager_stats is not None:
            if stats is None:
                stats = manager_stats
            else:
                stats.aggregate(manager_stats)

        return stats

    def request_finished(
        self,
        request: Request,
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Called when a request has finished, before its blocks are freed.

        Returns:
            True if the request is being saved/sent asynchronously and blocks
            should not be freed until the request_id is returned from
            get_finished().
            Optional KVTransferParams to be included in the request outputs
            returned by the engine.
        """
        # TODO(orozery): possibly kickoff offload for last block
        # which may have been deferred due to async scheduling
        req_status = self._req_status.get(request.request_id)

        if req_status is None:
            # Untracked request (offloading never started): no in-flight jobs,
            # nothing was deferred, so finalize immediately.
            req_context = _create_req_context(request)
            self.manager.on_new_request(req_context)
            self.manager.on_request_finished(req_context)
            return False, None

        self.manager.on_request_finished(req_status.req_context)
        self._maybe_observe_lookup_async_delay(req_status)
        if not req_status.transfer_jobs:
            # No in-flight jobs: no later complete_store()/complete_load() calls
            # need this request's state.
            del self._req_status[request.request_id]
            return False, None

        # In-flight jobs remain after the request stopped. Their completion may
        # still call manager.complete_store()/complete_load(), so keep req_status.
        # Pending stores outlive the request's block ownership; register them so
        # future reuse of those blocks triggers a flush.
        for job_id in req_status.transfer_jobs:
            job_status = self._jobs[job_id]
            for bid in job_status.non_sliding_window_block_ids or ():
                self._block_id_to_pending_jobs.setdefault(bid, set()).add(job_id)
        return False, None

    def take_events(self) -> Iterable[KVCacheEvent]:
        """Drain pending KV cache events.

        Complete metadata is available only when self-describing KV events
        are enabled, and only for full-attention groups. Other shapes retain
        the previous placeholder payload so consumers can ignore them.

        Yields:
            ``BlockStored`` or ``BlockRemoved`` events corresponding to
            the underlying :class:`OffloadingEvent` stream.
        """
        yield from self._events_tracker.take_events(self.manager.take_events())

    def reset_cache(self) -> None:
        """Reset the offloading manager cache, evicting all stored blocks."""

        # reset_cache cannot be called in the middle of a schedule step
        assert not self._current_batch_load_jobs
        assert not self._current_batch_jobs_to_flush
        assert not self._current_batch_allocated_block_ids

        # Flush all in-flight jobs
        self._current_batch_jobs_to_flush.update(self._jobs.keys())

        for req_id, status in list(self._req_status.items()):
            if status.req.is_finished():
                del self._req_status[req_id]

        # Reset offloading manager cache
        self.manager.reset_cache()

        # Reset store progress so active requests re-offload from block 0
        for status in self._req_status.values():
            for group_state in status.group_states:
                group_state.next_stored_chunk_idx = 0
            status.transfer_jobs.clear()

        # Release GPU blocks reserved by in-flight write-back jobs before we drop
        # their records: their CPU copies are discarded by manager.reset_cache
        # above, so unpin the source blocks (stored=False keeps them GPU-cached)
        # and clear the in-flight counter so the pin cap doesn't wedge shut.
        if self._gpu_block_pool is not None:
            for job_status in self._jobs.values():
                if job_status.is_writeback:
                    for block_id in job_status.writeback_block_ids or ():
                        self._gpu_block_pool.release_writeback_victim(block_id, stored=False)
        self._writeback_inflight_blocks = 0

        # Discard jobs and save job_counter to be able to discard worker responses
        self._stale_job_threshold = self._job_counter
        self._jobs.clear()
        self._block_id_to_pending_jobs.clear()

        # The manager pool is empty; pending event payloads and announced
        # reference counts are stale.
        self._events_tracker.reset()

        # Note: _current_batch_jobs_to_flush is intentionally NOT cleared.
        # The load flush IDs collected above must be delivered to workers.
        if self._chunks_being_loaded is not None:
            self._chunks_being_loaded.clear()

    def shutdown(self) -> None:
        self.manager.shutdown()
