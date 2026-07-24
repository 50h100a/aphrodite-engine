# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import OrderedDict
from collections.abc import Iterable

from typing_extensions import override

from aphrodite.logger import init_logger
from aphrodite.v1.kv_offload.base import OffloadKey, ReqContext
from aphrodite.v1.kv_offload.cpu.policies.base import BlockStatus, CachePolicy

logger = init_logger(__name__)


class LRUCachePolicy(CachePolicy):
    """LRU policy with a dedicated evictable list for O(1) eviction.

    When GPU-residency-awareness is enabled the evictable population is split
    into two ordered segments, evicted in this order:

      - ``duplicates``: blocks last confirmed GPU-resident, ordered by
        confirmation recency. Evicted first, most-recently-confirmed first.
      - ``evictable_blocks`` (exclusive): blocks not known to be GPU-resident.
        Evicted last, LRU (oldest) first.

    When disabled ``duplicates`` stays empty and behavior is identical to a
    single-list LRU.
    """

    supports_gpu_residency = True

    def __init__(self, cache_capacity: int):
        # ref_cnt-0 blocks in LRU order; the exclusive segment when residency-aware.
        self.evictable_blocks: OrderedDict[OffloadKey, None] = OrderedDict()
        # ref_cnt-0 blocks confirmed GPU-resident, MRU at the end.
        self.duplicates: OrderedDict[OffloadKey, None] = OrderedDict()
        self.blocks: dict[OffloadKey, BlockStatus] = {}

        self._gpu_residency_aware: bool = False
        # Keys believed GPU-resident; picks the segment a key lands in when it
        # (re)becomes evictable.
        self._gpu_resident_hint: set[OffloadKey] = set()

    def set_gpu_residency_aware(self, enabled: bool) -> None:
        self._gpu_residency_aware = enabled
        logger.info(
            "LRUCachePolicy: GPU-residency-aware eviction %s.",
            "enabled" if enabled else "disabled",
        )

    def _place_evictable(self, key: OffloadKey) -> None:
        if self._gpu_residency_aware and key in self._gpu_resident_hint:
            self.duplicates[key] = None
        else:
            self.evictable_blocks[key] = None

    @override
    def get(self, key: OffloadKey) -> BlockStatus | None:
        return self.blocks.get(key)

    @override
    def insert(self, key: OffloadKey, block: BlockStatus) -> None:
        self.blocks[key] = block
        if block.ref_cnt == 0:
            self._place_evictable(key)

    @override
    def remove(self, key: OffloadKey) -> None:
        del self.blocks[key]
        self.evictable_blocks.pop(key, None)
        self.duplicates.pop(key, None)
        self._gpu_resident_hint.discard(key)

    @override
    def touch(self, keys: Iterable[OffloadKey], req_context: ReqContext) -> None:
        for key in reversed(list(keys)):
            if key in self.evictable_blocks:
                self.evictable_blocks.move_to_end(key)
            elif key in self.duplicates:
                self.duplicates.move_to_end(key)

    @override
    def clear(self) -> None:
        self.evictable_blocks.clear()
        self.duplicates.clear()
        self.blocks.clear()
        self._gpu_resident_hint.clear()

    @override
    def evict(self, n: int, protected: set[OffloadKey]) -> list[tuple[OffloadKey, BlockStatus]] | None:
        if n == 0:
            return []

        candidates: list[tuple[OffloadKey, BlockStatus]] = []

        # Duplicates first, most-recently-confirmed (MRU) first.
        if self.duplicates:
            for key in reversed(self.duplicates):
                if key in protected:
                    continue
                block = self.blocks[key]
                assert block.ref_cnt == 0
                candidates.append((key, block))
                if len(candidates) == n:
                    break

        # Then exclusive blocks, LRU (oldest) first.
        if len(candidates) < n:
            for key in self.evictable_blocks:
                if key in protected:
                    continue
                block = self.blocks[key]
                assert block.ref_cnt == 0
                candidates.append((key, block))
                if len(candidates) == n:
                    break

        if len(candidates) < n:
            return None

        for key, _ in candidates:
            self.evictable_blocks.pop(key, None)
            self.duplicates.pop(key, None)
            del self.blocks[key]
            self._gpu_resident_hint.discard(key)
        return candidates

    @override
    def mark_evictable(self, key: OffloadKey) -> None:
        self._place_evictable(key)

    @override
    def mark_non_evictable(self, key: OffloadKey) -> None:
        if self.evictable_blocks.pop(key, None) is None:
            self.duplicates.pop(key, None)

    @override
    def iter_keys(self) -> Iterable[OffloadKey]:
        return self.blocks.keys()

    @override
    def mark_gpu_resident(self, keys: Iterable[OffloadKey]) -> None:
        if not self._gpu_residency_aware:
            return
        for key in keys:
            if key not in self.blocks:
                continue
            self._gpu_resident_hint.add(key)
            if key in self.evictable_blocks:
                del self.evictable_blocks[key]
                self.duplicates[key] = None
            elif key in self.duplicates:
                self.duplicates.move_to_end(key)

    @override
    def mark_gpu_evicted(self, keys: Iterable[OffloadKey]) -> None:
        if not self._gpu_residency_aware:
            return
        for key in keys:
            self._gpu_resident_hint.discard(key)
            if key in self.duplicates:
                del self.duplicates[key]
                self.evictable_blocks[key] = None

    @override
    def segment_counts(self) -> tuple[int, int]:
        return (len(self.duplicates), len(self.evictable_blocks))
