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

    ``evictable_blocks`` holds ref_cnt-0 blocks in LRU order (oldest first);
    ``blocks`` holds all tracked blocks regardless of ref_cnt.
    """

    def __init__(self, cache_capacity: int):
        # ref_cnt-0 blocks in LRU order (oldest first).
        self.evictable_blocks: OrderedDict[OffloadKey, None] = OrderedDict()
        self.blocks: dict[OffloadKey, BlockStatus] = {}

    @override
    def get(self, key: OffloadKey) -> BlockStatus | None:
        return self.blocks.get(key)

    @override
    def insert(self, key: OffloadKey, block: BlockStatus) -> None:
        self.blocks[key] = block
        if block.ref_cnt == 0:
            self.evictable_blocks[key] = None

    @override
    def remove(self, key: OffloadKey) -> None:
        del self.blocks[key]
        self.evictable_blocks.pop(key, None)

    @override
    def touch(self, keys: Iterable[OffloadKey], req_context: ReqContext) -> None:
        for key in reversed(list(keys)):
            if key in self.evictable_blocks:
                self.evictable_blocks.move_to_end(key)

    @override
    def clear(self) -> None:
        self.evictable_blocks.clear()
        self.blocks.clear()

    @override
    def evict(self, n: int, protected: set[OffloadKey]) -> list[tuple[OffloadKey, BlockStatus]] | None:
        if n == 0:
            return []

        candidates: list[tuple[OffloadKey, BlockStatus]] = []
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
            del self.blocks[key]
        return candidates

    @override
    def mark_evictable(self, key: OffloadKey) -> None:
        self.evictable_blocks[key] = None

    @override
    def mark_non_evictable(self, key: OffloadKey) -> None:
        self.evictable_blocks.pop(key, None)

    @override
    def iter_keys(self) -> Iterable[OffloadKey]:
        return self.blocks.keys()
