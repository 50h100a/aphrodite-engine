# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for BlockPool write-back victim reservation.

These back the offload-on-eviction ("true offload") design: cached blocks are
reserved out of the free queue so their KV data can be copied to the CPU tier
before the physical block is reused, then evicted from the GPU prefix cache on
completion so the content lives only on the lower tier.
"""
from aphrodite.v1.core.block_pool import BlockPool
from aphrodite.v1.core.kv_cache_utils import (
    BlockHash,
    make_block_hash_with_group_id,
)


def make_pool_with_cached_blocks(num_blocks: int) -> BlockPool:
    """A pool whose non-null blocks are each cached under a distinct hash,
    still idle in the free queue (i.e. eviction candidates), in block-id order.

    Block 0 is the null block. Blocks are laid out in the free queue in
    eviction order head->tail = block 1 (oldest) .. block N (newest), matching
    how free_blocks appends freshly-freed cached blocks to the tail.
    """
    pool = BlockPool(num_gpu_blocks=num_blocks, enable_caching=True, hash_block_size=16)
    for block in pool.blocks:
        if block.is_null:
            continue
        block_hash = make_block_hash_with_group_id(plain_hash(block.block_id), 0)
        block.set_block_hash(block_hash)
        pool.cached_block_hash_to_block.insert(block_hash, block)
        pool.cached_block_hashes_by_block.setdefault(block.block_id, set()).add(block_hash)
    return pool


def plain_hash(block_id: int) -> BlockHash:
    """The plain (group-less) BlockHash used by get_cached_block lookups."""
    return BlockHash(str(block_id).encode())


def free_queue_ids(pool: BlockPool) -> list[int]:
    ids = []
    block = pool.free_block_queue.fake_free_list_head.next_free_block
    tail = pool.free_block_queue.fake_free_list_tail
    while block is not None and block is not tail:
        ids.append(block.block_id)
        block = block.next_free_block
    return ids


def test_reserve_picks_eviction_soonest_first():
    pool = make_pool_with_cached_blocks(5)  # blocks 1..4 cached (0 is null)
    victims = pool.reserve_writeback_victims(2)
    assert [bid for bid, _ in victims] == [1, 2]  # head of the free queue
    # Reserved blocks are pinned out of the free queue.
    assert free_queue_ids(pool) == [3, 4]
    assert pool.blocks[1].ref_cnt == 1
    assert pool.blocks[2].ref_cnt == 1


def test_reserved_blocks_are_not_reallocated():
    pool = make_pool_with_cached_blocks(5)
    pool.reserve_writeback_victims(2)  # reserve 1, 2
    # get_new_blocks must draw only from the unreserved blocks 3, 4.
    allocated = pool.get_new_blocks(2)
    assert {b.block_id for b in allocated} == {3, 4}


def test_reserve_skips_already_reserved_and_null():
    pool = make_pool_with_cached_blocks(5)
    first = pool.reserve_writeback_victims(2)
    second = pool.reserve_writeback_victims(10)  # more than remain
    assert [bid for bid, _ in first] == [1, 2]
    assert [bid for bid, _ in second] == [3, 4]  # never re-offers 1,2 or the null block
    assert pool.reserve_writeback_victims(1) == []


def test_release_stored_evicts_and_frees_reuse_first():
    pool = make_pool_with_cached_blocks(5)
    (bid, _block_hash), _ = pool.reserve_writeback_victims(2)
    pool.release_writeback_victim(bid, stored=True)

    block = pool.blocks[bid]
    assert block.ref_cnt == 0
    # Evicted from the prefix cache: no longer hittable, hash cleared.
    assert block.block_hash is None
    assert pool.get_cached_block(plain_hash(bid), [0]) is None
    # Returned to the free queue as reuse-first (head).
    assert free_queue_ids(pool)[0] == bid


def test_release_stored_reuse_first_ordering():
    pool = make_pool_with_cached_blocks(5)
    victims = pool.reserve_writeback_victims(2)  # 1, 2
    for bid, _ in victims:
        pool.release_writeback_victim(bid, stored=True)
    # Offloaded blocks are pure-free now and should be handed out before the
    # still-cached blocks 3, 4.
    allocated = pool.get_new_blocks(2)
    assert {b.block_id for b in allocated} == {1, 2}


def test_release_failed_store_keeps_block_cached():
    pool = make_pool_with_cached_blocks(5)
    (bid, block_hash), _ = pool.reserve_writeback_victims(2)
    pool.release_writeback_victim(bid, stored=False)

    block = pool.blocks[bid]
    assert block.ref_cnt == 0
    # Still cached and hittable; returned to the free queue as an eviction
    # candidate (tail), not reuse-first.
    assert block.block_hash is block_hash
    assert pool.get_cached_block(plain_hash(bid), [0]) == [block]
    assert free_queue_ids(pool)[-1] == bid


def test_revived_by_cache_hit_is_not_evicted():
    pool = make_pool_with_cached_blocks(5)
    (bid, block_hash), _ = pool.reserve_writeback_victims(2)

    # A concurrent request hits the reserved block and pins it.
    hit = pool.get_cached_block(plain_hash(bid), [0])
    assert hit == [pool.blocks[bid]]
    pool.touch(hit)  # ref_cnt 1 (reservation) -> 2
    assert pool.blocks[bid].ref_cnt == 2

    # Completing the store must NOT evict a block a live request now holds.
    pool.release_writeback_victim(bid, stored=True)
    block = pool.blocks[bid]
    assert block.ref_cnt == 1  # only the reviving request remains
    assert block.block_hash is block_hash  # still cached
    assert bid not in free_queue_ids(pool)  # in use, not free

    # When that request frees it, it re-enters the free queue normally.
    pool.free_blocks([block])
    assert block.ref_cnt == 0
    assert bid in free_queue_ids(pool)


def test_num_free_blocks_balanced_across_reserve_release():
    pool = make_pool_with_cached_blocks(5)
    before = pool.get_num_free_blocks()
    victims = pool.reserve_writeback_victims(3)
    assert pool.get_num_free_blocks() == before - 3
    for i, (bid, _) in enumerate(victims):
        pool.release_writeback_victim(bid, stored=(i % 2 == 0))
    assert pool.get_num_free_blocks() == before


def test_reserve_noop_without_caching():
    pool = BlockPool(num_gpu_blocks=5, enable_caching=False, hash_block_size=16)
    assert pool.reserve_writeback_victims(3) == []
