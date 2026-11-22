# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from aphrodite.models.deepseek_v4.nvidia import flashinfer_sparse as backend
from aphrodite.platforms import current_platform


def make_config(max_model_len=1048576, tokens=16384):
    return SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=tokens),
        model_config=SimpleNamespace(
            max_model_len=max_model_len,
            hf_text_config=SimpleNamespace(index_topk=2048, sliding_window=128),
        ),
        attention_config=SimpleNamespace(use_non_causal=False),
        speculative_config=None,
    )


@pytest.mark.cpu_test
@pytest.mark.parametrize("heads", [8, 16, 32, 64, 128])
@pytest.mark.parametrize("context", [4096, 1048576, 1048577])
def test_workspace_covers_c128_decode(heads, context):
    size = backend._sm120_dsv4_workspace_bytes(make_config(context), heads)
    c128_keys = ((context + 127) // 128 + 127) // 128 * 128
    splits = 2 + max(2048, c128_keys) // 64
    needed = 64 * heads * splits * 1028 + 64 * heads * 4
    assert size >= needed
    assert size >= backend._FLASHINFER_DSV4_WORKSPACE_BUFFER_SIZE


@pytest.mark.cpu_test
def test_workspace_accounts_for_speculative_window():
    config = make_config()
    normal = backend._sm120_dsv4_workspace_bytes(config, 32)
    config.attention_config.use_non_causal = True
    config.speculative_config = SimpleNamespace(num_speculative_tokens=65)
    assert backend._sm120_dsv4_workspace_bytes(config, 32) > normal


@pytest.mark.skipif(not current_platform.is_device_capability_family(120), reason="Requires SM120 FlashInfer")
@torch.inference_mode()
def test_workspace_fits_real_flashinfer_decode_plan():
    execution = pytest.importorskip("flashinfer.mla._sparse_mla_sm120._execution")
    prepared = pytest.importorskip("flashinfer.mla._sparse_mla_sm120._prepared")
    props = torch.cuda.get_device_properties(0)
    plan = execution.resolve_attention(
        model=1,
        tokens=64,
        heads=32,
        topk=128,
        extra_topk=8192,
        page_size=256,
        extra_page_size=2,
        page_stride_bytes=256 * 584,
        extra_page_stride_bytes=2 * 584,
        row_stride_bytes=584,
        indices_stride=128,
        extra_indices_stride=8192,
        lse_stride=32,
        has_lengths=True,
        has_extra_lengths=True,
        has_sink=True,
        extra_fp4=False,
        variant=0,
        precision="default",
        cpb=0,
        sm_count=props.multi_processor_count,
        max_shared_bytes=props.shared_memory_per_block_optin,
    )
    required = list(plan.workspace())
    assert sum(int(entry[2]) for entry in required) > backend._FLASHINFER_DSV4_WORKSPACE_BUFFER_SIZE
    workspace = torch.empty(backend._sm120_dsv4_workspace_bytes(make_config(), 32), device="cuda", dtype=torch.uint8)
    offset = 0
    for shape, dtype, _, alignment in required:
        view, offset = prepared._workspace_tensor_view(
            workspace,
            byte_offset=offset,
            shape=tuple(shape),
            dtype=getattr(torch, str(dtype)),
            alignment=int(alignment),
        )
        assert view is not None


@pytest.mark.skipif(not current_platform.is_device_capability_family(120), reason="Requires SM120 FlashInfer")
@torch.inference_mode()
def test_long_context_sparse_attention_cuda_graph():
    from aphrodite.utils.flashinfer import flashinfer_trtllm_batch_decode_sparse_mla_dsv4

    q = torch.zeros(64, 32, 512, device="cuda", dtype=torch.bfloat16)
    out = torch.empty_like(q)
    swa = torch.zeros(1, 256, 1, 584, device="cuda", dtype=torch.uint8)
    extra = torch.zeros(1, 2, 1, 584, device="cuda", dtype=torch.uint8)
    # Packed cache data is zero, with unit UE8M0 scales in the page footer.
    swa.view(-1)[256 * 576 :] = 127
    extra.view(-1)[2 * 576 :] = 127
    indices = torch.zeros(64, 1, 128, device="cuda", dtype=torch.int32)
    extra_indices = torch.zeros(64, 1, 8192, device="cuda", dtype=torch.int32)
    lengths = torch.ones(64, device="cuda", dtype=torch.int32)
    workspace = torch.empty(backend._sm120_dsv4_workspace_bytes(make_config(), 32), device="cuda", dtype=torch.uint8)

    def run():
        flashinfer_trtllm_batch_decode_sparse_mla_dsv4(
            query=q,
            swa_kv_cache=swa,
            compressed_kv_cache=extra,
            out=out,
            workspace_buffer=workspace,
            sparse_indices=indices,
            extra_sparse_indices=extra_indices,
            swa_topk_lens=lengths,
            extra_sparse_topk_lens=lengths,
            bmm1_scale=1.0,
            sinks=torch.zeros(32, device="cuda", dtype=torch.float32),
            kv_layout="NHD",
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out, torch.zeros_like(out), rtol=0, atol=0)
