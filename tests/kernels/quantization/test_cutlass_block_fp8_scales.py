# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from aphrodite.model_executor.kernels.linear.scaled_mm.cutlass import CutlassFp8BlockScaledMMKernel
from aphrodite.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel import FP8ScaledMMLinearLayerConfig
from aphrodite.model_executor.layers.quantization.utils.quant_utils import GroupShape, create_fp8_quant_key
from aphrodite.platforms import current_platform

pytestmark = pytest.mark.usefixtures("default_aphrodite_config")


def make_kernel():
    return CutlassFp8BlockScaledMMKernel(
        FP8ScaledMMLinearLayerConfig(
            weight_quant_key=create_fp8_quant_key(static=True, group_shape=GroupShape(128, 128)),
            activation_quant_key=create_fp8_quant_key(static=False, group_shape=GroupShape(1, 128)),
            weight_shape=(256, 256),
            input_dtype=torch.bfloat16,
            out_dtype=torch.bfloat16,
        )
    )


def make_layer(scale_attr, scale_dtype, device):
    torch.manual_seed(0)
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(torch.randn(256, 256, device=device).to(torch.float8_e4m3fn), requires_grad=False)
    scales = torch.tensor([[0.25, 0.5], [1.0, 2.0]], device=device)
    encoded = scales.to(torch.float8_e8m0fnu)
    if scale_dtype == torch.uint8:
        encoded = encoded.view(torch.uint8)
    elif scale_dtype == torch.float32:
        encoded = scales
    layer.register_parameter(scale_attr, torch.nn.Parameter(encoded, requires_grad=False))
    return layer, scales


@pytest.mark.cpu_test
@pytest.mark.parametrize("scale_attr", ["weight_scale_inv", "weight_scale"])
@pytest.mark.parametrize("scale_dtype", [torch.float8_e8m0fnu, torch.uint8, torch.float32])
def test_cutlass_block_fp8_scale_conversion(scale_attr, scale_dtype):
    layer, expected = make_layer(scale_attr, scale_dtype, "cpu")
    original_weight = layer.weight.view(torch.uint8).clone()
    kernel = make_kernel()
    kernel.process_weights_after_loading(layer)
    scale = getattr(layer, scale_attr)
    assert scale.dtype == torch.float32
    assert scale.is_contiguous()
    assert not scale.requires_grad
    torch.testing.assert_close(scale, expected, rtol=0, atol=0)
    torch.testing.assert_close(layer.weight.view(torch.uint8), original_weight)
    kernel.process_weights_after_loading(layer)
    torch.testing.assert_close(getattr(layer, scale_attr), expected, rtol=0, atol=0)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUTLASS requires CUDA")
@pytest.mark.parametrize("scale_attr", ["weight_scale_inv", "weight_scale"])
@pytest.mark.parametrize("num_tokens", [1, 83, 16384])
@torch.inference_mode()
def test_cutlass_block_fp8_e8m0_linear(scale_attr, num_tokens):
    supported, reason = CutlassFp8BlockScaledMMKernel.is_supported()
    if not supported:
        pytest.skip(reason)
    layer, scales = make_layer(scale_attr, torch.float8_e8m0fnu, "cuda")
    kernel = make_kernel()
    x = torch.randn(num_tokens, 256, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(256, device="cuda", dtype=torch.bfloat16)
    # Isolate GEMM and scale interpretation from activation quantization error.
    qx, sx = kernel.quant_fp8(x)
    x_dequant = qx.float() * sx.repeat_interleave(128, dim=1)
    w_dequant = layer.weight.float() * scales.repeat_interleave(128, 0).repeat_interleave(128, 1)
    expected = (x_dequant @ w_dequant.T).to(torch.bfloat16) + bias
    kernel.process_weights_after_loading(layer)
    actual = kernel.apply_weights(layer, x, bias)
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.125)
