# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the Aphrodite project
"""Warm up FA4 CuTeDSL MLA prefill compile keys."""

from __future__ import annotations

from typing import TYPE_CHECKING

from aphrodite.v1.attention.backends.mla.prefill import get_mla_prefill_backend

if TYPE_CHECKING:
    from aphrodite.v1.worker.gpu_worker import Worker


def fa4_cutedsl_warmup(worker: Worker) -> None:
    runner = worker.model_runner
    if runner.is_pooling_model:
        return

    aphrodite_config = runner.aphrodite_config
    if not aphrodite_config.model_config.use_mla:
        return

    # Selection raises when no backend supports the model's MLA dimensions,
    # which is a supported state rather than an error: sparse-MLA models such as
    # DeepSeek V4 expose no qk_nope/v head dims and run the top-k MQA path with
    # no dense prefill backend at all (see MLAAttention.__init__, which handles
    # the same ValueError). Either way FA4 is not in use, so nothing to warm up.
    try:
        backend_cls = get_mla_prefill_backend(aphrodite_config)
    except ValueError:
        return
    if backend_cls.get_name() != "FLASH_ATTN":
        return

    from aphrodite.v1.attention.backends.mla.prefill import flash_attn

    flash_attn.FA4_MLA_PREFILL_KERNEL.warmup(aphrodite_config)
