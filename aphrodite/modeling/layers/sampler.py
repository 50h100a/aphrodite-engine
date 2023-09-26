"""A layer that samples the next tokens from the model's outputs."""
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn

from aphrodite.modeling.metadata import InputMetadata
from aphrodite.modeling.megatron.tensor_parallel import (
    gather_from_tensor_model_parallel_region)
from aphrodite.common.sampling_params import SamplingParams
from aphrodite.common.sequence import SamplerOutput, SequenceOutputs

_SAMPLING_EPS = 1e-5


class Sampler(nn.Module):
    """Samples the next tokens from the model's outputs.

    This layer does the following:
    1. Discard the hidden states that are not used for sampling (i.e., all
        tokens except the final one in each prompt).
    2. Compute the logits for the next tokens.
    3. Apply presence and frequency penalties.
    4. Apply temperature scaling.
    5. Apply top-p and top-k truncation.
    6. Sample the next tokens.
    Here, each sequence group within the batch can have different sampling
    parameters (e.g., sampling method, temperature, top-p, top-k, etc.).
    """

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size

    def forward(
        self,
        embedding: torch.Tensor,
        hidden_states: torch.Tensor,
        input_metadata: InputMetadata,
        embedding_bias: Optional[torch.Tensor] = None,
    ) -> SamplerOutput:
        # Get the hidden states that we use for sampling.
        hidden_states = _prune_hidden_states(hidden_states, input_metadata)

        # Get the logits for the next tokens.
        logits = torch.matmul(hidden_states, embedding.t())
        if embedding_bias is not None:
            logits += embedding_bias
        logits = gather_from_tensor_model_parallel_region(logits)
        # Remove paddings in vocab (if any).
        logits = logits[:, :self.vocab_size]

        # Apply presence and frequency penalties.
        output_tokens = _get_output_tokens(input_metadata)
        assert len(output_tokens) == logits.shape[0]
        presence_penalties, frequency_penalties = _get_penalties(
            input_metadata)
        assert len(presence_penalties) == logits.shape[0]
        assert len(frequency_penalties) == logits.shape[0]
        logits = _apply_penalties(logits, output_tokens, presence_penalties,
                                  frequency_penalties, self.vocab_size)

        logits = _apply_logits_processors(input_metadata, logits, output_tokens)

        # Apply temperature scaling.
        temperatures = _get_temperatures(input_metadata)
        assert len(temperatures) == logits.shape[0]
        if any(t != 1.0 for t in temperatures):
            t = torch.tensor(temperatures,
                             dtype=logits.dtype,
                             device=logits.device)
            # Use in-place division to avoid creating a new tensor.
            logits.div_(t.unsqueeze(dim=1))

        # Apply top-p, top-k, and top-a truncation.
        top_ps, top_ks, top_as = _get_top_p_top_k_top_a(input_metadata, self.vocab_size)
        assert len(top_ps) == len(top_ks) == logits.shape[0]
        do_top_p = any(p < 1.0 - _SAMPLING_EPS for p in top_ps)
        do_top_k = any(k != self.vocab_size for k in top_ks)
        do_top_a = any(a > _SAMPLING_EPS for a in top_as)
        if do_top_p or do_top_k or do_top_a:
            logits = _apply_top_ap_top_k(logits, top_ps, top_ks, top_as)

        # input_metadata.context_lens # is current token index. or something like that.
        # print("InputMetadata", input_metadata)

        # We use float32 for probabilities and log probabilities.
        # Compute the probabilities.
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        # Compute the log probabilities (before applying top-p and top-k).
        # Use log_softmax to ensure numerical stability
        logprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float)

        # Sample the next tokens.
        return _sample(probs, logprobs, input_metadata)


def _prune_hidden_states(
    hidden_states: torch.Tensor,
    input_metadata: InputMetadata,
) -> torch.Tensor:
    start_idx = 0
    last_token_indicies: List[int] = []
    for prompt_len in input_metadata.prompt_lens:
        last_token_indicies.append(start_idx + prompt_len - 1)
        start_idx += prompt_len
    last_token_indicies.extend(
        range(start_idx, start_idx + input_metadata.num_generation_tokens))
    return hidden_states.index_select(
        0, torch.tensor(last_token_indicies, device=hidden_states.device))


def _get_penalties(
        input_metadata: InputMetadata) -> Tuple[List[float], List[float]]:
    # Collect the presence and frequency penalties.
    presence_penalties: List[float] = []
    frequency_penalties: List[float] = []
    for i, seq_group in enumerate(input_metadata.seq_groups):
        seq_ids, sampling_params = seq_group
        p = sampling_params.presence_penalty
        f = sampling_params.frequency_penalty
        if i < input_metadata.num_prompts:
            # A prompt input.
            presence_penalties.append(p)
            frequency_penalties.append(f)
        else:
            # A generation token.
            presence_penalties += [p] * len(seq_ids)
            frequency_penalties += [f] * len(seq_ids)
    return presence_penalties, frequency_penalties


def _get_output_tokens(input_metadata: InputMetadata) -> List[List[int]]:
    output_tokens: List[List[int]] = []
    for i, seq_group in enumerate(input_metadata.seq_groups):
        seq_ids, _ = seq_group
        if i < input_metadata.num_prompts:
            # A prompt input.
            # NOTE: While the prompt input usually has no output tokens,
            # it may have output tokens in the case of recomputation.
            seq_id = seq_ids[0]
            seq_data = input_metadata.seq_data[seq_id]
            output_tokens.append(seq_data.output_token_ids)
        else:
            # A generation token.
            for seq_id in seq_ids:
                seq_data = input_metadata.seq_data[seq_id]
                output_tokens.append(seq_data.output_token_ids)
    return output_tokens

def _apply_logits_processors(
        input_metadata: InputMetadata,
        logits: torch.Tensor,
        output_tokens: List[List[int]]
) -> torch.Tensor:
    for _, seq_group in enumerate(input_metadata.seq_groups):
        _, sampling_params = seq_group
        logits_processors = sampling_params.logits_processors

        if logits_processors is not None:
            for logits_processor in logits_processors:
                logits = logits_processor(logits, output_tokens)
    return logits

def _apply_penalties(
    logits: torch.Tensor,
    output_tokens: List[List[int]],
    presence_penalties: List[float],
    frequency_penalties: List[float],
    vocab_size: int,
) -> torch.Tensor:
    num_seqs = logits.shape[0]
    # Collect the indices of sequences that have non-zero penalties.
    indices = []
    for i in range(num_seqs):
        if not output_tokens[i]:
            continue
        p = presence_penalties[i]
        f = frequency_penalties[i]
        if abs(p) < _SAMPLING_EPS and abs(f) < _SAMPLING_EPS:
            continue
        indices.append(i)

    # Return early if all sequences have zero penalties.
    if not indices:
        return logits

    bin_counts = []
    for i in indices:
        bin_counts.append(np.bincount(output_tokens[i], minlength=vocab_size))
    bin_counts = np.stack(bin_counts, axis=0)
    bin_counts = torch.from_numpy(bin_counts).to(dtype=logits.dtype,
                                                 device=logits.device)

    frequency_penalties = [frequency_penalties[i] for i in indices]
    frequency_penalties = torch.tensor(frequency_penalties,
                                       dtype=logits.dtype,
                                       device=logits.device)
    presence_penalties = [presence_penalties[i] for i in indices]
    presence_penalties = torch.tensor(presence_penalties,
                                      dtype=logits.dtype,
                                      device=logits.device)

    # We follow the definition in OpenAI API.
    # Refer to https://platform.openai.com/docs/api-reference/parameter-details
    logits[indices] -= frequency_penalties.unsqueeze(dim=1) * bin_counts
    presence_mask = (bin_counts > 0.0).to(dtype=logits.dtype)
    logits[indices] -= presence_penalties.unsqueeze(dim=1) * presence_mask
    return logits


def _get_temperatures(input_metadata: InputMetadata) -> List[float]:
    # Collect the temperatures for the logits.
    temperatures: List[float] = []
    for i, seq_group in enumerate(input_metadata.seq_groups):
        seq_ids, sampling_params = seq_group
        temperature = sampling_params.temperature
        if temperature < _SAMPLING_EPS:
            # NOTE: Zero temperature means deterministic sampling
            # (i.e., greedy sampling or beam search).
            # Set the temperature to 1 to avoid division by zero.
            temperature = 1.0

        if i < input_metadata.num_prompts:
            # A prompt input.
            temperatures.append(temperature)
        else:
            # A generation token.
            temperatures += [temperature] * len(seq_ids)
    return temperatures


def _get_top_p_top_k_top_a(
    input_metadata: InputMetadata,
    vocab_size: int,
) -> Tuple[List[float], List[int], List[float]]:
    top_ps: List[float] = []
    top_ks: List[int] = []
    top_as: List[float] = []
    for i, seq_group in enumerate(input_metadata.seq_groups):
        seq_ids, sampling_params = seq_group
        # k should not be greater than the vocab size.
        top_k = min(sampling_params.top_k, vocab_size)
        # k=-1 means no truncation.
        top_k = vocab_size if top_k == -1 else top_k

        # Prompt inputs repeat once, generation tokens can repeat N times
        repeats = 1 if i < input_metadata.num_prompts else len(seq_ids)
        top_ps += [sampling_params.top_p] * repeats
        top_ks += [top_k] * repeats
        top_as += [sampling_params.top_a] * repeats

    return top_ps, top_ks, top_as


def _apply_top_ap_top_k(
    logits: torch.Tensor,   # [n_samples, n_vocab]
    top_ps: List[float],    # [n_samples]
    top_ks: List[int],      # [n_samples]
    top_as: List[float],    # [n_samples]
) -> torch.Tensor:
    ts_a = torch.tensor(top_as, dtype=logits.dtype, device=logits.device)
    ts_p = torch.tensor(top_ps, dtype=logits.dtype, device=logits.device)
    ts_k = torch.tensor(top_ks, dtype=torch.int, device=logits.device)
    logits_sort:torch.Tensor
    logits_sort, logits_idx = logits.sort(dim=-1, descending=True)

    # Apply top-k.
    # Create a mask for the top-k elements.
    top_k_mask = torch.arange(logits_idx.shape[-1], device=logits_idx.device)
    top_k_mask = top_k_mask.expand(logits_idx.shape[0], -1)
    top_k_mask = top_k_mask >= ts_k.unsqueeze(dim=1)
    logits_sort[top_k_mask] = -float("inf")

    # Apply top-a and top-p.
    probs_sort = logits_sort.softmax(dim=-1)
    probs_sum = probs_sort.cumsum(dim=-1)
    top_a_thresholds = torch.pow(probs_sort[:, 0], 2) * ts_a
    top_ap_mask = (probs_sort < top_a_thresholds.unsqueeze(1)) # Cull logits below the top-a threshold
    top_ap_mask.logical_or_(probs_sum > ts_p.unsqueeze(dim=1)) # Cull logits above the top-p summation threshold
    top_ap_mask.scatter_(0, torch.tensor([0], device=logits.device).unsqueeze(1), False) # Guarantee at least one token is pickable
    logits_sort[top_ap_mask] = -float("inf")

    # print("ROW", ts_p[0].item(), ts_a[0].item(), probs_sort[0][0].item(), (logits.shape[1] - top_ap_mask.long().sum(dim=1)[0].item()))
    

    # Re-sort the probabilities.
    logits = torch.gather(logits_sort,
                          dim=-1,
                          index=torch.argsort(logits_idx, dim=-1))
    return logits


def _get_topk_logprobs(
    logprobs: torch.Tensor,
    num_logprobs: Optional[int],
) -> Dict[int, float]:
    if num_logprobs is None or num_logprobs == 0:
        return {}

    topk_logprobs, topk_ids = torch.topk(logprobs, num_logprobs)
    if num_logprobs == 1:
        topk_logprobs = [topk_logprobs.item()]
        topk_ids = [topk_ids.item()]
    else:
        topk_logprobs = topk_logprobs.tolist()
        topk_ids = topk_ids.tolist()

    token_to_logprob: Dict[int, float] = {}
    for token_id, logprob in zip(topk_ids, topk_logprobs):
        token_to_logprob[token_id] = logprob
    return token_to_logprob


def _sample_from_prompt(
    prob: torch.Tensor,
    sampling_params: SamplingParams,
) -> List[int]:
    if sampling_params.use_beam_search:
        # Beam search.
        beam_width = sampling_params.best_of
        # Sample 2 * beam_width candidates to make sure that with high
        # probability we can get `beam_width` candidates in addition to
        # the finished sequences for the next iteration. See
        # https://github.com/tensorflow/tensor2tensor/blob/bafdc1b67730430d38d6ab802cbd51f9d053ba2e/tensor2tensor/utils/beam_search.py#L557-L563
        # for details. See also HF reference:
        # https://github.com/huggingface/transformers/blob/a4dd53d88e4852f023332d284ff07a01afcd5681/src/transformers/generation/utils.py#L3063-L3065
        _, next_token_ids = torch.topk(prob, 2 * beam_width)
        next_token_ids = next_token_ids.tolist()
    elif sampling_params.temperature < _SAMPLING_EPS:
        # Greedy sampling.
        assert sampling_params.best_of == 1
        next_token_id = torch.argmax(prob)
        next_token_ids = [next_token_id.item()]
    else:
        # Random sampling.
        # Sample `best_of` tokens for the prompt.
        num_seqs = sampling_params.best_of
        next_token_ids = torch.multinomial(prob,
                                           num_samples=num_seqs,
                                           replacement=True)
        next_token_ids = next_token_ids.tolist()
    return next_token_ids


def _sample_from_generation_tokens(
    seq_ids: List[int],
    probs: torch.Tensor,
    logprobs: torch.Tensor,
    seq_logprobs: List[float],
    sampling_params: SamplingParams,
) -> Tuple[List[int], List[int]]:
    # NOTE: sampling_params.best_of can be greater than
    # len(seq_ids) because some sequences in the group might have
    # been already terminated.
    if sampling_params.use_beam_search:
        # Beam search.
        # Add cumulative logprobs for the sequences in the group.
        seq_logprobs = torch.tensor(seq_logprobs,
                                    dtype=torch.float,
                                    device=logprobs.device)
        logprobs = logprobs + seq_logprobs.unsqueeze(dim=1)

        vocab_size = logprobs.size(-1)
        beam_width = len(seq_ids)
        _, topk_ids = torch.topk(logprobs.flatten(), 2 * beam_width)
        topk_ids = topk_ids.tolist()
        seq_idx = [i // vocab_size for i in topk_ids]
        parent_seq_ids = [seq_ids[i] for i in seq_idx]
        next_token_ids = [i % vocab_size for i in topk_ids]
    elif sampling_params.temperature < _SAMPLING_EPS:
        # Greedy sampling.
        assert len(seq_ids) == 1
        next_token_id = torch.argmax(probs, dim=-1)
        next_token_ids = [int(next_token_id.item())]
        parent_seq_ids = seq_ids
    else:
        # Random sampling.
        # Sample 1 token for each sequence in the group.
        next_token_ids = torch.multinomial(probs,
                                           num_samples=1,
                                           replacement=True)
        next_token_ids = next_token_ids.squeeze(dim=-1).tolist()
        parent_seq_ids = seq_ids
    return parent_seq_ids, next_token_ids


def _sample(
    probs: torch.Tensor,
    logprobs: torch.Tensor,
    input_metadata: InputMetadata,
) -> SamplerOutput:
    seq_outputs: SamplerOutput = []

    # TODO: Optimize.
    idx = 0
    for i, seq_group in enumerate(input_metadata.seq_groups):
        seq_group_outputs: List[SequenceOutputs] = []
        seq_ids, sampling_params = seq_group
        if i < input_metadata.num_prompts:
            # Generate the next tokens for a prompt input.
            assert len(seq_ids) == 1, "Prompt input should have only one seq."
            parent_seq_id = seq_ids[0]
            prob = probs[idx]
            logprob = logprobs[idx]
            idx += 1

            # Sample the next tokens.
            next_token_ids = _sample_from_prompt(prob, sampling_params)
            # Get top-k log probabilities for the next tokens.
            next_logprobs = _get_topk_logprobs(logprob,
                                               sampling_params.logprobs)

            # Build the output.
            for next_token_id in next_token_ids:
                output_logprobs = next_logprobs.copy()
                output_logprobs[next_token_id] = logprob[next_token_id].item()
                seq_group_outputs.append(
                    SequenceOutputs(parent_seq_id, next_token_id,
                                    output_logprobs))
        else:
            # Generate the next tokens for generation tokens.
            num_parent_seqs = len(seq_ids)
            prob = probs[idx:idx + num_parent_seqs]
            logprob = logprobs[idx:idx + num_parent_seqs]
            idx += num_parent_seqs

            # Sample the next tokens.
            seq_logprobs = [
                input_metadata.seq_data[seq_id].cumulative_logprob
                for seq_id in seq_ids
            ]
            parent_seq_ids, next_token_ids = _sample_from_generation_tokens(
                seq_ids, prob, logprob, seq_logprobs, sampling_params)

            # Get top-k log probabilities for the next tokens.
            next_logprobs: Dict[int, Dict[int, float]] = {}
            for j, seq_id in enumerate(seq_ids):
                next_logprobs[seq_id] = _get_topk_logprobs(
                    logprob[j], sampling_params.logprobs)

            # Build the output.
            for parent_seq_id, next_token_id in zip(parent_seq_ids,
                                                    next_token_ids):
                j = seq_ids.index(parent_seq_id)
                output_logprobs = next_logprobs[parent_seq_id].copy()
                output_logprobs[next_token_id] = logprob[j,
                                                         next_token_id].item()
                seq_group_outputs.append(
                    SequenceOutputs(parent_seq_id, next_token_id,
                                    output_logprobs))
        seq_outputs.append(seq_group_outputs)

    return seq_outputs