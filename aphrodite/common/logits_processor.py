from abc import ABC, abstractmethod
import torch
from collections import defaultdict, Counter


class LogitsProcessor(ABC):

    @abstractmethod
    def __call__(self, logits: torch.Tensor, output_tokens: list[list[int]]) -> None:
        """Logits are edited in-place"""
        pass


class BiasLogitsProcessor(LogitsProcessor):
    """This is to enable logit_bias in the OpenAI server.
    biases is a dict where each value is -100 to 100
      according to the OpenAI API docs.
    Args:
      biases: Dict of values from -100 to 100 to scale the
        probability of a token being generated.
        Each key of the dict corresponds to the the token id.
    """

    def __init__(self, biases: dict[int, float]):
        self.biases = biases

        if not biases:
            return

        self.keys = torch.tensor(list(self.biases.keys()), dtype=torch.long)
        self.values = torch.tensor(list(self.biases.values()),
                                   dtype=torch.long)

    def __call__(self, logits: torch.Tensor, output_tokens: list[list[int]]) -> None:
        if not self.biases:
            return

        values = self.values.to(logits.device)
        keys = self.keys.to(logits.device)

        update_factors = torch.where(values >= 0, 1 + (values / 100),
                                     1 / (1 - (values / 100)))
        logits[0, keys] *= update_factors


class BanEOSUntil(LogitsProcessor):
    """Bans the EOS token until a certain condition is met.
    In this case, 'number of output tokens'.

    With this condition, both 'min_tokens' and 'ignore_eos'
    parameters can be handled gracefully."""

    def __init__(self, min_tokens:int, eos_token_id:int):
        self._min_tokens = min_tokens
        self._eos_token_id = eos_token_id

    def __call__(self, logits: torch.Tensor, output_tokens: list[list[int]]) -> None:
        for i in range(len(output_tokens)):
            if len(output_tokens[i]) < self._min_tokens:
                logits[i][self._eos_token_id] = -float("inf")


class BanTokens(LogitsProcessor):
    """Bans specified tokens."""
    def __init__(self, banned_token_ids:list[int]):
        self._banned_token_ids = banned_token_ids

    def __call__(self, logits: torch.Tensor, output_tokens: list[list[int]]) -> None:
        logits[:, self._banned_token_ids] = -float("inf")


class DynamicTemperature(LogitsProcessor):
    def __init__(self, tmin:float, tmax:float):
        self._tmin = tmin
        self._tvar = tmax - tmin

    def __call__(self, logits: torch.Tensor, output_tokens: list[list[int]]) -> None:
        probs = logits.softmax(dim=-1)
        entropy = (probs * probs.clamp(min=1e-7).log()).sum(dim=-1).neg_() # why the hell are there NaNs?
        norm_term = torch.count_nonzero(probs, dim=-1).float().log()
        norm_entropy = entropy / norm_term
        dyn_temps = self._tmin + self._tvar * norm_entropy
        dyn_temps[dyn_temps.isnan()] = 1.0
        
        # print(f"Geh.\n{entropy.tolist()}\n{norm_term.tolist()}\n{norm_entropy.tolist()}\n{dyn_temps.tolist()}")
        logits /= dyn_temps.clamp(min=1e-2) # multinomials are containing inf, NaN, or negative later on. What the hell?


class TopPolynomial(LogitsProcessor):  # Can be min_p, linear_a, or top_a depending on exponent provided
    def __init__(self, p:float, exp:float):
        self._p = p
        self._exp = exp

    def __call__(self, logits: torch.Tensor, output_tokens: list[list[int]]) -> None:
        logits_sort, logits_idx = logits.sort(dim=-1, descending=True)
        
        probs_sort = logits_sort.softmax(dim=-1)
        top_thresholds = torch.pow(probs_sort[:, 0], self._exp) * self._p
        top_mask = (probs_sort < top_thresholds.unsqueeze(1))  # Cull logits below the top-a threshold
        top_mask[:, 0] = False  # Guarantee at least one token is pickable
        logits_sort[top_mask] = -float("inf")

        # Put the masked logits back where they came from
        torch.gather(logits_sort, dim=-1,
                     index=torch.argsort(logits_idx, dim=-1),
                     out=logits)





class NGramPenaltyProcessor(LogitsProcessor):
    """Use ngrams to apply repetition penalties to token sequences."""
    def __init__(self, pres_pen:float, freq_pen:float,prompt:list[int],
                 immune_sequences:list[list[int]] = [],
                 n_min:int=2, n_max:int=4, tokenizer=None):
        self._pres_pen = pres_pen
        self._freq_pen = freq_pen
        self._n_min = n_min
        self._n_max = n_max
        self._tail = prompt[-n_max:]

        # What if, instead, I use word ngrams? That seems slightly insane.
        
        can_penalize = [True] * len(prompt)  # If False, token is part of an ignored sequence, and should not be penalized.
        for seq in immune_sequences:
            iseq = 0
            iprompt = 0
            while iprompt < len(prompt):
                if seq[iseq] == prompt[iprompt]:
                    iseq += 1
                    if iseq == len(seq):
                        can_penalize[iprompt-len(seq)+1:iprompt+1] = [False] * len(seq)
                        iprompt -= len(seq)-1  # Rewind so we can catch self-overlapping sequences.
                        # print(f"Found {repr(seq)} at {iprompt}, rewinding {len(seq)} ")
                        iseq = 0
                else:
                    iseq = 0
                iprompt += 1
        # print(f"Masked out {len(can_penalize) - sum(can_penalize)} tokens")

        # I'm not sure that's what we want. We DO want penalties for starting every reply the exact same way.
        # What we DON'T want is penalties on the instruct sequence _itself_.
        #   Does that mean the 'weight' is moved to the other tokens in those sequences? Or just not counted at all?
        #Ahh, hm. We want to nullify the penalty only when RESULTING TOKENS are part of the protected sequences.
        #   Protected sequences can still *contribute* to penalties for other tokens.

        penalties = defaultdict(lambda:Counter())  # seq -> penalized id -> penalty
        top = Counter()
        for ng in range(n_min,n_max):
            for i in range(ng, len(prompt)):
                geh = tuple(prompt[i-ng:i])
                if can_penalize[i]:
                    penalties[geh].update([prompt[i]])
                    top.update([tuple(prompt[i-ng:i+1])])

        # print(f"created {len(penalties)} ngrams jfc, {top.total()} instances")
        # print(f"Immunes (tok):", [tokenizer.convert_ids_to_tokens(x) for x in immune_sequences])
        # print('\n'.join([f"{tokenizer.convert_ids_to_tokens(x[0])}: {x[1]} ({self._pres_pen + self._freq_pen * (x[1] / (self._n_max - self._n_min)):.02f})" for x in top.most_common()[:10]]))
        self._penalties = dict(penalties)

    def __call__(self, logits: torch.Tensor, output_tokens: list[list[int]]) -> None:
        for i,outputs in enumerate(output_tokens):
            tail = (self._tail + outputs)[-self._n_max:]
            for n in range(self._n_min, self._n_max):
                key = tuple(tail[-n:])
                if key in self._penalties:
                    for tok,pen in self._penalties[key].items():
                        # pval = self._penalty ** (pen / (self._n_max - self._n_min))
                        # logits[i,tok] /= pval if logits[i,tok] > 0 else 1/pval
                        logits[i,tok] -= self._pres_pen + self._freq_pen * (pen / (self._n_max - self._n_min))



