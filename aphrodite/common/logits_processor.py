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


class NGramPenaltyProcessor(LogitsProcessor):
    """Use ngrams to apply repetition penalties to token sequences."""
    def __init__(self, penalty:float, prompt:list[int],
                 immune_sequences:list[list[int]] = [],
                 n_min:int=2, n_max:int=4, tokenizer=None):
        self._penalty = penalty
        self._n_min = n_min
        self._n_max = n_max
        self._tail = prompt[-n_max:]

        # What if, instead, I use word ngrams? That seems slightly insane.
        
        prompt_mask = [True] * len(prompt)  # If False, token is part of an ignored sequence, and should not be included in any penalties?
        for seq in immune_sequences:
            sidx = 0
            i = 0
            while i < len(prompt):
                if seq[sidx] == prompt[i]:
                    sidx += 1
                    if sidx == len(seq):
                        prompt_mask[i-len(seq):i] = False
                        i -= len(seq)  # Rewind so we can catch self-overlapping sequences.
                        sidx = 0
                else:
                    sidx = 0
                i += 1

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
                if prompt_mask[i]:
                    penalties[geh].update([prompt[i]])
                    top.update([tuple(prompt[i-ng:i+1])])
        print(f"created {len(penalties)} ngrams jfc, {top.total()} instances")
        print(f"Immunes:", immune_sequences)
        print('\n'.join([f"{tokenizer.convert_ids_to_tokens(x[0])}: {x[1]} ({self._penalty ** (x[1] / (self._n_max - self._n_min)):.02f})" for x in top.most_common()[:10]]))
        self._penalties = dict(penalties)

    def __call__(self, logits: torch.Tensor, output_tokens: list[list[int]]) -> None:
        for i,outputs in enumerate(output_tokens):
            # print("b4", logits.sort(dim=1, descending=True)[0][0,:10].tolist())
            tail = (self._tail + outputs)[-self._n_max:]
            for n in range(self._n_min, self._n_max):
                key = tuple(tail[-n:])
                if key in self._penalties:
                    for tok,pen in self._penalties[key].items():
                        pval = self._penalty ** (pen / (self._n_max - self._n_min))
                        logits[i,tok] /= pval if logits[i,tok] > 0 else 1/pval
            # print("AF", logits.sort(dim=1, descending=True)[0][0,:10].tolist())
        # logits[:, self._banned_token_ids] = -float("inf")



