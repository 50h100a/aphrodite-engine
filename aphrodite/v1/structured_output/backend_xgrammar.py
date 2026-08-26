# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

import aphrodite.envs
from aphrodite.logger import init_logger
from aphrodite.sampling_params import SamplingParams
from aphrodite.utils.import_utils import LazyLoader
from aphrodite.utils.mistral import is_mistral_tokenizer
from aphrodite.v1.structured_output.backend_types import (
    StructuredOutputBackend,
    StructuredOutputGrammar,
    StructuredOutputOptions,
)
from aphrodite.v1.structured_output.utils import (
    choice_as_grammar,
    compile_regex_with_timeout,
    convert_lark_to_ebnf,
    grammar_is_likely_lark,
)

if TYPE_CHECKING:
    import xgrammar as xgr
else:
    xgr = LazyLoader("xgr", globals(), "xgrammar")

logger = init_logger(__name__)


@dataclass
class XgrammarBackend(StructuredOutputBackend):
    def __post_init__(self):
        self.disable_any_whitespace = self.aphrodite_config.structured_outputs_config.disable_any_whitespace

        self._engine_stop_ids: list[int] | None = None
        self.compiler = self._build_compiler(None)
        # Keyed by the stop token ids handed to xgrammar. Only populated for
        # tags that need the default stop token as a grammar terminal, so most
        # deployments never build a second one. See _compiler_for_structural_tag.
        self._tag_compilers: dict[frozenset[int], xgr.GrammarCompiler] = {}

        self.num_speculative_tokens = 0
        if self.aphrodite_config.speculative_config is not None:
            self.num_speculative_tokens = self.aphrodite_config.speculative_config.num_speculative_tokens

    def _build_tokenizer_info(self, stop_token_ids: list[int] | None):
        if is_mistral_tokenizer(self.tokenizer):
            # NOTE: ideally, xgrammar should handle this accordingly.
            # refer to https://github.com/mlc-ai/xgrammar/blob/d77c0a0173ef14779c918e3be7966ba852f7910f/python/xgrammar/tokenizer_info.py#L98
            if stop_token_ids is None:
                stop_token_ids = [self.tokenizer.eos_token_id]

            # not self.tokenizer.vocab_size as self.tokenizer.vocab
            # collapses all decoded errors into a single token.
            self.vocab_size = len(self.tokenizer.vocab)
            return xgr.TokenizerInfo(  # type: ignore
                encoded_vocab=self.tokenizer.vocab,
                # NOTE: https://github.com/mlc-ai/xgrammar/blob/5e141f6ff1ca02bc31f9e512e68b61f2a8ae88e5/tests/python/test_tokenizer_info.py#L43 # noqa: E501
                vocab_type=xgr.VocabType.RAW if self.tokenizer.is_tekken else xgr.VocabType.BYTE_FALLBACK,
                vocab_size=self.vocab_size,
                stop_token_ids=stop_token_ids,
                add_prefix_space=True,
            )
        # `stop_token_ids=None` is what xgrammar defaults to, so this stays the
        # behaviour it has always had unless a caller asks for something else.
        return xgr.TokenizerInfo.from_huggingface(
            self.tokenizer,
            vocab_size=self.vocab_size,
            stop_token_ids=stop_token_ids,
        )

    def _build_compiler(self, stop_token_ids: list[int] | None) -> xgr.GrammarCompiler:
        return xgr.GrammarCompiler(
            self._build_tokenizer_info(stop_token_ids),
            max_threads=8,
            cache_enabled=True,
            cache_limit_bytes=aphrodite.envs.APHRODITE_XGRAMMAR_CACHE_MB * 1024 * 1024,
        )

    def _engine_stop_token_ids(self) -> list[int]:
        """The token ids the engine itself will stop on, primary EOS first.

        Mirrors what `SamplingParams.update_from_generation_config` assembles:
        the tokenizer's EOS, plus every id in the model's
        `generation_config.json`. gpt-oss contributes `<|return|>`,
        `<|endoftext|>` and `<|call|>` here.
        """
        if self._engine_stop_ids is not None:
            return self._engine_stop_ids

        stop_ids: list[int] = []
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if eos_token_id is not None:
            stop_ids.append(eos_token_id)

        try:
            generation_config = self.aphrodite_config.model_config.try_get_generation_config()
        except Exception:
            # Resolved lazily, on the first tag that needs it. A model whose
            # generation config cannot be read is not a reason to fail the
            # request -- the tokenizer's EOS alone is what we had before.
            logger.warning("Could not read the generation config; falling back to the tokenizer's EOS.")
            generation_config = {}

        configured = generation_config.get("eos_token_id")
        if isinstance(configured, int):
            configured = [configured]
        for token_id in configured or ():
            if token_id not in stop_ids:
                stop_ids.append(token_id)

        self._engine_stop_ids = stop_ids
        return stop_ids

    def _token_text(self, token_id: int) -> str | None:
        convert = getattr(self.tokenizer, "convert_ids_to_tokens", None)
        if convert is not None:
            return convert(token_id)
        return None

    def _compiler_for_structural_tag(self, grammar_spec: str) -> xgr.GrammarCompiler:
        """Pick a compiler whose stop tokens the tag does not need to emit.

        xgrammar reserves its stop tokens: they are unmasked only once the
        grammar is in an accepting state, and can never be produced as a
        terminal *inside* a rule. That is fine for a schema, which ends by
        running out of grammar, but not for a tag whose own terminator is the
        model's EOS -- the harmony tag closes a `final` message with
        `<|return|>`, which is exactly gpt-oss's EOS. Compiled against the
        default tokenizer info, that tag can never be closed: the model sits in
        the message with its intended next token masked and runs to max_tokens.

        So when the tag's text contains the default stop token, compile it
        against the remaining engine stop ids instead (for harmony, that leaves
        `<|endoftext|>`). Every other grammar type keeps the default compiler,
        which it must: a completed JSON schema leaves the real EOS as the only
        legal token, and substituting it there would reintroduce the same hang.
        """
        stop_ids = self._engine_stop_token_ids()
        if not stop_ids:
            return self.compiler

        usable = [
            token_id
            for token_id in stop_ids
            if (text := self._token_text(token_id)) is None or text not in grammar_spec
        ]
        # The default stop token is not something this tag has to emit, so the
        # compiler everything else uses is already correct for it.
        if stop_ids[0] in usable:
            return self.compiler
        if not usable:
            logger.warning(
                "Structural tag uses every engine stop token (%s) as a grammar terminal, so none "
                "can be reserved for xgrammar. The tag may be unable to terminate.",
                stop_ids,
            )
            return self.compiler

        key = frozenset(usable)
        compiler = self._tag_compilers.get(key)
        if compiler is None:
            logger.info(
                "Structural tag needs stop token %s as a grammar terminal; compiling it against "
                "stop token ids %s so it can be emitted.",
                stop_ids[0],
                usable,
            )
            compiler = self._tag_compilers[key] = self._build_compiler(sorted(usable))
        return compiler

    def compile_grammar(self, request_type: StructuredOutputOptions, grammar_spec: str) -> StructuredOutputGrammar:
        if request_type == StructuredOutputOptions.JSON:
            ctx = self.compiler.compile_json_schema(grammar_spec, any_whitespace=not self.disable_any_whitespace)
        elif request_type == StructuredOutputOptions.JSON_OBJECT:
            ctx = self.compiler.compile_json_schema(
                '{"type": "object"}', any_whitespace=not self.disable_any_whitespace
            )
        elif request_type == StructuredOutputOptions.GRAMMAR:
            ctx = self.compiler.compile_grammar(grammar_spec)
        elif request_type == StructuredOutputOptions.REGEX:
            ctx = compile_regex_with_timeout(
                self.compiler.compile_regex,
                grammar_spec,
            )
        elif request_type == StructuredOutputOptions.STRUCTURAL_TAG:
            # A tag can name the model's EOS as one of its own terminators, and
            # xgrammar will not hand back a token it has reserved for stopping.
            compiler = self._compiler_for_structural_tag(grammar_spec)
            s_tag = json.loads(grammar_spec)
            if "structures" in s_tag:
                # Falling back to deprecated method of compiling structural tag
                tags = [
                    xgr.StructuralTagItem(
                        begin=s["begin"],
                        schema=json.dumps(s["schema"]),
                        end=s["end"],
                    )
                    for s in s_tag["structures"]
                ]
                ctx = compiler.compile_structural_tag(tags, s_tag["triggers"])
            else:
                ctx = compiler.compile_structural_tag(grammar_spec)
        else:
            logger.error("Validation should have already occurred. Please file an issue.")
            raise ValueError(f"grammar is not of valid supported types. ({request_type!s})")

        return XgrammarGrammar(
            matcher=xgr.GrammarMatcher(
                ctx,
                max_rollback_tokens=self.num_speculative_tokens,
            ),
            vocab_size=self.vocab_size,
            ctx=ctx,
        )

    def allocate_token_bitmask(self, max_num_seqs: int):
        return xgr.allocate_token_bitmask(max_num_seqs, self.vocab_size)

    def destroy(self):
        del self.compiler
        self._tag_compilers.clear()


@dataclass
class XgrammarGrammar(StructuredOutputGrammar):
    # NOTE: This would be a generic-enough class for
    # supporting different backends, in the future.
    # For now, just xgrammar.
    #
    # https://xgrammar.mlc.ai/docs/api/python/index.html#xgrammar.GrammarMatcher.find_jump_forward_string
    # for jump-forward decoding

    vocab_size: int
    matcher: xgr.GrammarMatcher = field(hash=False)
    ctx: xgr.CompiledGrammar = field(hash=False)
    num_processed_tokens: int = field(default_factory=lambda: 0, repr=False, hash=False, init=False)
    _is_terminated: bool = field(default=False, repr=False, hash=False)

    def accept_tokens(self, request_id: str, tokens: list[int]) -> bool:
        """Accepts a list of tokens and advances the FSM.

        Returns True if the FSM was advanced successfully.
        Returns False if the FSM failed to advance.
        """
        if self._is_terminated:
            return False
        for token in tokens:
            if not self.matcher.accept_token(token):
                logger.error(
                    "Failed to advance FSM for request %s for tokens %s. Please file an issue.",
                    request_id,
                    token,
                )
                return False
            self.num_processed_tokens += 1
            if self.matcher.is_terminated():
                # Specdec may emit EOS and then some. Stop before then.
                break
        self._is_terminated = self.matcher.is_terminated()
        return True

    def validate_tokens(self, tokens: list[int]) -> list[int]:
        """Checks if the list of tokens are accepted by the FSM in sequence.
        Will not advance the FSM.

        Returns the prefix list of tokens that are accepted by the FSM.
        """
        if self._is_terminated:
            return []
        accepted_tokens = []
        for token in tokens:
            if not self.matcher.accept_token(token):
                break
            accepted_tokens.append(token)
            if self.matcher.is_terminated():
                # stop probing a finished matcher
                break
        if len(accepted_tokens) > 0:
            # Rollback the FSM to the initial state
            self.matcher.rollback(len(accepted_tokens))
        return accepted_tokens

    def rollback(self, num_tokens: int) -> None:
        self.matcher.rollback(num_tokens)
        self.num_processed_tokens -= num_tokens
        self._is_terminated = self.matcher.is_terminated()

    def fill_bitmask(self, bitmask: torch.Tensor, idx: int) -> None:
        # xgrammar rejects a bitmask whose rows are not exactly its own vocab
        # width. The bitmask is shared with the other backends and sized for
        # the widest of them, so hand xgrammar a view of the right width when
        # they differ.
        num_words = (self.vocab_size + 31) // 32
        if bitmask.shape[-1] != num_words:
            self.matcher.fill_next_token_bitmask(bitmask[idx : idx + 1, :num_words], 0)
            return
        self.matcher.fill_next_token_bitmask(bitmask, idx)

    def is_terminated(self) -> bool:
        return self._is_terminated

    def reset(self):
        self.num_processed_tokens = 0
        self.matcher.reset()


# cf https://github.com/mlc-ai/xgrammar/blob/a32ac892676d2eedc0327416105b9b06edfb94b2/cpp/json_schema_converter.cc
STRING_SUPPORTED_FORMATS = {
    "email",
    "date",
    "time",
    "date-time",
    "duration",
    "ipv4",
    "ipv6",
    "hostname",
    "uuid",
    "uri",
    "uri-reference",
    "uri-template",
    "json-pointer",
    "relative-json-pointer",
}


def has_xgrammar_unsupported_json_features(schema: dict[str, Any]) -> bool:
    """Check if JSON schema contains features unsupported by xgrammar."""

    def check_object(obj: dict[str, Any]) -> bool:
        if not isinstance(obj, dict):
            return False

        # Check for numeric ranges
        if obj.get("type") in ("integer", "number") and ("multipleOf" in obj):
            return True

        # Check for array unsupported keywords
        if obj.get("type") == "array" and any(
            key in obj for key in ("uniqueItems", "contains", "minContains", "maxContains")
        ):
            return True

        # Unsupported keywords for strings
        if obj.get("type") == "string" and "format" in obj and obj["format"] not in STRING_SUPPORTED_FORMATS:
            return True

        # Unsupported keywords for objects
        if obj.get("type") == "object" and any(key in obj for key in ("patternProperties", "propertyNames")):
            return True

        # Recursively check all nested objects and arrays
        for value in obj.values():
            if isinstance(value, dict):
                if check_object(value):
                    return True
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict) and check_object(item):
                        return True

        return False

    return check_object(schema)


def validate_xgrammar_grammar(sampling_params: SamplingParams) -> None:
    """Validate that the request is supported by structured output.

    Raises ValueError if the request is not supported.
    """
    if sampling_params.structured_outputs is None:
        return

    so_params = sampling_params.structured_outputs

    if so_params.regex:
        try:
            compile_regex_with_timeout(
                xgr.Grammar.from_regex,
                so_params.regex,
            )
        except Exception as err:
            raise ValueError(f"Failed to transform regex into a grammar: {err}") from err

    if so_params.choice:
        choice_grammar = choice_as_grammar(so_params.choice)
        try:
            xgr.Grammar.from_ebnf(choice_grammar)
        except Exception as err:
            raise ValueError(f"Failed to transform choices into a grammar: {err}") from err
        so_params.choice = None
        so_params.grammar = choice_grammar
        return

    if so_params.json:
        if isinstance(so_params.json, str):
            try:
                schema = json.loads(so_params.json)
            except json.JSONDecodeError as e:
                raise ValueError("Invalid JSON grammar specification.") from e
        else:
            schema = so_params.json

        if has_xgrammar_unsupported_json_features(schema):
            raise ValueError("The provided JSON schema contains features not supported by xgrammar.")

        try:
            xgr.Grammar.from_json_schema(schema)
        except Exception as err:
            raise ValueError(f"Failed to transform json schema into a grammar: {err}") from err
        return

    if so_params.grammar:
        if grammar_is_likely_lark(so_params.grammar):
            # xgrammar supports EBNF grammars only
            try:
                so_params.grammar = convert_lark_to_ebnf(so_params.grammar)
            except ValueError as e:
                raise ValueError("Failed to convert the grammar from Lark to EBNF. ") from e

        # Test parsing EBNF grammar, possibly already converted from Lark
        try:
            # parse the grammar, but we aren't compiling it.
            xgr.Grammar.from_ebnf(so_params.grammar)
        except Exception as e:
            raise ValueError("Invalid grammar specification.") from e
        return

    if so_params.structural_tag:
        try:
            s_tag = json.loads(so_params.structural_tag)

            # Using the deprecated method of compiling structural tag
            if "structures" in s_tag:
                tags = [
                    xgr.StructuralTagItem(
                        begin=s["begin"],
                        schema=json.dumps(s["schema"]),
                        end=s["end"],
                    )
                    for s in s_tag["structures"]
                ]
                xgr.Grammar.from_structural_tag(tags, s_tag["triggers"])
            else:
                xgr.Grammar.from_structural_tag(so_params.structural_tag)
        except Exception as e:
            raise ValueError(f"Invalid structural tag specification: {e}") from e
