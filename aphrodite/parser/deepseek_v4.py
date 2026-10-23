# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4 parser: ``<think>``/``</think>``
reasoning plus DSML tool calls in a single state machine.

DeepSeek V4 output format::

    <think>
    ...reasoning...
    </think>
    <｜DSML｜tool_calls>
    <｜DSML｜invoke name="func_name">
    <｜DSML｜parameter name="location" string="true">杭州</｜DSML｜parameter>
    <｜DSML｜parameter name="count" string="false">5</｜DSML｜parameter>
    </｜DSML｜invoke>
    </｜DSML｜tool_calls>
"""

from __future__ import annotations

import contextlib
import functools
import json
from typing import TYPE_CHECKING

import regex as re

from aphrodite.parser.engine.events import EventType
from aphrodite.parser.engine.parser_engine import ParserEngine
from aphrodite.parser.engine.parser_engine_config import (
    ParserEngineConfig,
    ParserState,
    Transition,
)
from aphrodite.tool_parsers.utils import find_tool_properties

if TYPE_CHECKING:
    from aphrodite.tokenizers import TokenizerLike
    from aphrodite.tool_parsers.abstract_tool_parser import Tool

_DSML = "｜DSML｜"

DSML_THINK_START = "<think>"
DSML_THINK_END = "</think>"
DSML_TOOL_START = f"<{_DSML}tool_calls>"
DSML_TOOL_END = f"</{_DSML}tool_calls>"
DSML_INVOKE_PREFIX = f'<{_DSML}invoke name="'
DSML_INVOKE_NAME_END = '">'
DSML_INVOKE_END = f"</{_DSML}invoke>"
DSML_PARAM_START = f"<{_DSML}parameter"
DSML_PARAM_CLOSE = f"</{_DSML}parameter>"

# Spellings variants of ``DSML_TOOL_START`` observed in production.
DSML_TOOL_START_VARIANTS: tuple[str, ...] = (
    f"<{_DSML}toolcalls>",
    f"<{_DSML}tool>",
)


def _param_patterns(param_start: str, param_close: str) -> tuple[re.Pattern, re.Pattern]:
    """Complete and trailing-partial parameter regexes for one DSML dialect.

    The ``string`` attribute is optional: the model sometimes omits it, and
    dropping such a parameter hands the client a tool call with no arguments.
    """
    start, close = re.escape(param_start), re.escape(param_close)
    head = rf'{start}\s+name="([^"]+)"(?:\s+string="([^"]*)")?\s*>'
    return (
        re.compile(rf"{head}(.*?)(?:{close}|(?={start}\s+name=))", re.DOTALL),
        re.compile(rf"{head}(.*)$", re.DOTALL),
    )


_PARAM_RE, _PARTIAL_PARAM_RE = _param_patterns(DSML_PARAM_START, DSML_PARAM_CLOSE)


# The whitespace xgrammar's `deepseek_xml` style allows around a parameter
# value. Its generated rule for one parameter reads
#
#     "\">" [ \n\t]* <value> [ \n\t]* "</｜DSML｜parameter>"
#
# so a model generating under the grammar may indent its value and still be
# inside the grammar. That padding is formatting, not data -- but the parser
# writes the slot out verbatim, which turns e.g. a tab before an enum member
# into part of the member and produces a tool call whose arguments violate
# the tool's own schema.
_DSML_VALUE_PADDING = " \t\n"


def _dsml_declared_type(prop: object) -> str | None:
    """The single JSON type this parameter is declared to hold, if it has one.

    ``None`` whenever the schema leaves any room -- absent, composed with
    ``anyOf``/``oneOf``/``allOf``, or a union of types. The value slot is then
    ambiguous enough that the safe reading of both the padding and the model's
    ``string=`` flag is the literal one.
    """
    if not isinstance(prop, dict):
        return None
    if any(key in prop for key in ("anyOf", "oneOf", "allOf")):
        return None
    declared = prop.get("type")
    return declared if isinstance(declared, str) else None


def _dsml_param_value(name: str, is_str: str, raw: str, properties: dict[str, object] | None) -> object:
    """Read one parameter's value slot, letting the schema settle what it is.

    Three cases, and the schema rather than the model's ``string=`` flag picks
    between them, because the grammar constrains the value slot but leaves the
    flag free -- the model may mark a number as a string, or a string as a
    number, without ever leaving the grammar.

    * A string with an ``enum``/``const``: the padding cannot be part of any
      value the schema admits, so it comes off.
    * Any other declared type: the slot holds a JSON literal, so the padding is
      formatting and the text parses.
    * A free string, or nothing declared: the padding is indistinguishable from
      the value -- ``xml_string`` matches it either way -- so it stays, as
      minimax_m2 keeps it for the same reason.
    """
    declared = _dsml_declared_type(properties.get(name) if properties else None)
    prop = properties.get(name) if properties else None

    if declared == "string":
        pinned = isinstance(prop, dict) and ("enum" in prop or "const" in prop)
        return raw.strip(_DSML_VALUE_PADDING) if pinned else raw

    if declared is not None and declared != "string":
        text = raw.strip(_DSML_VALUE_PADDING)
        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return text

    if is_str == "true":
        return raw
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return raw


def _dsml_arg_converter(
    raw_args: str,
    partial: bool,
    *,
    param_re: re.Pattern = _PARAM_RE,
    partial_param_re: re.Pattern = _PARTIAL_PARAM_RE,
    properties: dict[str, object] | None = None,
) -> str:
    params: dict[str, object] = {}

    last_end = 0
    for match in param_re.finditer(raw_args):
        name, is_str, value = match.group(1), match.group(2), match.group(3)
        params[name] = _dsml_param_value(name, is_str, value, properties)
        last_end = match.end()

    if partial:
        partial_match = partial_param_re.search(raw_args, last_end)
        if partial_match:
            name = partial_match.group(1)
            is_str = partial_match.group(2)
            value = partial_match.group(3)
            declared = _dsml_declared_type((properties or {}).get(name) if properties else None)
            if declared == "string" or (declared is None and is_str == "true"):
                # Text either way, so a half-written value reads like a finished
                # one. Taking the padding off a prefix stays right as the prefix
                # grows, which keeps the streamed argument deltas a lengthening
                # string rather than one that jumps backwards.
                params[name] = _dsml_param_value(name, is_str, value, properties)
            else:
                # A literal, and half of one does not parse -- `1.` is not yet a
                # number. Leave the parameter out until it does rather than
                # publish the fragment as a string and retract it a token later.
                with contextlib.suppress(json.JSONDecodeError, ValueError):
                    params[name] = json.loads(value.strip(_DSML_VALUE_PADDING) if declared else value)

    return json.dumps(params, ensure_ascii=False)


def _unwrap_wrapper_args(
    args_json: str,
    tools: list[Tool] | None,
    func_name: str | None,
) -> str:
    if not tools or not func_name:
        return args_json
    try:
        args = json.loads(args_json)
    except (json.JSONDecodeError, ValueError):
        return args_json
    if not isinstance(args, dict):
        return args_json
    properties = find_tool_properties(tools, func_name)
    if not properties:
        return args_json
    allowed = set(properties.keys())
    for wrapper in ("arguments", "input"):
        if set(args.keys()) != {wrapper} or wrapper in allowed:
            continue
        inner = args[wrapper]
        if isinstance(inner, str):
            try:
                inner = json.loads(inner)
            except json.JSONDecodeError:
                return args_json
        if isinstance(inner, dict) and set(inner.keys()).issubset(allowed):
            return json.dumps(inner, ensure_ascii=False)
    return args_json


@functools.cache
def deepseek_v4_config(thinking: bool = False) -> ParserEngineConfig:
    return ParserEngineConfig(
        name="deepseek_v4",
        initial_state=ParserState.REASONING if thinking else ParserState.CONTENT,
        wait_for_reasoning=thinking,
        terminals={
            "THINK_START": DSML_THINK_START,
            "THINK_END": DSML_THINK_END,
            "TOOL_START": (DSML_TOOL_START, *DSML_TOOL_START_VARIANTS),
            "TOOL_END": DSML_TOOL_END,
            "INVOKE_PREFIX": DSML_INVOKE_PREFIX,
            "INVOKE_NAME_END": DSML_INVOKE_NAME_END,
            "INVOKE_END": DSML_INVOKE_END,
            "PARAM_START": DSML_PARAM_START,
            "PARAM_CLOSE": DSML_PARAM_CLOSE,
        },
        token_id_terminals={
            "THINK_START": DSML_THINK_START,
            "THINK_END": DSML_THINK_END,
            "TOOL_START": DSML_TOOL_START,
            "TOOL_END": DSML_TOOL_END,
        },
        transitions={
            (ParserState.CONTENT, "THINK_START"): Transition(
                ParserState.REASONING,
                (EventType.REASONING_START,),
            ),
            (ParserState.CONTENT, "THINK_END"): Transition(
                ParserState.CONTENT,
                (),
            ),
            (ParserState.REASONING, "THINK_START"): Transition(
                ParserState.REASONING,
                (),
            ),
            (ParserState.REASONING, "THINK_END"): Transition(
                ParserState.CONTENT,
                (EventType.REASONING_END,),
            ),
            (ParserState.REASONING, "TOOL_START"): Transition(
                ParserState.TOOL_PREAMBLE,
                (EventType.REASONING_END,),
            ),
            (ParserState.CONTENT, "TOOL_START"): Transition(
                ParserState.TOOL_PREAMBLE,
                (),
            ),
            (ParserState.TOOL_PREAMBLE, "INVOKE_PREFIX"): Transition(
                ParserState.TOOL_NAME,
                (EventType.TOOL_CALL_START,),
            ),
            (ParserState.CONTENT, "INVOKE_PREFIX"): Transition(
                ParserState.TOOL_NAME,
                (EventType.TOOL_CALL_START,),
            ),
            (ParserState.TOOL_NAME, "INVOKE_NAME_END"): Transition(
                ParserState.TOOL_ARGS,
                (),
            ),
            (ParserState.TOOL_ARGS, "INVOKE_END"): Transition(
                ParserState.TOOL_BETWEEN,
                (EventType.TOOL_CALL_END,),
            ),
            (ParserState.TOOL_ARGS, "TOOL_END"): Transition(
                ParserState.TOOL_BETWEEN,
                (EventType.TOOL_CALL_END,),
            ),
            (ParserState.TOOL_BETWEEN, "INVOKE_PREFIX"): Transition(
                ParserState.TOOL_NAME,
                (EventType.TOOL_CALL_START,),
            ),
            # A tool call ends the turn: stay in TOOL_BETWEEN so any text
            # after the block is dropped.
            (ParserState.TOOL_BETWEEN, "TOOL_END"): Transition(
                ParserState.TOOL_BETWEEN,
                (),
            ),
            (ParserState.TOOL_BETWEEN, "TOOL_START"): Transition(
                ParserState.TOOL_PREAMBLE,
                (),
            ),
        },
        content_events={
            ParserState.CONTENT: EventType.TEXT_CHUNK,
            ParserState.REASONING: EventType.REASONING_CHUNK,
            ParserState.TOOL_NAME: EventType.TOOL_NAME,
            ParserState.TOOL_ARGS: EventType.ARG_VALUE_CHUNK,
        },
        arg_converter=_dsml_arg_converter,
        arg_structural_chars=frozenset(">"),
        strip_content_whitespace_with_tools=False,
        tool_args_json=False,
    )


class DeepSeekV4Parser(ParserEngine):
    parser_config = staticmethod(deepseek_v4_config)

    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
        **kwargs,
    ) -> None:
        chat_kwargs = kwargs.pop("chat_template_kwargs", None) or {}
        thinking = bool(chat_kwargs.get("thinking") or chat_kwargs.get("enable_thinking"))
        if "thinking" not in chat_kwargs and "enable_thinking" not in chat_kwargs:
            thinking = True
        thinking = thinking and chat_kwargs.get("reasoning_effort") != "none"
        super().__init__(
            tokenizer,
            tools,
            parser_engine_config=self.parser_config(thinking=thinking),
            **kwargs,
        )
        self._arg_converter = self._convert_args

    def _convert_args(self, raw_args: str, partial: bool) -> str:
        converter = self.parser_engine_config.arg_converter
        assert converter is not None
        func_name = next((slot.name for slot in self._tool_slots if slot.args == raw_args), None)
        properties = find_tool_properties(self._tools, func_name or "")
        # Native and V4.1 converters accept the same schema context.
        result = converter(raw_args, partial, properties=properties)
        if not self._tools:
            return result
        func_name = next((s.name for s in self._tool_slots if s.args == raw_args), None)
        return _unwrap_wrapper_args(result, self._tools, func_name)
