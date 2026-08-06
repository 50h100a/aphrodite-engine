# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from typing import TYPE_CHECKING

from aphrodite.entrypoints.openai.engine.protocol import (
    DeltaMessage,
    ExtractedToolCallInformation,
)
from aphrodite.tool_parsers.abstract_tool_parser import Tool, ToolParser

if TYPE_CHECKING:
    from aphrodite.tokenizers import TokenizerLike


_COMMENTARY_PREAMBLE = "<|channel|>commentary<|message|>"
_FINAL_CHANNEL = "<|channel|>final<|message|>"


def _tag_begin(tag) -> str | None:
    begin = tag.begin
    return begin if isinstance(begin, str) else None


def _admit_commentary_preamble(tag):
    """Add the recipient-less commentary channel to a harmony structural tag.

    gpt-oss narrates on `commentary` with no recipient before it calls a tool,
    and ``HarmonyParser`` reads that segment back as ordinary content. The
    xgrammar harmony template lists only `analysis`, `final`, and the
    `commentary to=functions.*` call forms, and the tag is a closed alternation
    rather than a set of triggers -- so leaving the preamble out does not make
    it optional, it makes it unreachable.

    Only added when the template already admits a free-text reply. Under
    `tool_choice="required"` it drops the `final` channel to push the model
    toward a call, and a preamble is free text by another name: adding it there
    would hand back the escape the template just took away.
    """
    from xgrammar.structural_tag import AnyTextFormat, TagFormat, TagsWithSeparatorFormat

    fmt = tag.format
    if not isinstance(fmt, TagsWithSeparatorFormat):
        return tag

    begins = {_tag_begin(t) for t in fmt.tags}
    if _COMMENTARY_PREAMBLE in begins or _FINAL_CHANNEL not in begins:
        return tag

    # Borrow the terminators the template already uses for the free-text reply.
    final = next(t for t in fmt.tags if _tag_begin(t) == _FINAL_CHANNEL)

    tag = tag.model_copy(deep=True)
    tag.format.tags.append(TagFormat(begin=_COMMENTARY_PREAMBLE, content=AnyTextFormat(), end=final.end))
    return tag


class GptOssToolParser(ToolParser):
    """
    Stub tool parser for gpt-oss/harmony models.

    All output parsing is handled by HarmonyParser. This stub exists as a
    capability declaration via HarmonyParser.tool_parser_cls.
    """

    structural_tag_model = "harmony"

    def __init__(self, tokenizer: "TokenizerLike", tools: list[Tool] | None = None):
        super().__init__(tokenizer, tools)

    def get_structural_tag(self, request, *, reasoning: bool = False):
        tag = super().get_structural_tag(request, reasoning=reasoning)
        if tag is None:
            return None
        return _admit_commentary_preamble(tag)

    def extract_tool_calls(self, model_output, request, **kwargs) -> ExtractedToolCallInformation:
        raise NotImplementedError("GptOssToolParser is a stub. Use HarmonyParser for tool parsing.")

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request,
    ) -> DeltaMessage | None:
        raise NotImplementedError("GptOssToolParser is a stub. Use HarmonyParser for tool parsing.")
