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


def _admit_commentary_preamble(tag):
    """Add the recipient-less commentary channel to a harmony structural tag.

    gpt-oss narrates on `commentary` with no recipient before it calls a tool,
    and ``HarmonyParser`` reads that segment back as ordinary content. The
    xgrammar harmony template lists only `analysis`, `final`, and the
    `commentary to=functions.*` call forms, and the tag is a closed alternation
    rather than a set of triggers -- so leaving the preamble out does not make
    it optional, it makes it unreachable.
    """
    from xgrammar.structural_tag import AnyTextFormat, TagFormat, TagsWithSeparatorFormat

    fmt = tag.format
    if not isinstance(fmt, TagsWithSeparatorFormat):
        return tag
    if any(getattr(t.begin, "begin", t.begin) == _COMMENTARY_PREAMBLE for t in fmt.tags):
        return tag

    # Borrow the terminators the template already uses for free text.
    ends = [t.end for t in fmt.tags if isinstance(t.content, AnyTextFormat)]
    end = ends[0] if ends else ["<|end|>", "<|return|>"]

    tag = tag.model_copy(deep=True)
    tag.format.tags.append(
        TagFormat(begin=_COMMENTARY_PREAMBLE, content=AnyTextFormat(), end=end)
    )
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
