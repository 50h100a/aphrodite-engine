# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""The grammar that makes a tool call unrepresentable.

`exclusion_grammar` must accept exactly the strings containing none of its
markers: no more (or a tool call slips through) and no less (or ordinary prose
that happens to look like one gets denied mid-sentence).

The language is checked by simulating the generated EBNF directly, against
`marker in text` as the oracle. The simulator below reads the grammar back as
a transition table; it shares no code with the Aho-Corasick construction that
produced it, so agreement means something.
"""

import random
import re

import pytest

from aphrodite.parser.tool_call_exclusion import exclusion_grammar

QWEN3_MARKERS = ["<tool_call>", "<function="]
DEEPSEEK_MARKERS = ["<｜DSML｜function_calls>"]

_RULE = re.compile(r"^s(\d+) ::= (.*)$")
_LITERAL = re.compile(r'^"(.*)" s(\d+)$')
_CLASS = re.compile(r"^\[\^(.*)\] s(\d+)$")


def _unescape(body: str) -> str:
    out, i = [], 0
    while i < len(body):
        if body[i] == "\\":
            if body[i + 1] == "u":
                out.append(chr(int(body[i + 2 : i + 6], 16)))
                i += 6
                continue
            out.append(body[i + 1])
            i += 2
            continue
        out.append(body[i])
        i += 1
    return "".join(out)


def accepts(grammar: str, text: str) -> bool:
    """Whether the generated grammar derives ``text``."""
    literal: dict[tuple[int, str], int] = {}
    default: dict[int, int] = {}
    excluded: dict[int, set[str]] = {}
    start = None
    for line in grammar.splitlines():
        if line.startswith("root ::= "):
            start = int(line.removeprefix("root ::= s"))
            continue
        state, body = _RULE.match(line).groups()
        state = int(state)
        for alt in body.split(" | "):
            if alt == '""':
                continue
            if m := _CLASS.match(alt):
                excluded[state] = set(_unescape(m.group(1)))
                default[state] = int(m.group(2))
                continue
            m = _LITERAL.match(alt)
            literal[(state, _unescape(m.group(1)))] = int(m.group(2))

    state = start
    for ch in text:
        if (state, ch) in literal:
            state = literal[(state, ch)]
        elif ch not in excluded[state]:
            state = default[state]
        else:
            return False  # the transition was deleted: a marker completes here
    return True


class TestLanguage:
    """Accepted iff no marker occurs."""

    @pytest.mark.parametrize("markers", [QWEN3_MARKERS, DEEPSEEK_MARKERS, ["aa"], ["ab", "ba"]])
    @pytest.mark.parametrize(
        "text",
        [
            "",
            "3",
            "There are 3 orange circles.",
            "<tool_call>",
            "a<tool_call>b",
            "<function=foo>",
            "<function",
            "<tool_",
            "<div>tool_call</div>",
            "<<tool_call>",
            "aa",
            "aba",
            "abab",
            "baba",
            "<｜DSML｜function_calls>",
            "<｜DSML｜function_call",
        ],
    )
    def test_matches_substring_oracle(self, markers, text):
        grammar = exclusion_grammar(markers)

        assert accepts(grammar, text) == (not any(m in text for m in markers))

    @pytest.mark.parametrize("markers", [QWEN3_MARKERS, ["aa"], ["ab", "ba"], ["abc", "bc", "c"]])
    def test_fuzz_against_oracle(self, markers):
        """Hand-picked strings only cover what was thought of."""
        rng = random.Random(20240907)
        alphabet = sorted({ch for m in markers for ch in m} | set("xy>"))

        for _ in range(4000):
            text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 14)))
            grammar = exclusion_grammar(markers)

            assert accepts(grammar, text) == (not any(m in text for m in markers)), text

    def test_overlapping_markers(self):
        """A marker that is a suffix of another must still be caught.

        The fail links carry the shorter marker's accepting state into the
        longer one's; without that propagation `xaa` slips through.
        """
        grammar = exclusion_grammar(["baa", "aa"])

        assert accepts(grammar, "xaa") is False
        assert accepts(grammar, "aba") is True


class TestDegenerateInput:
    def test_no_markers(self):
        assert exclusion_grammar([]) is None

    def test_empty_marker_refused(self):
        """An empty marker excludes every string, the empty one included --
        the model could not answer at all."""
        assert exclusion_grammar([""]) is None
        assert exclusion_grammar(["<tool_call>", ""]) is None

    def test_duplicate_markers_collapse(self):
        once = exclusion_grammar(["<tool_call>"])

        assert exclusion_grammar(["<tool_call>", "<tool_call>"]) == once


class TestBackendsAccept:
    """Both grammar backends must compile it: a request picks one, and which
    one is the operator's choice, not this code's."""

    @pytest.mark.parametrize("markers", [QWEN3_MARKERS, DEEPSEEK_MARKERS])
    def test_xgrammar(self, markers):
        xgr = pytest.importorskip("xgrammar")

        xgr.Grammar.from_ebnf(exclusion_grammar(markers))

    @pytest.mark.parametrize("markers", [QWEN3_MARKERS, DEEPSEEK_MARKERS])
    def test_llguidance(self, markers):
        llguidance = pytest.importorskip("llguidance")

        grammar = llguidance.grammar_from("grammar", exclusion_grammar(markers))

        assert llguidance.LLMatcher.validate_grammar(grammar, None) == ""
