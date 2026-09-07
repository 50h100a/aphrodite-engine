# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Grammar that makes a tool call unrepresentable."""

from collections import deque
from collections.abc import Sequence

__all__ = ["exclusion_grammar"]


def _quoted(ch: str) -> str:
    """A single character as an EBNF string literal body.

    Escapes are kept to the two that must be escaped. `\\uXXXX` is spelled out
    only for control characters, because llguidance rejects the escape inside a
    quoted literal where xgrammar accepts it, and printable characters -- the
    full-width pipes in DeepSeek's markers included -- can simply be
    themselves.
    """
    if ch in '"\\':
        return "\\" + ch
    if ord(ch) < 0x20 or ord(ch) == 0x7F:
        return f"\\u{ord(ch):04x}"
    return ch


def _classed(ch: str) -> str:
    """A single character as a member of an EBNF character class."""
    if ch in "\\]^-[":
        return "\\" + ch
    if ord(ch) < 0x20 or ord(ch) == 0x7F:
        return f"\\u{ord(ch):04x}"
    return ch


def exclusion_grammar(markers: Sequence[str]) -> str | None:
    """EBNF accepting every string that contains none of ``markers``.

    Returns None when there is nothing to exclude, or when a marker is empty --
    an empty marker excludes every string, including the empty one, which would
    leave the model unable to say anything at all.
    """
    markers = [m for m in dict.fromkeys(markers)]
    if not markers or any(not m for m in markers):
        return None

    # Aho-Corasick trie over the markers.
    goto: list[dict[str, int]] = [{}]
    fail: list[int] = [0]
    forbidden: list[bool] = [False]
    for marker in markers:
        state = 0
        for ch in marker:
            if ch not in goto[state]:
                goto.append({})
                fail.append(0)
                forbidden.append(False)
                goto[state][ch] = len(goto) - 1
            state = goto[state][ch]
        forbidden[state] = True

    queue = deque(goto[0].values())
    while queue:
        state = queue.popleft()
        # A state that ends one marker also ends any marker that is a suffix of
        # it, which is what the fail link points at.
        forbidden[state] = forbidden[state] or forbidden[fail[state]]
        for ch, target in goto[state].items():
            back = fail[state]
            while back and ch not in goto[back]:
                back = fail[back]
            fail[target] = goto[back].get(ch, 0)
            queue.append(target)

    alphabet = sorted({ch for marker in markers for ch in marker})

    def delta(state: int, ch: str) -> int:
        while True:
            if ch in goto[state]:
                return goto[state][ch]
            if state == 0:
                return 0
            state = fail[state]

    # A character outside the alphabet appears in no marker, so from any state
    # it returns the automaton to its start: no marker has a prefix ending in
    # it, so no partial match survives it. One class covers all of them, and
    # every state sends them to the same place.
    others = "[^" + "".join(_classed(ch) for ch in alphabet) + "] s0"

    lines = ["root ::= s0"]
    for state in range(len(goto)):
        if forbidden[state]:
            continue
        # Every surviving state is accepting: the model may stop wherever it
        # likes, so long as it has not spelled a marker.
        alts = ['""', others]
        for ch in alphabet:
            target = delta(state, ch)
            if not forbidden[target]:
                alts.append(f'"{_quoted(ch)}" s{target}')
        lines.append(f"s{state} ::= " + " | ".join(alts))
    return "\n".join(lines) + "\n"
