# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import enum
from typing import Literal

ToolStrictLevelName = Literal["auto", "function", "parameter"]


class ToolStrictLevel(enum.IntEnum):
    """Server-side floor for tool-call structural tags (``--tool-strict-level``).

    AUTO:      constrain the call format and resolve argument-schema strictness
               from the client field or the environment default.
    FUNCTION:  explicitly require the tool-call envelope.
    PARAMETER: default omitted ``strict`` to true, preserving explicit false.
    """

    AUTO = 0
    FUNCTION = 1
    PARAMETER = 2

    @classmethod
    def from_name(cls, name: str) -> "ToolStrictLevel":
        try:
            return cls[name.strip().upper()]
        except KeyError:
            expected = ", ".join(level.name.lower() for level in cls)
            raise ValueError(f"Unknown tool strict level {name!r}; expected one of {expected}.") from None
