# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from aphrodite.tool_parsers.hermes_tool_parser import Hermes2ProToolParser


@pytest.fixture(autouse=True)
def hermes_supports_required_and_named(monkeypatch):
    """Pin the legacy required/named tool-choice path these tests exercise."""
    monkeypatch.setattr(Hermes2ProToolParser, "supports_required_and_named", True)
