# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The exact Harmony final-JSON envelope understood by postconditions.

Recognizing the entire envelope is deliberate: arbitrary structural tags,
especially tool parameters, do not have the same document boundaries.
"""

import json
from typing import Any


def make_harmony_json_tag(schema: Any):
    from xgrammar import StructuralTag
    from xgrammar.structural_tag import (
        AnyTextFormat,
        ConstStringFormat,
        JSONSchemaFormat,
        OptionalFormat,
        OrFormat,
        SequenceFormat,
        TagFormat,
    )

    return StructuralTag(
        format=SequenceFormat(
            elements=[
                OptionalFormat(
                    content=SequenceFormat(
                        elements=[
                            TagFormat(begin="<|channel|>analysis<|message|>", content=AnyTextFormat(), end="<|end|>"),
                            ConstStringFormat(value="<|start|>assistant"),
                        ]
                    )
                ),
                OrFormat(
                    elements=[
                        TagFormat(
                            begin=f"<|channel|>final{constrain}<|message|>",
                            content=JSONSchemaFormat(json_schema=schema),
                            end=["<|end|>", ""],
                        )
                        for constrain in (" json", " <|constrain|>json")
                    ]
                ),
            ]
        )
    )


def harmony_json_schema(spec: Any) -> dict | None:
    """Return the single final-body schema only for our supported envelope."""
    if isinstance(spec, str):
        try:
            spec = json.loads(spec)
        except ValueError:
            return None
    if not isinstance(spec, dict):
        return None
    try:
        schema = spec["format"]["elements"][1]["elements"][0]["content"]["json_schema"]
        if not isinstance(schema, dict):
            return None
        from xgrammar import StructuralTag

        if StructuralTag.model_validate(spec) != make_harmony_json_tag(schema):
            return None
    except (KeyError, IndexError, TypeError, ValueError, ImportError):
        return None
    return schema
