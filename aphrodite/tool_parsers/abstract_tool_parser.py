# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
import json
import os
from collections.abc import Callable, Sequence
from functools import cached_property
from typing import Any, NoReturn

from openai.types.responses import (
    ResponseFormatTextJSONSchemaConfig,
    ResponseTextConfig,
)
from openai.types.responses.function_tool import FunctionTool

from aphrodite.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
)
from aphrodite.entrypoints.openai.engine.protocol import (
    DeltaMessage,
    ExtractedToolCallInformation,
)
from aphrodite.entrypoints.openai.responses.protocol import (
    ResponsesRequest,
)
from aphrodite.exceptions import AphroditeValidationError
from aphrodite.logger import init_logger
from aphrodite.sampling_params import (
    StructuredOutputsParams,
)
from aphrodite.tokenizers import TokenizerLike
from aphrodite.tool_parsers.utils import Tool, get_json_schema_from_tools
from aphrodite.utils.collection_utils import is_list_of
from aphrodite.utils.import_utils import import_from_path

__all__ = ["Tool"]

logger = init_logger(__name__)


def reply_schema_for_tool_grammar(
    request: ChatCompletionRequest | ResponsesRequest,
) -> dict[str, Any] | bool | None:
    """The reply constraint the caller asked for, as a schema the tool grammar
    can carry alongside its tool calls. None when the caller asked for none.

    Only a JSON schema is carried. The tool tag holds the reply in a slot shaped
    like a schema, and a regex, a choice list or a caller's own structural tag
    has nothing to sit in it.
    """
    if isinstance(request, ResponsesRequest):
        reply_format = getattr(request.text, "format", None)
        parameter = "text.format"
    else:
        reply_format = request.response_format
        parameter = "response_format"

    schema: dict[str, Any] | bool | None = None
    kind = getattr(reply_format, "type", None)
    if kind == "json_schema":
        schema = _reply_json_schema(reply_format)
    elif kind == "json_object":
        schema = True
    elif kind == "structural_tag":
        raise _reply_schema_refused(parameter, "a structural tag of its own")

    structured_outputs = getattr(request, "structured_outputs", None)
    if structured_outputs is not None:
        for name in ("regex", "choice", "grammar", "structural_tag"):
            if getattr(structured_outputs, name, None) is not None:
                raise _reply_schema_refused("structured_outputs", f"`{name}`")
        if structured_outputs.json is not None:
            schema = structured_outputs.json
        elif structured_outputs.json_object:
            schema = True

    return schema


def _reply_json_schema(reply_format: Any) -> dict[str, Any] | None:
    """The schema body out of either API's spelling of a json_schema format."""
    # Responses states it inline; Chat Completions nests it one deeper.
    if (schema := getattr(reply_format, "schema_", None)) is not None:
        return schema
    return getattr(getattr(reply_format, "json_schema", None), "json_schema", None)


def _reply_schema_refused(parameter: str, what: str) -> AphroditeValidationError:
    return AphroditeValidationError(
        f"`{parameter}` asking for {what} cannot be combined with tool calling.",
        parameter=parameter,
    )


def reject_unmergeable_reply_schema(
    request: ChatCompletionRequest | ResponsesRequest,
) -> NoReturn:
    """Refuse a reply schema the tool grammar for this model cannot carry."""
    parameter = "text.format" if isinstance(request, ResponsesRequest) else "response_format"
    raise AphroditeValidationError(
        f"`{parameter}` cannot be combined with tool calling for this model.",
        parameter=parameter,
    )


def reject_reply_schema_without_tool_grammar(
    request: ChatCompletionRequest | ResponsesRequest,
) -> NoReturn:
    """Refuse a reply schema for a request whose tool calls get no grammar.

    Without one the reply schema is the only grammar there is, and it spans the
    whole reply -- so the model cannot spell a tool call at all, and the tools
    are silently gone.
    """
    parameter = "text.format" if isinstance(request, ResponsesRequest) else "response_format"
    raise AphroditeValidationError(
        f"`{parameter}` cannot be combined with tool calling for this model.",
        parameter=parameter,
    )


class ToolParser:
    """
    Abstract ToolParser class that should not be used directly. Provided
    properties and methods should be used in
    derived classes.
    """

    # When True (default), the serving layer uses the standard JSON-based
    # parsing for tool_choice="required" and named function tool_choice,
    # which works for models where guided decoding produces well-formed
    # JSON output (e.g. Hermes).
    # Subclasses set False when the standard parsing does not work for
    # their model's output format (e.g. GLM models that use XML).  When
    # False, the serving layer falls back to the tool_parser's
    # extract_tool_calls / extract_tool_calls_streaming methods for
    # required/named tool_choice, treating them the same as "auto".
    supports_required_and_named: bool = True
    # xgrammar builtin structural tag model key. Subclasses set this when
    # their parsed tool-call syntax matches a builtin xgrammar format.
    structural_tag_model: str | None = None
    # If True, `adjust_request` reads the reply constraints off the request
    # and builds its own grammar to handle it alongside tool-calling.
    merges_reply_schema: bool = False
    engine_based_streaming: bool = False

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls.structural_tag_model is not None:
            cls.supports_required_and_named = False

    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
    ):
        self.prev_tool_call_arr: list[dict] = []
        # the index of the tool call that is currently being parsed
        self.current_tool_id: int = -1
        self.current_tool_name_sent: bool = False
        self.streamed_args_for_tool: list[str] = []

        self.model_tokenizer = tokenizer
        if tools:
            self.tools: list[ChatCompletionToolsParam | FunctionTool] = [
                tool for tool in tools if isinstance(tool, (ChatCompletionToolsParam, FunctionTool))
            ]
        else:
            self.tools = []

    def get_remaining_unstreamed_args(self) -> str:
        """Return tool call arguments parsed but not yet streamed."""
        if not self.prev_tool_call_arr:
            return ""
        index = len(self.prev_tool_call_arr) - 1
        args = self.prev_tool_call_arr[index].get("arguments", {})
        if isinstance(args, str):
            expected = args
        else:
            expected = json.dumps(args, ensure_ascii=False)
        actual = self.streamed_args_for_tool[index] if index < len(self.streamed_args_for_tool) else ""
        if expected.startswith(actual):
            return expected[len(actual) :]
        return ""

    @cached_property
    def vocab(self) -> dict[str, int]:
        # NOTE: Only TokenizersBackend is guaranteed to have .vocab
        # whereas all tokenizers have .get_vocab()
        return self.model_tokenizer.get_vocab()

    def adjust_request(
        self,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> ChatCompletionRequest | ResponsesRequest:
        # If there are no tools, return the request as is.
        if not request.tools:
            return request

        # Set structured output params when tool constraints are derived from
        # the tool schema. Unified parsers handle model-specific structural
        # tags before calling into the tool parser.
        structured_outputs = getattr(request, "structured_outputs", None)
        if structured_outputs is not None and structured_outputs.structural_tag is not None:
            return request

        json_schema_from_tool = get_json_schema_from_tools(tool_choice=request.tool_choice, tools=request.tools)
        # Set structured output params for tool calling
        if json_schema_from_tool is not None:
            # Reaching here means the tool call is forced, so the reply is a
            # tool call and nothing else. A reply constraint has no reply left
            # to constrain and is dropped rather than reported.
            if isinstance(request, ChatCompletionRequest):
                # tool_choice: "Forced Function" or "required" will override
                # structured output json settings to make tool calling work correctly
                request.structured_outputs = StructuredOutputsParams(
                    json=json_schema_from_tool  # type: ignore[call-arg]
                )
                request.response_format = None
            if isinstance(request, ResponsesRequest):
                # Single-shot construction so Pydantic v2 tracks `format`
                # in __fields_set__ — assigning to `.format` after the bare
                # `ResponseTextConfig()` constructor does not, which can
                # drop the nested config from `model_dump`. Also drop the
                # `description` kwarg: it is not a field on
                # ResponseFormatTextJSONSchemaConfig and was being silently
                # passed through as extra.
                request.text = ResponseTextConfig(
                    format=ResponseFormatTextJSONSchemaConfig(
                        type="json_schema",
                        name="tool_calling_response",
                        schema=json_schema_from_tool,
                        strict=True,
                    )
                )
                request.structured_outputs = None

        return request

    def get_structural_tag(
        self,
        request: ChatCompletionRequest | ResponsesRequest,
        *,
        reasoning: bool = False,
    ):
        if self.structural_tag_model is None:
            return None
        from aphrodite.tool_parsers.structural_tag_registry import get_model_structural_tag

        return get_model_structural_tag(
            model=self.structural_tag_model,
            tools=request.tools,
            tool_choice=request.tool_choice,
            reasoning=reasoning,
        )

    def extract_tool_calls(self, model_output: str, request: ChatCompletionRequest) -> ExtractedToolCallInformation:
        """
        Static method that should be implemented for extracting tool calls from
        a complete model-generated string.
        Used for non-streaming responses where we have the entire model response
        available before sending to the client.
        Static because it's stateless.
        """
        raise NotImplementedError("AbstractToolParser.extract_tool_calls has not been implemented!")

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        """
        Instance method that should be implemented for extracting tool calls
        from an incomplete response; for use when handling tool calls and
        streaming. Has to be an instance method because  it requires state -
        the current tokens/diffs, but also the information about what has
        previously been parsed and extracted (see constructor)
        """
        raise NotImplementedError("AbstractToolParser.extract_tool_calls_streaming has not been implemented!")


class ToolParserManager:
    """
    Central registry for ToolParser implementations.

    Supports two modes:
      - Eager (immediate) registration via `register_module`
      - Lazy registration via `register_lazy_module`
    """

    tool_parsers: dict[str, type[ToolParser]] = {}
    lazy_parsers: dict[str, tuple[str, str]] = {}  # name -> (module_path, class_name)

    @classmethod
    def get_tool_parser(cls, name: str) -> type[ToolParser]:
        """
        Retrieve a registered or lazily registered ToolParser class.

        If the parser is lazily registered,
        it will be imported and cached on first access.
        Raises KeyError if not found.
        """
        if name in cls.tool_parsers:
            return cls.tool_parsers[name]

        if name in cls.lazy_parsers:
            return cls._load_lazy_parser(name)

        raise KeyError(f"Tool parser '{name}' not found.")

    @classmethod
    def _load_lazy_parser(cls, name: str) -> type[ToolParser]:
        """Import and register a lazily loaded parser."""
        module_path, class_name = cls.lazy_parsers[name]
        try:
            mod = importlib.import_module(module_path)
            parser_cls = getattr(mod, class_name)
            if not issubclass(parser_cls, ToolParser):
                raise TypeError(f"{class_name} in {module_path} is not a ToolParser subclass.")
            cls.tool_parsers[name] = parser_cls  # cache
            return parser_cls
        except Exception as e:
            logger.exception(
                "Failed to import lazy tool parser '%s' from %s: %s",
                name,
                module_path,
                e,
            )
            raise

    @classmethod
    def _register_module(
        cls,
        module: type[ToolParser],
        module_name: str | list[str] | None = None,
        force: bool = True,
    ) -> None:
        """Register a ToolParser class immediately."""
        if not issubclass(module, ToolParser):
            raise TypeError(f"module must be subclass of ToolParser, but got {type(module)}")

        if module_name is None:
            module_name = module.__name__

        if isinstance(module_name, str):
            module_names = [module_name]
        elif is_list_of(module_name, str):
            module_names = module_name
        else:
            raise TypeError("module_name must be str, list[str], or None.")

        for name in module_names:
            if not force and name in cls.tool_parsers:
                existed = cls.tool_parsers[name]
                raise KeyError(f"{name} is already registered at {existed.__module__}")
            cls.tool_parsers[name] = module

    @classmethod
    def register_lazy_module(cls, name: str, module_path: str, class_name: str) -> None:
        """
        Register a lazy module mapping.

        Example:
            ToolParserManager.register_lazy_module(
                name="kimi_k2",
                module_path="aphrodite.tool_parsers.kimi_k2_parser",
                class_name="KimiK2ToolParser",
            )
        """
        cls.lazy_parsers[name] = (module_path, class_name)

    @classmethod
    def register_module(
        cls,
        name: str | list[str] | None = None,
        force: bool = True,
        module: type[ToolParser] | None = None,
    ) -> type[ToolParser] | Callable[[type[ToolParser]], type[ToolParser]]:
        """
        Register module immediately or lazily (as a decorator).

        Usage:
            @ToolParserManager.register_module("kimi_k2")
            class KimiK2ToolParser(ToolParser):
                ...

        Or:
            ToolParserManager.register_module(module=SomeToolParser)
        """
        if not isinstance(force, bool):
            raise TypeError(f"force must be a boolean, but got {type(force)}")

        # Immediate registration
        if module is not None:
            cls._register_module(module=module, module_name=name, force=force)
            return module

        # Decorator usage
        def _decorator(obj: type[ToolParser]) -> type[ToolParser]:
            module_path = obj.__module__
            class_name = obj.__name__

            if isinstance(name, str):
                names = [name]
            elif name is not None and is_list_of(name, str):
                names = name
            else:
                names = [class_name]

            for n in names:
                # Lazy mapping only: do not import now
                cls.lazy_parsers[n] = (module_path, class_name)

            return obj

        return _decorator

    @classmethod
    def list_registered(cls) -> list[str]:
        """Return names of all eagerly and lazily registered tool parsers."""
        return sorted(set(cls.tool_parsers.keys()) | set(cls.lazy_parsers.keys()))

    @classmethod
    def import_tool_parser(cls, plugin_path: str) -> None:
        """Import a user-defined parser file from arbitrary path."""

        module_name = os.path.splitext(os.path.basename(plugin_path))[0]
        try:
            import_from_path(module_name, plugin_path)
        except Exception:
            logger.exception("Failed to load module '%s' from %s.", module_name, plugin_path)
