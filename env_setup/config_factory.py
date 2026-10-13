import inspect
from dataclasses import dataclass
from typing import Any, Union, get_args, get_origin

from langchain_core.tools import StructuredTool

from .datasets import DATASETS
from .goals import GOALS
from .packages import PYTHON_PACKAGE_SETS
from .tools import SUPPORTED_TOOLS, make_tools, retrieve_tool_signatures

COLLAB_AGENT_TOOLS_WHITELIST = [
    "search_internal_archive",
    "arxiv",
]


@dataclass
class ToolInfo:
    """One tool: executable object + exportable code + signature/help."""

    object: StructuredTool
    tool_signature: str
    code: str
    code_signature: str


@dataclass
class EnvConfig:
    """Configuration for an environment."""

    dataset_description: dict[str, str]
    python_packages: dict[str, str]
    primary_agent_tools: dict[str, ToolInfo]
    collab_agent_tools: dict[str, ToolInfo]
    discovery_goal: str


def _type_str(tp: Any) -> str:
    """Render a Python/typing annotation into a compact, human-readable string.

    Handles:
      - Builtins and typing aliases (e.g., ``str``, ``int``, ``typing.Any``).
      - ``Union`` / ``Optional[T]`` normalization (e.g., ``Optional[int]``).
      - Common containers: ``list[T]``, ``tuple[T]``, ``set[T]``, ``dict[K, V]``.

    Args:
        tp: A type annotation object (from typing or a builtin).

    Returns:
        str: A readable representation of the type (e.g., ``"dict[str, int]"``).
    """
    if tp is None:
        return "Any"
    origin = get_origin(tp)
    if origin is None:
        # Bare type or typing alias (e.g., str, int, typing.Any)
        return getattr(tp, "__name__", str(tp).replace("typing.", ""))
    args = get_args(tp)

    # Optional[T] / Union
    if origin is Union:
        # Optional[T]?
        if type(None) in args and len(args) == 2:
            inner = [a for a in args if a is not type(None)][0]
            return f"Optional[{_type_str(inner)}]"
        return " | ".join(_type_str(a) for a in args)

    # Containers
    if origin in (list, tuple, set):
        inner = _type_str(args[0]) if args else "Any"
        return f"{origin.__name__}[{inner}]"
    if origin is dict:
        k, v = (args + (Any, Any))[:2]
        return f"dict[{_type_str(k)}, {_type_str(v)}]"

    return str(tp).replace("typing.", "")


def tool_signature_line(tool) -> str:
    """Construct a single-line, human-readable signature for a tool.

    Inspects the tool's argument schema to build a compact spec listing arg
    names and types, plus a short description and per-argument notes when
    available. Supports:
      - Pydantic v2 models (``model_fields``)
      - Pydantic v1 models (``__fields__``)
      - Fallback JSON-schema-like ``tool.args`` dict (with ``type``/``description``)

    Args:
        tool: A LangChain ``StructuredTool`` (or similar) exposing schema metadata.

    Returns:
        str: A one-line signature like:
             ``- tool_name: args = { "a": int, "b": str } (Short desc) a = foo b = bar``
    """
    fields = []  # for types, e.g.  ["\"a\": int", "\"b\": int"]
    field_notes = []  # for per-arg descriptions, e.g. ["a = first addend", "b = second addend"]

    schema = getattr(tool, "args_schema", None)

    if schema is not None:
        # Pydantic v2
        if hasattr(schema, "model_fields"):
            for name, f in schema.model_fields.items():
                fields.append(f'"{name}": {_type_str(f.annotation)}')
                desc = getattr(f, "description", None)
                if desc:
                    field_notes.append(f"{name} = {desc}")
        # Pydantic v1
        elif hasattr(schema, "__fields__"):
            for name, f in schema.__fields__.items():
                typ = getattr(f, "outer_type_", None) or getattr(f, "type_", Any)
                fields.append(f'"{name}": {_type_str(typ)}')
                finfo = getattr(f, "field_info", None)
                if finfo and getattr(finfo, "description", None):
                    field_notes.append(f"{name} = {finfo.description}")

    # Fallback: JSON-schema-like dict on tool.args (common on some community tools)
    if not fields and getattr(tool, "args", None):
        for name, info in tool.args.items():
            t = info.get("type", "Any")
            # Normalize common JSON types to Python-ish
            t = {"string": "str", "integer": "int", "number": "float", "boolean": "bool"}.get(t, t)
            fields.append(f'"{name}": {t}')
            if info.get("description"):
                field_notes.append(f"{name} = {info['description']}")

    args_block = "{ " + ", ".join(fields) + " }" if fields else "{ }"

    # Short tool description (first line only)
    desc = (tool.description or inspect.getdoc(tool) or "").strip()
    short_desc = desc.splitlines()[0] if desc else ""

    # Compose final line
    line = f"- {tool.name}: args = {args_block}"
    if short_desc:
        line += f" ({short_desc})"
    if field_notes:
        line += " " + " ".join(field_notes)

    return line


def build_config(
    dataset_keys: list[str], python_package_set_key: str, tools_cfg: dict[str, Any], goal_key: str
) -> EnvConfig:
    """Validate inputs, materialize tools, and assemble an `EnvConfig`.

    Steps:
      1) Validate dataset keys, package-set key, tool names, and goal key.
      2) Collect dataset descriptions and Python package specs from registries.
      3) Instantiate tools via ``make_tools`` and retrieve code/signatures.
      4) Populate ``primary_agent_tools`` for all requested tools and
         ``collab_agent_tools`` only for those on the whitelist.

    Args:
        dataset_keys: Keys selecting dataset descriptions from ``DATASETS``.
        python_package_set_key: Key selecting a package set from ``PYTHON_PACKAGE_SETS``.
        tools_cfg: Tool configuration mapping (passed to ``make_tools``).
        goal_key: Key selecting the discovery goal text from ``GOALS``.

    Returns:
        EnvConfig: Fully constructed environment configuration for agents.

    Raises:
        ValueError: If any provided key is unknown or a tool is unsupported.
    """
    for key in dataset_keys:
        if key not in DATASETS:
            raise ValueError(f"Unknown dataset key: {key}")
    if python_package_set_key not in PYTHON_PACKAGE_SETS:
        raise ValueError(f"Unknown python package key: {python_package_set_key}")
    for name in tools_cfg:
        if name not in SUPPORTED_TOOLS:
            raise ValueError(f"Unsupported tool: {name}")
    if goal_key not in GOALS:
        raise ValueError(f"Unknown goal key: {goal_key}")

    dataset_description = {key: DATASETS[key] for key in dataset_keys}
    python_packages = PYTHON_PACKAGE_SETS[python_package_set_key]
    discovery_goal = GOALS[goal_key]

    # tools
    tool_objects = make_tools(tools_cfg)
    tool_signatures = retrieve_tool_signatures(tools_cfg)
    primary_agent_tools: dict[str, ToolInfo] = {}
    collab_agent_tools: dict[str, ToolInfo] = {}
    for name, obj in tool_objects.items():
        sig = tool_signatures[name]
        primary_agent_tools[name] = ToolInfo(
            object=obj,
            tool_signature=tool_signature_line(obj),
            code=sig["code"],
            code_signature=sig["signature"],
        )
        if name in COLLAB_AGENT_TOOLS_WHITELIST:
            collab_agent_tools[name] = ToolInfo(
                object=obj,
                tool_signature=tool_signature_line(obj),
                code=sig["code"],
                code_signature=sig["signature"],
            )

    return EnvConfig(
        dataset_description=dataset_description,
        python_packages=python_packages,
        primary_agent_tools=primary_agent_tools,
        collab_agent_tools=collab_agent_tools,
        discovery_goal=discovery_goal,
    )
