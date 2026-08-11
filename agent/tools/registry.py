"""工具注册表。

每个工具模块通过 @register 装饰器自动注册到全局注册表。
agent 层调用 list_tools() 即可拿到全部工具列表。

新增工具步骤：
    1. 新建 agent/tools/<tool>.py
    2. 用 @register + @tool 装饰函数
    3. 在 agent/tools/__init__.py 中 import 该模块
删除工具：在 __init__.py 中注释掉对应 import。
"""

from langchain.tools import BaseTool

_tools: dict[str, BaseTool] = {}
_cached_format: str | None = None


def register(tool: BaseTool) -> BaseTool:
    """将工具注册到全局注册表。

    用法：
        @register
        @tool
        def my_tool(param: str) -> str: ...
    """
    global _cached_format
    _tools[tool.name] = tool
    _cached_format = None
    return tool


def get_tool(name: str) -> BaseTool | None:
    """按名称获取工具。"""
    return _tools.get(name)


def list_tools() -> list[BaseTool]:
    """返回已注册的全部工具列表。"""
    return list(_tools.values())


def get_tool_names() -> list[str]:
    """返回已注册的工具名称列表。"""
    return list(_tools.keys())


def format_tools_for_prompt() -> str:
    """格式化工具列表供 LLM 提示词使用（工具名 + 首行描述）。

    结果在注册时缓存，工具列表变化时自动失效。
    """
    global _cached_format
    if _cached_format is not None:
        return _cached_format
    tools = list_tools()
    if not tools:
        _cached_format = "（无可用工具）"
        return _cached_format
    lines: list[str] = []
    for t in tools:
        first_line = t.description.split("\n")[0]
        lines.append(f"- {t.name}: {first_line}")
    _cached_format = "\n".join(lines)
    return _cached_format