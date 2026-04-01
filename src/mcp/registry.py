"""Tool Registry — 工具注册中心.

MCP 协议核心：所有可用工具在此声明名称、描述、参数 schema、所属 capability。
支持按 capability 分类查询，供 GenericWorker 动态绑定工具集。
"""

from typing import TypedDict


class ToolInfo(TypedDict):
    """工具注册条目."""
    name: str
    description: str
    schema: dict       # JSON Schema for parameters
    capability: str     # 所属 capability 标签


class ToolRegistry:
    """
    全局工具注册中心。

    使用方法：
        ToolRegistry.register(
            name="web_search",
            description="从互联网搜索最新信息",
            schema={"type": "object", "properties": {"query": {"type": "string"}}},
            capability="web_search",
        )
        tools = ToolRegistry.list_all()
        web_tools = ToolRegistry.get_by_capability("web_search")
    """

    _tools: dict[str, ToolInfo] = {}

    @classmethod
    def register(
        cls,
        name: str,
        description: str,
        schema: dict,
        capability: str,
    ) -> None:
        """注册一个工具。重复注册会覆盖。"""
        cls._tools[name] = {
            "name": name,
            "description": description,
            "schema": schema,
            "capability": capability,
        }

    @classmethod
    def list_all(cls) -> list[ToolInfo]:
        """列出所有已注册的工具。"""
        return list(cls._tools.values())

    @classmethod
    def get_by_capability(cls, capability: str) -> list[ToolInfo]:
        """按 capability 过滤工具。"""
        return [t for t in cls._tools.values() if t["capability"] == capability]

    @classmethod
    def get(cls, name: str) -> ToolInfo | None:
        """按名称获取工具信息。"""
        return cls._tools.get(name)

    @classmethod
    def get_capabilities(cls) -> list[str]:
        """列出所有不重复的 capability 标签。"""
        return sorted(set(t["capability"] for t in cls._tools.values()))

    @classmethod
    def get_by_names(cls, names: list[str]) -> list[ToolInfo]:
        """按名称列表批量获取工具。"""
        return [cls._tools[n] for n in names if n in cls._tools]

    @classmethod
    def clear(cls) -> None:
        """清空所有注册（主要用于测试）。"""
        cls._tools.clear()
