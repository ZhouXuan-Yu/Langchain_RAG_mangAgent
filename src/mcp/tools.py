"""MCP 工具声明 — 将现有 src/tools/ 中的所有工具注册到 ToolRegistry."""

from src.mcp.registry import ToolRegistry


def _register_all_tools() -> None:
    """注册所有可用工具到 ToolRegistry。"""
    ToolRegistry.register(
        name="web_search",
        description="从互联网搜索最新信息（支持 Tavily），返回标题、摘要和 URL",
        schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索查询词"},
            },
            "required": ["query"],
        },
        capability="web_search",
    )

    ToolRegistry.register(
        name="browse_page",
        description="使用 Playwright 浏览器访问指定 URL，返回页面文本内容",
        schema={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "目标网页 URL"},
            },
            "required": ["url"],
        },
        capability="web_search",
    )

    ToolRegistry.register(
        name="memory_search",
        description="在用户长期记忆（ChromaDB 向量数据库）中检索相关内容",
        schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "检索查询"},
                "top_k": {"type": "integer", "description": "返回结果数量", "default": 5},
                "category": {"type": "string", "description": "记忆类别: project/conversation/technical/general"},
            },
            "required": ["query"],
        },
        capability="memory_search",
    )

    ToolRegistry.register(
        name="save_memory",
        description="将重要信息保存到用户长期记忆（ChromaDB）中",
        schema={
            "type": "object",
            "properties": {
                "fact": {"type": "string", "description": "要保存的事实/记忆内容"},
                "category": {
                    "type": "string",
                    "description": "记忆类别: project/tech_stack/hardware/preference/decision",
                    "default": "general",
                },
                "importance": {"type": "integer", "description": "重要性 1-5", "default": 3},
                "tags": {"type": "array", "items": {"type": "string"}, "description": "标签列表"},
            },
            "required": ["fact"],
        },
        capability="memory_search",
    )

    ToolRegistry.register(
        name="knowledge_base_search",
        description="在用户上传的文档知识库（ChromaDB category=document）中检索",
        schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "检索查询"},
                "top_k": {"type": "integer", "description": "返回结果数量", "default": 5},
            },
            "required": ["query"],
        },
        capability="knowledge_base_search",
    )

    ToolRegistry.register(
        name="calculator",
        description="执行安全的数学表达式计算（支持 sqrt, sin, cos, log, pow 等）",
        schema={
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "数学表达式，如 sqrt(2) + 1"},
            },
            "required": ["expression"],
        },
        capability="code_generation",
    )

    ToolRegistry.register(
        name="process_image",
        description="分析本地图片，提取特征描述并自动分类存储到记忆",
        schema={
            "type": "object",
            "properties": {
                "image_path": {"type": "string", "description": "本地图片路径"},
            },
            "required": ["image_path"],
        },
        capability="multimodal",
    )

    ToolRegistry.register(
        name="orchestrate",
        description="触发 Crayfish 多 Agent 编排，执行复杂任务规划与执行",
        schema={
            "type": "object",
            "properties": {
                "requirement": {"type": "string", "description": "用户需求描述"},
                "enabled_agents": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "启用的 Agent 列表",
                    "default": ["search", "rag", "coder"],
                },
                "quality_threshold": {
                    "type": "number",
                    "description": "质量阈值 0-10",
                    "default": 8.0,
                },
            },
            "required": ["requirement"],
        },
        capability="orchestration",
    )


# 模块加载时自动注册所有工具
_register_all_tools()
