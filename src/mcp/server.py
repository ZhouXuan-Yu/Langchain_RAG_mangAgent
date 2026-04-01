"""MCP Server — 基于 SSE 传输的 Model Context Protocol 实现.

支持标准 MCP 协议（SSE + JSON-RPC）：
- 工具发现：GET /tools
- 工具调用：POST /tools/{name}
- Agent 资源：GET /resources/agents
- Episode 资源：GET /resources/episodes

连接方式（Cursor MCP 配置示例）：
{
  "mcpServers": {
    "crayfish": {
      "command": "curl",
      "args": ["http://localhost:8000/mcp/tools"]
    }
  }
}
或通过 SSE 端点实时推送（见 MCP 规范 v1）。
"""

import json
import logging
import uuid
from datetime import datetime
from typing import Any, AsyncGenerator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from src.config import ENABLE_MCP_SERVER
from src.mcp.registry import ToolRegistry

logger = logging.getLogger(__name__)

# 延迟导入 tools 以触发注册
if ENABLE_MCP_SERVER:
    try:
        from src.mcp import tools as _  # noqa: F401
    except Exception as e:
        logger.warning(f"[mcp] failed to load tools: {e}")

router = APIRouter(prefix="/mcp", tags=["MCP"])


# ── MCP Protocol Types ────────────────────────────────────────────────────────

class MCPMessage:
    """MCP JSON-RPC 2.0 消息构造器."""

    @staticmethod
    def success(req_id: str | int, result: Any) -> dict:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": result,
        }

    @staticmethod
    def error(req_id: str | int, code: int, message: str, data: Any = None) -> dict:
        payload = {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}
        if data is not None:
            payload["error"]["data"] = data
        return payload

    @staticmethod
    def notification(method: str, params: dict) -> dict:
        return {"jsonrpc": "2.0", "method": method, "params": params}


# ── MCP Endpoints ─────────────────────────────────────────────────────────────

@router.get("/tools")
async def list_tools() -> JSONResponse:
    """
    列出所有已注册的工具（符合 MCP protocol/tools/list 规范）。

    返回格式：
    {
      "tools": [
        { "name": "web_search", "description": "...", "inputSchema": {...} },
        ...
      ]
    }
    """
    if not ENABLE_MCP_SERVER:
        raise HTTPException(status_code=503, detail="MCP server is disabled (set ENABLE_MCP_SERVER=true)")

    tools = ToolRegistry.list_all()
    return JSONResponse({
        "tools": [
            {
                "name": t["name"],
                "description": t["description"],
                "inputSchema": t["schema"],
            }
            for t in tools
        ]
    })


@router.get("/capabilities")
async def list_capabilities() -> JSONResponse:
    """列出所有不重复的 capability 标签。"""
    if not ENABLE_MCP_SERVER:
        raise HTTPException(status_code=503, detail="MCP server is disabled")

    return JSONResponse({
        "capabilities": ToolRegistry.get_capabilities(),
    })


@router.get("/tools/{tool_name}")
async def get_tool(tool_name: str) -> JSONResponse:
    """获取指定工具的详细信息。"""
    if not ENABLE_MCP_SERVER:
        raise HTTPException(status_code=503, detail="MCP server is disabled")

    tool = ToolRegistry.get(tool_name)
    if not tool:
        raise HTTPException(status_code=404, detail=f"Tool not found: {tool_name}")

    return JSONResponse({
        "name": tool["name"],
        "description": tool["description"],
        "inputSchema": tool["schema"],
        "capability": tool["capability"],
    })


@router.post("/tools/{tool_name}")
async def call_tool(tool_name: str, request: Request) -> JSONResponse:
    """
    调用指定工具（符合 MCP protocol/tools/call 规范）。

    请求体（MCP JSON-RPC 2.0）：
    {
      "jsonrpc": "2.0",
      "id": "req_1",
      "method": "tools/call",
      "params": { "name": "web_search", "arguments": { "query": "..." } }
    }
    """
    if not ENABLE_MCP_SERVER:
        raise HTTPException(status_code=503, detail="MCP server is disabled")

    tool = ToolRegistry.get(tool_name)
    if not tool:
        raise HTTPException(status_code=404, detail=f"Tool not found: {tool_name}")

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    req_id = body.get("id", str(uuid.uuid4()))
    arguments = body.get("params", {}).get("arguments", {})

    try:
        result = await _execute_tool(tool_name, arguments)
        return JSONResponse(MCPMessage.success(req_id, {
            "content": [
                {
                    "type": "text",
                    "text": result if isinstance(result, str) else json.dumps(result, ensure_ascii=False),
                }
            ],
            "isError": False,
        }))
    except Exception as e:
        logger.error(f"[mcp] tool {tool_name} failed: {e}")
        return JSONResponse(MCPMessage.error(req_id, -32603, f"Tool execution failed: {e}"))


async def _execute_tool(tool_name: str, arguments: dict) -> str | dict:
    """根据工具名分发到对应实现。"""
    if tool_name == "web_search":
        from src.tools.browser_tools import TavilySearch
        result = await TavilySearch().execute(arguments.get("query", ""))
        return str(result) if result else ""

    if tool_name == "memory_search":
        from src.tools.memory_tools import get_memory_store
        store = get_memory_store()
        results = store.search(
            arguments.get("query", ""),
            top_k=arguments.get("top_k", 5),
            category=arguments.get("category"),
        )
        return results if isinstance(results, list) else str(results)

    if tool_name == "knowledge_base_search":
        from src.tools.memory_tools import knowledge_base_search
        result = await knowledge_base_search.invoke({"query": arguments.get("query", ""), "top_k": arguments.get("top_k", 5)})
        return str(result) if result else ""

    if tool_name == "save_memory":
        from src.tools.memory_tools import get_memory_store, MemoryRecord
        from datetime import datetime
        store = get_memory_store()
        record = MemoryRecord(
            fact=arguments.get("fact", ""),
            category=arguments.get("category", "general"),
            importance=arguments.get("importance", 3),
            tags=arguments.get("tags", []),
            timestamp=datetime.now().isoformat(),
        )
        status = store.upsert_record(record)
        return f"Memory saved: {status}"

    if tool_name == "calculator":
        from src.tools.calc_tools import calculator
        result = await calculator.invoke({"expression": arguments.get("expression", "")})
        return str(result) if result else ""

    if tool_name == "orchestrate":
        from src.graph.orchestrator import get_orchestrator
        orch = get_orchestrator()
        result = await orch.orchestrate(
            requirement=arguments.get("requirement", ""),
            enabled_agents=arguments.get("enabled_agents", ["search", "rag", "coder"]),
            quality_threshold=arguments.get("quality_threshold", 8.0),
        )
        return {
            "summary": result.get("summary", ""),
            "quality_score": result.get("quality_score", 0),
            "passed": result.get("passed", False),
            "output_dir": result.get("output_dir", ""),
        }

    if tool_name == "browse_page":
        from src.tools.browser_tools import browse_page
        result = await browse_page.invoke({"url": arguments.get("url", "")})
        return str(result) if result else ""

    raise ValueError(f"Tool {tool_name} has no implementation registered")


# ── MCP Resources ─────────────────────────────────────────────────────────────

@router.get("/resources/agents")
async def list_agent_resources() -> JSONResponse:
    """列出所有 Agent 资源（符合 MCP protocol/resources/list 规范）。"""
    if not ENABLE_MCP_SERVER:
        raise HTTPException(status_code=503, detail="MCP server is disabled")

    from src.multi_agent.orchestrator import get_agent_registry
    agents = get_agent_registry().list_agents(is_active=True)
    return JSONResponse({
        "resources": [
            {
                "uri": f"agents://{a['id']}",
                "name": a.get("name", a["id"]),
                "description": a.get("description", ""),
                "mimeType": "application/json",
            }
            for a in agents
        ]
    })


@router.get("/resources/agents/{agent_id}")
async def get_agent_resource(agent_id: str) -> JSONResponse:
    """获取指定 Agent 的详细信息。"""
    if not ENABLE_MCP_SERVER:
        raise HTTPException(status_code=503, detail="MCP server is disabled")

    from src.multi_agent.orchestrator import get_agent_registry
    agent = get_agent_registry().get(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent not found: {agent_id}")

    return JSONResponse({
        "uri": f"agents://{agent_id}",
        "name": agent.get("name", agent_id),
        "description": agent.get("description", ""),
        "capabilities": agent.get("capabilities", []),
        "worker_kind": agent.get("worker_kind"),
        "is_active": agent.get("is_active", True),
    })


@router.get("/resources/episodes")
async def list_episode_resources(limit: int = 20) -> JSONResponse:
    """列出最近的编排 episode（符合 MCP protocol/resources/list 规范）。"""
    if not ENABLE_MCP_SERVER:
        raise HTTPException(status_code=503, detail="MCP server is disabled")

    try:
        from src.memory.episode_store import get_episode_store
        store = get_episode_store()
        episodes = store.get_recent(limit=limit)
        return JSONResponse({
            "resources": [
                {
                    "uri": f"episodes://{e['id']}",
                    "name": f"Episode {e.get('requirement', '')[:40]}...",
                    "description": f"质量 {e.get('quality_score', 0):.1f} | {e.get('agents_used', [])}",
                    "mimeType": "application/json",
                }
                for e in episodes
            ]
        })
    except Exception:
        return JSONResponse({"resources": []})


# ── MCP SSE Event Stream ─────────────────────────────────────────────────────

@router.get("/events")
async def mcp_event_stream(request: Request) -> StreamingResponse:
    """
    MCP SSE 事件流（符合 MCP protocol/events 规范）。

    推送事件：
    - tools/list_changed: 工具列表变更通知
    - resources/list_changed: 资源列表变更通知

    也可用作 MCP Client 连接端点（标准 MCP over SSE）。
    """
    if not ENABLE_MCP_SERVER:
        raise HTTPException(status_code=503, detail="MCP server is disabled")

    async def event_generator() -> AsyncGenerator[str, None]:
        # 发送初始能力声明
        yield f"event: endpoint\ndata: {json.dumps({'endpoint': '/mcp/events'})}\n\n"

        # 定期发送心跳 + 能力更新通知
        import asyncio
        try:
            while True:
                # 发送 capabilities 列表
                caps = ToolRegistry.get_capabilities()
                yield f"event: notification\ndata: {json.dumps(MCPMessage.notification('capabilities/list', {'capabilities': caps}))}\n\n"
                await asyncio.sleep(30)
        except asyncio.CancelledError:
            pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
