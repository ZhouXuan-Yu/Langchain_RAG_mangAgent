"""Test agent — 16-18: LangGraph 节点、路由和 Graph 组装."""

import pytest


class TestRouterDecision:
    """测试条件路由逻辑."""

    def test_route_memory_retrieve(self):
        """测试记忆检索路由."""
        from src.graph.router import route_decision
        from src.graph.state import AgentState
        from langchain_core.messages import HumanMessage

        state = AgentState(
            messages=[HumanMessage(content="我之前的项目是怎么做的？")],
            thread_id="test",
            memory_context=[],
            web_context=[],
            pending_memory=[],
            memory_updated=False,
            last_tool_result=None,
            turn_count=0,
            route_decision="memory_retrieve",
        )
        assert route_decision(state) == "retrieve_memory"

    def test_route_web_search(self):
        """测试网页搜索路由."""
        from src.graph.router import route_decision
        from src.graph.state import AgentState
        from langchain_core.messages import HumanMessage

        state = AgentState(
            messages=[HumanMessage(content="DeepSeek 最新版本是多少？2026年的更新")],
            thread_id="test",
            memory_context=[],
            web_context=[],
            pending_memory=[],
            memory_updated=False,
            last_tool_result=None,
            turn_count=0,
            route_decision="web_search",
        )
        assert route_decision(state) == "retrieve_web"

    def test_route_direct_reason(self):
        """测试直接推理路由."""
        from src.graph.router import route_decision
        from src.graph.state import AgentState
        from langchain_core.messages import HumanMessage

        state = AgentState(
            messages=[HumanMessage(content="你好，帮我介绍一下 LangGraph")],
            thread_id="test",
            memory_context=[],
            web_context=[],
            pending_memory=[],
            memory_updated=False,
            last_tool_result=None,
            turn_count=0,
            route_decision="direct_reason",
        )
        assert route_decision(state) == "reason_node"


class TestRouterNode:
    """测试 router_node 动态路由判断."""

    def test_router_detects_project_query(self):
        """测试路由识别项目查询."""
        from src.graph.nodes import router_node
        from src.graph.state import AgentState
        from langchain_core.messages import HumanMessage

        state = AgentState(
            messages=[HumanMessage(content="我记得我的智程导航项目用的是微服务架构？")],
            thread_id="test",
            memory_context=[],
            web_context=[],
            pending_memory=[],
            memory_updated=False,
            last_tool_result=None,
            turn_count=0,
            route_decision=None,
        )
        result = router_node(state)
        assert result["route_decision"] == "memory_retrieve"

    def test_router_detects_latest_info(self):
        """测试路由识别最新信息查询."""
        from src.graph.nodes import router_node
        from src.graph.state import AgentState
        from langchain_core.messages import HumanMessage

        state = AgentState(
            messages=[HumanMessage(content="LangGraph 0.4 版本有哪些新特性？")],
            thread_id="test",
            memory_context=[],
            web_context=[],
            pending_memory=[],
            memory_updated=False,
            last_tool_result=None,
            turn_count=0,
            route_decision=None,
        )
        result = router_node(state)
        assert result["route_decision"] == "web_search"

    def test_router_detects_update_request(self):
        """测试路由识别更新请求."""
        from src.graph.nodes import router_node
        from src.graph.state import AgentState
        from langchain_core.messages import HumanMessage

        state = AgentState(
            messages=[HumanMessage(content="我想修改之前的项目配置")],
            thread_id="test",
            memory_context=[],
            web_context=[],
            pending_memory=[],
            memory_updated=False,
            last_tool_result=None,
            turn_count=0,
            route_decision=None,
        )
        result = router_node(state)
        assert result["route_decision"] == "memory_update"

    def test_router_default_to_reason(self):
        """测试默认路由到推理."""
        from src.graph.nodes import router_node
        from src.graph.state import AgentState
        from langchain_core.messages import HumanMessage

        state = AgentState(
            messages=[HumanMessage(content="2 + 3 等于多少？")],
            thread_id="test",
            memory_context=[],
            web_context=[],
            pending_memory=[],
            memory_updated=False,
            last_tool_result=None,
            turn_count=0,
            route_decision=None,
        )
        result = router_node(state)
        assert result["route_decision"] == "direct_reason"


class TestShouldContinue:
    """测试循环控制."""

    def test_should_stop_at_max_turns(self):
        """测试超过最大回合停止."""
        from src.graph.nodes import should_continue
        from src.graph.state import AgentState
        from langchain_core.messages import HumanMessage, AIMessage

        state = AgentState(
            messages=[AIMessage(content="response")],
            thread_id="test",
            memory_context=[],
            web_context=[],
            pending_memory=[],
            memory_updated=False,
            last_tool_result=None,
            turn_count=10,  # 达到阈值
            route_decision=None,
        )
        assert should_continue(state) == "__end__"

    def test_should_stop_after_ai_response(self):
        """测试 AI 回复后停止."""
        from src.graph.nodes import should_continue
        from src.graph.state import AgentState
        from langchain_core.messages import AIMessage

        state = AgentState(
            messages=[AIMessage(content="Here is the answer.")],
            thread_id="test",
            memory_context=[],
            web_context=[],
            pending_memory=[],
            memory_updated=False,
            last_tool_result=None,
            turn_count=2,
            route_decision=None,
        )
        assert should_continue(state) == "__end__"


class TestAgentGraphBuild:
    """测试 Agent Graph 构建."""

    def test_build_agent_graph_imports(self):
        """测试 Graph 构建所需的所有依赖可正确导入."""
        from src.graph.agent_graph import build_agent_graph, build_react_agent
        from src.graph.state import AgentState

        assert callable(build_agent_graph)
        assert callable(build_react_agent)

    def test_agent_state_structure(self):
        """测试 AgentState 结构正确."""
        from src.graph.state import AgentState
        from langchain_core.messages import HumanMessage

        state = AgentState(
            messages=[HumanMessage(content="test")],
            thread_id="test_thread",
            memory_context=[],
            web_context=[],
            pending_memory=[],
            memory_updated=False,
            last_tool_result=None,
            turn_count=0,
            route_decision=None,
        )

        assert state["thread_id"] == "test_thread"
        assert state["turn_count"] == 0
        assert len(state["messages"]) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
