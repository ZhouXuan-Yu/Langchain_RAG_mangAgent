"""Test LLM client — 01-03: DeepSeek API 调用与消息结构."""

import pytest


class TestDeepseekClient:
    """测试 DeepSeek LLM 初始化和消息构建."""

    def test_init_deepseek_llm_import(self):
        """验证所有依赖可正确导入."""
        from src.llm.deepseek_client import init_deepseek_llm, build_messages
        from langchain_core.messages import SystemMessage, HumanMessage
        assert callable(init_deepseek_llm)
        assert callable(build_messages)

    def test_build_messages_system_only(self):
        """测试仅系统提示的消息构建."""
        from src.llm.deepseek_client import build_messages

        msgs = build_messages(
            system_prompt="You are a helpful assistant.",
            user_input="Hello!",
        )
        assert len(msgs) == 2
        assert msgs[0].type == "system"
        assert msgs[1].type == "human"
        assert "Hello!" in msgs[1].content

    def test_build_messages_with_history(self):
        """测试带历史记录的消息构建."""
        from src.llm.deepseek_client import build_messages
        from langchain_core.messages import AIMessage, HumanMessage

        history = [
            HumanMessage(content="What is LangGraph?"),
            AIMessage(content="LangGraph is a framework for building agents."),
        ]

        msgs = build_messages(
            system_prompt="You are a helpful assistant.",
            user_input="Tell me more.",
            history=history,
        )

        assert len(msgs) == 4  # system + history(2) + human
        assert msgs[0].type == "system"
        assert msgs[1].type == "human"
        assert msgs[1].content == "What is LangGraph?"
        assert msgs[2].type == "ai"
        assert msgs[3].type == "human"
        assert msgs[3].content == "Tell me more."

    def test_system_prompt_template(self):
        """测试 System Prompt 模板."""
        from src.graph.prompt import build_system_prompt

        prompt = build_system_prompt(user_name="测试用户")
        assert "测试用户" in prompt
        assert "记忆" in prompt
        assert "Protocol" in prompt


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
