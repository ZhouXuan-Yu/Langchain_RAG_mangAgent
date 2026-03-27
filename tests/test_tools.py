"""Test tools — 04-06: web_search, calculator, memory_tools."""

import pytest


class TestCalculator:
    """测试 Calculator 安全计算工具."""

    def test_calculator_basic(self):
        """测试基础运算."""
        from src.tools.calc_tools import calculator

        assert calculator.invoke({"expression": "2 + 3"}) == "5"
        assert calculator.invoke({"expression": "10 - 4"}) == "6"
        assert calculator.invoke({"expression": "3 * 7"}) == "21"
        assert calculator.invoke({"expression": "20 / 4"}) == "5.0"

    def test_calculator_power(self):
        """测试幂运算."""
        from src.tools.calc_tools import calculator

        assert calculator.invoke({"expression": "2 ** 8"}) == "256"

    def test_calculator_sqrt(self):
        """测试数学函数."""
        from src.tools.calc_tools import calculator

        assert "2.0" in calculator.invoke({"expression": "sqrt(4)"})
        assert calculator.invoke({"expression": "pi"}) == str(__import__("math").pi)

    def test_calculator_safe(self):
        """测试安全性：拒绝危险表达式."""
        from src.tools.calc_tools import calculator

        # 尝试导入 os 模块应该失败
        result = calculator.invoke({"expression": "os.system('ls')"})
        assert "计算错误" in result


class TestPIIRedactor:
    """测试 PII 脱敏工具."""

    def test_redact_phone(self):
        """测试手机号脱敏."""
        from src.middleware.pii_redactor import redact_pii

        text = "我的手机号是13812345678，请联系我。"
        redacted = redact_pii(text)
        assert "13812345678" not in redacted
        assert "[手机号]" in redacted

    def test_redact_email(self):
        """测试邮箱脱敏."""
        from src.middleware.pii_redactor import redact_pii

        text = "邮箱: zhou@example.com"
        redacted = redact_pii(text)
        assert "zhou@example.com" not in redacted
        assert "[邮箱]" in redacted

    def test_redact_id_card(self):
        """测试身份证号脱敏."""
        from src.middleware.pii_redactor import redact_pii

        text = "身份证: 110101199001011234"
        redacted = redact_pii(text)
        assert "110101199001011234" not in redacted
        assert "[身份证号]" in redacted

    def test_before_model_state(self):
        """测试 before_model 中间件状态修改."""
        from src.middleware.pii_redactor import before_model
        from langchain_core.messages import HumanMessage

        state = {
            "messages": [
                HumanMessage(content="我的邮箱: test@email.com，手机: 13900001111")
            ]
        }
        result = before_model(state)
        assert "messages" in result
        new_content = result["messages"][0].content
        assert "test@email.com" not in new_content
        assert "13900001111" not in new_content


class TestInputGuard:
    """测试输入验证工具."""

    def test_validate_search_query_empty(self):
        """测试空搜索关键词."""
        from src.middleware.input_guard import validate_search_query

        assert validate_search_query("") is not None
        assert validate_search_query("   ") is not None

    def test_validate_search_query_too_long(self):
        """测试超长搜索关键词."""
        from src.middleware.input_guard import validate_search_query

        long_query = "a" * 600
        result = validate_search_query(long_query)
        assert result is not None
        assert "过长" in result

    def test_validate_search_query_injection(self):
        """测试注入攻击检测."""
        from src.middleware.input_guard import validate_search_query

        assert validate_search_query("UNION SELECT * FROM users") is not None
        assert validate_search_query("<script>alert(1)</script>") is not None

    def test_validate_search_query_ok(self):
        """测试正常搜索关键词."""
        from src.middleware.input_guard import validate_search_query

        assert validate_search_query("LangGraph 最新版本") is None
        assert validate_search_query("DeepSeek API 使用教程") is None


class TestMemorySchema:
    """测试 Pydantic MemoryRecord Schema."""

    def test_memory_record_creation(self):
        """测试 MemoryRecord 创建."""
        from src.memory.memory_schema import MemoryRecord

        record = MemoryRecord(
            fact="用户使用 RTX 5060 进行深度学习训练",
            category="hardware",
            importance=5,
            project_ref="智眸千析",
        )

        assert record.fact == "用户使用 RTX 5060 进行深度学习训练"
        assert record.category == "hardware"
        assert record.importance == 5
        assert record.project_ref == "智眸千析"

    def test_memory_record_to_document(self):
        """测试 MemoryRecord 转为文档字符串."""
        from src.memory.memory_schema import MemoryRecord

        record = MemoryRecord(
            fact="使用 Rust 开发后端服务",
            category="tech_stack",
        )
        doc = record.to_document()
        assert "[tech_stack]" in doc
        assert "Rust" in doc

    def test_memory_record_to_metadata(self):
        """测试 MemoryRecord 转为 metadata."""
        from src.memory.memory_schema import MemoryRecord

        record = MemoryRecord(fact="测试", category="project", importance=3)
        meta = record.to_metadata()
        assert meta["fact"] == "测试"
        assert meta["category"] == "project"
        assert meta["importance"] == 3
        assert "timestamp" in meta

    def test_memory_record_comparison(self):
        """测试记忆比较逻辑."""
        from src.memory.memory_schema import MemoryRecord

        short = MemoryRecord(fact="简短描述", category="project")
        long = MemoryRecord(fact="这是一个更详细和全面的描述内容" * 3, category="project")
        other_cat = MemoryRecord(fact="详细描述" * 5, category="tech_stack")

        assert long.is_more_comprehensive_than(short) is True
        assert short.is_more_comprehensive_than(long) is False
        # 不同类别不比较
        assert long.is_more_comprehensive_than(other_cat) is False


class TestTokenTracker:
    """测试 Token 追踪器."""

    def test_token_count(self):
        """测试 token 计数."""
        from src.utils.token_tracker import TokenTracker

        tracker = TokenTracker()
        count = tracker.count("Hello, world!")
        assert count > 0

    def test_token_record(self):
        """测试记录调用."""
        from src.utils.token_tracker import TokenTracker

        tracker = TokenTracker()
        tracker.record(prompt_tokens=100, completion_tokens=50, label="test")
        assert tracker.total_tokens == 150
        assert len(tracker.history) == 1
        assert tracker.history[0]["label"] == "test"

    def test_cost_report(self):
        """测试成本报告."""
        from src.utils.token_tracker import TokenTracker

        tracker = TokenTracker()
        tracker.record(prompt_tokens=1000, completion_tokens=500, model="deepseek-chat")
        summary = tracker.summary()
        assert summary["total_tokens"] == 1500
        assert summary["num_calls"] == 1
        assert summary["total_cost_usd"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
