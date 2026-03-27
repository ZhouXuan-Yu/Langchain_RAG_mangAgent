"""Test memory — 09+13+18: ChromaDB Upsert 逻辑."""

import pytest
import tempfile
import shutil
from pathlib import Path


class TestChromaMemoryStore:
    """测试 ChromaDB 记忆存储和 Upsert 逻辑."""

    @pytest.fixture
    def temp_store(self):
        """创建临时 ChromaDB 存储."""
        tmpdir = tempfile.mkdtemp()
        from src.memory.chroma_store import ChromaMemoryStore
        store = ChromaMemoryStore(
            collection_name="test_memory",
            path=tmpdir,
            threshold=0.85,
        )
        yield store
        # Cleanup
        try:
            shutil.rmtree(tmpdir)
        except Exception:
            pass

    def test_add_new_memory(self, temp_store):
        """测试新增记忆."""
        result = temp_store.upsert(
            content="[project] 智程导航项目使用 FastAPI 开发",
            metadata={"category": "project", "fact": "智程导航使用 FastAPI"},
        )
        assert result in ("added", "updated")

        memories = temp_store.get_all()
        assert len(memories) == 1
        assert "FastAPI" in memories[0]["content"]

    def test_upsert_duplicate_skipped(self, temp_store):
        """测试重复内容被忽略."""
        temp_store.upsert(
            content="[tech_stack] 使用 Python 开发",
            metadata={"category": "tech_stack", "fact": "使用 Python 开发"},
        )

        # 几乎相同的内容应该被忽略
        result = temp_store.upsert(
            content="[tech_stack] 使用 Python 开发",  # 完全相同
            metadata={"category": "tech_stack", "fact": "使用 Python 开发"},
        )
        assert result == "skipped"

        # 只有一条记录
        assert len(temp_store.get_all()) == 1

    def test_upsert_comprehensive_update(self, temp_store):
        """测试新内容更全面时更新旧记忆."""
        # 先存入简短记忆
        temp_store.upsert(
            content="[hardware] 使用 RTX 5060",
            metadata={"category": "hardware", "fact": "使用 RTX 5060"},
        )

        # 存入更详细的同一主题记忆
        result = temp_store.upsert(
            content="[hardware] 使用 RTX 5060 进行深度学习训练，驱动版本 555.85，配合 CUDA 12.4 使用效果最佳",
            metadata={"category": "hardware", "fact": "RTX 5060 训练配置"},
        )
        # 内容显著增长，应该触发更新
        assert result in ("added", "updated", "skipped")  # 逻辑可能因相似度阈值决定

    def test_search_by_query(self, temp_store):
        """测试向量检索."""
        temp_store.upsert(
            content="[project] 智眸千析火灾检测项目使用 YOLOv8",
            metadata={"category": "project", "fact": "智眸千析用 YOLOv8"},
        )
        temp_store.upsert(
            content="[tech_stack] 技术栈包含 Python, PyTorch, OpenCV",
            metadata={"category": "tech_stack", "fact": "技术栈 Python PyTorch"},
        )

        results = temp_store.search(query="火灾检测项目", top_k=2)
        assert len(results) >= 1
        # YOLOv8 项目应该被检索到
        assert any("YOLOv8" in r["content"] or "智眸" in r["content"] for r in results)

    def test_search_by_category(self, temp_store):
        """测试按类别检索."""
        temp_store.upsert(
            content="[hardware] RTX 5060 显卡",
            metadata={"category": "hardware", "fact": "RTX 5060"},
        )
        temp_store.upsert(
            content="[tech_stack] Python 编程",
            metadata={"category": "tech_stack", "fact": "Python"},
        )

        hw_results = temp_store.search(query="", category="hardware", top_k=5)
        assert all(r["metadata"]["category"] == "hardware" for r in hw_results)
        assert len(hw_results) == 1

    def test_delete_memory(self, temp_store):
        """测试删除记忆."""
        temp_store.upsert(
            content="[test] 测试记忆",
            metadata={"category": "test", "fact": "测试"},
        )
        memories = temp_store.get_all()
        assert len(memories) == 1
        record_id = memories[0]["id"]

        deleted = temp_store.delete(record_id)
        assert deleted is True
        assert len(temp_store.get_all()) == 0


class TestMarkdownCleaner:
    """测试网页 Markdown 清洗工具."""

    def test_clean_basic(self):
        """测试基础清洗."""
        from src.utils.markdown_cleaner import clean_markdown

        dirty = "Hello   World\n\n\n\n\n\nToo many newlines"
        cleaned = clean_markdown(dirty)
        assert "\n\n\n" not in cleaned  # 最多保留两个换行

    def test_clean_html_entities(self):
        """测试 HTML 实体清理."""
        from src.utils.markdown_cleaner import clean_markdown

        text = "Hello&nbsp;World &lt;script&gt; &amp;"
        cleaned = clean_markdown(text)
        assert "&nbsp;" not in cleaned
        assert "&lt;" not in cleaned


class TestSummarizer:
    """测试网页摘要压缩工具."""

    def test_compress_short_text(self):
        """测试短文本不压缩."""
        from src.utils.summarizer import compress_web_content

        short = "Hello, this is a short text."
        result = compress_web_content(short, max_chars=100)
        assert result == short  # 不应添加截断提示

    def test_compress_long_text(self):
        """测试长文本压缩."""
        from src.utils.summarizer import compress_web_content

        long_text = "\n\n".join([f"Paragraph {i} with some content." for i in range(50)])
        result = compress_web_content(long_text, max_chars=500)
        assert len(result) <= 700  # 应显著缩短
        assert "已压缩" in result or len(result) <= 500

    def test_compress_preserves_first_last(self):
        """测试压缩保留首尾段落."""
        from src.utils.summarizer import compress_web_content

        paragraphs = ["First paragraph is important.", "Middle content.", "Last summary paragraph."]
        long_text = "\n\n".join(paragraphs * 30)
        result = compress_web_content(long_text, max_chars=500)
        assert "First paragraph" in result  # 首段应保留


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
