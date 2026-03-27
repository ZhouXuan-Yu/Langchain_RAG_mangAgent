"""文档处理器 — 支持 PDF/DOCX/TXT/Markdown/CSV/图片/代码 的文本提取与分块."""

from __future__ import annotations

import io
import logging
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import BinaryIO, Optional

from src.config import CHROMA_PATH

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DOCS_DIR = DATA_DIR / "documents"
DOCS_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DB = DATA_DIR / "documents.db"


def _uuid() -> str:
    return "doc_" + uuid.uuid4().hex[:12]


# ═══════════════════════════════════════════════════════════════════════════════
#  SQLite 连接
# ═══════════════════════════════════════════════════════════════════════════════

_docs_lock = sqlite3


@contextmanager
def _conn():
    conn = sqlite3.connect(str(DOCS_DB), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _init_db():
    with _conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id           TEXT PRIMARY KEY,
                filename     TEXT NOT NULL,
                doc_type     TEXT NOT NULL,
                size_bytes   INTEGER DEFAULT 0,
                chunk_count  INTEGER DEFAULT 0,
                status       TEXT DEFAULT 'uploading',
                error        TEXT,
                storage_path TEXT,
                uploaded_at  TEXT
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS doc_chunks (
                id          TEXT PRIMARY KEY,
                doc_id      TEXT NOT NULL,
                chunk_index INTEGER DEFAULT 0,
                chunk_text  TEXT NOT NULL,
                FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE
            )
        """)
        c.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_doc ON doc_chunks(doc_id)
        """)


_init_db()


# ═══════════════════════════════════════════════════════════════════════════════
#  文档类型检测
# ═══════════════════════════════════════════════════════════════════════════════

EXT_TO_TYPE = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".doc": "docx",
    ".txt": "txt",
    ".md": "markdown",
    ".markdown": "markdown",
    ".csv": "csv",
    ".py": "code",
    ".js": "code",
    ".ts": "code",
    ".java": "code",
    ".cpp": "code",
    ".c": "code",
    ".go": "code",
    ".rs": "code",
    ".sh": "code",
    ".sql": "code",
    ".json": "code",
    ".yaml": "code",
    ".yml": "code",
    ".xml": "code",
    ".html": "code",
    ".css": "code",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".gif": "image",
    ".webp": "image",
    ".bmp": "image",
}


def detect_doc_type(filename: str) -> str:
    """根据文件扩展名检测文档类型."""
    _, ext = os.path.splitext(filename.lower())
    return EXT_TO_TYPE.get(ext, "txt")


ALLOWED_TYPES = set(EXT_TO_TYPE.values())


def is_allowed_type(doc_type: str) -> bool:
    return doc_type in ALLOWED_TYPES


# ═══════════════════════════════════════════════════════════════════════════════
#  文本提取
# ═══════════════════════════════════════════════════════════════════════════════

def extract_text(content: bytes, doc_type: str) -> str:
    """
    根据文档类型从二进制内容中提取纯文本。

    参数:
        content: 文件二进制内容
        doc_type: 文档类型标识符

    返回:
        提取的纯文本字符串
    """
    if not content:
        return ""

    try:
        if doc_type == "pdf":
            return _extract_pdf(content)
        elif doc_type == "docx":
            return _extract_docx(content)
        elif doc_type == "txt":
            return content.decode("utf-8", errors="replace")
        elif doc_type == "markdown":
            return content.decode("utf-8", errors="replace")
        elif doc_type == "csv":
            return _extract_csv(content)
        elif doc_type == "image":
            return _extract_image_description(content, filename="")
        elif doc_type == "code":
            return _extract_code(content)
        else:
            return content.decode("utf-8", errors="replace")
    except Exception as e:
        logger.warning(f"[doc_processor] extract failed for type {doc_type}: {e}")
        return ""


def _extract_pdf(content: bytes) -> str:
    try:
        import pypdf

        reader = pypdf.PdfReader(io.BytesIO(content))
        texts = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                texts.append(text)
        return "\n\n".join(texts)
    except ImportError:
        logger.warning("[doc_processor] pypdf not installed, returning empty")
        return ""
    except Exception as e:
        logger.warning(f"[doc_processor] PDF extraction error: {e}")
        return ""


def _extract_docx(content: bytes) -> str:
    try:
        from docx import Document

        doc = Document(io.BytesIO(content))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n".join(paragraphs)
    except ImportError:
        logger.warning("[doc_processor] python-docx not installed, returning empty")
        return ""
    except Exception as e:
        logger.warning(f"[doc_processor] DOCX extraction error: {e}")
        return ""


def _extract_csv(content: bytes) -> str:
    try:
        import pandas as pd

        df = pd.read_csv(io.BytesIO(content))
        return df.to_string(index=False)
    except ImportError:
        # 降级：逐行读取
        try:
            text = content.decode("utf-8", errors="replace")
            return text[:5000]  # 截断避免过长
        except Exception:
            return ""
    except Exception as e:
        logger.warning(f"[doc_processor] CSV extraction error: {e}")
        return ""


def _extract_image_description(content: bytes, filename: str) -> str:
    """
    图片暂存，返回占位符文本。
    实际描述由 Agent 的 process_image 工具生成。
    """
    # 将图片保存到临时路径，由 Agent 处理
    tmp_path = DOCS_DIR / f"img_{uuid.uuid4().hex[:8]}.tmp"
    tmp_path.write_bytes(content)
    return f"[IMAGE_PLACEHOLDER: {tmp_path.name}]"


def _extract_code(content: bytes) -> str:
    """代码文件：保留原始文本，最多 5000 字符."""
    text = content.decode("utf-8", errors="replace")
    return text[:5000]


# ═══════════════════════════════════════════════════════════════════════════════
#  文本分块
# ═══════════════════════════════════════════════════════════════════════════════

CHUNK_SIZE = 500  # 每个 chunk 的字符数
CHUNK_OVERLAP = 50  # 重叠字符数


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE) -> list[str]:
    """
    将长文本切分为重叠的小块。

    策略：按段落切分，单段落过长则按字符硬切。
    """
    if not text:
        return []

    # 先按段落分割
    paragraphs = text.split("\n\n")
    chunks = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current) + len(para) + 2 <= chunk_size:
            current += ("\n\n" if current else "") + para
        else:
            if current:
                chunks.append(current)
            # 如果单个段落就超过 chunk_size，按字符硬切
            if len(para) > chunk_size:
                for i in range(0, len(para), chunk_size - CHUNK_OVERLAP):
                    sub = para[i : i + chunk_size]
                    if sub.strip():
                        chunks.append(sub)
                current = ""
            else:
                current = para

    if current:
        chunks.append(current)

    return [c.strip() for c in chunks if c.strip()]


# ═══════════════════════════════════════════════════════════════════════════════
#  DocumentStore — SQLite 文档元数据管理
# ═══════════════════════════════════════════════════════════════════════════════

class DocumentStore:
    """文档元数据 + chunk 存储."""

    def list_docs(self, doc_type: Optional[str] = None) -> list[dict]:
        with _conn() as c:
            if doc_type:
                rows = c.execute(
                    "SELECT * FROM documents WHERE doc_type=? ORDER BY uploaded_at DESC",
                    (doc_type,),
                ).fetchall()
            else:
                rows = c.execute(
                    "SELECT * FROM documents ORDER BY uploaded_at DESC"
                ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get(self, doc_id: str) -> Optional[dict]:
        with _conn() as c:
            row = c.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
        if not row:
            return None
        return self._row_to_dict(row)

    def save(
        self,
        filename: str,
        doc_type: str,
        size_bytes: int,
        content: bytes,
    ) -> dict:
        """保存文档：写入文件 + 记录元数据."""
        doc_id = _uuid()
        storage_name = f"{doc_id}_{filename}"
        storage_path = DOCS_DIR / storage_name
        storage_path.write_bytes(content)

        now = datetime.now().isoformat()
        with _conn() as c:
            c.execute(
                "INSERT INTO documents (id,filename,doc_type,size_bytes,status,storage_path,uploaded_at) "
                "VALUES (?,?,?,?,?,?,?)",
                (doc_id, filename, doc_type, size_bytes, "processing", str(storage_path), now),
            )

        logger.info(f"[doc_store] saved document {doc_id}: {filename}")
        return self.get(doc_id)

    def update_status(
        self,
        doc_id: str,
        status: str,
        chunk_count: Optional[int] = None,
        error: Optional[str] = None,
    ) -> Optional[dict]:
        updates = ["status=?"]
        params: list = [status]
        if chunk_count is not None:
            updates.append("chunk_count=?")
            params.append(chunk_count)
        if error is not None:
            updates.append("error=?")
            params.append(error)
        params.append(doc_id)

        with _conn() as c:
            c.execute(f"UPDATE documents SET {','.join(updates)} WHERE id=?", params)
        return self.get(doc_id)

    def delete(self, doc_id: str) -> bool:
        """删除文档：移除文件 + 删除元数据（cascade 删除 chunks）."""
        doc = self.get(doc_id)
        if not doc:
            return False
        # 删除存储文件
        if doc.get("storage_path"):
            p = Path(doc["storage_path"])
            if p.exists():
                p.unlink()
        with _conn() as c:
            c.execute("DELETE FROM documents WHERE id=?", (doc_id,))
        logger.info(f"[doc_store] deleted document {doc_id}")
        return True

    def add_chunks(self, doc_id: str, chunks: list[str]) -> int:
        """为文档存储文本块."""
        with _conn() as c:
            for idx, chunk in enumerate(chunks):
                chunk_id = f"{doc_id}_c{idx}"
                c.execute(
                    "INSERT INTO doc_chunks (id,doc_id,chunk_index,chunk_text) VALUES (?,?,?,?)",
                    (chunk_id, doc_id, idx, chunk),
                )
        return len(chunks)

    def get_chunks(self, doc_id: str) -> list[str]:
        """获取文档的所有文本块."""
        with _conn() as c:
            rows = c.execute(
                "SELECT chunk_text FROM doc_chunks WHERE doc_id=? ORDER BY chunk_index",
                (doc_id,),
            ).fetchall()
        return [r["chunk_text"] for r in rows]

    def ingest_to_chroma(self, doc_id: str) -> int:
        """
        将文档 chunks 批量写入 ChromaDB，关联 document_id tag。
        """
        from src.memory.chroma_store import ChromaMemoryStore

        chunks = self.get_chunks(doc_id)
        if not chunks:
            # 否则 status 会永远停在 processing（extract 失败 / 空文本 / 未写入 chunk）
            self.update_status(
                doc_id,
                "failed",
                chunk_count=0,
                error="未生成文本块：文件可能为空，或缺少依赖（如 Word 需 python-docx、PDF 需 pypdf）",
            )
            return 0

        store = ChromaMemoryStore()
        doc = self.get(doc_id)
        filename = doc["filename"] if doc else doc_id

        count = 0
        for chunk in chunks:
            store.upsert(
                content=chunk[:1000],  # ChromaDB 有长度限制
                metadata={
                    "category": "document",
                    "importance": 5,
                    "document_id": doc_id,
                    "filename": filename,
                    "source": "document_ingestion",
                },
            )
            count += 1

        self.update_status(doc_id, "ready", chunk_count=count)
        logger.info(f"[doc_store] ingested {count} chunks to ChromaDB for doc {doc_id}")
        return count

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict:
        return {
            "id": row["id"],
            "filename": row["filename"],
            "doc_type": row["doc_type"],
            "size_bytes": row["size_bytes"],
            "chunk_count": row["chunk_count"],
            "status": row["status"],
            "error": row["error"],
            "storage_path": row["storage_path"],
            "uploaded_at": row["uploaded_at"],
        }


# ── 全局单例 ──────────────────────────────────────────────────────────────
_doc_store: Optional[DocumentStore] = None


def get_document_store() -> DocumentStore:
    global _doc_store
    if _doc_store is None:
        _doc_store = DocumentStore()
    return _doc_store
