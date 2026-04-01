"""Output Manager — 编排任务输出文件夹管理模块.

职责：
- 为每个编排任务创建独立的输出文件夹（outputs/<plan_id>/）
- 按文件类型自动分类保存 Worker 产出（.py/.html/.md/.docx 等）
- 生成 SUMMARY.md 总结文档
- 提供文件元信息查询接口
"""

import logging
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── 项目根路径 ──────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUTS_ROOT = _PROJECT_ROOT / "outputs"

# ── 文件类型映射 ──────────────────────────────────────────────────────────────
FILE_TYPE_EXTENSIONS: dict[str, list[str]] = {
    "code":     [".py", ".js", ".ts", ".jsx", ".tsx", ".rs", ".go", ".java", ".cpp", ".c", ".h", ".sh", ".bat"],
    "html":     [".html", ".htm", ".css", ".scss", ".sass"],
    "markdown": [".md", ".mdx", ".txt"],
    "doc":      [".docx"],
    "data":     [".json", ".csv", ".yaml", ".yml", ".xml", ".toml"],
    "report":   [".pdf"],
    "image":    [".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"],
    "other":    [],
}

# 扩展名 -> 文件类型（反向索引）
_EXT_TO_TYPE: dict[str, str] = {}
for ftype, exts in FILE_TYPE_EXTENSIONS.items():
    for ext in exts:
        _EXT_TO_TYPE[ext.lower()] = ftype

# ── 文件大小上限 ─────────────────────────────────────────────────────────────
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10MB


class OutputFileInfo:
    """单个输出文件的元信息."""

    def __init__(
        self,
        filename: str,
        file_type: str,
        file_path: str,
        size_bytes: int,
        created_at: str,
        agent_id: str = "",
        task_id: str = "",
        description: str = "",
    ):
        self.filename = filename
        self.file_type = file_type
        self.file_path = file_path
        self.size_bytes = size_bytes
        self.created_at = created_at
        self.agent_id = agent_id
        self.task_id = task_id
        self.description = description

    def to_dict(self) -> dict:
        return {
            "filename":    self.filename,
            "file_type":  self.file_type,
            "file_path":  self.file_path,
            "size_bytes": self.size_bytes,
            "size_display": self._format_size(self.size_bytes),
            "created_at": self.created_at,
            "agent_id":   self.agent_id,
            "task_id":    self.task_id,
            "description": self.description,
        }

    @staticmethod
    def _format_size(size: int) -> str:
        if size < 1024:
            return f"{size}B"
        if size < 1024 * 1024:
            return f"{size / 1024:.1f}KB"
        return f"{size / (1024 * 1024):.1f}MB"


class OutputManager:
    """
    编排任务输出管理器 — 为每个 plan 创建独立的输出文件夹.

    文件夹结构：
        outputs/<plan_id>/
        ├── SUMMARY.md               ← 任务总结文档
        ├── <task_id>_<agent_id>_1.py
        ├── <task_id>_<agent_id>_2.html
        └── ...

    使用方式：
        om = OutputManager("plan_abc123")
        file_info = om.save_file("print('hello')", "code", "coder", "task_1", filename="hello.py")
        summary = om.generate_summary_md(results)
    """

    def __init__(self, plan_id: str, base_dir: Path | None = None):
        self.plan_id = plan_id
        self.base_dir = (base_dir or OUTPUTS_ROOT) / plan_id
        self._files: list[OutputFileInfo] = []
        self._seq_counter: dict[str, int] = {}  # agent_id -> seq
        self._created = False

    # ── 生命周期 ────────────────────────────────────────────────────────────

    def ensure_dir(self) -> Path:
        """确保输出目录存在，返回目录路径."""
        if not self.base_dir.exists():
            self.base_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"[output_manager] created output dir: {self.base_dir}")
        self._created = True
        return self.base_dir

    @property
    def output_dir(self) -> Path:
        """返回输出目录路径（若不存在则自动创建）."""
        return self.ensure_dir()

    @property
    def is_created(self) -> bool:
        return self._created and self.base_dir.exists()

    # ── 文件类型检测 ──────────────────────────────────────────────────────────

    @staticmethod
    def detect_file_type(filename: str) -> str:
        """根据文件名扩展名检测文件类型."""
        ext = Path(filename).suffix.lower()
        return _EXT_TO_TYPE.get(ext, "other")

    @staticmethod
    def infer_extension(output_type: str, filename_hint: str = "") -> str:
        """
        根据 output_type 推断文件扩展名。

        Args:
            output_type: "code" | "html" | "markdown" | "doc" | "data" | "report" | "other"
            filename_hint: 可选的文件名提示（用于从已有扩展名推断）

        Returns:
            带点的扩展名，如 ".py"，若无法推断则返回 ".txt"
        """
        if filename_hint:
            hint_ext = Path(filename_hint).suffix.lower()
            if hint_ext in _EXT_TO_TYPE:
                return hint_ext

        defaults: dict[str, str] = {
            "code":     ".py",
            "html":     ".html",
            "markdown": ".md",
            "doc":      ".docx",
            "data":     ".json",
            "report":   ".pdf",
            "other":    ".txt",
        }
        return defaults.get(output_type, ".txt")

    @staticmethod
    def is_text_type(file_type: str) -> bool:
        """判断文件类型是否为纯文本（可在前端预览）."""
        return file_type in ("code", "html", "markdown", "data")

    # ── 文件保存 ──────────────────────────────────────────────────────────────

    def _next_seq(self, agent_id: str) -> int:
        seq = self._seq_counter.get(agent_id, 0) + 1
        self._seq_counter[agent_id] = seq
        return seq

    def _sanitize_filename(self, name: str) -> str:
        """移除文件名中的非法字符."""
        name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name)
        name = name.strip(". ")
        if not name:
            name = "unnamed"
        return name[: 200]  # 限制长度

    def save_file(
        self,
        content: str,
        output_type: str,
        agent_id: str,
        task_id: str,
        filename: str = "",
        description: str = "",
    ) -> OutputFileInfo | None:
        """
        将内容写入输出文件夹。

        Args:
            content: 文件内容（字符串）
            output_type: 文件类型标识（"code" | "html" | "markdown" | "doc" | "data" | "report" | "other"）
            agent_id: 生成该文件的 Agent ID
            task_id: 对应的任务 ID
            filename: 可选，指定文件名（不含扩展名）；若为空则自动生成
            description: 可选，文件描述

        Returns:
            OutputFileInfo 对象，或失败时返回 None
        """
        try:
            self.ensure_dir()

            # 确定扩展名
            ext = self.infer_extension(output_type, filename)

            # 确定文件名
            if not filename:
                seq = self._next_seq(agent_id)
                safe_agent = self._sanitize_filename(agent_id)
                safe_task = self._sanitize_filename(task_id)
                filename = f"{safe_task}_{safe_agent}_{seq}{ext}"
            else:
                # 用户指定了文件名，确保有正确扩展名
                filename = self._sanitize_filename(filename)
                if Path(filename).suffix.lower() not in _EXT_TO_TYPE:
                    filename += ext

            filepath = self.base_dir / filename

            # 写入内容（处理大文件）
            content_bytes = content.encode("utf-8", errors="replace")
            if len(content_bytes) > MAX_FILE_SIZE_BYTES:
                logger.warning(
                    f"[output_manager] file {filename} exceeds {MAX_FILE_SIZE_BYTES} bytes, "
                    f"truncating to 10MB"
                )
                content = content[: MAX_FILE_SIZE_BYTES].decode("utf-8", errors="replace")

            filepath.write_text(content, encoding="utf-8")

            # 构建元信息
            stat = filepath.stat()
            file_info = OutputFileInfo(
                filename=filename,
                file_type=self.detect_file_type(filename),
                file_path=str(filepath),
                size_bytes=stat.st_size,
                created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                agent_id=agent_id,
                task_id=task_id,
                description=description,
            )
            self._files.append(file_info)

            logger.info(
                f"[output_manager] saved file: {filename} ({file_info.size_bytes} bytes) "
                f"for task={task_id} agent={agent_id}"
            )
            return file_info

        except Exception as e:
            logger.error(f"[output_manager] failed to save file: {e}")
            return None

    def save_file_bytes(
        self,
        content: bytes,
        output_type: str,
        agent_id: str,
        task_id: str,
        filename: str,
        description: str = "",
    ) -> OutputFileInfo | None:
        """保存二进制文件（用于 .docx、.pdf 等）."""
        try:
            self.ensure_dir()

            filepath = self.base_dir / self._sanitize_filename(filename)
            if len(content) > MAX_FILE_SIZE_BYTES:
                logger.warning(f"[output_manager] binary file {filename} exceeds size limit")
                content = content[: MAX_FILE_SIZE_BYTES]

            filepath.write_bytes(content)

            stat = filepath.stat()
            file_info = OutputFileInfo(
                filename=filepath.name,
                file_type=self.detect_file_type(filepath.name),
                file_path=str(filepath),
                size_bytes=stat.st_size,
                created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                agent_id=agent_id,
                task_id=task_id,
                description=description,
            )
            self._files.append(file_info)

            logger.info(f"[output_manager] saved binary file: {filepath.name}")
            return file_info

        except Exception as e:
            logger.error(f"[output_manager] failed to save binary file: {e}")
            return None

    def read_file(self, filename: str) -> str | None:
        """读取已保存的文件内容."""
        try:
            filepath = self.base_dir / filename
            if not filepath.exists():
                return None
            return filepath.read_text(encoding="utf-8")
        except Exception as e:
            logger.error(f"[output_manager] failed to read {filename}: {e}")
            return None

    def file_exists(self, filename: str) -> bool:
        """检查文件是否存在."""
        return (self.base_dir / filename).exists()

    # ── SUMMARY.md 生成 ───────────────────────────────────────────────────────

    def generate_summary_md(self, all_results: list[dict], requirement: str = "") -> OutputFileInfo | None:
        """
        在输出文件夹根目录生成 SUMMARY.md 总结文档。

        Args:
            all_results: 所有 Worker 的执行结果列表
            requirement: 用户原始需求描述

        Returns:
            生成的 SUMMARY.md 的 OutputFileInfo
        """
        lines: list[str] = []
        lines.append(f"# 任务执行总结\n")
        lines.append(f"**计划ID**: `{self.plan_id}`\n")
        lines.append(f"**生成时间**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC\n")
        lines.append(f"**状态**: ✅ 执行完成\n")
        if requirement:
            lines.append(f"\n## 用户需求\n\n{requirement}\n")

        # ── 产出文件列表 ────────────────────────────────────────────────────
        lines.append("\n## 产出文件\n\n")
        if self._files:
            lines.append("| 文件名 | 类型 | 大小 | Agent | 任务 | 说明 |\n")
            lines.append("|--------|------|------|-------|------|------|\n")
            for f in self._files:
                desc = f.description or "—"
                size = OutputFileInfo._format_size(f.size_bytes)
                lines.append(
                    f"| `{f.filename}` | {f.file_type} | {size} | {f.agent_id} | {f.task_id} | {desc} |\n"
                )
        else:
            lines.append("*（无文件产出）*\n")

        # ── 各 Worker 结果摘要 ───────────────────────────────────────────────
        lines.append("\n## 执行详情\n\n")
        for result in all_results:
            agent = result.get("agent", "unknown")
            task_id = result.get("task_id", "")
            score = result.get("quality_score", 0.0)
            raw = result.get("raw_data", "") or result.get("result", "")

            lines.append(f"### [{agent}] 任务: {task_id}\n")
            lines.append(f"- **质量评分**: {score:.1f}/10\n")
            lines.append(f"- **来源**: {result.get('source', 'unknown')}\n")

            # 显示部分原始内容
            if raw:
                preview = raw[: 800].replace("```", "\\`\\`\\`")
                lines.append(f"\n**内容预览（前800字符）**:\n\n```\n{preview}\n```\n")
            lines.append("\n")

        # ── 使用说明 ──────────────────────────────────────────────────────
        lines.append("\n## 使用说明\n\n")
        code_files = [f for f in self._files if f.file_type == "code"]
        html_files = [f for f in self._files if f.file_type == "html"]
        md_files = [f for f in self._files if f.file_type == "markdown"]

        if code_files:
            lines.append("### 代码文件\n")
            for f in code_files:
                lines.append(f"- `{f.filename}` — {f.description or '代码文件'}\n")
            lines.append("\n")

        if html_files:
            lines.append("### 可直接在浏览器打开\n")
            for f in html_files:
                lines.append(f"- `{f.filename}` — {f.description or 'HTML文件'}\n")
            lines.append("\n")

        if md_files:
            lines.append("### Markdown 文档\n")
            for f in md_files:
                lines.append(f"- `{f.filename}` — {f.description or '文档'}\n")
            lines.append("\n")

        summary_content = "\n".join(lines)
        return self.save_file(
            content=summary_content,
            output_type="markdown",
            agent_id="system",
            task_id="summary",
            filename="SUMMARY.md",
            description="任务执行总结文档",
        )

    # ── 查询接口 ──────────────────────────────────────────────────────────────

    def list_files(self, file_type: str | None = None) -> list[dict]:
        """返回所有已保存文件的元信息列表."""
        files = self._files
        if file_type:
            files = [f for f in files if f.file_type == file_type]
        return [f.to_dict() for f in files]

    def get_file_info(self, filename: str) -> dict | None:
        """根据文件名查询文件元信息."""
        for f in self._files:
            if f.filename == filename:
                return f.to_dict()
        return None

    def get_output_info(self) -> dict:
        """返回完整的输出信息（用于 API 返回）."""
        return {
            "plan_id":    self.plan_id,
            "output_dir": str(self.base_dir),
            "output_url": f"/api/orchestrate/outputs/{self.plan_id}",
            "file_count": len(self._files),
            "files":      self.list_files(),
            "total_size": sum(f.size_bytes for f in self._files),
        }

    # ── 打包下载 ──────────────────────────────────────────────────────────────

    def create_archive(self) -> Path | None:
        """
        将输出文件夹打包为 .zip 存档。

        Returns:
            .zip 文件路径，或失败时返回 None
        """
        try:
            if not self.is_created:
                return None
            archive_path = OUTPUTS_ROOT / f"{self.plan_id}.zip"
            if archive_path.exists():
                archive_path.unlink()

            shutil.make_archive(
                base_name=str(OUTPUTS_ROOT / self.plan_id),
                format="zip",
                root_dir=str(self.base_dir.parent),
                base_dir=self.plan_id,
            )
            logger.info(f"[output_manager] created archive: {archive_path}")
            return archive_path

        except Exception as e:
            logger.error(f"[output_manager] failed to create archive: {e}")
            return None

    def cleanup(self) -> None:
        """删除输出文件夹（清理）."""
        try:
            if self.base_dir.exists():
                shutil.rmtree(self.base_dir)
                logger.info(f"[output_manager] cleaned up: {self.base_dir}")
        except Exception as e:
            logger.error(f"[output_manager] cleanup failed: {e}")

    def __repr__(self) -> str:
        return f"<OutputManager plan_id={self.plan_id!r} files={len(self._files)}>"


# ── 全局缓存（plan_id -> OutputManager 实例）───────────────────────────────────
_managers: dict[str, OutputManager] = {}


def get_output_manager(plan_id: str) -> OutputManager:
    """获取或创建指定 plan_id 的 OutputManager 实例."""
    if plan_id not in _managers:
        _managers[plan_id] = OutputManager(plan_id)
    return _managers[plan_id]


def cleanup_output_manager(plan_id: str) -> None:
    """清理指定 plan_id 的 OutputManager."""
    _managers.pop(plan_id, None)
