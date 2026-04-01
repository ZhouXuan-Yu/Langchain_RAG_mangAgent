# 多智能体协同学习笔记

> 记录日期：2026-04-01
> 项目：Crayfish Multi-Agent System

---

## 一、Supervisor 模式是什么？

Supervisor（监督者）是一种常见的多智能体架构模式，由一个**中央协调者**负责分析用户需求、拆解任务、并分发给下属 Worker Agent 执行。

### 项目中的 Supervisor 实现

在 `src/graph/orchestrator.py` 的 `CrayfishOrchestrator` 类中，Supervisor 的核心职责由 `_create_plan()` 方法实现：

```python
# src/graph/orchestrator.py 第 223-382 行
async def _create_plan(self, requirement: str, enabled_agents: list[str]) -> list[TaskItem]:
    """
    Supervisor Planning Node — 分析需求，生成 JSON Plan。
    使用"最小完备原则"：只拆解当前最紧迫的 2-3 个子任务。
    """
```

**工作流程：**
1. 接收用户需求描述
2. 收集当前启用的 Agent 档案（participants）
3. 构建 prompt，包含 Agent 描述和任务拆分原则
4. 调用 LLM 生成 JSON 格式的任务清单（最多 3 个子任务）
5. 校验每个 task 的 worker_kind 是否在 enabled_kinds 中
6. 失败时回退到基于关键词的自动拆解

### Supervisor 的核心价值

| 能力 | 说明 |
|------|------|
| 任务拆解 | 将复杂需求分解为可并行/顺序执行的子任务 |
| 智能路由 | 根据 Agent 档案匹配最适合的 Worker |
| 结果聚合 | 汇总各 Worker 输出，生成最终回复 |
| 质量把控 | 评估结果质量，不达标时触发自修复 |

---

## 二、DAG 依赖是什么？---满足代码的要求，可完成性

DAG = **Directed Acyclic Graph**（有向无环图）

这是计算机科学中的一种数据结构，由**节点（Node）**和**有向边（Edge）**组成：
- **有向**：边有方向，表示依赖关系（A → B 表示 B 依赖 A）
- **无环**：不存在循环依赖（A → B → C → A 是不允许的）

### 在多智能体协同中的意义

传统模式下，任务之间的关系只有两种：**并行**或**顺序**。

引入 DAG 后，可以表达更复杂的依赖关系：

```
示例：一个需要深度研究的任务

用户需求："帮我调研 RAG 技术的最新进展，并写一个基于 LangChain 的 RAG 示例代码"

DAG 依赖图：

    [search_latest_papers] ──────────┐
           ↓                          │
    [rag_query_my_knowledge]         │  （并行执行）
           ↓                          │
    [analyze_findings] ──────────────→│──→ [write_rag_code]
           ↑                          │         ↓
    [search_related_implementations] ─┘   [review_code]
                                               ↓
                                         [final_output]
```

在这个 DAG 中：
- `analyze_findings` 必须等待 `search_latest_papers` 和 `rag_query_my_knowledge` 都完成
- `write_rag_code` 必须等待 `analyze_findings` 完成
- `review_code` 必须等待 `write_rag_code` 完成

### 当前项目 vs DAG 理想状态

**当前项目**（`src/graph/orchestrator.py` 第 98-137 行）：

```python
# 按 worker_kind 分组：内置并行的 vs 顺序的
builtin_parallel = {"search_worker", "rag_worker"}
sequential = {"coder"}

parallel_tasks = [t for t in plan_tasks if ...]
coder_tasks = [t for t in plan_tasks if ...]
# 顺序执行 coder（依赖前两个任务的结果）
for coder_task in coder_tasks:
    context = all_results if all_results else None
```

局限性：
- 只支持「所有并行 + coder 顺序」的简单模式
- 无法表达「A 和 B 并行，但 C 必须等 A，B 必须等 C」这样的复杂依赖

**DAG 增强后**的 TaskItem 结构（规划中）：

```python
TaskItem {
    task_id: str,
    description: str,
    assigned_agent: str,
    worker_kind: str | None,
    depends_on: list[str] | None,      # ★ 新增：依赖的任务 ID 列表
    execution_mode: "parallel" | "sequential",  # ★ 新增：执行模式
    status: str,
    result: Any,
    quality_score: float,
}
```

---

## 三、如何改进 Supervisor 以支持 DAG 依赖

### 改进路线图

#### Phase 1: 数据模型升级

```python
# 1. 修改 TaskItem，增加依赖字段
class TaskItem(TypedDict):
    task_id: str
    description: str
    assigned_agent: str
    worker_kind: str | None
    depends_on: list[str] | None    # 依赖的任务 ID 列表
    execution_mode: str | None      # "parallel" | "sequential"
    status: str
    result: Any
    quality_score: float
```

#### Phase 2: Supervisor Prompt 增强

```python
prompt = f"""你是一个任务规划专家（Supervisor）。请分析用户需求，将其拆解为可执行的子任务。

用户需求:
{requirement}

可用的 Agent:
{agents_desc}

拆分原则:
1. 只拆分确实需要的子任务，不要过度拆分
2. 互不干扰的任务（如搜索和记忆检索）可以并行
3. 代码生成任务依赖搜索/记忆结果，应放在最后
4. 每个任务描述要清晰、具体
5. ★ 新增：为每个任务指定 depends_on 字段（依赖的其他 task_id 列表）

请输出 JSON 格式的 Plan:
{{
  "plan_id": "plan_xxx",
  "tasks": [
    {{
      "task_id": "task_1",
      "description": "具体任务描述",
      "assigned_agent": "agent_id",
      "depends_on": [],  // 依赖的任务 ID 列表
      "execution_mode": "parallel"  // 或 "sequential"
    }}
  ]
}}"""
```

#### Phase 3: DAG 调度引擎

```python
from collections import defaultdict, deque

class DAGScheduler:
    """基于 DAG 的任务调度器"""
    
    def __init__(self, tasks: list[TaskItem]):
        self.tasks = {t["task_id"]: t for t in tasks}
        self.graph = defaultdict(list)  # task_id -> [依赖它的任务]
        self.in_degree = defaultdict(int)  # 入度计数
        
        self._build_graph()
    
    def _build_graph(self):
        """根据 depends_on 构建 DAG"""
        for task_id, task in self.tasks.items():
            depends = task.get("depends_on", []) or []
            for dep_id in depends:
                if dep_id in self.tasks:
                    self.graph[dep_id].append(task_id)
                    self.in_degree[task_id] += 1
    
    def get_executable_tasks(self) -> list[TaskItem]:
        """获取所有入度为 0（无依赖）的任务"""
        return [
            task for task_id, task in self.tasks.items()
            if self.in_degree[task_id] == 0
            and task["status"] == TASK_STATUS_PENDING
        ]
    
    def mark_completed(self, task_id: str):
        """标记任务完成，更新依赖它的任务的入度"""
        for dependent_id in self.graph[task_id]:
            self.in_degree[dependent_id] -= 1
    
    def is_complete(self) -> bool:
        """检查是否所有任务都已完成"""
        return all(t["status"] == TASK_STATUS_COMPLETED for t in self.tasks.values())
```

#### Phase 4: 调度执行

```python
async def orchestrate_with_dag(self, requirement, enabled_agents):
    # 1. Supervisor 规划
    plan_tasks = await self._create_plan(requirement, enabled_agents)
    
    # 2. 构建 DAG 调度器
    scheduler = DAGScheduler([dict(t) for t in plan_tasks])
    
    # 3. 按 DAG 顺序执行
    while not scheduler.is_complete():
        # 获取可执行的任务（入度为 0）
        executable = scheduler.get_executable_tasks()
        
        # 并行执行所有可执行任务
        if executable:
            coroutines = [
                self._execute_single_task(t, progress_callback)
                for t in executable
            ]
            results = await asyncio.gather(*coroutines, return_exceptions=True)
            
            # 标记完成，更新 DAG
            for result in results:
                if not isinstance(result, Exception):
                    scheduler.mark_completed(result["task_id"])
        
        await asyncio.sleep(0.1)  # 防止忙等待
```

---

## 四、对话总结

### 本次对话核心议题

1. **多智能体协同的整体设计**
   - Supervisor 驱动的 Plan-then-Execute 模式
   - 三省六部制的隐喻架构
   - Worker 的并行/顺序混合执行

2. **质量保障机制**
   - 多源置信度权重（Web 0.95 > RAG 0.70 > Memory 0.50）
   - 各 Worker 自评估质量分（0-10）
   - LLM 驱动的自修复循环（最多 2 次）

3. **Agent 注册与管理**
   - AgentRegistry 的 CRUD 操作
   - 按需构建和缓存 Agent Graph 实例
   - SQLite + ChromaDB 双层持久化

4. **项目已知局限**
   - Supervisor 硬编码白名单
   - 自定义 Agent 无专门执行器
   - 前端只渲染内置 Agent 步骤
   - 无跨 Agent 状态共享
   - 规划能力受限（最多 3 任务，无 DAG）

5. **改进方向探讨**
   - Supervisor DAG 依赖支持
   - GenericWorkerExecutor 统一执行层
   - AgentMessageBus 消息总线

### 下一步探索方向

- [ ] 实现 DAG 调度引擎
- [ ] 增强 Supervisor Prompt 支持依赖声明
- [ ] 设计 AgentMessageBus 跨 Agent 通信
- [ ] 前端动态步骤渲染

---

## 五、MCP、Skills 与 AI Agent 热门术语梳理

> 更新日期：2026-04-01
> 数据来源：基于 2025-2026 年最新行业动态整理

---

### 5.1 MCP — Model Context Protocol（模型上下文协议）

**一句话理解：AI 领域的"USB-C 接口"**

MCP 是由 **Anthropic**（Claude 的开发公司）于 **2024年11月** 发布的开放标准。2025年12月已移交 Linux Foundation 治理，确保不被任何单一公司控制。

#### 核心价值

MCP 解决了一个根本问题：**每个 AI 应用都要自己写一套代码去连接外部工具**（浏览器、数据库、文件系统等），导致生态割裂。

```
传统方式：
  Claude App ──→ 专门代码 ──→ 浏览器
  Cursor AI  ──→ 专门代码 ──→ 浏览器    （各自独立，重复造轮子）
  GPT App    ──→ 专门代码 ──→ 浏览器

MCP 方式：
  Claude App ─┐
  Cursor AI  ─┼──→ MCP Server ──→ 浏览器/数据库/文件系统
  GPT App    ─┘
```

#### 技术架构

MCP Server 暴露三种能力：

| 能力类型 | 说明 | 项目中的对应 |
|---------|------|-------------|
| **Tools（工具）** | AI 可以调用的函数 | `src/tools/` 中的 `web_search`、`memory_search` 等 |
| **Resources（资源）** | AI 可读取的结构化数据 | ChromaDB 记忆、SQLite 会话数据 |
| **Prompts（提示模板）** | 可复用的 prompt 模板 | 项目的 prompt.py |

#### 2025-2026 最新进展

- **生态爆发**：截至 2026 年初，已有 **500+** 公共 MCP Server，SDK 月下载量超 **9700万次**
- **巨头全面支持**：OpenAI、Google（Gemini）、Microsoft、AWS 均已支持 MCP
- **MCP Apps（2026.01）**：允许外部应用直接在 Claude 界面中返回**可交互的 UI 组件**（仪表盘、表单、多步骤工作流）
- **MCP v2 Beta（2026.03）**：新增 OAuth 2.0 认证、多 Agent 通信支持、 elicitation（工具选择确认）

#### Cursor 对 MCP 的支持

```json5
// .cursor/mcp.json 配置示例
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/project"]
    },
    "browser": {
      "command": "python",
      "args": ["/path/to/browser-server.py"]
    }
  }
}
```

**传输方式**：`stdio`（进程通信）、`SSE`、`Streamable HTTP`（当前项目使用的 `cursor-ide-browser` MCP 即属此类）。

---

### 5.2 Cursor Skills（技能）

**一句话理解：教 Cursor AI 做特定事情的"说明书"**

Skills 是 Cursor 中的**可复用 Markdown 文件**，用于指导 Cursor Agent 处理特定的多步骤工作流。

#### Skills vs Rules（规则）

| | Skills | Rules |
|--|--------|-------|
| **用途** | 多步骤工作流和流程 | 编码规范和约束 |
| **粒度** | 复杂、详细 | 简短、精炼 |
| **触发** | `/skill-name` 或自动发现 | 每次对话自动生效 |
| **存储** | `.cursor/skills/your-skill-name/SKILL.md` | `.cursor/rules/*.md` 或 `AGENTS.md` |
| **适用场景** | 部署流程、安全审计、测试流程 | 代码风格、命名规范、安全要求 |

#### Skill 文件结构

```
.cursor/skills/
├── create-skill/
│   ├── SKILL.md          ← 必需：核心定义文件
│   ├── scripts/           ← 可选：执行的脚本
│   └── references/       ← 可选：参考资料
```

```yaml
# SKILL.md 的 YAML Frontmatter 示例
---
name: create-rule
description: 在 Cursor 中创建 AI 编码规则和指导
---
# 下面是技能的具体内容
```

#### 调用方式

```
输入 /create-rule    ← 通过 / 斜杠命令调用
或：@skill-name      ← 通过 @ 引用作为上下文
```

---

### 5.3 其他热门术语速查

#### 框架类

| 术语 | 全称 | 一句话说明 | 项目中的对应 |
|------|------|-----------|-------------|
| **LangGraph** | — | 构建有状态、多步骤 AI Agent 的图结构框架（节点=动作，边=条件跳转） | `src/graph/agent_graph.py` 的 `build_agent_graph()` |
| **LangChain** | — | LLM 应用的"标准库"，提供工具、prompt、链式调用等抽象 | 项目底层依赖 |
| **CrewAI** | — | 团队式多 Agent 框架，~20 行代码即可实现多 Agent 协作 | 项目对标的参考模式 |
| **AutoGen** | Microsoft | 微软的多 Agent 对话框架，支持 Agent 间自然语言协作 | — |
| **Temporal** | — | 企业级工作流引擎，用于任务编排和状态持久化 | 项目的编排端点可对标 |

#### Agent 模式类

| 术语 | 全称 | 一句话说明 |
|------|------|-----------|
| **ReAct** | Reasoning + Acting | Agent 在推理和行动之间交替执行（思考→行动→观察→思考） |
| **Agentic RAG** | — | 传统 RAG 的进化版，Agent 能自主决定检索策略、检索次数、结果验证 |
| **Tool Use** | — | Agent 调用外部工具的能力（搜索、代码执行等） |
| **Streaming / SSE** | Server-Sent Events | 服务器主动向客户端推送事件，项目中大量使用（`/chat/stream`、`/orchestrate/jobs/{id}/events`） |

#### 评估与质量类

| 术语 | 全称 | 项目中的对应 |
|------|------|-------------|
| **Self-Healing** | 自修复 | `src/graph/self_healer.py` |
| **Confidence Score** | 置信度 | `src/graph/conflict_resolver.py` 中的权重系统 |
| **Quality Threshold** | 质量阈值 | `orchestrator.py` 中的 `quality_threshold=8.0` |
| **Loop Counter** | 循环计数器 | `MAX_LOOP_COUNT=15` 防死循环 |

#### 持久化与记忆类

| 术语 | 全称 | 项目中的对应 |
|------|------|-------------|
| **Checkpoint** | 检查点 | SQLite持久化 (`src/memory/sqlite_store.py`) |
| **Vector Store** | 向量数据库 | ChromaDB (`src/memory/chroma_store.py`) |
| **Session** | 会话 | `sqlite_store.py` 中的 `upsert_session` |
| **Memory** | 长期记忆 | ChromaDB `user_memory` collection |

#### 前端与协作类

| 术语 | 全称 | 项目中的对应 |
|------|------|-------------|
| **SSE** | Server-Sent Events | `src/server/api.py` 的流式端点 |
| **Long Polling** | 长轮询 | `stream_orch_job_events()` 每 0.5s 检查一次 |
| **Orchestration UI** | 编排台 | `pages/orchestrate.html` |

---

### 5.4 MCP 与 Skills 的关系

```
MCP 解决的问题：AI 如何连接外部世界（工具/数据/服务）
Skills 解决的问题：AI 如何按照特定流程做事（流程/规范/模板）

MCP = 工具层标准化
Skills = 行为层标准化

项目中的映射：
  MCP ──────────→ 工具能力（browser_tools.py, memory_tools.py）
  Skills ───────→ 编码规范（可创建 .cursor/rules/ 补充）
  Supervisor ───→ 任务协调（MCP Server 的调度层）
  Worker ───────→ 具体执行器（类比 MCP Server 的具体功能）
```

---

### 5.5 术语生态全景图

```
                    ┌─────────────────────────────────────┐
                    │         AI Agent 系统全景            │
                    └─────────────────────────────────────┘
                                        │
            ┌─────────────────────────────┼─────────────────────────────┐
            │                             │                             │
     ┌──────▼──────┐            ┌───────▼──────┐            ┌────────▼──────┐
     │  协议/标准层  │            │   框架层      │            │   应用层       │
     └─────────────┘            └──────────────┘            └───────────────┘

协议/标准层：
  MCP ──────────── Model Context Protocol（工具/资源/提示标准化）
  SSE ──────────── Server-Sent Events（服务端推送）
  WebSocket ────── 双向实时通信（项目中未用）

框架层：
  LangGraph ────── 图结构 Agent 框架
  LangChain ────── LLM 应用"瑞士军刀"
  CrewAI ───────── 团队式多 Agent 编排
  Temporal ─────── 企业级工作流引擎

应用层：
  Supervisor ───── 任务规划协调者（项目核心）
  Worker ───────── 专项执行器（Search/RAG/Coder）
  ReAct ────────── 推理+行动交替模式
  Agentic RAG ───── 自主决策的 RAG

规范层：
  Rules ────────── 编码规范和约束（Cursor）
  Skills ────────── 多步骤工作流说明（Cursor）
  AGENTS.md ─────── 项目级 AI 指导文件

项目对应：
  MCP ──→ cursor-ide-browser MCP Server（已集成）
  Skills ─→ .cursor/skills-cursor/（项目已有 3 个 skill）
  LangGraph ─→ src/graph/agent_graph.py
  Supervisor ─→ src/graph/orchestrator.py CrayfishOrchestrator
  Worker ─→ src/graph/workers/
  ReAct ─→ src/graph/nodes.py（router → tool → reason 循环）
```

---

## 六、对话总结（第二轮）

### 新增议题

1. **MCP 生态**
   - Model Context Protocol = AI 领域的 USB-C
   - 项目已集成 `cursor-ide-browser` MCP Server
   - MCP Apps 支持在 Claude 中返回可交互 UI 组件
   - MCP v2 Beta 支持多 Agent 通信

2. **Cursor Skills**
   - 可复用的 Markdown 工作流说明
   - 与 Rules 的区别：Skills 管"怎么做"，Rules 管"写成什么样"
   - 项目的 `.cursor/skills-cursor/` 已有 3 个 skill

3. **术语生态全景**
   - 协议/标准层：MCP、SSE、WebSocket
   - 框架层：LangGraph、LangChain、CrewAI、Temporal
   - 模式层：ReAct、Agentic RAG、Tool Use
   - 规范层：Rules、Skills、AGENTS.md

### 项目与行业热点的对照

| 行业趋势 | 项目现状 |
|---------|---------|
| MCP 统一工具标准 | 已集成 browser MCP，正向 GenericWorker 演进 |
| LangGraph 图结构 Agent | 已用 StateGraph 构建 agent_graph |
| Multi-Agent 编排 | 已实现 Supervisor + 3 Worker，但缺 DAG |
| Skills 工作流复用 | 项目已有 3 个 Cursor skill |
| 企业级工作流（Temporal） | 项目的 BackgroundJobManager + TaskScheduler 可对标 |

---

*本文件记录多智能体协同系统的学习与思考过程*


