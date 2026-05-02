# LangChain RAG 记忆智能体

> 基于 LangGraph + ChromaDB + SQLite + DeepSeek，实现具备主动记忆 Upsert 的 AI 智能体系统，支持多智能体编排、Web UI 和任务调度。

---

## 目录

- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [完整安装详解](#完整安装详解)
- [配置说明](#配置说明)
- [运行项目](#运行项目)
- [项目结构](#项目结构)
- [核心模块说明](#核心模块说明)
- [Web UI 界面](#web-ui-界面)
- [API 接口](#api-接口)
- [常见问题](#常见问题)

---

## 环境要求

| 要求 | 版本 | 说明 |
|------|------|------|
| **Python** | **3.11+** | 必须，LangGraph/LangChain 生态要求 |
| conda | 任意版本 | 用于创建和管理 Python 环境 |
| DeepSeek API Key | 有效密钥 | 用于 LLM 对话能力 |
| Anthropic API Key | 可选 | 用于 Claude 模型 |
| OpenAI API Key | 可选 | 用于 GPT-4o 等模型 |
| Google API Key | 可选 | 用于 Gemini 模型 |
| Tavily API Key | 可选 | 用于网页搜索工具 |
| LangSmith API Key | 可选 | 用于链路追踪和监控 |

### 为什么选择 Python 3.11？

LangGraph 0.4+、LangChain 0.4+ 等核心依赖包对 Python 版本有明确要求：

- `langgraph>=0.4.0` 推荐 Python 3.11+
- `chromadb>=0.5.0` 在 Python 3.11 下性能最优
- `langchain-deepseek>=0.1.0` 使用 Pydantic V2，需要 Python 3.9+，推荐 3.11+
- Python 3.11 相比 3.10 有约 10-25% 的性能提升，对 AI 推理场景尤为重要
- 不支持 Python 3.8 及以下版本

---

## 快速开始

```bash
# 1. 克隆 / 进入项目目录
cd D:\Aprogress\Langchain

# 2. 使用 conda 创建 Python 3.11 环境
conda create -n langchain python=3.11 -y

# 3. 激活环境
conda activate langchain

# 4. 安装依赖
pip install -r requirements.txt

# 5. 安装 Playwright 浏览器驱动
playwright install chromium

# 6. 配置 API 密钥
copy .env.example .env
# 编辑 .env 文件，填入你的 DeepSeek API Key

# 7. 运行 Web 服务
python run.py
```

---

## 完整安装详解

### 第一步：创建 conda 环境

打开 **Anaconda Prompt**（或任意终端），执行：

```bash
# 使用 conda 创建名为 langchain 的环境，Python 版本为 3.11
conda create -n langchain python=3.11 -y
```

参数说明：
- `-n langchain`：环境名称为 `langchain`
- `python=3.11`：指定 Python 3.11 版本
- `-y`：自动确认，无需手动输入 yes

### 第二步：激活环境

```bash
# Windows / Linux / macOS 通用
conda activate langchain
```

验证 Python 版本：

```bash
python --version
# 预期输出：Python 3.11.x
```

### 第三步：安装依赖

```bash
pip install -r requirements.txt
```

> 注意：如果安装过程中出现网络问题，可以添加国内镜像源：
>
> ```bash
> pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
> ```

### 第四步：安装 Playwright 浏览器

```bash
playwright install chromium
```

Playwright 用于浏览器自动化工具（Tavily 网页搜索等功能依赖）。

如果安装失败，可以尝试：

```bash
# 设置浏览器下载源（国内加速）
playwright install chromium --with-deps
```

### 第五步：获取 API Key

#### DeepSeek API Key（必须）

1. 访问 [DeepSeek 开放平台](https://platform.deepseek.com/)
2. 注册 / 登录账号
3. 进入 **API Keys** 页面
4. 点击 **Create API Key**，复制生成的密钥
5. 格式类似：`sk-e2bc13304a6249cf956a36`

> 注意：DeepSeek API Key 是必须的，系统启动时会检查。若暂不配置其他模型，先只填这一项即可使用。

#### Anthropic API Key（可选，用于 Claude 模型）

1. 访问 [Anthropic Console](https://console.anthropic.com/)
2. 注册后进入 **API Keys** 页面
3. 创建密钥后复制
4. 在设置页面输入后，对应模型变为可选

#### OpenAI API Key（可选，用于 GPT-4o 等模型）

1. 访问 [OpenAI Platform](https://platform.openai.com/)
2. 进入 **API Keys** 页面创建密钥
3. 若使用代理或第三方兼容接口，可在设置中填写自定义 `base_url`
4. 在设置页面输入后，对应模型变为可选

#### Google API Key（可选，用于 Gemini 模型）

1. 访问 [Google AI Studio](https://aistudio.google.com/app/apikey)
2. 创建 API Key 后复制
3. 在设置页面输入后，对应模型变为可选

#### Tavily API Key（可选，用于网页搜索）

1. 访问 [Tavily AI](https://tavily.com/)
2. 注册后获取免费 API Key
3. 每日有免费配额（1000 次/天）

#### LangSmith API Key（可选，用于链路追踪）

1. 访问 [LangSmith](https://smith.langchain.com/)
2. 注册后获取 API Key
3. 用于监控和调试 Agent 执行链路

### 第六步：配置文件

```bash
# 在项目根目录执行
copy .env.example .env
```

编辑 `.env` 文件：

```env
# ===== 必须填写 =====
DEEPSEEK_API_KEY=sk-your-deepseek-api-key-here

# ===== 可选填写（多模型支持）=====
ANTHROPIC_API_KEY=sk-ant-your-anthropic-api-key-here
OPENAI_API_KEY=sk-your-openai-api-key-here
OPENAI_BASE_URL=https://api.openai.com/v1
GOOGLE_API_KEY=your-google-api-key-here

# ===== 可选填写 =====
TAVILY_API_KEY=tvly-your-tavily-api-key-here
LANGSMITH_API_KEY=your-langsmith-api-key-here

# ===== 模型参数（可选，可使用默认值）=====
DEFAULT_MODEL=deepseek-chat
TEMPERATURE=0.7
MAX_TOKENS=8000

# ===== 存储路径（可选）=====
CHROMA_PATH=./data/chroma_db
CHECKPOINT_PATH=./data/checkpointer/checkpoints.db
```

---

## 配置说明

| 配置项 | 必须 | 默认值 | 说明 |
|--------|------|--------|------|
| `DEEPSEEK_API_KEY` | ✅ | - | DeepSeek API 密钥 |
| `ANTHROPIC_API_KEY` | ❌ | 空 | Anthropic API 密钥（Claude 模型） |
| `OPENAI_API_KEY` | ❌ | 空 | OpenAI API 密钥（GPT-4o 等） |
| `OPENAI_BASE_URL` | ❌ | `https://api.openai.com/v1` | OpenAI 兼容端点（支持代理） |
| `GOOGLE_API_KEY` | ❌ | 空 | Google API 密钥（Gemini 模型） |
| `TAVILY_API_KEY` | ❌ | 空 | 网页搜索（可选） |
| `LANGSMITH_API_KEY` | ❌ | 空 | 链路追踪（可选） |
| `DEFAULT_MODEL` | ❌ | `deepseek-chat` | 模型名称 |
| `TEMPERATURE` | ❌ | `0.7` | 生成温度（0-1） |
| `MAX_TOKENS` | ❌ | `8000` | 最大输出 Token 数 |
| `CHROMA_PATH` | ❌ | `./data/chroma_db` | ChromaDB 存储路径 |
| `CHECKPOINT_PATH` | ❌ | `./data/checkpointer/checkpoints.db` | SQLite 状态持久化路径 |

---

## 运行项目

### Web 服务模式（推荐）

```bash
python run.py
```

启动后访问 `http://localhost:8000`

### 命令行模式

```bash
python src/main.py
```

### 运行测试

```bash
# 运行全部测试
pytest tests/

# 运行指定测试文件
pytest tests/test_memory.py -v

# 查看测试覆盖率
pytest tests/ --cov=src --cov-report=html
```

---

## 项目结构

```
D:\Aprogress\Langchain\
│
├── src/                         # 项目源代码
│   ├── __init__.py
│   ├── config.py                # 全局配置管理
│   ├── main.py                  # 命令行主入口
│   ├── document_processor.py    # 文档处理模块
│   │
│   ├── llm/                     # LLM 客户端
│   │   ├── __init__.py
│   │   └── deepseek_client.py  # DeepSeek API 调用封装
│   │
│   ├── tools/                   # 工具定义
│   │   ├── __init__.py
│   │   ├── browser_tools.py     # 浏览器搜索工具（Tavily）
│   │   ├── calc_tools.py       # 计算器工具
│   │   ├── memory_tools.py     # 记忆存储/检索工具
│   │   └── multimodal_tools.py # 多模态处理工具
│   │
│   ├── graph/                   # LangGraph 状态机
│   │   ├── __init__.py
│   │   ├── agent_graph.py      # Agent 图构建
│   │   ├── orchestrator.py     # 多任务编排器
│   │   ├── self_healer.py      # 自愈修复机制
│   │   ├── conflict_resolver.py # 冲突解决器
│   │   ├── nodes.py            # 图节点定义
│   │   ├── prompt.py           # Prompt 模板
│   │   ├── router.py           # 路由逻辑
│   │   └── state.py            # 状态定义
│   │   └── workers/            # 专用 Worker
│   │       ├── coder_worker.py  # 编码任务 Worker
│   │       ├── rag_worker.py    # RAG 任务 Worker
│   │       └── search_worker.py # 搜索任务 Worker
│   │
│   ├── multi_agent/             # 多智能体系统
│   │   ├── __init__.py
│   │   └── orchestrator.py     # 多智能体编排器
│   │
│   ├── memory/                  # 记忆系统
│   │   ├── __init__.py
│   │   ├── chroma_store.py     # ChromaDB 长期记忆
│   │   ├── memory_schema.py    # 记忆数据模型
│   │   └── sqlite_store.py     # SQLite 状态持久化
│   │
│   ├── middleware/              # 中间件
│   │   ├── __init__.py
│   │   ├── input_guard.py      # 输入验证
│   │   └── pii_redactor.py     # PII 脱敏处理
│   │
│   ├── server/                  # Web API 服务
│   │   ├── __init__.py
│   │   ├── main_server.py      # FastAPI 主服务
│   │   ├── api.py              # API 路由定义
│   │   ├── models.py           # Pydantic 数据模型
│   │   ├── dependencies.py     # 依赖注入
│   │   ├── orch_jobs.py        # 编排任务管理
│   │   └── task_scheduler.py   # 任务调度器
│   │
│   ├── supervision/            # 监控集成
│   │   ├── __init__.py
│   │   └── langsmith_client.py # LangSmith 追踪
│   │
│   └── utils/                   # 工具函数
│       ├── __init__.py
│       ├── debug_ndjson.py      # NDJSON 调试工具
│       ├── markdown_cleaner.py # Markdown 清洗
│       ├── summarizer.py       # 内容摘要
│       └── token_tracker.py    # Token 用量追踪
│
├── pages/                        # Web UI 页面
│   ├── index.html              # 首页/仪表盘
│   ├── agents.html             # 智能体管理
│   ├── tasks.html              # 任务管理
│   ├── sessions.html           # 会话历史
│   ├── kb.html                 # 知识库管理
│   ├── orchestrate.html         # 任务编排
│   ├── settings.html           # 系统设置
│   └── costs.html              # 成本统计
│
├── tests/                        # 测试目录
│   ├── test_agent.py
│   ├── test_llm_client.py
│   ├── test_memory.py
│   └── test_tools.py
│
├── data/                         # 数据存储（运行时生成）
│   ├── chroma_db/              # ChromaDB 向量数据库
│   ├── checkpointer/           # SQLite 检查点
│   └── documents/              # 文档存储
│
├── run.py                        # Web 服务入口
├── .env.example                  # 环境变量示例
├── requirements.txt            # Python 依赖
└── README.md                   # 项目文档
```

---

## 核心模块说明

### 1. LLM 模块 (`src/llm/`)

支持多 Provider 模型调用（DeepSeek / Claude / OpenAI / Gemini），核心功能：
- 初始化各 Provider 的 Chat 模型
- 构建系统 Prompt
- 流式响应处理

| 文件 | Provider | 说明 |
|------|----------|------|
| `deepseek_client.py` | DeepSeek | OpenAI 兼容端点（V3 / R1） |
| `claude_client.py` | Anthropic | Claude 3.5 Sonnet / Haiku / Opus |
| `openai_client.py` | OpenAI | GPT-4o / GPT-4o Mini / GPT-4 Turbo |
| `google_client.py` | Google | Gemini 2.0 Flash / 1.5 Flash / 1.5 Pro |

> **动态模型切换**：模型字符串格式为 `provider/model`（如 `claude/claude-3-5-sonnet-20241022`），后端根据前缀自动路由到对应 LLM 客户端并缓存 Agent 实例。

### 2. 工具模块 (`src/tools/`)

Agent 可调用的工具集：

| 工具 | 功能 | 依赖 |
|------|------|------|
| `browser_tools` | 网页搜索和内容抓取 | Tavily API |
| `calc_tools` | 数学计算 | 无 |
| `memory_tools` | 记忆存储与检索 | ChromaDB |
| `multimodal_tools` | 图片理解 | DeepSeek 多模态 |

### 3. Graph 模块 (`src/graph/`)

LangGraph 状态机核心：

- **state.py** — 定义 AgentState，包含消息历史、工具状态等
- **nodes.py** — 定义各处理节点（LLM 调用、工具执行、记忆存储等）
- **router.py** — 根据状态决定下一步流向
- **prompt.py** — 动态 Prompt 构建，支持上下文注入
- **agent_graph.py** — 将节点和边组装成完整图
- **orchestrator.py** — 多任务编排，协调多个 Worker
- **self_healer.py** — 自愈修复机制，自动检测并修复错误
- **conflict_resolver.py** — 冲突解决器，处理并发任务冲突

### 4. Worker 系统 (`src/graph/workers/`)

专用任务处理器：

| Worker | 功能 |
|--------|------|
| `coder_worker` | 代码编写、调试和优化任务 |
| `rag_worker` | RAG 检索和生成任务 |
| `search_worker` | 网页搜索和信息提取任务 |

### 5. 多智能体系统 (`src/multi_agent/`)

- **orchestrator.py** — 多智能体协调器，支持并行/串行任务执行

### 6. 记忆模块 (`src/memory/`)

两层记忆系统：

- **ChromaDB (长期记忆)** — 向量数据库，存储用户偏好、历史对话摘要
- **SQLite (会话状态)** — Checkpointer，支持断电恢复和会话回溯

### 7. 中间件模块 (`src/middleware/`)

- **input_guard.py** — 输入格式验证、危险内容拦截
- **pii_redactor.py** — 自动脱敏姓名、身份证、电话、邮箱等 PII 信息

### 8. 服务端模块 (`src/server/`)

FastAPI Web 服务：

| 文件 | 功能 |
|------|------|
| `main_server.py` | FastAPI 应用入口，静态文件服务 |
| `api.py` | RESTful API 路由定义 |
| `models.py` | Pydantic 请求/响应模型 |
| `dependencies.py` | 依赖注入工具 |
| `orch_jobs.py` | 编排任务状态管理 |
| `task_scheduler.py` | 定时任务调度器 |

### 9. 监控模块 (`src/supervision/`)

LangSmith 集成，实时追踪：
- Agent 执行链路
- 各节点耗时
- Token 消耗统计
- 工具调用记录

### 10. 工具函数 (`src/utils/`)

| 工具 | 功能 |
|------|------|
| `token_tracker.py` | Token 使用量追踪和成本报告 |
| `summarizer.py` | 长文本摘要压缩 |
| `markdown_cleaner.py` | 网页 Markdown 清洗 |
| `debug_ndjson.py` | NDJSON 日志调试工具 |

---

## Web UI 界面

启动 `python run.py` 后访问 `http://localhost:8000`

### 页面说明

| 页面 | 路径 | 功能 |
|------|------|------|
| 首页仪表盘 | `/` | 系统概览、快速入口、模型切换下拉框 |
| 智能体管理 | `/agents` | 查看和管理 AI 智能体 |
| 任务管理 | `/tasks` | 创建、监控、终止任务 |
| 会话历史 | `/sessions` | 查看历史对话记录 |
| 知识库 | `/kb` | 文档管理和 RAG 配置 |
| 任务编排 | `/orchestrate` | 可视化编排多步骤任务 |
| 系统设置 | `/settings` | API 密钥管理、模型参数配置 |
| 成本统计 | `/costs` | Token 使用量和成本报告 |

### 多模型支持与运行时切换

系统支持 DeepSeek（默认）、Claude（Anthropic）、OpenAI（GPT-4o）、Gemini（Google）四大模型族。在 **设置页面** 可：

1. **查看 Provider 状态** — 每个 Provider 显示绿色圆点（已配置 API Key）或灰色圆点（未配置）
2. **配置 API Key** — 在对应输入框中填入 Key，保存后立即生效（无需重启服务）
3. **切换当前模型** — 工具栏下拉框动态显示所有已配置 Provider 的模型，选择后即时切换

> **模型标识格式**：`provider/model`，例如 `claude/claude-3-5-sonnet-20241022`、`deepseek-chat`（DeepSeek 可省略前缀）。Key 仅存储在服务器内存中，服务重启后会恢复为 `.env` 中的值。

### 功能截图

> 以下截图展示系统各核心页面功能。

#### 首页仪表盘
![首页仪表盘](Demophoto/对话页面.png)
系统概览、模型切换下拉框、快捷操作入口

#### 任务编排 - 多步骤编排
![任务编排](Demophoto/多agent编排页面.png)
可视化编排多步骤任务流程

#### 编排页面 - 任务流程配置
![编排页面](Demophoto/多agent编排结果页面.png)
可视化编排多步骤任务流程，支持卡片式节点配置

#### 任务管理 - 任务创建与监控
![任务管理](Demophoto/任务调度中心.png)
创建、监控、终止任务，支持实时状态更新

#### 会话历史 - 多会话管理
![会话历史](Demophoto/上传的文档内容.png)
查看和管理历史对话记录，支持多会话并行

#### 设置页面 - 多模型配置
![设置页面](Demophoto/设置.png)
配置 DeepSeek、Claude、OpenAI、Gemini 等多模型 API Key

#### 设置页面 - 工具配置
![工具配置](Demophoto/设置.png)
配置 Tavily 搜索、Playwright 浏览器、ChromaDB 记忆等工具

#### 记忆系统 - ChromaDB 向量存储
![记忆系统](Demophoto/任务协调页面.png)
ChromaDB 长期记忆存储，支持向量检索和语义相似度匹配

#### 记忆系统 - SQLite 状态持久化
![状态持久化](Demophoto/任务协调页面.png)
SQLite Checkpointer 支持断电恢复和会话回溯

#### 成本统计 - Token 使用追踪
![成本统计](Demophoto/成本报表页面.png)
Token 使用量和成本报告统计

#### 知识库管理
![知识库](Demophoto/知识库页面.png)
文档管理和 RAG 配置

#### Agent 组织架构
![Agent组织架构](Demophoto/agent组织架构页面.png)
查看和管理 AI 智能体，支持多模型切换

#### 任务协调中心
![任务协调](Demophoto/任务协调页面.png)
多任务协调与状态同步

#### 对话调用工具
![对话调用工具](Demophoto/对话调用工具页面.png)
实时工具调用展示与执行结果

---

## API 接口

服务启动后可访问 `http://localhost:8000/docs` 查看完整的 Swagger API 文档。

### 主要接口

| 方法 | 路径 | 功能 |
|------|------|------|
| POST | `/api/chat` | 发送对话消息 |
| POST | `/api/model/switch` | 动态切换 LLM 模型（支持 `provider/model` 格式） |
| GET | `/api/config` | 获取当前配置（含可用模型列表，按 Provider 分组） |
| POST | `/api/config/keys` | 更新运行时 API Key（立即生效） |
| POST | `/api/config/model` | 更新模型参数（Temperature、MaxTokens） |
| POST | `/api/orchestrate` | 创建编排任务 |
| GET | `/api/tasks` | 获取任务列表 |
| GET | `/api/tasks/{id}` | 获取任务详情 |
| DELETE | `/api/tasks/{id}` | 终止任务 |
| GET | `/api/sessions` | 获取会话列表 |
| GET | `/api/memory` | 获取记忆内容 |
| POST | `/api/documents` | 上传文档 |
| GET | `/api/stats` | 获取统计数据 |

---

## 常见问题

### Q1: 提示 "DEEPSEEK_API_KEY is not set"

**原因**：`.env` 文件中未配置 API Key，或环境变量未加载。

**解决**：
1. 确认 `.env` 文件存在于项目根目录
2. 确认文件内容格式正确：`DEEPSEEK_API_KEY=sk-xxxxxx`
3. 重启终端后再次运行

### Q2: 提示 "ModuleNotFoundError: No module named 'xxxx'"

**原因**：依赖未安装，或环境激活错误。

**解决**：
```bash
# 确认在正确的 conda 环境中
conda activate langchain

# 重新安装依赖
pip install -r requirements.txt
```

### Q3: Playwright 安装失败

**解决**：
```bash
# 使用国内镜像
playwright install chromium --with-deps

# 或手动下载 Chromium
playwright install chromium
```

### Q4: Python 版本不对

**检查**：
```bash
python --version
# 必须输出 Python 3.11.x

# 如果版本不对，切换环境
conda activate langchain
```

### Q5: LangSmith 追踪不生效

**原因**：`LANGSMITH_API_KEY` 未配置。

**解决**：在 `.env` 中添加：
```env
LANGSMITH_API_KEY=your-key-here
```

### Q6: 端口被占用 / 数据库锁定

默认 Web 端口为 **8000**（可在 `.env` 中设置 `APP_PORT`）。若占用，请结束旧 `python`/`uvicorn` 进程或改端口。

**解决**：
```bash
# 删除锁文件
del /f data\checkpointer\checkpoints.db.lock 2>nul

# 或删除数据库重新开始
del /f data\checkpointer\checkpoints.db
del /f data\chroma_db\*.sqlite 2>nul /s
```

### Q7: 如何切换到其他模型（如 Claude / GPT-4o / Gemini）？

**方法一（推荐 — 运行时配置）**：
1. 进入 **设置页面** → **API Keys & Models** 区块
2. 在对应 Provider 的输入框中填入 API Key，点击「保存设置」
3. 圆点变为绿色后，在工具栏模型下拉框中选择目标模型即可

**方法二（永久配置）**：
直接在 `.env` 文件中添加对应 Key，然后重启服务。

> 支持的模型列表见 [LLM 模块说明](#1-llm-模块-srcllm)。

### Q8: 任务执行失败或卡住

**解决**：
1. 在任务管理页面查看错误详情
2. 检查 API Key 配额是否充足
3. 查看日志文件定位问题
4. 使用自愈机制：`/api/tasks/{id}/retry`

---

## 开发计划

| 阶段 | 状态 | 内容 |
|------|------|------|
| Phase 1 | ✅ 已完成 | DeepSeek 调用 + ReAct Agent 跑通 |
| Phase 2 | ✅ 已完成 | ChromaDB 记忆 + SQLite 持久化 + 上下文裁剪 |
| Phase 3 | ✅ 已完成 | 高级 LangGraph 编排 + 多模态 + LangSmith 监控 |
| Phase 4 | ✅ 已完成 | 多智能体系统 + Web UI + 任务编排 |
| Phase 5 | ✅ 已完成 | 多 Agent 系统优化（消息总线 + DAG 依赖 + 能力路由） |
| Phase 6 | ✅ 已完成 | 线程池架构 + 并发优化 + 前端编排增强 |
| Phase 7 | ✅ 已完成 | 自愈机制增强 + 错误恢复 |
| Phase 8 | ✅ 已完成 | 长期记忆（Episodic Memory）+ 可观测性 + Token 追踪 |
| Phase 9 | ✅ 已完成 | 代码清理 + MCP 协议集成框架 |
| Phase 10 | 🚧 进行中 | 前端增强 + 性能优化 |

### Phase 5 更新内容（2026-04-01）

本次更新实现了多 Agent 系统的三大核心优化：

#### 1. AgentMessageBus 消息总线 ✅

**功能说明**：构建声明式消息传递层，支持 Agent 间结构化通信。

**核心能力**：
- `store_result / get_context / mark_done` — 结果存储与上下文注入
- `add_dependency / _has_cycle` — 依赖管理与循环检测
- DAG 拓扑排序执行 — 按依赖关系智能调度任务

**代码位置**：`src/multi_agent/orchestrator.py`

#### 2. DAG 任务规划 ✅

**功能说明**：支持有向无环图依赖声明，实现复杂的任务执行顺序。

**核心能力**：
- `depends_on` — 声明任务依赖关系
- `execution_mode` — 指定并行/顺序执行模式
- 环形依赖自动检测与报错
- 拓扑排序调度引擎

**代码位置**：`src/graph/orchestrator.py`、`src/graph/prompt.py`

#### 3. Capability-Based Routing ✅

**功能说明**：基于能力声明的智能路由，替代硬编码的 worker_kind 匹配。

**核心能力**：
- `capabilities` 字段 — Agent 声明自身能力标签
- Supervisor 能力匹配 — 根据任务需求匹配最合适的 Agent
- capability overlap 算法 — 最优/次优匹配策略

**代码位置**：`src/server/dependencies.py`、`src/graph/prompt.py`

### Phase 6 更新内容（2026-04-02）

本次更新实现了线程池架构和并发优化，解决多页面并发执行任务时的阻塞问题：

#### 1. 线程池隔离架构 ✅

**问题背景**：页面中执行任务时，其他页面功能无法使用。

**根本原因**：
- 单一事件循环阻塞 — 所有 LLM 调用共同竞争同一事件循环
- 后台任务缺乏隔离 — BackgroundJob 是进程内单例
- ThreadPoolExecutor 未使用 — 存在死代码

**解决方案**：三层隔离架构
- **L1 事件循环层** — FastAPI/Uvicorn 主事件循环
- **L2 线程池层** — LLM 调用、数据库操作、向量检索各自独立线程池
- **L3 任务队列层** — 任务入队、按优先级调度

#### 2. 前端编排页面增强 ✅

**功能优化**：
- 实时步骤展示 — 执行过程可视化
- 多任务并行显示 — 支持多个任务同时展示
- SSE 流式更新 — 实时状态同步
- 错误恢复机制 — 自修复和重试支持

**代码位置**：`pages/orchestrate.html`

#### 3. 服务端 API 增强 ✅

**新增功能**：
- 任务状态流式推送
- 后台任务管理增强
- 并发执行控制

**代码位置**：`src/server/api.py`、`src/server/orch_jobs.py`

### Phase 7 更新内容（2026-04-03）

本次更新增强了自愈机制和错误恢复能力：

#### 1. 自愈机制增强 ✅

**功能优化**：
- LLM 驱动的自我修复循环 — 分析错误原因并提出修复方案
- 多轮重试策略 — 最多 2 次自修复尝试
- 修复结果验证 — 评估修复后结果的实际质量
- 人工介入机制 — 自修复失败时触发人工介入提示

**核心能力**：
- `_llm_self_heal()` — 调用 LLM 进行自我修复
- `_evaluate_healed_quality()` — 评估修复后质量
- `_is_repair_valid()` — 验证修复结果有效性
- 上下文感知 — 结合之前成功执行的结果辅助修复

**代码位置**：`src/graph/self_healer.py`

#### 2. 自修复工作流 ✅

```
任务执行失败
    ↓
捕获错误日志
    ↓
自修复循环（最多2次）
    ├─ 尝试 #1: 调用 LLM 分析并修复
    │   ├─ 修复成功 → 评估质量 → 返回结果
    │   └─ 修复失败 → 进入尝试 #2
    └─ 尝试 #2: 重新分析 + 修复
        ├─ 修复成功 → 评估质量 → 返回结果
        └─ 修复失败 → 触发人工介入
```

#### 3. 编排器优化 ✅

**新增功能**：
- 自修复任务集成 — 失败任务自动进入自修复流程
- 上下文传递 — 修复时携带之前成功的结果
- 进度回调 — 实时反馈自修复状态

**代码位置**：`src/graph/orchestrator.py`、`src/graph/prompt.py`

### Phase 10 更新内容（2026-04-08）

本次更新实现了三大核心功能：线程池隔离、跨会话长期记忆（Episodic Memory）、以及异步任务执行架构。

#### 1. 线程池隔离架构 ✅

**问题背景**：FastAPI 异步事件循环与 LLM 同步调用之间的竞争问题。

**解决方案**：三层隔离架构
- **L1 事件循环层** — FastAPI/Uvicorn 主事件循环处理 HTTP 请求
- **L2 线程池层** — LLM 调用、数据库操作、向量检索各自独立线程池
- **L3 任务队列层** — 任务入队、按优先级调度

**新增文件**：
- `src/utils/thread_pool_manager.py` — 全局线程池生命周期管理器
  - `llm_pool`：LLM 推理调用（网络 I/O + JSON 解析）
  - `db_pool`：SQLite / 文件 I/O 操作
  - `vector_pool`：ChromaDB / embedding 操作

#### 2. 异步任务执行器 ✅

**新增文件**：`src/server/chat_task_executor.py`

**核心能力**：
- `ChatTaskExecutor` — 在 LLM 线程池中运行 agent.astream_events()
- 支持任务提交（`submit`）、结果查询（`get_result`）、任务取消（`cancel`）
- SSE 流式推送 — 前端实时显示工具调用和模型响应
- 事件去重 — 避免重复的工具开始/结束标记

**新增文件**：`src/server/chat_result_store.py`

**核心能力**：
- 内存中的 asyncio.Lock 保护结果存储
- TTL 自动清理（1小时后自动删除已完成/失败的任务）
- 线程安全的 chunk 追加（使用 call_soon_threadsafe）

#### 3. 跨会话长期记忆（Episodic Memory） ✅

**新增文件**：`src/memory/episode_store.py`

**四层记忆架构**：
| 层级 | 名称 | 生命周期 | 技术选型 |
|------|------|----------|----------|
| L1 | Session Memory | 单次编排 | 当前 `all_results` 内存 |
| L2 | Working Memory | 单次编排，Agent 间共享 | `AgentMessageBus._results` |
| L3 | Episodic Memory | 跨会话，事件序 | SQLite `agent_episodes` 表 |
| L4 | Semantic Memory | 跨会话，向量检索 | ChromaDB（已有） |

**核心能力**：
- `save()` — 保存编排 episode（含任务清单、执行结果、质量评分、耗时）
- `get_recent()` — 获取最近的 N 条 episode
- `query_similar()` — 模糊匹配相似的历史 episode（关键词分词 + LIKE 匹配）
- `get_stats()` — 全局 episode 统计（平均质量分、平均耗时、通过率）
- `get_agent_usage_ranking()` — 跨 episode 的 agent 使用频率排名

**数据库表**：`agent_episodes`
```sql
CREATE TABLE agent_episodes (
    id TEXT PRIMARY KEY,
    job_id TEXT,
    requirement TEXT,
    tasks JSON,
    results JSON,
    quality_score REAL,
    duration_ms INTEGER,
    agents_used JSON,
    max_depth INTEGER,
    healing_attempts INTEGER,
    passed INTEGER,
    created_at TEXT
);
```

### Phase 9 更新内容（2026-04-08）

本次更新完成了代码清理和 MCP 协议集成的初始框架。

#### 1. 代码清理 ✅

**移除内容**：
- 调试日志代码（`_dlog` 函数调用）
- `src/utils/debug_ndjson.py` 文件（已从 git 中删除）
- 散布在各模块中的临时调试日志

**涉及文件**：
- `src/graph/orchestrator.py` — 移除多层级递归调试日志
- `src/server/orch_jobs.py` — 移除错误追踪日志

#### 2. MCP 协议集成框架 ✅

**新增目录**：`src/mcp/`

**核心文件**：
- `__init__.py` — MCP 模块初始化
- `server.py` — MCP Server 实现，支持 SSE 传输协议
- `registry.py` — 工具注册中心，管理工具元数据
- `tools.py` — 内置工具定义

**功能说明**：
- 工具动态发现：`GET /mcp/tools` 返回可用工具列表
- 工具按需绑定：根据 task capability 动态加载工具集
- 跨框架互操作：其他 MCP 客户端可直接调用 Crayfish 工具

**代码位置**：`src/mcp/`

### Phase 10 远期规划

| 模块 | 说明 |
|------|------|
| 多级编排（递归 Supervisor） | 总任务 → 子任务 → 孙任务，最多 3 层 |
| 可观测性（LangSmith） | 分布式追踪 + Token 成本追踪 |
| MCP 协议集成 | 工具动态发现与按需绑定 |

---

*文档最后更新：2026-04-08*
