# LangChain RAG 记忆智能体

> 基于 LangGraph + ChromaDB + SQLite + DeepSeek，实现具备主动记忆 Upsert 的 AI 智能体。

---

## 目录

- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [完整安装详解](#完整安装详解)
- [配置说明](#配置说明)
- [运行项目](#运行项目)
- [项目结构](#项目结构)
- [核心模块说明](#核心模块说明)
- [使用说明](#使用说明)
- [常见问题](#常见问题)

---

## 环境要求

| 要求 | 版本 | 说明 |
|------|------|------|
| **Python** | **3.11+** | 必须，LangGraph/LangChain 生态要求 |
| conda | 任意版本 | 用于创建和管理 Python 环境 |
| DeepSeek API Key | 有效密钥 | 用于 LLM 对话能力 |
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

# 7. 运行
python src/main.py
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
5. 格式类似：`sk-e2bc13304a6249cf956a369a21c18b2c`

> ⚠️ 请勿将真实 API Key 上传至公开仓库！`.env` 文件已被 `.gitignore` 忽略。

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
| `TAVILY_API_KEY` | ❌ | 空 | 网页搜索（可选） |
| `LANGSMITH_API_KEY` | ❌ | 空 | 链路追踪（可选） |
| `DEFAULT_MODEL` | ❌ | `deepseek-chat` | 模型名称 |
| `TEMPERATURE` | ❌ | `0.7` | 生成温度（0-1） |
| `MAX_TOKENS` | ❌ | `8000` | 最大输出 Token 数 |
| `CHROMA_PATH` | ❌ | `./data/chroma_db` | ChromaDB 存储路径 |
| `CHECKPOINT_PATH` | ❌ | `./data/checkpointer/checkpoints.db` | SQLite 状态持久化路径 |

---

## 运行项目

### 基本运行

```bash
python src/main.py
```

### 带 LangSmith 追踪运行

```bash
# 确保 .env 中配置了 LANGSMITH_API_KEY
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
│   ├── main.py                  # 主入口脚本
│   │
│   ├── llm/                     # LLM 客户端
│   │   ├── __init__.py
│   │   └── deepseek_client.py   # DeepSeek API 调用封装
│   │
│   ├── tools/                   # 工具定义
│   │   ├── __init__.py
│   │   ├── browser_tools.py     # 浏览器搜索工具（Tavily）
│   │   ├── calc_tools.py        # 计算器工具
│   │   ├── memory_tools.py      # 记忆存储/检索工具
│   │   └── multimodal_tools.py  # 多模态处理工具
│   │
│   ├── graph/                   # LangGraph 状态机
│   │   ├── __init__.py
│   │   ├── agent_graph.py       # Agent 图构建
│   │   ├── nodes.py             # 图节点定义
│   │   ├── prompt.py            # Prompt 模板
│   │   ├── router.py            # 路由逻辑
│   │   └── state.py             # 状态定义
│   │
│   ├── memory/                  # 记忆系统
│   │   ├── __init__.py
│   │   ├── chroma_store.py      # ChromaDB 长期记忆
│   │   ├── memory_schema.py     # 记忆数据模型
│   │   └── sqlite_store.py      # SQLite 状态持久化
│   │
│   ├── middleware/              # 中间件
│   │   ├── __init__.py
│   │   ├── input_guard.py       # 输入验证
│   │   └── pii_redactor.py       # PII 脱敏处理
│   │
│   ├── supervision/             # 监控集成
│   │   ├── __init__.py
│   │   └── langsmith_client.py  # LangSmith 追踪
│   │
│   └── utils/                    # 工具函数
│       ├── __init__.py
│       ├── markdown_cleaner.py  # Markdown 清洗
│       ├── summarizer.py         # 内容摘要
│       └── token_tracker.py     # Token 用量追踪
│
├── tests/                        # 测试目录
│   ├── test_memory.py
│   ├── test_tools.py
│   └── test_graph.py
│
├── data/                         # 数据存储（运行时生成）
│   ├── chroma_db/               # ChromaDB 向量数据库
│   └── checkpointer/            # SQLite 检查点
│
├── .env.example                  # 环境变量示例
├── requirements.txt             # Python 依赖
└── README.md                    # 项目文档
```

---

## 核心模块说明

### 1. LLM 模块 (`src/llm/`)

DeepSeek API 的调用封装，支持流式输出。核心功能：
- 初始化 DeepSeek Chat 模型
- 构建系统 Prompt
- 流式响应处理

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

### 4. 记忆模块 (`src/memory/`)

两层记忆系统：

- **ChromaDB (长期记忆)** — 向量数据库，存储用户偏好、历史对话摘要
- **SQLite (会话状态)** — Checkpointer，支持断电恢复和会话回溯

### 5. 中间件模块 (`src/middleware/`)

- **input_guard.py** — 输入格式验证、危险内容拦截
- **pii_redactor.py** — 自动脱敏姓名、身份证、电话、邮箱等 PII 信息

### 6. 监控模块 (`src/supervision/`)

LangSmith 集成，实时追踪：
- Agent 执行链路
- 各节点耗时
- Token 消耗统计
- 工具调用记录

### 7. 工具函数 (`src/utils/`)

- **token_tracker.py** — Token 使用量追踪和成本报告
- **summarizer.py** — 长文本摘要压缩
- **markdown_cleaner.py** — 网页 Markdown 清洗

---

## 使用说明

### 交互命令

启动后，在终端输入：

| 命令 | 功能 |
|------|------|
| 直接输入问题 | 与 Agent 对话 |
| `quit` / `exit` | 退出并打印 Token 报告 |
| `reset` | 重置当前会话（新建 thread_id） |
| `cost` | 打印 Token 使用量报告 |
| `memory` | 查看所有长期记忆条目 |

### 对话示例

```
╔══════════════════════════════════════════════╗
║   LangGraph RAG Agent — DeepSeek + ChromaDB  ║
║   用户: 周暄                                  ║
║   输入 'quit' 或 'exit' 退出                   ║
║   输入 'reset' 重置当前会话                    ║
║   输入 'cost' 查看 Token 成本报告              ║
║   输入 'memory' 查看所有长期记忆               ║
╚══════════════════════════════════════════════╝

Assistant: 你好！有什么我可以帮助你的吗？

You: 你好，我叫周暄，我喜欢用 Rust 写高性能代码
Assistant: 你好周暄！我记住你了。很高兴认识一位 Rust 开发者！

You: memory
--- 长期记忆 (1 条) ---
  [user_preference] 用户名为周暄，擅长 Rust 编程...

You: quit
Token 使用报告:
  prompt_tokens:  3200
  completion_tokens: 850
  总花费估算: $0.0023
再见！
```

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
which python  # 确认路径
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

---

## 分阶段开发计划  OVO ZhouXuan

| 阶段 | 时间 | 内容 |
|------|------|------|
| Time 1-2 | 已完成 | DeepSeek 调用 + ReAct Agent 跑通 |
| Time 3-4 | 已完成 | ChromaDB 记忆 + SQLite 持久化 + 上下文裁剪 |
| Time 5-6 | 已完成 | 高级 LangGraph 编排 + 多模态 + LangSmith 监控 |

---

*文档最后更新：2026-03-27*
