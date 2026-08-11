# AI Agent — Plan-and-Execute 智能体

基于 AutoGen 多智能体协作 + LangChain 原生工具调用的实用型 AI Agent。
支持双模式 LLM（远端 API / 本地 Ollama 一键切换）、短期对话记忆 + SQLite 长期记忆、丰富工具生态（浏览器自动化、代码搜索/分析、多引擎搜索等），提供 PyQt6 现代桌面 GUI、Tkinter 桌面 GUI 和 FastAPI REST API 三种交互方式。

---

## 特性

- **双路径推理**：简单查询自动路由到快速 ReAct 路径，复杂多步骤任务走 AutoGen GroupChat Plan-and-Execute
- **多智能体协作**：Planner → Executor → Verifier → UserProxy 四智能体确定性状态机协作
- **双 LLM 模式**：远端 API（兼容 OpenAI 的任何 API）/ 本地 Ollama，`.env` 一键切换
- **短期 + 长期记忆**：滑动窗口短期记忆 + SQLite 持久化长期记忆（跨会话、可搜索）
- **丰富工具生态**：浏览器自动化（Playwright）、代码搜索（grep/glob/outline）、代码分析（lint/typecheck）、多引擎搜索、文件读写、笔记管理等
- **三种交互方式**：PyQt6 GUI、Tkinter GUI、FastAPI REST API（支持 SSE 流式响应）
- **Docker 部署**：生产级 Dockerfile + docker-compose.yml（含可选 Ollama 服务）

---

## 架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                    交互层 (UI Layer)                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │  app_qt.py   │  │  app_tk.py   │  │  agent/api.py (FastAPI)   │  │
│  │  PyQt6 GUI   │  │  Tkinter GUI │  │  REST + SSE Streaming    │  │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────────┘  │
├─────────┴─────────────────┴─────────────────────┴─────────────────┤
│                    Agent 核心层                                    │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  agent/agent.py — AgentSession                              │   │
│  │  ┌─────────────────────────────────────────────────────┐    │   │
│  │  │  查询分类器 (LLM-based classifier)                   │    │   │
│  │  │  simple ──► Executor (ReAct 快速路径)                │    │   │
│  │  │  complex ─► AutoGenOrchestrator (Plan-and-Execute)  │    │   │
│  │  └─────────────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────────┤
│                    Orchestrator 层 (AutoGen)                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐       │
│  │ Planner  │─►│ Executor │─►│ Verifier │─►│ UserProxy    │       │
│  │ (规划)   │  │ (执行)   │  │ (验证)   │  │ (工具执行)   │       │
│  └──────────┘  └────┬─────┘  └──────────┘  └──────────────┘       │
│       ▲              │ ▲            │ ▲                              │
│       └──────────────┘ └────────────┘                               │
│              确定性状态机 speaker 选择                                │
├─────────────────────────────────────────────────────────────────────┤
│                    基础设施层                                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐       │
│  │  llm/    │  │ memory/  │  │  tools/  │  │   utils/     │       │
│  │ 工厂模式 │  │ 短期+SQLite│  │ 注册表模式│  │ 缓存/重试等 │       │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────┘       │
└─────────────────────────────────────────────────────────────────────┘
```

### 数据流（单次对话）

```
用户输入 "读取 README.md，分析项目结构，然后写一份简要的项目说明"
  │
  ▼
AgentSession.invoke(user_input, chat_history)
  │
  ├── 1. LLM 分类器判定: "complex" → Plan-and-Execute 路径
  │
  ├── 2. AutoGenOrchestrator.run()
  │     ├── Planner 制定执行计划 (3 步)
  │     ├── Executor 逐步调用工具
  │     │     ├── tool: file_reader("README.md")
  │     │     ├── tool: code_analysis("agent/agent.py")
  │     │     └── tool: file_writer("summary.md", ...)
  │     └── Verifier 验证完成 → TERMINATE
  │
  ├── 3. Memory 写入 (短期 + 长期)
  │
  ▼
UI 层接收 progress events → 流式渲染工具调用过程 + 最终回答
```

---

## 项目结构

```
PythonProject/
├── app_qt.py                      # PyQt6 桌面 GUI 入口（推荐）
├── app_tk.py                      # Tkinter 桌面 GUI 入口
├── Dockerfile                     # 生产级 Docker 镜像
├── docker-compose.yml             # Docker Compose 编排（含可选 Ollama）
├── pyproject.toml                 # 项目元数据 + 依赖 + 工具配置
├── requirements.txt               # 精简依赖列表
├── .env                           # 运行时配置（不提交 git）
├── .env.example                   # 配置模板
├── .gitignore
│
├── agent/                         # 核心包
│   ├── __init__.py
│   ├── agent.py                   # AgentSession 主编排入口（双路径路由）
│   ├── api.py                     # FastAPI REST API 服务
│   ├── config.py                  # 统一配置（pydantic-settings + .env）
│   ├── executor.py                # ReAct 快速路径执行器
│   ├── i18n.py                    # 国际化字符串
│   ├── orchestrator.py            # AutoGen Plan-and-Execute 编排器
│   ├── progress.py                # 进度事件协议 + ProgressTracker
│   ├── prompts.py                 # Agent 提示词模板
│   ├── runner.py                  # 共享工具调用执行逻辑
│   │
│   ├── llm/                       # LLM 对话模型层
│   │   ├── base.py                #   抽象接口 BaseLLMProvider
│   │   ├── factory.py             #   create_llm() 工厂函数
│   │   ├── ollama_llm.py          #   本地 Ollama 实现
│   │   └── openai_compatible.py   #   远端 API 兼容实现
│   │
│   ├── memory/                    # 记忆层
│   │   ├── short_term.py          #   短期滑动窗口（内存）
│   │   ├── long_term.py           #   长期 SQLite 持久化
│   │   └── memory_manager.py      #   统一记忆管理器
│   │
│   ├── tools/                     # 工具生态
│   │   ├── registry.py            #   @register 装饰器注册表
│   │   ├── calculator.py          #   安全数学计算器
│   │   ├── web_search.py          #   多引擎搜索（百度/必应/搜狗/360/Google）
│   │   ├── web_fetcher.py         #   网页抓取
│   │   ├── web_content_fetcher.py #   增强版网页内容抓取
│   │   ├── file_reader.py         #   读取本地文件
│   │   ├── file_writer.py         #   写入/编辑文件
│   │   ├── file_opener.py         #   系统默认方式打开文件
│   │   ├── code_executor.py       #   Python 代码沙盒执行
│   │   ├── code_search.py         #   代码搜索（grep/glob/outline）
│   │   ├── code_analysis.py       #   代码静态分析（lint/typecheck）
│   │   ├── datetime_tool.py       #   日期时间查询
│   │   ├── browser.py             #   浏览器自动化（Playwright）
│   │   └── note_manager.py        #   持久笔记管理
│   │
│   └── utils/                     # 通用工具
│       ├── retry.py               #   LLM 调用重试保护
│       ├── cache.py               #   工具响应缓存
│       ├── json_extractor.py      #   JSON 提取器
│       ├── path_guard.py          #   路径安全守护
│       └── http_fetcher.py        #   HTTP 请求封装
│
├── agent_notes/                   # 笔记存储目录
├── agent_workspace/               # 代码执行沙盒目录
├── agent_long_term.db             # 长期记忆 SQLite 数据库（运行时生成）
└── tests/                         # 单元测试
    ├── test_cache.py
    ├── test_config.py
    ├── test_json_extractor.py
    └── test_retry.py
```

---

## 安装

### 1. 环境要求

- Python >= 3.11
- pip（或 uv 包管理器）

### 2. 克隆与安装

```bash
# 克隆项目
git clone <repo-url>
cd PythonProject

# 方式 A：pip 安装
pip install -r requirements.txt

# 方式 B：使用 uv（推荐，更快）
uv sync --no-dev
```

### 3. 配置

```bash
cp .env.example .env
# 编辑 .env，填入 API Key 或切换为 Ollama
```

### 4. 启动

```bash
# PyQt6 桌面 GUI（推荐）
python app_qt.py

# Tkinter 桌面 GUI（轻量替代）
python app_tk.py

# FastAPI REST API 服务
python -m agent.api
# 或
uvicorn agent.api:app --host 0.0.0.0 --port 8080
```

---

## Docker 部署

### 仅部署 Agent API

```bash
docker compose up -d
# API 服务运行在 http://localhost:8080
```

### 包含本地 Ollama（完全离线）

```bash
docker compose --profile ollama up -d
ollama pull qwen2.5
```

### 手动构建

```bash
docker build -t ai-agent .
docker run -p 8080:8080 --env-file .env -v ./agent_notes:/app/agent_notes -v ./agent_workspace:/app/agent_workspace ai-agent
```

---

## 配置说明 (.env)

### LLM —— 对话模型

| 变量 | 说明 | 默认值 |
|---|---|---|
| `LLM_PROVIDER` | `openai`（远端）或 `ollama`（本地） | `openai` |
| `OPENAI_API_KEY` | API 密钥 | 必填 |
| `OPENAI_BASE_URL` | API 地址 | `http://127.0.0.1:13000/v1` |
| `OPENAI_MODEL` | 模型名 | `deepseek-v4-pro` |
| `OPENAI_TEMPERATURE` | 温度（0-1） | `0.1` |
| `OLLAMA_BASE_URL` | Ollama 地址 | `http://localhost:11434` |
| `OLLAMA_MODEL` | 本地模型名 | `qwen2.5` |
| `OLLAMA_TEMPERATURE` | 温度 | `0.1` |

### Agent —— 行为控制

| 变量 | 说明 | 默认值 |
|---|---|---|
| `MAX_ITERATIONS` | 最大工具调用轮次 | `99` |
| `MAX_EXECUTION_TIME_SEC` | 执行超时（秒） | `120` |
| `MAX_SHORT_TERM_ROUNDS` | 短期记忆保留轮数 | `6` |
| `MAX_RETRIES` | LLM 调用重试次数 | `1` |

### Tools —— 工具配置

| 变量 | 说明 | 默认值 |
|---|---|---|
| `FILE_READER_ROOT` | 文件操作沙盒根目录 | `.` |
| `NOTES_DIR` | 笔记存储目录 | `./agent_notes` |
| `CODE_WORKDIR` | 代码执行沙盒目录 | `./agent_workspace` |
| `BROWSER_HEADLESS` | 浏览器无头模式 | `false` |

### Server —— 服务

（桌面应用无需服务端配置；API 模式默认监听 `0.0.0.0:8080`）

---

## 切换 LLM 模式

### 远端 API → 本地 Ollama

```ini
# .env
LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen2.5
```

确保本地 Ollama 已拉取对应模型：

```bash
ollama pull qwen2.5
```

重启应用即生效。

---

## 可用工具

### 文件操作

| 工具 | 用途 | 示例 |
|---|---|---|
| `file_reader` | 读取本地文本文件 | `读一下 README.md` |
| `file_writer` | 创建或覆写文件 | `把代码保存为 main.py` |
| `file_editor` | 替换首次匹配的内容 | `把 config.py 里的端口改成 8080` |
| `file_editor_all` | 替换所有匹配的内容 | `把所有 old_path 替换为 new_path` |
| `file_editor_exact` | 精确字符串替换 | `把第一个函数的参数改一下` |
| `file_opener` | 用系统默认应用打开文件 | `打开桌面上的 report.pdf` |

### 代码分析

| 工具 | 用途 | 示例 |
|---|---|---|
| `grep` | 内容搜索（正则） | `搜索代码中的 TODO` |
| `glob` | 文件名匹配 | `找所有 .py 文件` |
| `code_outline` | 提取文件结构大纲 | `看看 agent.py 的结构` |
| `code_lint` | Python AST 静态检查 | `检查 main.py 有没有语法错误` |
| `code_typecheck` | 类型标注检查 | `检查 agent.py 的类型注解` |
| `code_executor` | Python 代码沙盒执行 | `计算 1+2+...+100` |

### 浏览器自动化

| 工具 | 用途 |
|---|---|
| `browser_navigate` | 导航到 URL |
| `browser_snapshot` | 读取页面文本内容（无障碍树） |
| `browser_click` | 点击元素 |
| `browser_type` | 输入文本 |
| `browser_press_key` | 按键（Enter/Escape 等） |
| `browser_screenshot` | 截图保存 |
| `browser_list_interactive` | 列出所有可交互元素 |
| `browser_close` | 关闭浏览器 |

### 搜索与抓取

| 工具 | 用途 | 示例 |
|---|---|---|
| `web_search` | 多引擎搜索（百度/必应/搜狗/360/Google） | `搜索 2026 AI 趋势` |
| `web_fetcher` | 抓取网页正文 | `抓取 https://example.com` |
| `web_content_fetcher` | 增强版网页抓取（标题/meta/链接/层级） | `抓取这篇新闻的关键信息` |

### 其他

| 工具 | 用途 | 示例 |
|---|---|---|
| `math_calculator` | 安全数学表达式求值 | `计算 128*36+2^8` |
| `datetime_query` | 日期/时间查询 | `今天周几` / `30天后几号` |
| `note_save` | 保存笔记 | `记下来：会议要点...` |
| `note_read` | 读取笔记 | `查一下之前的会议要点` |
| `note_list` | 列出所有笔记 | `我记过哪些笔记` |

---

## API 服务

启动 API 服务后（`python -m agent.api`），提供以下端点：

| 方法 | 端点 | 说明 |
|---|---|---|
| `GET` | `/health` | 健康检查（返回模型信息） |
| `POST` | `/chat` | 发送消息，获取非流式响应 |
| `POST` | `/chat/stream` | 发送消息，SSE 流式响应 |
| `POST` | `/chat/cancel` | 取消当前运行中的请求 |
| `GET` | `/memory/stats` | 记忆统计 |
| `DELETE` | `/memory/long-term` | 清空长期记忆 |
| `DELETE` | `/memory/short-term` | 清空短期记忆和对话历史 |

### 请求/响应示例

```bash
# 非流式
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "你好，帮我计算 2^10"}'

# SSE 流式
curl -N -X POST http://localhost:8080/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "读取 README.md 并总结"}'
```

---

## 扩展指南

### 添加新工具

1. 新建 `agent/tools/my_tool.py`：

```python
from langchain.tools import tool
from agent.tools.registry import register

@register
@tool
def my_tool(param: str) -> str:
    """工具描述（会出现在提示词中）。

    Args:
        param: 参数说明
    """
    return f"结果：{param}"
```

2. 在 `agent/tools/__init__.py` 加一行：

```python
from agent.tools import my_tool  # noqa: F401
```

3. 重启即可。Agent 自动获得新工具。

### 添加新的 LLM 提供商

1. 新建 `agent/llm/your_provider.py`，继承 `BaseLLMProvider` 实现
2. 在 `agent/llm/factory.py` 中添加对应分支
3. 在 `agent/config.py` 中添加相关配置字段
4. 在 `.env.example` 中添加对应配置项

### 添加新语言 (i18n)

1. 在 `agent/i18n.py` 中添加新的语言字典（如 `_EN`）
2. 通过 `LANG` 环境变量或 `I18N.locale` 切换

---

## 开发

### 运行测试

```bash
pytest -v
```

### 代码风格

```bash
# 检查
ruff check .

# 自动修复
ruff check . --fix
ruff format .

# 类型检查
mypy agent/
```

---

## 技术栈

| 组件 | 技术 |
|---|---|
| 语言 | Python 3.11+ |
| LLM 框架 | LangChain 0.3+ |
| 多智能体协作 | AutoGen (AG2) 0.14+ |
| 本地 LLM | Ollama |
| 远端 LLM | OpenAI Compatible API |
| GUI | PyQt6 / Tkinter |
| Web 框架 | FastAPI + Uvicorn |
| 浏览器自动化 | Playwright |
| 配置管理 | pydantic-settings + python-dotenv |
| 长期记忆 | SQLite |
| 包管理 | uv / pip |
| 容器化 | Docker + Docker Compose |

---

## License

MIT#   P y t h o n P r o j e c t  
 #   a u r o r a  
 #   a u r o r a  
 