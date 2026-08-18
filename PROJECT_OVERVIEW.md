# AI Agent — 项目说明

## 项目简介

**AI Agent** 是一个基于 **LangGraph Plan-and-Execute** 架构的桌面端 AI 智能体应用，使用 **PyQt6** 构建图形界面。它支持通过 LLM（远程 API 或本地 Ollama）驱动的多步骤任务规划与执行，内置丰富的工具生态（文件操作、网页搜索、代码执行、浏览器自动化等），并具备短期/长期记忆能力。

- **版本**: 6.0.0
- **Python 要求**: ≥ 3.11
- **许可证**: MIT
- **入口**: `app_qt.py`（PyQt6 桌面 GUI）

---

## 核心架构

### 1. 编排层（Orchestration）

采用 **LangGraph StateGraph** 实现 Plan-and-Execute 工作流：

```
classify → simple → react_fast → END
classify → complex → plan → execute_step → check_steps → verify → END / re-plan
```

- **classify**: 判断用户查询是简单（单步）还是复杂（多步）
- **react_fast**: 简单查询直接走 ReAct 工具调用快速路径
- **plan**: LLM 生成 2-5 步执行计划
- **execute_step**: 每步通过 LangChain ReAct 工具调用执行
- **verify**: 验证目标是否达成，未达成则触发 re-plan（最多 3 轮）

核心文件：
- `agent/graph_orchestrator.py` — LangGraph 编排器（~25KB）
- `agent/agent.py` — AgentSession 会话管理
- `agent/runner.py` — 共享 ReAct 执行器（工具调用、流式输出、取消支持）

### 2. LLM 层

支持两种 LLM 提供者，通过 `.env` 切换：

| 提供者 | 说明 | 默认模型 |
|--------|------|----------|
| `openai` | OpenAI 兼容 API（DeepSeek、GPT 等） | deepseek-v4-pro |
| `ollama` | 本地 Ollama 服务（完全离线） | qwen2.5 |

核心文件：
- `agent/llm/factory.py` — LLM 工厂函数
- `agent/llm/base.py` — 抽象接口
- `agent/llm/openai_compatible.py` — OpenAI 兼容实现
- `agent/llm/ollama_llm.py` — Ollama 本地实现

### 3. 记忆层（Memory）

| 类型 | 存储 | 说明 |
|------|------|------|
| 短期记忆 | 内存滑动窗口 | 最近 N 轮对话（默认 6 轮），重启丢失 |
| 长期记忆 | SQLite | 跨会话持久化，支持关键词搜索 |

核心文件：
- `agent/memory/memory_manager.py` — 统一记忆管理器
- `agent/memory/short_term.py` — 短期滑动窗口
- `agent/memory/long_term.py` — SQLite 长期存储

### 4. 工具层（Tools）

通过装饰器注册系统管理，共 **14 个内置工具**：

| 工具 | 功能 |
|------|------|
| `file_reader` | 读取文件内容 |
| `file_writer` | 写入文件 |
| `file_opener` | 用系统默认应用打开文件 |
| `web_search` | 多引擎搜索（百度/必应/搜狗/360/Google） |
| `web_fetcher` | 抓取网页正文文本 |
| `web_content_fetcher` | 结构化网页内容提取（标题/正文/链接/图片） |
| `code_executor` | 沙箱执行 Python 代码（AST 安全校验） |
| `code_search` | 代码搜索（grep/glob/outline） |
| `code_analysis` | 代码静态分析（lint/typecheck） |
| `browser_navigate` | Playwright 浏览器导航 |
| `browser_click/type/press_key` | 浏览器交互 |
| `browser_snapshot/screenshot` | 页面快照/截图 |
| `math_calculator` | 安全数学计算（AST 白名单） |
| `datetime_query` | 日期时间查询 |
| `note_save/read/list` | 持久笔记管理 |
| `calculate` | 表达式求值 |
| `http_fetcher` | HTTP 请求 |
| `directory_list` | 目录列表 |
| `system_info` | 系统信息 |

核心文件：
- `agent/tools/registry.py` — 工具注册中心 + 内置工具
- `agent/tools/browser.py` — Playwright 浏览器自动化
- `agent/tools/code_executor.py` — 沙箱代码执行
- `agent/tools/web_search.py` — 多引擎搜索
- `agent/tools/web_content_fetcher.py` — 结构化网页抓取

### 5. UI 层（PyQt6）

- **主窗口**: `app_qt.py` — 聊天界面 + 侧边栏（模型信息/会话统计/快捷操作）
- **事件驱动**: Agent 通过事件总线（EventBus）向 UI 推送进度事件
- **HTML 渲染**: 聊天消息、工具调用、日志等以 HTML 片段渲染
- **样式**: 统一 QSS 样式表 + 颜色调色板

核心文件：
- `app_qt.py` — 主窗口（~25KB）
- `agent/ui/worker.py` — QThread 工作线程
- `agent/ui/chat_html.py` — HTML 渲染
- `agent/ui/styles.py` — 样式定义
- `agent/ui/event_handler.py` — 事件处理

### 6. 工具/基础设施层（Utils）

| 模块 | 功能 |
|------|------|
| `cache.py` | LRU + TTL 缓存（LLM 响应 + 工具结果） |
| `retry.py` | LLM 调用超时 + 指数退避重试 |
| `classifier.py` | 查询复杂度分类（简单/复杂） |
| `path_guard.py` | 路径安全守卫（防路径穿越） |
| `rate_limiter.py` | 令牌桶限流器 |
| `json_extractor.py` | 从 LLM 响应中提取 JSON |
| `http_fetcher.py` | 共享 HTTP 抓取工具 |

---

## 项目目录结构

```
.
├── app_qt.py              # PyQt6 主入口（GUI）
├── pyproject.toml         # 项目配置（依赖/构建/工具）
├── requirements.txt       # Python 依赖
├── Dockerfile             # Docker 构建（多阶段）
├── docker-compose.yml     # Docker Compose（agent + ollama）
├── .env / .env.example    # 环境变量配置
├── agent/                 # 核心包
│   ├── __init__.py        # 包入口
│   ├── agent.py           # AgentSession 会话管理
│   ├── graph_orchestrator.py  # LangGraph 编排器
│   ├── runner.py          # ReAct 执行器
│   ├── config.py          # 统一配置（pydantic-settings）
│   ├── prompts.py         # 提示词模板
│   ├── i18n.py            # 国际化
│   ├── models.py          # 数据模型
│   ├── events.py          # 事件类型
│   ├── progress.py        # 进度事件工厂
│   ├── llm/               # LLM 提供者
│   │   ├── base.py
│   │   ├── factory.py
│   │   ├── openai_compatible.py
│   │   └── ollama_llm.py
│   ├── memory/            # 记忆系统
│   │   ├── memory_manager.py
│   │   ├── short_term.py
│   │   └── long_term.py
│   ├── tools/             # 工具生态
│   │   ├── registry.py    # 注册中心 + 内置工具
│   │   ├── browser.py     # Playwright 浏览器
│   │   ├── code_executor.py
│   │   ├── code_search.py
│   │   ├── code_analysis.py
│   │   ├── web_search.py
│   │   ├── web_fetcher.py
│   │   ├── web_content_fetcher.py
│   │   ├── calculator.py
│   │   ├── datetime_tool.py
│   │   ├── note_manager.py
│   │   └── ...
│   ├── ui/                # PyQt6 UI 组件
│   │   ├── worker.py
│   │   ├── chat_html.py
│   │   ├── styles.py
│   │   └── event_handler.py
│   └── utils/             # 基础设施
│       ├── cache.py
│       ├── retry.py
│       ├── classifier.py
│       ├── path_guard.py
│       ├── rate_limiter.py
│       ├── json_extractor.py
│       └── http_fetcher.py
├── tests/                 # 测试
│   ├── test_cache.py
│   ├── test_config.py
│   ├── test_json_extractor.py
│   └── test_retry.py
├── agent_notes/           # 笔记持久化目录
├── agent_workspace/       # 代码执行工作目录
└── agent_long_term.db     # SQLite 长期记忆数据库
```

---

## 技术栈

| 类别 | 技术 |
|------|------|
| 编排 | LangGraph (StateGraph) |
| Agent | LangChain (tool-calling agent) |
| LLM | OpenAI 兼容 API / Ollama |
| GUI | PyQt6 |
| 浏览器 | Playwright |
| 搜索 | 百度/必应/搜狗/360/Google |
| 配置 | pydantic-settings + .env |
| 持久化 | SQLite |
| 容器化 | Docker / Docker Compose |
| 测试 | pytest |
| 代码质量 | Ruff + Mypy |

---

## 快速开始

### 本地运行

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env，设置 LLM_PROVIDER 和 API 密钥

# 3. 启动 GUI
python app_qt.py
```

### Docker 运行

```bash
# 构建并启动
docker compose up --build

# 可选：启动本地 Ollama（完全离线）
docker compose --profile ollama up --build
```

---

## 设计亮点

1. **Plan-and-Execute 架构**: 复杂任务先规划再执行，支持验证和重新规划（最多 3 轮）
2. **双路径路由**: 简单查询走快速 ReAct 路径，复杂任务走完整 Plan-and-Execute 流程
3. **安全沙箱**: 代码执行通过 AST 白名单 + 子进程隔离，防止恶意代码
4. **路径守卫**: 所有文件操作通过 `safe_resolve()` 防止路径穿越
5. **LLM 缓存**: LRU + TTL 缓存减少重复 API 调用
6. **超时重试**: 指数退避重试 + 用户取消支持
7. **双 LLM 支持**: 远程 API 和本地 Ollama 一键切换
8. **事件驱动 UI**: Agent 通过事件总线实时推送进度，UI 非阻塞渲染
