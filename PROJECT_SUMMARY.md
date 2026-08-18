# AI Agent — 项目简要说明

## 项目概述

**AI Agent** 是一个基于 **LangGraph Plan-and-Execute** 架构的桌面端 AI 智能体应用，使用 **PyQt6** 构建图形界面。支持通过 LLM（远程 API 或本地 Ollama）驱动的多步骤任务规划与执行，内置丰富的工具生态，并具备短期/长期记忆能力。

- **版本**: 6.0.0
- **Python 要求**: ≥ 3.11
- **许可证**: MIT
- **入口**: `app_qt.py`（PyQt6 桌面 GUI）

---

## 核心架构

项目采用分层架构设计，共 6 层：

### 1. 编排层（Orchestration）
- 基于 **LangGraph StateGraph** 实现 Plan-and-Execute 工作流
- 双路径路由：简单查询走快速 ReAct 路径，复杂任务走完整规划-执行-验证流程
- 支持最多 3 轮重新规划
- 核心文件：`agent/graph_orchestrator.py`、`agent/agent.py`、`agent/runner.py`

### 2. LLM 层
- 支持两种 LLM 提供者：OpenAI 兼容 API（远程）和 Ollama（本地离线）
- 通过 `.env` 环境变量切换
- 核心文件：`agent/llm/factory.py`、`agent/llm/openai_compatible.py`、`agent/llm/ollama_llm.py`

### 3. 记忆层（Memory）
- **短期记忆**：内存滑动窗口（默认 6 轮），重启丢失
- **长期记忆**：SQLite 持久化，支持关键词搜索
- 核心文件：`agent/memory/memory_manager.py`、`agent/memory/short_term.py`、`agent/memory/long_term.py`

### 4. 工具层（Tools）
- 共 **14+ 个内置工具**，通过装饰器注册
- 涵盖：文件操作、网页搜索/抓取、代码执行（沙箱）、浏览器自动化（Playwright）、数学计算、笔记管理、系统信息等
- 核心文件：`agent/tools/registry.py` 及各工具模块

### 5. UI 层（PyQt6）
- 聊天界面 + 侧边栏（模型信息/会话统计/快捷操作）
- 事件驱动：Agent 通过事件总线向 UI 推送进度事件
- HTML 渲染聊天消息、工具调用、日志
- 核心文件：`app_qt.py`、`agent/ui/worker.py`、`agent/ui/chat_html.py`

### 6. 基础设施层（Utils）
- LRU + TTL 缓存、指数退避重试、查询复杂度分类、路径安全守卫、令牌桶限流、JSON 提取等
- 核心文件：`agent/utils/` 目录下各模块

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
│   ├── agent.py           # AgentSession 会话管理
│   ├── graph_orchestrator.py  # LangGraph 编排器
│   ├── runner.py          # ReAct 执行器
│   ├── config.py          # 统一配置（pydantic-settings）
│   ├── prompts.py         # 提示词模板
│   ├── i18n.py            # 国际化
│   ├── models.py          # 数据模型
│   ├── events.py          # 事件类型
│   ├── progress.py        # 进度事件工厂
│   ├── llm/               # LLM 提供者（base/factory/openai/ollama）
│   ├── memory/            # 记忆系统（短期/长期/管理器）
│   ├── tools/             # 工具生态（14+ 工具）
│   ├── ui/                # PyQt6 UI 组件（worker/chat_html/styles/event_handler）
│   └── utils/             # 基础设施（cache/retry/classifier/path_guard 等）
├── tests/                 # 测试（cache/config/json_extractor/retry）
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

## 设计亮点

1. **Plan-and-Execute 架构**：复杂任务先规划再执行，支持验证和重新规划
2. **双路径路由**：简单查询走快速 ReAct 路径，复杂任务走完整流程
3. **安全沙箱**：代码执行通过 AST 白名单 + 子进程隔离
4. **路径守卫**：所有文件操作通过 `safe_resolve()` 防止路径穿越
5. **LLM 缓存**：LRU + TTL 缓存减少重复 API 调用
6. **超时重试**：指数退避重试 + 用户取消支持
7. **双 LLM 支持**：远程 API 和本地 Ollama 一键切换
8. **事件驱动 UI**：Agent 通过事件总线实时推送进度，UI 非阻塞渲染

---

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env，设置 LLM_PROVIDER 和 API 密钥

# 3. 启动 GUI
python app_qt.py
```
