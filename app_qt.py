"""AI Agent — Modern Desktop GUI (PyQt6).

Thin UI layer: all business logic, rendering, and state management
live in agent/ modules. This file only handles:
  - Widget creation & layout
  - Signal routing between UI events and AgentSession
  - Displaying rendered HTML from the chat_html / markdown_renderer modules
"""

from __future__ import annotations

import html
import logging
import sys
import threading
from typing import Any

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QAction, QCursor, QFont, QIcon, QKeySequence, QPainter, QPixmap
from PyQt6.QtWidgets import (
    QApplication, QFrame, QHBoxLayout, QLabel, QMainWindow,
    QMenu, QMessageBox, QPushButton, QScrollArea, QSizePolicy,
    QSplitter, QTextEdit, QVBoxLayout, QWidget, QTextBrowser,
)

from agent.agent import AgentSession
from agent.config import settings
from agent.events import AgentEvent, ToolEvent
from agent.models import AgentResult, AgentStatus
from agent.ui.chat_html import (
    assistant_header_html,
    cancelled_html,
    error_html,
    event_to_html,
    final_answer_html,
    status_for_log_message,
    thinking_html,
    tool_html,
    user_message_html,
)
from agent.ui.markdown_renderer import render_markdown
from agent.ui.styles import COLORS, QSS
from agent.ui.worker import AgentWorker
from agent.ui.event_handler import DisplayState

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

QUICK_ACTIONS = [
    ("🧮 计算", "计算 128 * 56 / 8 + 2^10"),
    ("🔍 搜索", "搜索 Python asyncio 的用法，给我一个简单例子"),
    ("📖 读文件", "读取当前目录下的 README.md 文件内容并总结要点"),
    ("💻 写代码", "写一个 Python 脚本，输出 1-100 之间的素数并保存到文件"),
    ("📝 笔记", "帮我记一条笔记：我喜欢喝咖啡，然后把所有笔记列出来"),
    ("🌐 抓取", "抓取 https://example.com 的内容并总结关键信息"),
    ("📋 复杂任务", "读取 README.md，分析项目结构，然后写一份简要的项目说明"),
    ("📰 新闻搜索", "用百度搜索今天的 AI 新闻，选 3 条最重要的，统计每条的标题字数"),
]


class AIAgentWindow(QMainWindow):

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("AI Agent — Plan & Execute")
        self.resize(1200, 780)
        self.setMinimumSize(900, 600)

        self._session = AgentSession()
        self._worker: AgentWorker | None = None
        self._display = DisplayState()
        self._history: list[dict] = []
        self._is_running = False
        self._stop_requested = False
        self._streaming_active = False
        self._streaming_buffer = ""
        self._stream_start_pos = 0
        self._answer_rendered = False
        self._thinking_start_pos = 0
        self._thinking_end_pos = 0

        self._setup_ui()
        self._setup_connections()

    def _setup_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)

        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        self._build_header(root_layout)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(1)
        splitter.setStyleSheet(
            f"QSplitter::handle {{ background-color: {COLORS['border']}; }}"
        )

        self._chat_container = QWidget()
        chat_layout = QVBoxLayout(self._chat_container)
        chat_layout.setContentsMargins(12, 8, 6, 4)
        chat_layout.setSpacing(10)
        self._build_chat(chat_layout)

        self._sidebar = QWidget()
        self._sidebar.setObjectName("Sidebar")
        self._sidebar.setFixedWidth(290)

        self._sidebar_scroll = QScrollArea()
        self._sidebar_scroll.setWidgetResizable(True)
        self._sidebar_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._sidebar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._sidebar_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._sidebar_scroll.setStyleSheet(
            f"QScrollArea {{ background-color: transparent; border: none; }}"
        )

        self._sidebar_content = QWidget()
        sidebar_layout = QVBoxLayout(self._sidebar_content)
        sidebar_layout.setContentsMargins(10, 12, 10, 10)
        sidebar_layout.setSpacing(10)
        self._build_sidebar(sidebar_layout)
        self._sidebar_scroll.setWidget(self._sidebar_content)

        outer_layout = QVBoxLayout(self._sidebar)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.addWidget(self._sidebar_scroll)

        splitter.addWidget(self._chat_container)
        splitter.addWidget(self._sidebar)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([880, 320])
        root_layout.addWidget(splitter, 1)

        self._build_statusbar(root_layout)

    def _build_header(self, layout: QVBoxLayout) -> None:
        header = QFrame()
        header.setObjectName("Header")
        header.setFixedHeight(56)

        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(16, 0, 16, 0)
        header_layout.setSpacing(12)

        logo_label = QLabel("🤖")
        logo_label.setFixedSize(36, 36)
        logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        logo_label.setStyleSheet(
            f"background-color: {COLORS['accent']}; border-radius: 8px; font-size: 18px;"
        )

        title_block = QVBoxLayout()
        title_block.setSpacing(0)
        title_block.setContentsMargins(0, 0, 0, 0)

        title_main = QLabel("AI Agent")
        title_main.setStyleSheet(
            f"font-size: 15px; font-weight: 600; color: {COLORS['text']};"
        )

        title_sub = QLabel("Plan · ReAct · Reflection · Verification")
        title_sub.setStyleSheet(f"font-size: 10px; color: {COLORS['muted']};")

        title_block.addWidget(title_main)
        title_block.addWidget(title_sub)

        header_layout.addWidget(logo_label)
        header_layout.addLayout(title_block)
        header_layout.addStretch()

        llm_config = settings.get_llm_config()
        model_label = QLabel(f"{llm_config['provider'].upper()} · {llm_config['model']}")
        model_label.setStyleSheet(
            f"font-size: 12px; color: {COLORS['text_secondary']};"
        )
        header_layout.addWidget(model_label)
        layout.addWidget(header)

        accent = QFrame()
        accent.setObjectName("AccentBar")
        accent.setFixedHeight(3)
        layout.addWidget(accent)

    def _build_chat(self, layout: QVBoxLayout) -> None:
        container = QFrame()
        container.setObjectName("ChatContainer")

        inner = QVBoxLayout(container)
        inner.setContentsMargins(0, 0, 0, 0)
        inner.setSpacing(0)

        self._welcome = QWidget()
        welcome_layout = QVBoxLayout(self._welcome)
        welcome_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        welcome_layout.setContentsMargins(40, 40, 40, 40)
        welcome_layout.setSpacing(12)

        emoji_label = QLabel("🧠")
        emoji_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        emoji_label.setStyleSheet("font-size: 56px;")
        welcome_layout.addWidget(emoji_label)

        title = QLabel("Plan-and-Execute Agent")
        title.setObjectName("WelcomeTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        welcome_layout.addWidget(title)

        sub = QLabel("我能够理解目标、规划步骤、调用工具、反思调整、验证闭环。\n输入你的任务开始对话吧！")
        sub.setObjectName("WelcomeSub")
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        welcome_layout.addWidget(sub)

        chips_row = QHBoxLayout()
        chips_row.setSpacing(8)
        chips_row.setAlignment(Qt.AlignmentFlag.AlignCenter)
        for ct in [
            "计算 128 * 56 / 8 + 2^10",
            "搜索 Python asyncio 用法",
            "读取 README.md 并总结",
        ]:
            chip = QPushButton(ct)
            chip.setObjectName("Chip")
            chip.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
            chip.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            chip.clicked.connect(lambda _=False, t=ct: self._submit_message(t))
            chips_row.addWidget(chip)
        welcome_layout.addLayout(chips_row)
        inner.addWidget(self._welcome, 1)

        self._chat_text = QTextBrowser()
        self._chat_text.setObjectName("ChatText")
        self._chat_text.setOpenExternalLinks(True)
        self._chat_text.setOpenLinks(True)
        self._chat_text.setLineWrapMode(QTextBrowser.LineWrapMode.WidgetWidth)
        self._chat_text.setReadOnly(True)
        self._chat_text.setVisible(False)
        self._chat_text.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._chat_text.customContextMenuRequested.connect(self._show_context_menu)
        inner.addWidget(self._chat_text, 1)

        layout.addWidget(container, 1)
        self._build_input_area(layout)

    def _build_input_area(self, layout: QVBoxLayout) -> None:
        input_container = QFrame()
        input_container.setObjectName("InputContainer")

        input_layout = QHBoxLayout(input_container)
        input_layout.setContentsMargins(12, 10, 10, 10)
        input_layout.setSpacing(8)

        self._input_box = QTextEdit()
        self._input_box.setObjectName("InputBox")
        self._input_box.setPlaceholderText("输入你的问题，Enter 发送，Shift+Enter 换行...")
        self._input_box.setMinimumHeight(44)
        self._input_box.setMaximumHeight(150)
        self._input_box.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self._input_box.setAcceptRichText(False)
        self._input_box.textChanged.connect(self._adjust_input_height)
        input_layout.addWidget(self._input_box, 1)

        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(6)

        self._send_btn = QPushButton("Send")
        self._send_btn.setObjectName("SendButton")
        self._send_btn.setFixedSize(80, 36)
        self._send_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        btn_layout.addWidget(self._send_btn)

        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setObjectName("StopButton")
        self._stop_btn.setFixedSize(80, 36)
        self._stop_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self._stop_btn.setEnabled(False)
        btn_layout.addWidget(self._stop_btn)

        self._clear_btn = QPushButton("Clear")
        self._clear_btn.setObjectName("ClearButton")
        self._clear_btn.setFixedSize(80, 36)
        self._clear_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        btn_layout.addWidget(self._clear_btn)

        input_layout.addLayout(btn_layout)
        layout.addWidget(input_container)

    def _build_sidebar(self, layout: QVBoxLayout) -> None:
        card1 = self._create_card("🤖 模型信息")
        card1_layout = card1.layout()
        llm_config = settings.get_llm_config()
        rows = [
            ("服务商", llm_config["provider"].upper(), True),
            ("模型", llm_config["model"], False),
            ("温度", str(llm_config["temperature"]), False),
            ("模式", "Plan-and-Execute", True),
        ]
        for label_text, value_text, highlight in rows:
            row = QHBoxLayout()
            row.setContentsMargins(12, 4, 12, 4)
            lbl = QLabel(label_text)
            lbl.setStyleSheet(f"font-size: 12px; color: {COLORS['muted']};")
            val = QLabel(value_text)
            color = COLORS["accent"] if highlight else COLORS["text"]
            val.setStyleSheet(f"font-size: 12px; color: {color}; font-weight: 500;")
            val.setAlignment(Qt.AlignmentFlag.AlignRight)
            row.addWidget(lbl)
            row.addStretch()
            row.addWidget(val)
            card1_layout.addLayout(row)
        layout.addWidget(card1)

        card2 = self._create_card("📊 会话统计")
        card2_layout = card2.layout()
        stats_grid = QHBoxLayout()
        stats_grid.setContentsMargins(12, 8, 12, 8)
        stats_grid.setSpacing(8)
        self._stat_cells: dict[str, tuple[QLabel, QLabel]] = {}
        stats_data = [
            ("耗时", "0.0s", COLORS["accent"]),
            ("工具", "0", COLORS["text"]),
            ("步骤", "0", COLORS["text"]),
            ("状态", "●", COLORS["green"]),
        ]
        for i in range(0, len(stats_data), 2):
            col = QHBoxLayout()
            col.setSpacing(6)
            for j in range(2):
                name, val, color = stats_data[i + j]
                cell = QFrame()
                cell.setStyleSheet(
                    f"QFrame {{ background-color: {COLORS['surface2']}; border-radius: 8px; }}"
                )
                cell_layout = QVBoxLayout(cell)
                cell_layout.setContentsMargins(8, 8, 8, 8)
                cell_layout.setSpacing(2)
                val_label = QLabel(val)
                val_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                val_label.setStyleSheet(f"font-size: 18px; font-weight: 700; color: {color};")
                name_label = QLabel(name)
                name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                name_label.setStyleSheet(f"font-size: 10px; color: {COLORS['muted']};")
                cell_layout.addWidget(val_label)
                cell_layout.addWidget(name_label)
                col.addWidget(cell, 1)
                self._stat_cells[name] = (val_label, name_label)
            stats_grid.addLayout(col, 1)
        card2_layout.addLayout(stats_grid)
        layout.addWidget(card2)

        card3 = self._create_card("⚡ 快捷操作")
        card3_layout = card3.layout()
        btn_container = QVBoxLayout()
        btn_container.setContentsMargins(8, 4, 8, 8)
        btn_container.setSpacing(4)
        for label_text, prompt in QUICK_ACTIONS:
            btn = QPushButton(label_text)
            btn.setObjectName("QuickButton")
            btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
            btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            btn.clicked.connect(lambda _=False, p=prompt: self._run_quick_action(p))
            btn_container.addWidget(btn)
        card3_layout.addLayout(btn_container)
        layout.addWidget(card3, 1)

    def _create_card(self, title: str) -> QFrame:
        card = QFrame()
        card.setObjectName("Card")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(0, 0, 0, 0)
        card_layout.setSpacing(0)

        title_label = QLabel(title)
        title_label.setObjectName("CardTitle")
        title_label.setContentsMargins(12, 8, 12, 4)
        title_label.setStyleSheet(
            f"font-size: 12px; font-weight: 600; color: {COLORS['text_secondary']};"
        )
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet(f"background-color: {COLORS['border']}; max-height: 1px;")
        card_layout.addWidget(title_label)
        card_layout.addWidget(sep)
        return card

    def _build_statusbar(self, layout: QVBoxLayout) -> None:
        status = QFrame()
        status.setObjectName("StatusBar")
        status.setFixedHeight(32)
        status_layout = QHBoxLayout(status)
        status_layout.setContentsMargins(14, 0, 14, 0)

        self._status_dot = QLabel("●")
        self._status_dot.setObjectName("StatusDot")
        self._status_dot.setStyleSheet(f"color: {COLORS['green']}; font-size: 10px;")
        self._status_label = QLabel("就绪")
        self._status_label.setObjectName("StatusText")
        right_label = QLabel("Powered by ReAct Framework")
        right_label.setStyleSheet(f"font-size: 10px; color: {COLORS['muted']};")

        status_layout.addWidget(self._status_dot)
        status_layout.addWidget(self._status_label)
        status_layout.addStretch()
        status_layout.addWidget(right_label)
        layout.addWidget(status)

    def _setup_connections(self) -> None:
        self._send_btn.clicked.connect(self._on_submit)
        self._stop_btn.clicked.connect(self._on_stop)
        self._clear_btn.clicked.connect(self._on_clear)
        self._input_box.installEventFilter(self)

    def eventFilter(self, obj, event):
        if obj is self._input_box:
            if event.type() == event.Type.KeyPress:
                key = event.key()
                modifiers = event.modifiers()
                if key == Qt.Key.Key_Return and not (modifiers & Qt.KeyboardModifier.ShiftModifier):
                    self._on_submit()
                    return True
                elif key == Qt.Key.Key_Escape:
                    if self._is_running:
                        self._on_stop()
                    return True
        return super().eventFilter(obj, event)

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def _show_chat(self) -> None:
        if self._welcome.isVisible():
            self._welcome.setVisible(False)
            self._chat_text.setVisible(True)

    def _show_thinking_indicator(self) -> None:
        cursor = self._chat_text.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self._thinking_start_pos = cursor.position()
        self._chat_text.setTextCursor(cursor)
        self._chat_text.insertHtml(thinking_html())
        cursor = self._chat_text.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self._thinking_end_pos = cursor.position()
        sb = self._chat_text.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _remove_thinking_indicator(self) -> None:
        if self._thinking_start_pos <= 0 or self._thinking_end_pos <= 0:
            self._thinking_start_pos = 0
            self._thinking_end_pos = 0
            return
        cursor = self._chat_text.textCursor()
        cursor.setPosition(self._thinking_start_pos)
        cursor.setPosition(self._thinking_end_pos, cursor.MoveMode.KeepAnchor)
        cursor.removeSelectedText()
        self._thinking_start_pos = 0
        self._thinking_end_pos = 0

    def _start_streaming(self) -> None:
        self._streaming_active = True
        self._streaming_buffer = ""
        self._answer_rendered = False
        self._remove_thinking_indicator()
        cursor = self._chat_text.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self._stream_start_pos = cursor.position()
        self._chat_text.setTextCursor(cursor)
        self._chat_text.insertHtml(
            f'<div style="margin: 12px 0; text-align: left;">'
            f'<div style="display: inline-block; background-color: {COLORS["assistant_bubble"]};'
            f'border: 1px solid {COLORS["border"]}; border-radius: 12px;'
            f'padding: 14px 18px; line-height: 1.7; max-width: 85%; text-align: left;">'
            f'<div class="md-body" style="color: {COLORS["text"]}; font-size: 14px;">'
        )
        cursor = self._chat_text.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self._chat_text.setTextCursor(cursor)

    def _append_streaming_token(self, token: str) -> None:
        self._streaming_buffer += token
        escaped = html.escape(token)
        cursor = self._chat_text.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self._chat_text.setTextCursor(cursor)
        self._chat_text.insertHtml(escaped)

    def _end_streaming(self) -> None:
        if not self._streaming_active:
            return
        self._streaming_active = False
        cursor = self._chat_text.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self._chat_text.setTextCursor(cursor)
        self._chat_text.insertHtml('</div></div></div>')
        sb = self._chat_text.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _append_streaming_metadata(self) -> None:
        elapsed = self._display.elapsed
        tc = self._display.tool_count
        sec = f"{elapsed:.1f}s" if elapsed < 60 else f"{int(elapsed // 60)}m{int(elapsed % 60)}s"
        cursor = self._chat_text.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self._chat_text.setTextCursor(cursor)
        self._chat_text.insertHtml(
            f'<div style="margin-top: 6px; display: flex; gap: 16px; '
            f'font-size: 11px; color: {COLORS["muted"]};">'
            f'<span>⏱ {sec}</span>'
            f'<span>🔧 {tc} 次工具调用</span>'
            f'</div>'
        )
        sb = self._chat_text.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _append_html(self, html_str: str) -> None:
        if self._streaming_active:
            self._end_streaming()
        self._chat_text.append(html_str)
        sb = self._chat_text.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _set_status(self, text: str, running: bool = False) -> None:
        self._status_label.setText(text)
        if running:
            self._status_dot.setText("◉")
            self._status_dot.setStyleSheet(f"color: {COLORS['yellow']}; font-size: 10px;")
        else:
            self._status_dot.setText("●")
            self._status_dot.setStyleSheet(f"color: {COLORS['green']}; font-size: 10px;")

    def _update_stats(self) -> None:
        elapsed = self._display.elapsed
        sec = f"{elapsed:.1f}s" if elapsed < 60 else f"{int(elapsed // 60)}m{int(elapsed % 60)}s"
        self._set_stat("耗时", sec, COLORS["accent"])
        self._set_stat("工具", str(self._display.tool_count), COLORS["text"])
        self._set_stat("步骤", str(self._display.step_count), COLORS["text"])
        self._set_stat(
            "状态", "◉" if self._is_running else "●",
            COLORS["yellow"] if self._is_running else COLORS["green"],
        )

    def _set_stat(self, name: str, value: str, color: str) -> None:
        if name in self._stat_cells:
            val_label, _ = self._stat_cells[name]
            val_label.setText(value)
            val_label.setStyleSheet(f"font-size: 18px; font-weight: 700; color: {color};")

    def _reset_stats(self) -> None:
        self._display.reset()
        self._set_stat("耗时", "0.0s", COLORS["accent"])
        self._set_stat("工具", "0", COLORS["text"])
        self._set_stat("步骤", "0", COLORS["text"])
        self._set_stat("状态", "●", COLORS["green"])

    def _adjust_input_height(self) -> None:
        doc = self._input_box.document()
        h = int(doc.size().height()) + self._input_box.frameWidth() * 2 + 4
        h = max(44, min(h, 150))
        self._input_box.setFixedHeight(h)

    def _show_context_menu(self, pos) -> None:
        cursor = self._chat_text.textCursor()
        cursor.setPosition(self._chat_text.textCursor().position())
        has_selection = cursor.hasSelection()

        menu = QMenu(self)

        copy_action = QAction("复制", self)
        copy_action.setEnabled(has_selection)
        copy_action.setShortcut(QKeySequence.StandardKey.Copy)
        copy_action.triggered.connect(lambda: self._chat_text.copy())
        menu.addAction(copy_action)

        select_all_action = QAction("全选", self)
        select_all_action.setShortcut(QKeySequence.StandardKey.SelectAll)
        select_all_action.triggered.connect(lambda: self._chat_text.selectAll())
        menu.addAction(select_all_action)

        menu.addSeparator()

        copy_all_action = QAction("复制全部内容", self)
        copy_all_action.triggered.connect(self._copy_all_content)
        menu.addAction(copy_all_action)

        menu.addSeparator()

        clear_action = QAction("清空对话", self)
        clear_action.triggered.connect(self._on_clear)
        menu.addAction(clear_action)

        menu.exec(self._chat_text.mapToGlobal(pos))

    def _copy_all_content(self) -> None:
        all_text = self._chat_text.toPlainText()
        if all_text:
            QApplication.clipboard().setText(all_text)

    # ------------------------------------------------------------------
    # Event → rendering dispatch (uses typed AgentEvent directly)
    # ------------------------------------------------------------------

    def _replace_streaming_with_answer(self, answer: str) -> None:
        rendered = render_markdown(answer)
        html_str = (
            f'<div style="margin: 12px 0; text-align: left;">'
            f'<div style="display: inline-block; background-color: {COLORS["assistant_bubble"]};'
            f'border: 1px solid {COLORS["border"]}; border-radius: 12px;'
            f'padding: 14px 18px; line-height: 1.7; max-width: 85%; text-align: left;">'
            f'<div class="md-body" style="color: {COLORS["text"]}; font-size: 14px; line-height: 1.7;">'
            f'{rendered}'
            f'</div></div></div>'
        )
        cursor = self._chat_text.textCursor()
        cursor.setPosition(self._stream_start_pos)
        cursor.movePosition(cursor.MoveOperation.End, cursor.MoveMode.KeepAnchor)
        cursor.removeSelectedText()
        cursor.insertHtml(html_str)
        self._answer_rendered = True
        self._streaming_buffer = ""
        sb = self._chat_text.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _render_event(self, event: AgentEvent) -> None:
        from agent.ui.event_handler import handle_event

        event_dict = event if isinstance(event, dict) else {}
        event_type = event_dict.get("type", "")

        if event_type == "streaming_token":
            token = event_dict.get("token", "")
            if not self._streaming_active:
                self._remove_thinking_indicator()
                self._start_streaming()
            self._append_streaming_token(token)
            handle_event(self._display, event_dict)
            return

        if event_type in ("tool", "log", "done", "error"):
            self._remove_thinking_indicator()

        had_streaming = self._streaming_active
        if had_streaming and event_type not in ("streaming_token",):
            self._end_streaming()

        if event_type == "done":
            handle_event(self._display, event_dict)
            answer = event_dict.get("answer", "")
            if had_streaming or self._streaming_buffer:
                self._replace_streaming_with_answer(answer)
            else:
                rendered = render_markdown(answer)
                bubble = (
                    f'<div style="margin: 12px 0; text-align: left;">'
                    f'<div style="display: inline-block; background-color: {COLORS["assistant_bubble"]};'
                    f'border: 1px solid {COLORS["border"]}; border-radius: 12px;'
                    f'padding: 14px 18px; line-height: 1.7; max-width: 85%; text-align: left;">'
                    f'<div class="md-body" style="color: {COLORS["text"]}; font-size: 14px; line-height: 1.7;">'
                    f'{rendered}'
                    f'</div></div></div>'
                )
                self._append_html(bubble)
                self._answer_rendered = True
            return

        handle_event(self._display, event_dict)
        html_fragment = event_to_html(event_dict)
        if html_fragment:
            self._append_html(html_fragment)

    def _render_final(self, answer: str, elapsed: float, tc: int) -> None:
        self._append_html(final_answer_html(answer, elapsed, tc))

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _on_submit(self) -> None:
        if self._is_running:
            return
        message = self._input_box.toPlainText().strip()
        if not message:
            return
        self._input_box.clear()
        self._submit_message(message)

    def _submit_message(self, message: str) -> None:
        self._show_chat()
        self._append_html(user_message_html(message))
        self._history.append({"role": "user", "content": message})
        self._start_agent(message)

    def _start_agent(self, message: str) -> None:
        self._is_running = True
        self._stop_requested = False
        self._streaming_active = False
        self._streaming_buffer = ""
        self._stream_start_pos = 0
        self._answer_rendered = False
        self._display.reset()
        self._reset_stats()

        self._send_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._input_box.setEnabled(False)
        self._set_status("思考中...", running=True)

        self._worker = AgentWorker(self._session, message, self._history)
        self._worker.event_bus.subscribe(self._render_event)
        self._worker.result_signal.connect(self._on_result)
        self._worker.finished_signal.connect(self._on_finished)
        self._worker.error_signal.connect(self._on_error)
        self._worker.start()
        self._show_thinking_indicator()

    def _on_result(self, result: AgentResult) -> None:
        if result.status == AgentStatus.CANCELLED:
            self._append_html(cancelled_html())
        elif result.status == AgentStatus.FAILED:
            self._append_html(error_html(f"  ❌ {result.content}"))
        elif result.status == AgentStatus.PARTIAL:
            self._render_final(result.content, result.elapsed, len(result.tool_calls))
        else:
            pass

    def _on_finished(self, answer: str) -> None:
        self._is_running = False
        self._remove_thinking_indicator()

        if self._stop_requested:
            self._stop_requested = False
            self._send_btn.setEnabled(True)
            self._stop_btn.setEnabled(False)
            self._input_box.setEnabled(True)
            self._input_box.setFocus()
            self._set_status("就绪")
            self._update_stats()
            if self._worker:
                self._worker.deleteLater()
                self._worker = None
            return

        answer = answer or "*(no response)*"

        if self._answer_rendered:
            self._append_streaming_metadata()
        else:
            self._render_final(answer, self._display.elapsed, self._display.tool_count)

        self._answer_rendered = False
        self._history.append({"role": "assistant", "content": answer})

        self._send_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._input_box.setEnabled(True)
        self._input_box.setFocus()
        self._set_status("就绪")
        self._update_stats()

        if self._worker:
            self._worker.deleteLater()
            self._worker = None

    def _on_error(self, error_msg: str) -> None:
        self._is_running = False
        self._remove_thinking_indicator()

        if self._streaming_active:
            self._end_streaming()

        if self._stop_requested:
            self._stop_requested = False
            self._send_btn.setEnabled(True)
            self._stop_btn.setEnabled(False)
            self._input_box.setEnabled(True)
            self._input_box.setFocus()
            self._set_status("就绪")
            self._update_stats()
            if self._worker:
                self._worker.deleteLater()
                self._worker = None
            return

        self._append_html(error_html(f"  ❌ Error: {error_msg}"))
        self._send_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._input_box.setEnabled(True)
        self._input_box.setFocus()
        self._set_status("就绪")
        self._update_stats()

        if self._worker:
            self._worker.deleteLater()
            self._worker = None

    def _on_stop(self) -> None:
        if not self._is_running:
            return
        self._remove_thinking_indicator()
        if self._streaming_active:
            self._end_streaming()
        if self._worker:
            self._worker.cancel()
            QTimer.singleShot(5000, self._force_cleanup_if_needed)
        self._stop_requested = True
        self._append_html(cancelled_html())
        self._is_running = False
        self._send_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._input_box.setEnabled(True)
        self._input_box.setFocus()
        self._set_status("就绪")
        self._update_stats()

    def _force_cleanup_if_needed(self) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.quit()
            self._worker.wait(2000)
            if self._worker.isRunning():
                self._worker.terminate()
                self._worker.wait()
            self._worker.deleteLater()
            self._worker = None
            self._stop_requested = False
            self._is_running = False
            self._set_status("已强制终止")

    def _on_clear(self) -> None:
        if self._is_running:
            self._on_stop()
        self._chat_text.clear()
        self._welcome.setVisible(True)
        self._chat_text.setVisible(False)
        self._history = []
        self._session.memory.clear_short_term()
        self._streaming_active = False
        self._streaming_buffer = ""
        self._stream_start_pos = 0
        self._answer_rendered = False
        self._reset_stats()

    def _run_quick_action(self, prompt: str) -> None:
        if self._is_running:
            return
        self._submit_message(prompt)


def main() -> None:
    logger.info("Starting AI Agent Desktop (PyQt6)")
    app = QApplication(sys.argv)
    app.setStyleSheet(QSS)
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    window = AIAgentWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()