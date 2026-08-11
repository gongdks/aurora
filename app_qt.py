"""AI Agent — Modern Desktop GUI (PyQt6).

Primary entry point for the AI Agent application.
Usage: python app_qt.py
"""

from __future__ import annotations

import logging
import queue
import sys
import threading
from typing import Any

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import (
    QAction, QColor, QFont, QFontDatabase, QIcon, QKeySequence,
    QPainter, QPixmap, QCursor,
)
from PyQt6.QtWidgets import (
    QApplication, QFrame, QHBoxLayout, QLabel, QMainWindow,
    QMessageBox, QPushButton, QScrollArea, QSizePolicy,
    QSplitter, QTextEdit, QVBoxLayout, QWidget, QTextBrowser,
    QGraphicsDropShadowEffect, QSystemTrayIcon, QMenu,
)

from agent.agent import AgentSession
from agent.config import settings
from agent.progress import ProgressTracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# =============================================================================
# Color scheme — refined light theme
# =============================================================================
COLORS = {
    "bg": "#fafbfc",
    "surface": "#ffffff",
    "surface2": "#f5f6f8",
    "border": "#e1e4e8",
    "border_strong": "#d0d7de",
    "accent": "#2563eb",
    "accent_hover": "#1d4ed8",
    "accent_light": "#dbeafe",
    "green": "#16a34a",
    "green_light": "#dcfce7",
    "yellow": "#ca8a04",
    "yellow_light": "#fef9c3",
    "red": "#dc2626",
    "red_light": "#fee2e2",
    "text": "#0f172a",
    "text_secondary": "#475569",
    "muted": "#94a3b8",
    "sidebar_bg": "#f3f4f6",
    "header_bg": "#ffffff",
}

# =============================================================================
# Quick action examples
# =============================================================================
QUICK_ACTIONS = [
    ("🧮 计算", "计算 128 * 56 / 8 + 2^10"),
    ("🔍 搜索", "搜索 Python asyncio 的用法，给我一个简单例子"),
    ("📖 读文件", "读取当前目录下的 README.md 文件内容并总结要点"),
    ("💻 写代码", "写一个 Python 脚本，输出 1-100 之间的素数并保存到文件"),
    ("📝 笔记", "帮我记一条笔记：我喜欢喝咖啡，然后把所有笔记列出来"),
    ("🌐 抓取", "抓取 https://example.com 的内容并总结关键信息"),
    ("📋 复杂任务", "读取 README.md，分析项目结构，然后写一份简要的项目说明"),
    ("📰 新闻搜索", "用必应搜索今天的 AI 新闻，选 3 条最重要的，统计每条的标题字数"),
]

QSS = f"""
QWidget {{
    font-family: "Segoe UI", "Microsoft YaHei UI", sans-serif;
    color: {COLORS['text']};
}}

QMainWindow {{
    background-color: {COLORS['bg']};
}}

#Header {{
    background-color: {COLORS['header_bg']};
    border-bottom: 1px solid {COLORS['border']};
}}

#AccentBar {{
    background-color: {COLORS['accent']};
}}

#Sidebar {{
    background-color: {COLORS['sidebar_bg']};
    border-left: 1px solid {COLORS['border']};
}}

#ChatContainer {{
    background-color: {COLORS['surface']};
    border: 1px solid {COLORS['border']};
    border-radius: 12px;
}}

#WelcomeLabel {{
    color: {COLORS['muted']};
    font-size: 13px;
}}

#WelcomeTitle {{
    color: {COLORS['text']};
    font-size: 20px;
    font-weight: 600;
}}

#WelcomeSub {{
    color: {COLORS['text_secondary']};
    font-size: 13px;
}}

#Chip {{
    background-color: {COLORS['accent_light']};
    color: {COLORS['accent']};
    border: 1px solid transparent;
    border-radius: 14px;
    padding: 6px 14px;
    font-size: 12px;
}}

#Chip:hover {{
    background-color: {COLORS['accent']};
    color: white;
}}

#ChatText {{
    background-color: {COLORS['surface']};
    border: none;
    font-size: 13px;
    padding: 16px 20px;
}}

QScrollBar:vertical {{
    background-color: transparent;
    width: 10px;
    margin: 0;
}}

QScrollBar::handle:vertical {{
    background-color: {COLORS['border_strong']};
    border-radius: 5px;
    min-height: 30px;
}}

QScrollBar::handle:vertical:hover {{
    background-color: {COLORS['muted']};
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}

#InputContainer {{
    background-color: {COLORS['surface']};
    border: 1px solid {COLORS['border']};
    border-radius: 12px;
}}

#InputBox {{
    background-color: {COLORS['surface']};
    border: none;
    font-size: 13px;
    padding: 12px 14px;
    selection-background-color: {COLORS['accent_light']};
}}

#SendButton {{
    background-color: {COLORS['accent']};
    color: white;
    border: none;
    border-radius: 8px;
    padding: 8px 20px;
    font-size: 13px;
    font-weight: 600;
}}

#SendButton:hover {{
    background-color: {COLORS['accent_hover']};
}}

#SendButton:disabled {{
    background-color: {COLORS['muted']};
    color: white;
}}

#StopButton {{
    background-color: {COLORS['red']};
    color: white;
    border: none;
    border-radius: 8px;
    padding: 8px 14px;
    font-size: 13px;
    font-weight: 600;
}}

#StopButton:hover {{
    background-color: #b91c1c;
}}

#StopButton:disabled {{
    background-color: {COLORS['muted']};
    color: white;
}}

#ClearButton {{
    background-color: {COLORS['surface2']};
    color: {COLORS['text_secondary']};
    border: none;
    border-radius: 8px;
    padding: 8px 14px;
    font-size: 13px;
}}

#ClearButton:hover {{
    background-color: {COLORS['border']};
}}

#Card {{
    background-color: {COLORS['surface']};
    border: 1px solid {COLORS['border']};
    border-radius: 10px;
}}

#CardTitle {{
    color: {COLORS['text_secondary']};
    font-size: 12px;
    font-weight: 600;
}}

#StatValue {{
    color: {COLORS['accent']};
    font-size: 22px;
    font-weight: 700;
}}

#StatLabel {{
    color: {COLORS['muted']};
    font-size: 11px;
}}

#QuickButton {{
    background-color: {COLORS['surface']};
    color: {COLORS['text_secondary']};
    border: 1px solid {COLORS['border']};
    border-radius: 8px;
    padding: 8px 12px;
    font-size: 12px;
    text-align: left;
}}

#QuickButton:hover {{
    background-color: {COLORS['accent_light']};
    color: {COLORS['accent']};
    border-color: {COLORS['accent_light']};
}}

#StatusBar {{
    background-color: {COLORS['surface2']};
    border-top: 1px solid {COLORS['border']};
}}

#StatusDot {{
    color: {COLORS['green']};
    font-size: 10px;
}}

#StatusText {{
    color: {COLORS['text_secondary']};
    font-size: 11px;
}}

QToolTip {{
    background-color: {COLORS['text']};
    color: white;
    border: none;
    padding: 6px 10px;
    border-radius: 6px;
    font-size: 12px;
}}
"""


class AgentWorker(QThread):
    """Background worker thread for running the agent."""

    progress = pyqtSignal(dict)
    finished_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)

    def __init__(
        self,
        session: AgentSession,
        message: str,
        history: list,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._session = session
        self._message = message
        self._history = history
        self._cancel = threading.Event()

    def cancel(self) -> None:
        self._cancel.set()
        self._session.stop()

    def run(self) -> None:
        def _progress_cb(ev: dict) -> None:
            self.progress.emit(ev)

        try:
            answer = self._session.invoke(
                self._message,
                self._history,
                progress_callback=_progress_cb,
            )
            self.finished_signal.emit(answer)
        except Exception as exc:
            logger.error("Agent error: %s", exc, exc_info=True)
            self.error_signal.emit(str(exc))


class AIAgentWindow(QMainWindow):
    """Main application window."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("AI Agent — Plan & Execute")
        self.resize(1200, 780)
        self.setMinimumSize(900, 600)

        self._session = AgentSession()
        self._worker: AgentWorker | None = None
        self._tracker = ProgressTracker()
        self._history: list[dict] = []
        self._is_running = False
        self._stop_requested = False
        self._streaming_buffer = ""
        self._last_render_idx = 0

        self._setup_ui()
        self._setup_connections()

    # ------------------------------------------------------------------
    # UI Setup
    # ------------------------------------------------------------------
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
            f"background-color: {COLORS['accent']}; border-radius: 8px; "
            f"font-size: 18px;"
        )

        title_block = QVBoxLayout()
        title_block.setSpacing(0)
        title_block.setContentsMargins(0, 0, 0, 0)

        title_main = QLabel("AI Agent")
        title_main.setStyleSheet(
            f"font-size: 15px; font-weight: 600; color: {COLORS['text']};"
        )

        title_sub = QLabel("Plan · ReAct · Reflection · Verification")
        title_sub.setStyleSheet(
            f"font-size: 10px; color: {COLORS['muted']};"
        )

        title_block.addWidget(title_main)
        title_block.addWidget(title_sub)

        header_layout.addWidget(logo_label)
        header_layout.addLayout(title_block)
        header_layout.addStretch()

        llm_config = settings.get_llm_config()
        model_label = QLabel(
            f"{llm_config['provider'].upper()} · {llm_config['model']}"
        )
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
        welcome_layout.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )
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

        sub = QLabel(
            "我能够理解目标、规划步骤、调用工具、反思调整、验证闭环。\n输入你的任务开始对话吧！"
        )
        sub.setObjectName("WelcomeSub")
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        welcome_layout.addWidget(sub)

        chips_row = QHBoxLayout()
        chips_row.setSpacing(8)
        chips_row.setAlignment(Qt.AlignmentFlag.AlignCenter)

        chip_texts = [
            "计算 128 * 56 / 8 + 2^10",
            "搜索 Python asyncio 用法",
            "读取 README.md 并总结",
        ]
        for ct in chip_texts:
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
        self._chat_text.setLineWrapMode(
            QTextBrowser.LineWrapMode.WidgetWidth
        )
        self._chat_text.setReadOnly(True)
        self._chat_text.setVisible(False)
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
        self._input_box.setFixedHeight(44)
        self._input_box.setMaximumHeight(120)
        self._input_box.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self._input_box.setAcceptRichText(False)
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
            lbl.setStyleSheet(
                f"font-size: 12px; color: {COLORS['muted']};"
            )

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
                    f"QFrame {{ background-color: {COLORS['surface2']}; "
                    f"border-radius: 8px; }}"
                )
                cell_layout = QVBoxLayout(cell)
                cell_layout.setContentsMargins(8, 8, 8, 8)
                cell_layout.setSpacing(2)

                val_label = QLabel(val)
                val_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                val_label.setStyleSheet(
                    f"font-size: 18px; font-weight: 700; color: {color};"
                )

                name_label = QLabel(name)
                name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                name_label.setStyleSheet(
                    f"font-size: 10px; color: {COLORS['muted']};"
                )

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
            btn.clicked.connect(
                lambda _=False, p=prompt: self._run_quick_action(p)
            )
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
        self._status_dot.setStyleSheet(
            f"color: {COLORS['green']}; font-size: 10px;"
        )

        self._status_label = QLabel("就绪")
        self._status_label.setObjectName("StatusText")

        right_label = QLabel("Powered by ReAct Framework")
        right_label.setStyleSheet(
            f"font-size: 10px; color: {COLORS['muted']};"
        )

        status_layout.addWidget(self._status_dot)
        status_layout.addWidget(self._status_label)
        status_layout.addStretch()
        status_layout.addWidget(right_label)

        layout.addWidget(status)

    # ------------------------------------------------------------------
    # Connections
    # ------------------------------------------------------------------
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

                if key == Qt.Key.Key_Return and not (
                    modifiers & Qt.KeyboardModifier.ShiftModifier
                ):
                    self._on_submit()
                    return True
                elif key == Qt.Key.Key_Escape:
                    if self._is_running:
                        self._on_stop()
                    return True
        return super().eventFilter(obj, event)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _show_chat(self) -> None:
        if self._welcome.isVisible():
            self._welcome.setVisible(False)
            self._chat_text.setVisible(True)

    def _append_chat_html(self, html: str) -> None:
        self._chat_text.append(html)
        sb = self._chat_text.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _set_status(self, text: str, running: bool = False) -> None:
        self._status_label.setText(text)
        if running:
            self._status_dot.setText("◉")
            self._status_dot.setStyleSheet(
                f"color: {COLORS['yellow']}; font-size: 10px;"
            )
        else:
            self._status_dot.setText("●")
            self._status_dot.setStyleSheet(
                f"color: {COLORS['green']}; font-size: 10px;"
            )

    def _update_stats(self, tracker: ProgressTracker | None = None) -> None:
        if tracker is None:
            self._set_stat("耗时", "0.0s", COLORS["accent"])
            self._set_stat("工具", "0", COLORS["text"])
            self._set_stat("步骤", "0", COLORS["text"])
            self._set_stat("状态", "●", COLORS["green"])
            return

        elapsed = tracker.elapsed
        sec = (
            f"{elapsed:.1f}s"
            if elapsed < 60
            else f"{int(elapsed // 60)}m{int(elapsed % 60)}s"
        )
        tool_count = len(tracker.tool_calls)
        log_msgs = [
            e.get("message", "")
            for e in tracker.events
            if e.get("type") == "log"
        ]
        step_count = sum(
            1 for m in log_msgs if m.startswith("✅ Step")
        )

        self._set_stat("耗时", sec, COLORS["accent"])
        self._set_stat("工具", str(tool_count), COLORS["text"])
        self._set_stat("步骤", str(step_count), COLORS["text"])

        if self._is_running:
            self._set_stat("状态", "◉", COLORS["yellow"])
        else:
            self._set_stat("状态", "●", COLORS["green"])

    def _set_stat(self, name: str, value: str, color: str) -> None:
        if name in self._stat_cells:
            val_label, _ = self._stat_cells[name]
            val_label.setText(value)
            val_label.setStyleSheet(
                f"font-size: 18px; font-weight: 700; color: {color};"
            )

    # ------------------------------------------------------------------
    # HTML formatters for chat display
    # ------------------------------------------------------------------
    def _user_bubble_html(self, text: str) -> str:
        return (
            f'<div style="margin: 12px 0 4px 0;">'
            f'<span style="color: {COLORS["accent"]}; font-weight: 600; '
            f'font-size: 13px;">You</span>'
            f'</div>'
            f'<div style="color: {COLORS["text"]}; font-size: 13px; '
            f'line-height: 1.6; margin-bottom: 8px;">'
            f'{text}</div>'
        )

    def _assistant_header_html(self) -> str:
        return (
            f'<div style="margin: 16px 0 4px 0;">'
            f'<span style="color: {COLORS["accent"]}; font-weight: 600; '
            f'font-size: 13px;">Assistant</span>'
            f'</div>'
        )

    def _thought_html(self, msg: str) -> str:
        return (
            f'<div style="margin: 4px 0; padding: 6px 12px; '
            f'background-color: {COLORS["yellow_light"]}; '
            f'border-radius: 6px; color: #92400e; '
            f'font-size: 12px; font-style: italic;">'
            f'{msg}</div>'
        )

    def _log_info_html(self, msg: str) -> str:
        return (
            f'<div style="margin: 2px 0; padding: 2px 12px; '
            f'color: {COLORS["muted"]}; font-size: 11px;">'
            f'{msg}</div>'
        )

    def _tool_header_html(self, name: str) -> str:
        return (
            f'<div style="margin: 8px 0 2px 0; padding: 4px 12px; '
            f'color: {COLORS["accent"]}; font-size: 12px; font-weight: 600;">'
            f'⚙️ {name}</div>'
        )

    def _tool_body_html(self, text: str, is_error: bool = False) -> str:
        bg = COLORS["red_light"] if is_error else COLORS["surface2"]
        fg = COLORS["red"] if is_error else COLORS["text_secondary"]
        return (
            f'<div style="margin: 2px 0 4px 0; padding: 6px 12px; '
            f'background-color: {bg}; border-radius: 6px; '
            f'color: {fg}; font-size: 11px; font-family: Consolas, monospace;">'
            f'{text}</div>'
        )

    def _final_answer_html(self, answer: str, elapsed: float, tc: int) -> str:
        sec = (
            f"{elapsed:.1f}s"
            if elapsed < 60
            else f"{int(elapsed // 60)}m{int(elapsed % 60)}s"
        )
        return (
            f'<div style="margin: 8px 0 4px 0; color: {COLORS["text"]}; '
            f'font-size: 13px; line-height: 1.7;">'
            f'{answer}</div>'
            f'<div style="margin: 4px 0 12px 0; color: {COLORS["muted"]}; '
            f'font-size: 11px;">'
            f'⏱ {sec} · 🔧 {tc} 个工具已使用</div>'
        )

    def _error_html(self, msg: str) -> str:
        return (
            f'<div style="margin: 4px 0; padding: 6px 12px; '
            f'background-color: {COLORS["red_light"]}; '
            f'border-radius: 6px; color: {COLORS["red"]}; '
            f'font-size: 12px;">'
            f'{msg}</div>'
        )

    # ------------------------------------------------------------------
    # Event rendering
    # ------------------------------------------------------------------
    def _render_log(self, msg: str) -> None:
        if msg.startswith("📋"):
            html = self._thought_html(f"  {msg}")
        elif msg.startswith("▶️"):
            html = self._log_info_html(f"  {msg}")
        elif msg.startswith("✅ Step") or msg.startswith("✅ Verification"):
            html = self._log_info_html(f"  {msg}")
        elif msg.startswith("🤔") or msg.startswith("💭"):
            html = self._thought_html(f"  {msg}")
        elif msg.startswith("🎯"):
            html = self._log_info_html(f"  {msg}")
        elif msg.startswith("🔄"):
            html = self._thought_html(f"  {msg}")
        elif msg.startswith("⚠️"):
            html = self._error_html(f"  {msg}")
        elif msg.startswith("🔍"):
            html = self._log_info_html(f"  {msg}")
        elif msg.startswith("⏹") or msg.startswith("⏰") or msg.startswith("⏱") or msg.startswith("⏭"):
            html = self._log_info_html(f"  {msg}")
        else:
            html = self._log_info_html(f"  {msg}")
        self._append_chat_html(html)

    def _render_tool(self, event: dict) -> None:
        name = event.get("name", "?")
        inp = str(event.get("input", ""))[:300]
        out = event.get("output")

        if out is None:
            self._append_chat_html(self._tool_header_html(name))
            if inp:
                self._append_chat_html(
                    self._tool_body_html(f"输入: {inp}")
                )
        else:
            out_str = str(out)[:600]
            is_error = out_str.startswith("[ERR]") or out_str.startswith("[Timeout]")
            self._append_chat_html(
                self._tool_body_html(f"↳ 输出: {out_str}", is_error)
            )

    def _render_final(self, answer: str) -> None:
        elapsed = self._tracker.elapsed
        tc = len(self._tracker.tool_calls)
        html = self._final_answer_html(answer, elapsed, tc)
        self._append_chat_html(html)

    def _render_streaming_token(self, token: str) -> None:
        if not self._streaming_buffer:
            self._append_chat_html(self._assistant_header_html())
        self._streaming_buffer += token

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
        self._append_chat_html(self._user_bubble_html(message))
        self._history.append({"role": "user", "content": message})
        self._start_agent(message)

    def _start_agent(self, message: str) -> None:
        self._is_running = True
        self._stop_requested = False
        self._streaming_buffer = ""
        self._tracker = ProgressTracker()
        self._last_render_idx = 0

        self._send_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._input_box.setEnabled(False)
        self._set_status("思考中...", running=True)
        self._update_stats(self._tracker)

        self._worker = AgentWorker(self._session, message, self._history)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_signal.connect(self._on_finished)
        self._worker.error_signal.connect(self._on_error)
        self._worker.start()

    def _on_progress(self, ev: dict) -> None:
        if self._stop_requested:
            return

        self._tracker.feed(ev)

        all_events = self._tracker.events
        new_events = all_events[self._last_render_idx:]
        self._last_render_idx = len(all_events)

        for e in new_events:
            etype = e.get("type")
            if etype == "log":
                self._render_log(e.get("message", ""))
            elif etype == "tool":
                self._render_tool(e)
            elif etype == "streaming_token":
                self._render_streaming_token(e.get("token", ""))

        self._update_stats(self._tracker)

    def _on_finished(self, answer: str) -> None:
        self._is_running = False

        if self._stop_requested:
            self._stop_requested = False
            self._send_btn.setEnabled(True)
            self._stop_btn.setEnabled(False)
            self._input_box.setEnabled(True)
            self._input_box.setFocus()
            self._set_status("就绪")
            self._update_stats(self._tracker)
            if self._worker:
                self._worker.deleteLater()
                self._worker = None
            return

        if self._streaming_buffer:
            self._append_chat_html(
                f'<div style="color: {COLORS["text"]}; font-size: 13px; '
                f'line-height: 1.7; margin: 4px 0;">'
                f'{self._streaming_buffer}</div>'
            )

        answer = answer or "*(no response)*"
        self._render_final(answer)

        self._history.append({"role": "assistant", "content": answer})

        self._send_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._input_box.setEnabled(True)
        self._input_box.setFocus()
        self._set_status("就绪")
        self._update_stats(self._tracker)

        if self._worker:
            self._worker.deleteLater()
            self._worker = None

    def _on_error(self, error_msg: str) -> None:
        self._is_running = False

        if self._stop_requested:
            self._stop_requested = False
            self._send_btn.setEnabled(True)
            self._stop_btn.setEnabled(False)
            self._input_box.setEnabled(True)
            self._input_box.setFocus()
            self._set_status("就绪")
            self._update_stats(self._tracker)
            if self._worker:
                self._worker.deleteLater()
                self._worker = None
            return

        self._append_chat_html(
            self._error_html(f"  ❌ Error: {error_msg}")
        )

        self._send_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._input_box.setEnabled(True)
        self._input_box.setFocus()
        self._set_status("就绪")
        self._update_stats(self._tracker)

        if self._worker:
            self._worker.deleteLater()
            self._worker = None

    def _on_stop(self) -> None:
        if not self._is_running:
            return

        if self._worker:
            self._worker.cancel()
            QTimer.singleShot(5000, self._force_cleanup_if_needed)

        self._stop_requested = True

        self._append_chat_html(
            self._error_html("  ⏹ 已停止")
        )

        self._is_running = False
        self._send_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._input_box.setEnabled(True)
        self._input_box.setFocus()
        self._set_status("就绪")
        self._update_stats(self._tracker)

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
        self._update_stats(None)

    def _run_quick_action(self, prompt: str) -> None:
        if self._is_running:
            return
        self._submit_message(prompt)


# =============================================================================
# Entry point
# =============================================================================
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