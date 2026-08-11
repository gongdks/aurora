"""AI Agent — Native Desktop GUI (Tkinter).

Primary entry point for the AI Agent application.
Usage: python app_tk.py
"""

from __future__ import annotations

import logging
import queue
import threading
import tkinter as tk
from tkinter import ttk

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

FONTS = {
    "body": ("Segoe UI", 10),
    "small": ("Segoe UI", 9),
    "tiny": ("Segoe UI", 8),
    "mono": ("Cascadia Code", 9),
    "mono_small": ("Cascadia Code", 8),
    "title": ("Segoe UI", 14, "bold"),
    "header": ("Segoe UI", 10, "bold"),
    "section": ("Segoe UI", 9, "bold"),
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


# =============================================================================
# Main Application
# =============================================================================
class AIAgentApp:
    """Native desktop GUI for the Plan-and-Execute AI Agent."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("AI Agent — Plan & Execute")
        self.root.geometry("1200x780")
        self.root.minsize(900, 600)
        self.root.configure(bg=COLORS["bg"])

        # Agent session
        self._session = AgentSession()
        self._stop_event: threading.Event | None = None

        # Streaming state
        self._event_queue: queue.Queue = queue.Queue()
        self._tracker = ProgressTracker()
        self._final_answer: str = ""
        self._is_running = False
        self._streaming_buffer: str = ""

        # Chat history (list of {"role": "user"|"assistant", "content": str})
        self._history: list[dict] = []

        # Build UI
        self._setup_styles()
        self._build_layout()

        # Focus window
        self.root.after(200, lambda: (self.root.lift(), self.root.focus_force()))

    # -------------------------------------------------------------------------
    # Styles
    # -------------------------------------------------------------------------
    def _setup_styles(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")

        # Base
        style.configure(".", background=COLORS["bg"], foreground=COLORS["text"])
        style.configure("TFrame", background=COLORS["bg"])
        style.configure("TLabel", background=COLORS["bg"], foreground=COLORS["text"])

        # Sidebar
        style.configure("Sidebar.TFrame", background=COLORS["sidebar_bg"])
        style.configure("Sidebar.TLabel", background=COLORS["sidebar_bg"], foreground=COLORS["text"])

        # Card style (white background card on sidebar)
        style.configure("Card.TFrame", background=COLORS["surface"])
        style.configure("Card.TLabel", background=COLORS["surface"], foreground=COLORS["text"])

        # Send button — solid accent
        style.configure(
            "Send.TButton",
            background=COLORS["accent"],
            foreground="#ffffff",
            font=FONTS["header"],
            padding=(20, 8),
            borderwidth=0,
            focusthickness=0,
        )
        style.map(
            "Send.TButton",
            background=[("active", COLORS["accent_hover"]), ("disabled", COLORS["muted"])],
            foreground=[("disabled", "#ffffff")],
        )

        # Stop button — solid red
        style.configure(
            "Stop.TButton",
            background=COLORS["red"],
            foreground="#ffffff",
            font=FONTS["header"],
            padding=(14, 8),
            borderwidth=0,
            focusthickness=0,
        )
        style.map(
            "Stop.TButton",
            background=[("active", "#b91c1c"), ("disabled", COLORS["muted"])],
            foreground=[("disabled", "#ffffff")],
        )

        # Clear button — subtle ghost
        style.configure(
            "Clear.TButton",
            background=COLORS["surface2"],
            foreground=COLORS["text_secondary"],
            font=FONTS["body"],
            padding=(14, 8),
            borderwidth=0,
            focusthickness=0,
        )
        style.map(
            "Clear.TButton",
            background=[("active", COLORS["border"]), ("disabled", COLORS["surface2"])],
            foreground=[("disabled", COLORS["muted"])],
        )

        # Quick action buttons
        style.configure(
            "Quick.TButton",
            background=COLORS["surface"],
            foreground=COLORS["text_secondary"],
            font=FONTS["small"],
            padding=(12, 6),
            borderwidth=0,
            focusthickness=0,
        )
        style.map(
            "Quick.TButton",
            background=[("active", COLORS["accent_light"])],
            foreground=[("active", COLORS["accent"])],
        )

        # Section header label
        style.configure(
            "Section.TLabel",
            background=COLORS["surface"],
            foreground=COLORS["text_secondary"],
            font=FONTS["section"],
        )

        # Scrollbar
        style.configure(
            "Vertical.TScrollbar",
            background=COLORS["border_strong"],
            troughcolor=COLORS["bg"],
            bordercolor=COLORS["bg"],
            arrowcolor=COLORS["text_secondary"],
            relief=tk.FLAT,
            width=10,
        )
        style.map(
            "Vertical.TScrollbar",
            background=[("active", COLORS["text_secondary"])],
            arrowcolor=[("active", COLORS["text"])],
        )

    # -------------------------------------------------------------------------
    # Layout
    # -------------------------------------------------------------------------
    def _build_layout(self) -> None:
        self._build_header()

        # Main area: chat | sidebar
        main_pw = tk.PanedWindow(
            self.root, orient=tk.HORIZONTAL,
            bg=COLORS["bg"], sashwidth=6, sashrelief=tk.FLAT,
        )
        main_pw.pack(fill=tk.BOTH, expand=True, padx=12, pady=(8, 4))

        # Left: Chat
        chat_frame = ttk.Frame(main_pw)
        main_pw.add(chat_frame, stretch="always")
        self._build_chat(chat_frame)

        # Right: Sidebar
        sidebar = ttk.Frame(main_pw, style="Sidebar.TFrame")
        main_pw.add(sidebar, stretch="never")
        self._build_sidebar(sidebar)

        self.root.after(100, lambda: main_pw.sash_place(0, 880, 0))

        # Status bar
        self._build_statusbar()

    def _build_header(self) -> None:
        header = tk.Frame(self.root, bg=COLORS["header_bg"], height=56)
        header.pack(fill=tk.X)
        header.pack_propagate(False)

        # Accent bar at bottom
        accent_bar = tk.Frame(header, bg=COLORS["accent"], height=3)
        accent_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # Logo
        logo_wrap = tk.Frame(header, bg=COLORS["header_bg"])
        logo_wrap.pack(side=tk.LEFT, padx=(16, 10), pady=10)

        logo = tk.Frame(logo_wrap, bg=COLORS["accent"], width=36, height=36, highlightthickness=0)
        logo.pack()
        logo_label = tk.Label(logo, text="🤖", bg=COLORS["accent"], font=("Segoe UI", 16))
        logo_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        # Title block
        title_block = tk.Frame(header, bg=COLORS["header_bg"])
        title_block.pack(side=tk.LEFT, pady=10)

        tk.Label(
            title_block, text="AI Agent", font=FONTS["title"],
            bg=COLORS["header_bg"], fg=COLORS["text"],
        ).pack(anchor=tk.W)

        tk.Label(
            title_block,
            text="Plan · ReAct · Reflection · Verification",
            font=FONTS["tiny"], bg=COLORS["header_bg"], fg=COLORS["muted"],
        ).pack(anchor=tk.W)

        # Right side info
        right_info = tk.Frame(header, bg=COLORS["header_bg"])
        right_info.pack(side=tk.RIGHT, padx=16, pady=10)

        llm_config = settings.get_llm_config()
        tk.Label(
            right_info,
            text=f"{llm_config['provider'].upper()} · {llm_config['model']}",
            font=FONTS["small"], bg=COLORS["header_bg"], fg=COLORS["text_secondary"],
        ).pack(anchor=tk.E)

    def _build_chat(self, parent: ttk.Frame) -> None:
        # Chat container — white card with border
        chat_outer = tk.Frame(parent, bg=COLORS["border_strong"])
        chat_outer.pack(fill=tk.BOTH, expand=True, padx=(0, 6))

        chat_inner = tk.Frame(chat_outer, bg=COLORS["surface"])
        chat_inner.pack(fill=tk.BOTH, expand=True, padx=1, pady=(1, 0))

        # Welcome
        self._welcome = tk.Frame(chat_inner, bg=COLORS["surface"])
        self._welcome.pack(fill=tk.BOTH, expand=True)
        welcome_inner = tk.Frame(self._welcome, bg=COLORS["surface"])
        welcome_inner.place(relx=0.5, rely=0.45, anchor=tk.CENTER)

        tk.Label(
            welcome_inner, text="🧠", font=("Segoe UI", 48),
            bg=COLORS["surface"],
        ).pack(pady=(0, 16))

        tk.Label(
            welcome_inner, text="Plan-and-Execute Agent",
            font=("Segoe UI", 18, "bold"),
            bg=COLORS["surface"], fg=COLORS["text"],
        ).pack(pady=(0, 4))

        tk.Label(
            welcome_inner,
            text="我能够理解目标、规划步骤、调用工具、反思调整、验证闭环。\n输入你的任务开始对话吧！",
            font=FONTS["body"], bg=COLORS["surface"], fg=COLORS["text_secondary"],
            justify=tk.CENTER,
        ).pack(pady=(0, 16))

        # Suggestion chips row
        chips = ["计算 128 * 56 / 8 + 2^10", "搜索 Python asyncio 用法", "读取 README.md 并总结"]
        chip_row = tk.Frame(welcome_inner, bg=COLORS["surface"])
        chip_row.pack()
        for chip_text in chips:
            chip = tk.Label(
                chip_row, text=chip_text,
                font=FONTS["small"],
                bg=COLORS["accent_light"], fg=COLORS["accent"],
                padx=14, pady=6,
                cursor="hand2",
            )
            chip.pack(side=tk.LEFT, padx=4)
            chip.bind("<Button-1>", lambda e, t=chip_text: self._submit_message(t))

        # Chat text widget (inside the white card)
        chat_text_container = tk.Frame(chat_inner, bg=COLORS["surface"])
        self._chat_text = tk.Text(
            chat_text_container,
            bg=COLORS["surface"], fg=COLORS["text"],
            font=FONTS["body"],
            wrap=tk.WORD,
            state=tk.DISABLED,
            borderwidth=0,
            padx=20, pady=16,
            cursor="arrow",
            yscrollcommand=lambda *a: scrollbar.set(*a),
            highlightthickness=0,
            spacing1=0, spacing3=0,
        )
        scrollbar = ttk.Scrollbar(chat_text_container, orient=tk.VERTICAL, command=self._chat_text.yview)
        self._chat_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Text tags
        self._chat_text.tag_configure(
            "user_bubble",
            foreground=COLORS["text"],
            font=FONTS["header"],
            spacing1=12, spacing3=4,
        )
        self._chat_text.tag_configure(
            "assistant_text",
            foreground=COLORS["text"],
            font=FONTS["body"],
            spacing3=6,
        )
        self._chat_text.tag_configure(
            "assistant_header",
            foreground=COLORS["accent"],
            font=FONTS["header"],
            spacing1=16, spacing3=6,
        )
        self._chat_text.tag_configure(
            "thought",
            foreground="#92400e",
            font=("Segoe UI", 9, "italic"),
            background=COLORS["yellow_light"],
            lmargin1=20, lmargin2=20, rmargin=20,
            spacing1=4, spacing3=4,
        )
        self._chat_text.tag_configure(
            "tool_header",
            foreground=COLORS["accent"],
            font=("Segoe UI", 9, "bold"),
            lmargin1=20, lmargin2=20, rmargin=20,
            spacing1=8, spacing3=2,
        )
        self._chat_text.tag_configure(
            "tool_body",
            foreground=COLORS["text_secondary"],
            font=FONTS["mono_small"],
            background=COLORS["surface2"],
            lmargin1=20, lmargin2=20, rmargin=20,
            spacing1=2, spacing3=4,
        )
        self._chat_text.tag_configure(
            "error",
            foreground=COLORS["red"],
            font=FONTS["mono_small"],
            background=COLORS["red_light"],
            lmargin1=20, lmargin2=20, rmargin=20,
            spacing1=4, spacing3=4,
        )
        self._chat_text.tag_configure(
            "status_footer",
            foreground=COLORS["muted"],
            font=FONTS["tiny"],
            lmargin1=8, lmargin2=8, rmargin=8,
            spacing1=8,
        )
        self._chat_text.tag_configure(
            "log_info",
            foreground=COLORS["muted"],
            font=FONTS["tiny"],
            lmargin1=20, lmargin2=20, rmargin=20,
            spacing1=2, spacing3=2,
        )

        # --- Input area ---
        input_outer = tk.Frame(parent, bg=COLORS["border_strong"])
        input_outer.pack(fill=tk.X, padx=(0, 6), pady=(10, 0))

        input_inner = tk.Frame(input_outer, bg=COLORS["surface2"])
        input_inner.pack(fill=tk.X, padx=1, pady=(1, 1))

        self._input_box = tk.Text(
            input_inner,
            bg=COLORS["surface"],
            fg=COLORS["text"],
            font=FONTS["body"],
            wrap=tk.WORD,
            borderwidth=0,
            padx=14, pady=12,
            height=3,
            insertbackground=COLORS["accent"],
            relief=tk.FLAT,
            highlightthickness=0,
        )
        self._input_box.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(1, 0), pady=1)
        self._input_box.bind("<Return>", self._on_enter)
        self._input_box.bind("<Shift-Return>", self._on_shift_enter)

        btn_frame = tk.Frame(input_inner, bg=COLORS["surface2"])
        btn_frame.pack(side=tk.RIGHT, padx=8, pady=6)

        self._send_btn = ttk.Button(btn_frame, text="Send", style="Send.TButton", command=self._on_submit)
        self._send_btn.pack(side=tk.LEFT, padx=(0, 6))

        self._stop_btn = ttk.Button(btn_frame, text="Stop", style="Stop.TButton", command=self._on_stop)
        self._stop_btn.pack(side=tk.LEFT, padx=(0, 6))
        self._stop_btn.configure(state=tk.DISABLED)

        self._clear_btn = ttk.Button(btn_frame, text="Clear", style="Clear.TButton", command=self._on_clear)
        self._clear_btn.pack(side=tk.LEFT)

    def _build_sidebar(self, parent: ttk.Frame) -> None:
        parent.configure(width=280)
        parent.pack_propagate(False)

        # Inner scrollable-like container
        inner = tk.Frame(parent, bg=COLORS["sidebar_bg"])
        inner.pack(fill=tk.BOTH, expand=True, padx=10, pady=(12, 4))

        # --- Model Info Card ---
        card1 = tk.Frame(inner, bg=COLORS["surface"], highlightbackground=COLORS["border"], highlightthickness=1)
        card1.pack(fill=tk.X, pady=(0, 10))

        tk.Label(
            card1, text="🤖 模型信息",
            font=FONTS["section"],
            bg=COLORS["surface"], fg=COLORS["text_secondary"],
            padx=12, pady=8, anchor="w",
        ).pack(fill=tk.X)

        sep1 = tk.Frame(card1, bg=COLORS["border"], height=1)
        sep1.pack(fill=tk.X, padx=12)

        llm_config = settings.get_llm_config()
        rows = [
            ("服务商", llm_config["provider"].upper(), True),
            ("模型", llm_config["model"], False),
            ("温度", str(llm_config["temperature"]), False),
            ("模式", "Plan-and-Execute", True),
        ]
        for label_text, value_text, highlight in rows:
            row = tk.Frame(card1, bg=COLORS["surface"])
            row.pack(fill=tk.X, padx=12, pady=4)
            tk.Label(
                row, text=label_text, font=FONTS["small"],
                bg=COLORS["surface"], fg=COLORS["muted"],
            ).pack(side=tk.LEFT)
            color = COLORS["accent"] if highlight else COLORS["text"]
            tk.Label(
                row, text=value_text, font=FONTS["small"],
                bg=COLORS["surface"], fg=color,
            ).pack(side=tk.RIGHT)

        # --- Session Stats Card ---
        card2 = tk.Frame(inner, bg=COLORS["surface"], highlightbackground=COLORS["border"], highlightthickness=1)
        card2.pack(fill=tk.X, pady=(0, 10))

        tk.Label(
            card2, text="📊 会话统计",
            font=FONTS["section"],
            bg=COLORS["surface"], fg=COLORS["text_secondary"],
            padx=12, pady=8, anchor="w",
        ).pack(fill=tk.X)

        sep2 = tk.Frame(card2, bg=COLORS["border"], height=1)
        sep2.pack(fill=tk.X, padx=12)

        grid = tk.Frame(card2, bg=COLORS["surface"])
        grid.pack(fill=tk.X, padx=12, pady=10)

        stats = [
            ("0.0s", "耗时", COLORS["accent"]),
            ("0", "工具", COLORS["text"]),
            ("0", "步骤", COLORS["text"]),
            ("●", "状态", COLORS["green"]),
        ]
        self._stat_labels: dict[str, tk.Label] = {}
        self._stat_value_labels: dict[str, tk.Label] = {}

        for i, (val, label_text, color) in enumerate(stats):
            col = i % 2
            row_idx = i // 2

            cell = tk.Frame(grid, bg=COLORS["surface"])
            cell.grid(row=row_idx, column=col, padx=4, pady=6, sticky="nsew")

            value_label = tk.Label(
                cell, text=val, font=("Segoe UI", 18, "bold"),
                bg=COLORS["surface"], fg=color,
                anchor="center",
            )
            value_label.pack()

            label_lbl = tk.Label(
                cell, text=label_text, font=FONTS["tiny"],
                bg=COLORS["surface"], fg=COLORS["muted"],
                anchor="center",
            )
            label_lbl.pack()

            self._stat_labels[label_text] = label_lbl
            self._stat_value_labels[label_text] = value_label

        grid.columnconfigure(0, weight=1)
        grid.columnconfigure(1, weight=1)

        # --- Quick Actions Card ---
        card3 = tk.Frame(inner, bg=COLORS["surface"], highlightbackground=COLORS["border"], highlightthickness=1)
        card3.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        tk.Label(
            card3, text="⚡ 快捷操作",
            font=FONTS["section"],
            bg=COLORS["surface"], fg=COLORS["text_secondary"],
            padx=12, pady=8, anchor="w",
        ).pack(fill=tk.X)

        sep3 = tk.Frame(card3, bg=COLORS["border"], height=1)
        sep3.pack(fill=tk.X, padx=12)

        btn_container = tk.Frame(card3, bg=COLORS["surface"])
        btn_container.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        for i, (label_text, prompt) in enumerate(QUICK_ACTIONS):
            btn = ttk.Button(
                btn_container, text=label_text, style="Quick.TButton",
                command=lambda p=prompt: self._run_quick_action(p),
            )
            btn.pack(fill=tk.X, pady=2)

    def _build_statusbar(self) -> None:
        status_outer = tk.Frame(self.root, bg=COLORS["border_strong"])
        status_outer.pack(fill=tk.X, padx=12, pady=(0, 8))

        status_frame = tk.Frame(status_outer, bg=COLORS["surface2"])
        status_frame.pack(fill=tk.X, padx=1, pady=(1, 1))

        self._status_dot = tk.Label(
            status_frame, text="●", font=("Segoe UI", 8),
            bg=COLORS["surface2"], fg=COLORS["green"],
        )
        self._status_dot.pack(side=tk.LEFT, padx=(12, 6))

        self._status_label = tk.Label(
            status_frame, text="就绪", font=FONTS["tiny"],
            bg=COLORS["surface2"], fg=COLORS["text_secondary"],
        )
        self._status_label.pack(side=tk.LEFT)

        tk.Label(
            status_frame, text="Powered by ReAct Framework", font=FONTS["tiny"],
            bg=COLORS["surface2"], fg=COLORS["muted"],
        ).pack(side=tk.RIGHT, padx=12)

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    def _show_chat(self) -> None:
        if self._welcome.winfo_ismapped():
            self._welcome.pack_forget()
            self._chat_text.master.pack(fill=tk.BOTH, expand=True)

    def _append_chat(self, text: str, *tags: str) -> None:
        self._chat_text.configure(state=tk.NORMAL)
        self._chat_text.insert(tk.END, text, tags)
        self._chat_text.configure(state=tk.DISABLED)
        self._chat_text.see(tk.END)

    def _set_status(self, text: str, running: bool = False) -> None:
        self._status_label.configure(text=text)
        if running:
            self._status_dot.configure(fg=COLORS["yellow"], text="◉")
        else:
            self._status_dot.configure(fg=COLORS["green"], text="●")

    def _update_stats(self, tracker: ProgressTracker | None = None) -> None:
        if tracker is None:
            self._stat_value_labels["耗时"].configure(text="0.0s", fg=COLORS["accent"])
            self._stat_value_labels["工具"].configure(text="0", fg=COLORS["text"])
            self._stat_value_labels["步骤"].configure(text="0", fg=COLORS["text"])
            self._stat_value_labels["状态"].configure(text="●", fg=COLORS["green"])
            return

        elapsed = tracker.elapsed
        sec = f"{elapsed:.1f}s" if elapsed < 60 else f"{int(elapsed // 60)}m{int(elapsed % 60)}s"
        tool_count = len(tracker.tool_calls)
        log_msgs = [e.get("message", "") for e in tracker.events if e.get("type") == "log"]
        step_count = sum(1 for m in log_msgs if m.startswith("✅ Step"))

        self._stat_value_labels["耗时"].configure(text=sec, fg=COLORS["accent"])
        self._stat_value_labels["工具"].configure(text=str(tool_count), fg=COLORS["text"])
        self._stat_value_labels["步骤"].configure(text=str(step_count), fg=COLORS["text"])

        if self._is_running:
            self._stat_value_labels["状态"].configure(text="◉", fg=COLORS["yellow"])
        else:
            self._stat_value_labels["状态"].configure(text="●", fg=COLORS["green"])

    # -------------------------------------------------------------------------
    # Event rendering
    # -------------------------------------------------------------------------
    def _render_log(self, msg: str) -> None:
        if msg.startswith("📋"):
            self._append_chat(f"  {msg}\n", "thought")
        elif msg.startswith("▶️"):
            self._append_chat(f"  {msg}\n", "assistant_header")
        elif msg.startswith("✅ Step") or msg.startswith("✅ Verification"):
            self._append_chat(f"  {msg}\n", "log_info")
        elif msg.startswith("🤔") or msg.startswith("💭"):
            self._append_chat(f"  {msg}\n", "thought")
        elif msg.startswith("🎯"):
            self._append_chat(f"  {msg}\n", "assistant_header")
        elif msg.startswith("🔄"):
            self._append_chat(f"  {msg}\n", "thought")
        elif msg.startswith("⚠️"):
            self._append_chat(f"  {msg}\n", "error")
        elif msg.startswith("🔍"):
            self._append_chat(f"  {msg}\n", "log_info")
        elif msg.startswith("⏹") or msg.startswith("⏰") or msg.startswith("⏱") or msg.startswith("⏭"):
            self._append_chat(f"  {msg}\n", "log_info")
        else:
            self._append_chat(f"  {msg}\n", "log_info")

    def _render_tool(self, event: dict) -> None:
        name = event.get("name", "?")
        inp = str(event.get("input", ""))[:300]
        out = event.get("output")

        if out is None:
            self._append_chat(f"  ⚙️ {name}\n", "tool_header")
            if inp:
                self._append_chat(f"     输入: {inp}\n", "tool_body")
        else:
            out_str = str(out)[:600]
            is_error = out_str.startswith("[ERR]") or out_str.startswith("[Timeout]")
            tag = "error" if is_error else "tool_body"
            self._append_chat(f"     ↳ 输出: {out_str}\n", tag)

    def _render_final(self, answer: str) -> None:
        self._append_chat("\n", "assistant_text")
        for line in answer.split("\n"):
            self._append_chat(f"{line}\n", "assistant_text")

        elapsed = self._tracker.elapsed
        tc = len(self._tracker.tool_calls)
        sec = f"{elapsed:.1f}s" if elapsed < 60 else f"{int(elapsed // 60)}m{int(elapsed % 60)}s"
        self._append_chat(
            f"   ⏱ {sec}  ·  🔧 {tc} 个工具已使用\n\n", "status_footer",
        )

    def _render_streaming_token(self, token: str) -> None:
        if not self._streaming_buffer:
            self._append_chat("\nAssistant\n", "assistant_header")
        self._streaming_buffer += token
        self._chat_text.configure(state=tk.NORMAL)
        self._chat_text.insert(tk.END, token, "assistant_text")
        self._chat_text.configure(state=tk.DISABLED)
        self._chat_text.see(tk.END)

    # -------------------------------------------------------------------------
    # Streaming engine
    # -------------------------------------------------------------------------
    def _on_submit(self) -> None:
        if self._is_running:
            return

        message = self._input_box.get("1.0", "end-1c").strip()
        if not message:
            return

        self._input_box.delete("1.0", tk.END)
        self._submit_message(message)

    def _submit_message(self, message: str) -> None:
        self._show_chat()
        self._append_chat(f"You\n", "user_bubble")
        self._append_chat(f"{message}\n\n", "assistant_text")
        self._history.append({"role": "user", "content": message})
        self._start_agent(message)

    def _on_enter(self, event: tk.Event) -> str:
        self._on_submit()
        return "break"

    def _on_shift_enter(self, event: tk.Event) -> None:
        self._input_box.insert(tk.INSERT, "\n")

    def _start_agent(self, message: str) -> None:
        self._is_running = True
        self._final_answer = ""
        self._streaming_buffer = ""
        self._tracker = ProgressTracker()
        self._event_queue = queue.Queue()

        self._send_btn.configure(state=tk.DISABLED)
        self._stop_btn.configure(state=tk.NORMAL)
        self._input_box.configure(state=tk.DISABLED)
        self._set_status("思考中...", running=True)
        self._update_stats(self._tracker)

        self._stop_event = threading.Event()

        def _run() -> None:
            try:
                answer = self._session.invoke(
                    message, self._history,
                    progress_callback=lambda ev: self._event_queue.put(ev),
                )
                self._final_answer = answer
            except Exception as exc:
                logger.error("Agent error: %s", exc)
                self._final_answer = f"❌ Error: {exc}"
            finally:
                self._event_queue.put(None)

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

        self._last_render_idx = 0
        self._root_after_id = self.root.after(80, self._poll_events)

    def _poll_events(self) -> None:
        while True:
            try:
                event = self._event_queue.get_nowait()
            except queue.Empty:
                break

            if event is None:
                self._finish_stream()
                return

            self._tracker.feed(event)

        all_events = self._tracker.events
        new_events = all_events[self._last_render_idx:]
        self._last_render_idx = len(all_events)

        for ev in new_events:
            etype = ev.get("type")
            if etype == "log":
                self._render_log(ev.get("message", ""))
            elif etype == "tool":
                self._render_tool(ev)
            elif etype == "streaming_token":
                self._render_streaming_token(ev.get("token", ""))

        self._update_stats(self._tracker)
        self._root_after_id = self.root.after(80, self._poll_events)

    def _finish_stream(self) -> None:
        self._is_running = False

        answer = self._final_answer or "*(no response)*"
        self._render_final(answer)

        self._history.append({"role": "assistant", "content": answer})

        self._send_btn.configure(state=tk.NORMAL)
        self._stop_btn.configure(state=tk.DISABLED)
        self._input_box.configure(state=tk.NORMAL)
        self._input_box.focus_set()
        self._set_status("就绪")
        self._update_stats(self._tracker)
        self._stop_event = None

        if self._root_after_id:
            self.root.after_cancel(self._root_after_id)
            self._root_after_id = None

    # -------------------------------------------------------------------------
    # Actions
    # -------------------------------------------------------------------------
    def _on_stop(self) -> None:
        if not self._is_running:
            return

        self._session.stop()
        if self._stop_event:
            self._stop_event.set()

        self._final_answer = "⏹ *Stopped.*"
        self._append_chat("\n  ⏹ 已停止\n\n", "error")

        self._is_running = False
        self._send_btn.configure(state=tk.NORMAL)
        self._stop_btn.configure(state=tk.DISABLED)
        self._input_box.configure(state=tk.NORMAL)
        self._input_box.focus_set()
        self._set_status("就绪")
        self._update_stats(self._tracker)

        if self._root_after_id:
            self.root.after_cancel(self._root_after_id)
            self._root_after_id = None

    def _on_clear(self) -> None:
        if self._is_running:
            self._on_stop()

        self._chat_text.configure(state=tk.NORMAL)
        self._chat_text.delete("1.0", tk.END)
        self._chat_text.configure(state=tk.DISABLED)

        self._history = []
        self._session.memory.clear_short_term()
        self._update_stats(None)

    def _run_quick_action(self, prompt: str) -> None:
        if self._is_running:
            return
        self._submit_message(prompt)

    # -------------------------------------------------------------------------
    # Run
    # -------------------------------------------------------------------------
    def run(self) -> None:
        self.root.mainloop()


# =============================================================================
# Entry point
# =============================================================================
def main() -> None:
    logger.info("Starting AI Agent Desktop (tkinter)")

    root = tk.Tk()
    app = AIAgentApp(root)
    app.run()


if __name__ == "__main__":
    main()