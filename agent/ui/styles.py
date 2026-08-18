"""UI styles — color palette and Qt Style Sheets."""

from __future__ import annotations

COLORS: dict[str, str] = {
    "bg": "#f5f6f8",
    "surface": "#ffffff",
    "surface2": "#f0f2f5",
    "border": "#d9dce1",
    "text": "#1f2328",
    "text_secondary": "#656d76",
    "muted": "#8b949e",
    "accent": "#4a6cf7",
    "green": "#28a745",
    "yellow": "#d29922",
    "red": "#cf222e",
    "user_bubble": "#dce8ff",
    "assistant_bubble": "#f0f2f5",
    "tool_bubble": "#fff8e1",
    "code_bg": "#f6f8fa",
}

QSS = """
QMainWindow, QWidget {
    background-color: %(bg)s;
    color: %(text)s;
    font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
}

QFrame#Header {
    background-color: %(surface)s;
    border-bottom: 1px solid %(border)s;
}

QFrame#AccentBar {
    background-color: %(accent)s;
}

QLabel {
    color: %(text)s;
}

QLabel#WelcomeTitle {
    font-size: 22px;
    font-weight: 700;
    color: %(text)s;
}

QLabel#WelcomeSub {
    font-size: 13px;
    color: %(muted)s;
}

QLabel#CardTitle {
    font-size: 12px;
    font-weight: 600;
    color: %(text_secondary)s;
}

QFrame#Card {
    background-color: %(surface)s;
    border: 1px solid %(border)s;
    border-radius: 10px;
}

QFrame#StatusBar {
    background-color: %(surface)s;
    border-top: 1px solid %(border)s;
}

QLabel#StatusDot {
    font-size: 10px;
}

QLabel#StatusText {
    font-size: 11px;
    color: %(text_secondary)s;
}

QFrame#ChatContainer {
    background-color: transparent;
}

QTextBrowser#ChatText {
    background-color: transparent;
    border: none;
    color: %(text)s;
    font-size: 14px;
    line-height: 1.6;
}

QTextBrowser#ChatText a {
    color: %(accent)s;
}

QTextBrowser#ChatText code {
    background-color: %(code_bg)s;
    color: #24292f;
    border-radius: 4px;
    padding: 2px 6px;
    font-family: "Consolas", "Monaco", monospace;
    font-size: 13px;
}

QTextBrowser#ChatText pre {
    background-color: %(code_bg)s;
    border: 1px solid %(border)s;
    border-radius: 8px;
    padding: 12px;
    font-family: "Consolas", "Monaco", monospace;
    font-size: 13px;
}

QFrame#InputContainer {
    background-color: %(surface)s;
    border: 1px solid %(border)s;
    border-radius: 12px;
}

QTextEdit#InputBox {
    background-color: %(surface2)s;
    border: 1px solid %(border)s;
    border-radius: 8px;
    color: %(text)s;
    font-size: 14px;
    padding: 8px 12px;
    selection-background-color: %(accent)s;
}

QTextEdit#InputBox:focus {
    border-color: %(accent)s;
}

QPushButton#SendButton {
    background-color: %(accent)s;
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    font-size: 13px;
    padding: 0px 16px;
}

QPushButton#SendButton:hover {
    background-color: #4a6cf7;
}

QPushButton#SendButton:disabled {
    background-color: %(muted)s;
    color: %(text_secondary)s;
}

QPushButton#StopButton {
    background-color: transparent;
    color: %(red)s;
    border: 1px solid %(red)s;
    border-radius: 8px;
    font-weight: 600;
    font-size: 13px;
    padding: 0px 16px;
}

QPushButton#StopButton:hover {
    background-color: rgba(207, 34, 46, 0.1);
}

QPushButton#StopButton:disabled {
    border-color: %(muted)s;
    color: %(muted)s;
}

QPushButton#ClearButton {
    background-color: transparent;
    color: %(text_secondary)s;
    border: 1px solid %(border)s;
    border-radius: 8px;
    font-weight: 500;
    font-size: 13px;
    padding: 0px 16px;
}

QPushButton#ClearButton:hover {
    background-color: %(surface2)s;
    color: %(text)s;
}

QPushButton#QuickButton {
    background-color: %(surface2)s;
    color: %(text)s;
    border: 1px solid %(border)s;
    border-radius: 8px;
    font-size: 13px;
    padding: 8px 12px;
    text-align: left;
}

QPushButton#QuickButton:hover {
    background-color: %(border)s;
    border-color: %(accent)s;
}

QPushButton#Chip {
    background-color: %(surface2)s;
    color: %(text_secondary)s;
    border: 1px solid %(border)s;
    border-radius: 14px;
    font-size: 12px;
    padding: 6px 14px;
}

QPushButton#Chip:hover {
    background-color: %(surface)s;
    color: %(accent)s;
    border-color: %(accent)s;
}

QScrollBar:vertical {
    background-color: transparent;
    width: 8px;
    margin: 0;
}

QScrollBar::handle:vertical {
    background-color: %(border)s;
    border-radius: 4px;
    min-height: 30px;
}

QScrollBar::handle:vertical:hover {
    background-color: %(muted)s;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}

QSplitter::handle {
    background-color: %(border)s;
}

QToolTip {
    background-color: %(surface)s;
    color: %(text)s;
    border: 1px solid %(border)s;
    border-radius: 4px;
    padding: 6px 10px;
}
""" % COLORS