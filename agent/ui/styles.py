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
    text-decoration: underline;
}

QTextBrowser#ChatText .md-body {
    color: %(text)s;
    line-height: 1.7;
}

QTextBrowser#ChatText .md-body h1,
QTextBrowser#ChatText .md-body h2 {
    font-size: 18px;
    font-weight: 700;
    margin: 12px 0 8px 0;
    padding-bottom: 6px;
    border-bottom: 1px solid %(border)s;
    color: %(text)s;
}

QTextBrowser#ChatText .md-body h3 {
    font-size: 16px;
    font-weight: 700;
    margin: 10px 0 6px 0;
    color: %(text)s;
}

QTextBrowser#ChatText .md-body h4,
QTextBrowser#ChatText .md-body h5,
QTextBrowser#ChatText .md-body h6 {
    font-size: 14px;
    font-weight: 600;
    margin: 8px 0 4px 0;
    color: %(text_secondary)s;
}

QTextBrowser#ChatText .md-body p {
    margin: 6px 0;
}

QTextBrowser#ChatText .md-body ul,
QTextBrowser#ChatText .md-body ol {
    margin: 6px 0;
    padding-left: 24px;
}

QTextBrowser#ChatText .md-body li {
    margin: 3px 0;
}

QTextBrowser#ChatText .md-body blockquote {
    margin: 8px 0;
    padding: 6px 12px;
    border-left: 3px solid %(accent)s;
    background-color: %(surface2)s;
    color: %(text_secondary)s;
    border-radius: 0 6px 6px 0;
}

QTextBrowser#ChatText .md-body code {
    background-color: %(code_bg)s;
    color: #24292f;
    border-radius: 4px;
    padding: 2px 6px;
    font-family: "Consolas", "Monaco", monospace;
    font-size: 13px;
}

QTextBrowser#ChatText .md-body pre {
    background-color: %(code_bg)s;
    border: 1px solid %(border)s;
    border-radius: 8px;
    padding: 12px;
    margin: 8px 0;
    position: relative;
    font-family: "Consolas", "Monaco", monospace;
    font-size: 13px;
    white-space: pre-wrap;
    word-wrap: break-word;
}

QTextBrowser#ChatText .md-body pre code {
    background-color: transparent;
    padding: 0;
    border-radius: 0;
    border: none;
}

QTextBrowser#ChatText .md-body table {
    border-collapse: collapse;
    margin: 8px 0;
    width: 100%%;
}

QTextBrowser#ChatText .md-body th {
    background-color: %(surface2)s;
    border: 1px solid %(border)s;
    padding: 8px 12px;
    font-weight: 600;
    color: %(text)s;
    text-align: left;
}

QTextBrowser#ChatText .md-body td {
    border: 1px solid %(border)s;
    padding: 6px 12px;
}

QTextBrowser#ChatText .md-body hr {
    border: none;
    border-top: 1px solid %(border)s;
    margin: 12px 0;
}

QTextBrowser#ChatText .md-body strong {
    font-weight: 700;
    color: %(text)s;
}

QTextBrowser#ChatText .md-body em {
    font-style: italic;
}

QTextBrowser#ChatText .md-body del {
    color: %(muted)s;
}

QTextBrowser#ChatText .streaming-token {
    background-color: rgba(74, 108, 247, 0.15);
    border-radius: 3px;
    padding: 0 2px;
    animation: none;
}

QTextBrowser#ChatText::selection {
    background-color: rgba(74, 108, 247, 0.3);
}

QTextBrowser#ChatText tool-block:hover {
    background-color: %(surface2)s;
}

QFrame#InputContainer {
    background-color: %(surface)s;
    border: 1px solid %(border)s;
    border-radius: 16px;
}

QTextEdit#InputBox {
    background-color: transparent;
    border: none;
    color: %(text)s;
    font-size: 14px;
    padding: 4px 2px;
    selection-background-color: rgba(74, 108, 247, 0.3);
}

QTextEdit#InputBox:focus {
    border: none;
}

QPushButton#SendButton {
    background-color: %(accent)s;
    color: white;
    border: none;
    border-radius: 17px;
    font-size: 16px;
    font-weight: 700;
}

QPushButton#SendButton:hover {
    background-color: #3a5ce5;
}

QPushButton#SendButton:disabled {
    background-color: %(muted)s;
    color: %(text_secondary)s;
}

QPushButton#StopButton {
    background-color: %(red)s;
    color: white;
    border: none;
    border-radius: 17px;
    font-size: 14px;
    font-weight: 700;
}

QPushButton#StopButton:hover {
    background-color: #b91c1c;
}

QPushButton#StopButton:disabled {
    background-color: %(muted)s;
}

QPushButton#ClearButton {
    background-color: transparent;
    color: %(text_secondary)s;
    border: 1px solid %(border)s;
    border-radius: 14px;
    font-size: 12px;
    padding: 0px 14px;
}

QPushButton#ClearButton:hover {
    background-color: %(surface2)s;
    color: %(accent)s;
    border-color: %(accent)s;
}

QPushButton#FileButton {
    background-color: transparent;
    color: %(text_secondary)s;
    border: 1px solid %(border)s;
    border-radius: 14px;
    font-size: 14px;
}

QPushButton#FileButton:hover {
    background-color: %(accent)s;
    color: white;
    border-color: %(accent)s;
}

QFrame#FileChip {
    background-color: %(surface2)s;
    border: 1px solid %(border)s;
    border-radius: 12px;
}

QLabel#ChipLabel {
    font-size: 12px;
    color: %(text)s;
    max-width: 120px;
}

QPushButton#ChipClose {
    background-color: transparent;
    color: %(muted)s;
    border: none;
    border-radius: 10px;
    font-size: 14px;
    font-weight: 700;
}

QPushButton#ChipClose:hover {
    background-color: %(red)s;
    color: white;
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