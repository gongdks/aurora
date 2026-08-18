"""Browser automation tool — Playwright-based web interaction.

Supports: navigate, click, type, press_key, snapshot, screenshot, list_interactive, close.

Architecture:
  - BrowserSession class encapsulates all browser state (no module-level globals)
  - Module-level singleton for backward compatibility with @tool functions
  - Thread-safe via per-instance lock
  - Testable: instantiate BrowserSession directly in tests
"""

from __future__ import annotations

import atexit
import logging
import os
import threading
import time
from typing import Any

from langchain.tools import tool

from agent.config import settings
from agent.tools.registry import register
from agent.utils.path_guard import safe_resolve

logger = logging.getLogger(__name__)

# Timeout constants
_NAVIGATION_TIMEOUT = 30_000   # ms
_ELEMENT_TIMEOUT = 10_000      # ms
_MAX_SNAPSHOT_CHARS = 8_000
_MAX_SNAPSHOT_ELEMENTS = 100


def _build_launch_args(headless: bool) -> list[str]:
    """Build browser launch args based on headless mode."""
    args = [
        "--disable-blink-features=AutomationControlled",
        "--no-first-run",
        "--no-default-browser-check",
    ]
    if headless:
        args.append("--headless=new")
        args.append("--disable-gpu")
    return args


class BrowserSession:
    """Encapsulates Playwright browser lifecycle and page interactions.

    All browser state is instance-local — no module-level globals.
    Use `get_session()` for the module singleton, or instantiate directly for testing.
    """

    def __init__(self, headless: bool | None = None) -> None:
        self._lock = threading.Lock()
        self._browser: Any = None
        self._context: Any = None
        self._page: Any = None
        self._playwright_lib: Any = None
        self._headless = (
            headless if headless is not None
            else getattr(settings, "BROWSER_HEADLESS", False)
        )
        self._started = False

    # ---- Lazy playwright import ----

    def _get_playwright(self) -> Any:
        if self._playwright_lib is None:
            from playwright.sync_api import sync_playwright
            self._playwright_lib = sync_playwright
        return self._playwright_lib

    # ---- Page lifecycle ----

    def ensure_page(self) -> tuple[Any, str | None]:
        """Ensure browser and page exist. Returns (page, error_message)."""
        if self._page is not None:
            try:
                self._page.title()
                if not self._headless:
                    self._page.bring_to_front()
                return self._page, None
            except Exception:
                self._cleanup_internal()
                return self.ensure_page()

        pw = self._get_playwright()
        playwright_instance = pw().start()

        browser = None
        launch_method = "Edge"
        error_msgs: list[str] = []

        try:
            browser = playwright_instance.chromium.launch(
                channel="msedge",
                headless=self._headless,
                args=_build_launch_args(self._headless),
            )
            launch_method = "Edge"
        except Exception as edge_err:
            error_msgs.append(f"Edge: {edge_err}")
            logger.info("[Browser] Edge not available, trying Chromium...")
            try:
                browser = playwright_instance.chromium.launch(
                    headless=self._headless,
                    args=_build_launch_args(self._headless),
                )
                launch_method = "Chromium"
            except Exception as chromium_err:
                error_msgs.append(f"Chromium: {chromium_err}")
                return None, (
                    "Browser launch failed.\n"
                    f"- Edge: {error_msgs[0]}\n"
                    f"- Chromium: {error_msgs[-1]}\n"
                    "Ensure Edge is installed or run: python -m playwright install chromium"
                )

        context = browser.new_context(
            viewport={"width": 1280, "height": 720},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
        )
        page = context.new_page()
        page.set_default_timeout(_NAVIGATION_TIMEOUT)

        self._browser = browser
        self._context = context
        self._page = page
        self._started = True

        if not self._headless:
            page.bring_to_front()

        logger.info("[Browser] Browser started (%s, headless=%s)", launch_method, self._headless)
        return self._page, None

    def _cleanup_internal(self) -> None:
        """Internal cleanup without lock."""
        try:
            if self._page:
                self._page.close()
        except Exception:
            pass
        try:
            if self._context:
                self._context.close()
        except Exception:
            pass
        try:
            if self._browser:
                self._browser.close()
        except Exception:
            pass

        self._page = None
        self._context = None
        self._browser = None

    def close(self) -> str:
        """Close browser and release resources."""
        with self._lock:
            self._cleanup_internal()
            return "✅ Browser closed"

    def cleanup_atexit(self) -> None:
        """Called at process exit to prevent orphan browser processes."""
        try:
            if self._page:
                self._page.close()
        except Exception:
            pass
        try:
            if self._context:
                self._context.close()
        except Exception:
            pass
        try:
            if self._browser:
                self._browser.close()
        except Exception:
            pass
        self._page = None
        self._context = None
        self._browser = None

    # ---- Content extraction ----

    def extract_page_text(self, page: Any = None) -> str:
        """Extract readable text from a page (accessibility tree or body text)."""
        p = page or self._page
        if p is None:
            return "(no page)"
        try:
            snapshot = p.accessibility.snapshot()
            if snapshot:
                lines: list[str] = []
                self._walk_accessibility(snapshot, lines, depth=0)
                if lines:
                    return "\n".join(lines[:_MAX_SNAPSHOT_ELEMENTS])
        except Exception:
            pass
        try:
            text = p.inner_text("body")
            if text:
                return text[:_MAX_SNAPSHOT_CHARS]
        except Exception:
            pass
        return "(unable to extract page content)"

    @staticmethod
    def _walk_accessibility(node: dict, lines: list[str], depth: int) -> None:
        """Recursively walk accessibility tree to extract text."""
        if len(lines) >= _MAX_SNAPSHOT_ELEMENTS:
            return
        role = node.get("role", "unknown")
        name = (node.get("name") or "").strip()
        value = (node.get("value") or "").strip()
        indent = "  " * min(depth, 6)

        if role == "heading":
            level = node.get("level", 1)
            prefix = "#" * min(level, 4)
            lines.append(f"{indent}{prefix} {name}")
        elif role == "link":
            lines.append(f"{indent}🔗 {name} → {value}")
        elif role == "button":
            lines.append(f"{indent}[Button] {name}")
        elif role in ("textbox", "searchbox"):
            current = f' (value: "{value}")' if value else ""
            lines.append(f"{indent}[Input] {name or ''}{current}")
        elif role == "combobox":
            lines.append(f"{indent}[Dropdown] {name}: {value}")
        elif role == "checkbox":
            checked = "☑" if value == "true" else "☐"
            lines.append(f"{indent}{checked} {name}")
        elif role == "listitem":
            lines.append(f"{indent}• {name}")
        elif role == "image":
            if name:
                lines.append(f"{indent}[Image] {name}")
        elif role == "StaticText":
            if name:
                lines.append(f"{indent}{name}")
        elif role == "paragraph":
            if name:
                lines.append(f"{indent}{name}")
        elif name and role not in ("none", "generic", "group", "document", "application"):
            lines.append(f"{indent}[{role}] {name}")

        for child in node.get("children", []):
            BrowserSession._walk_accessibility(child, lines, depth + 1)

    # ---- Interactive element listing ----

    def list_interactive_elements(self, page: Any = None) -> str:
        """List clickable/typable elements on a page."""
        p = page or self._page
        if p is None:
            return "(no page)"
        lines: list[str] = []
        try:
            buttons = p.locator("button, [role='button'], input[type='submit']").all()
            for btn in buttons[:15]:
                try:
                    name = (btn.inner_text() or btn.get_attribute("aria-label")
                            or btn.get_attribute("value") or "").strip()
                    if name:
                        lines.append(f"  [Button] {name[:60]}")
                except Exception:
                    pass
            links = p.locator("a[href]").all()
            for link in links[:15]:
                try:
                    name = (link.inner_text() or "").strip()
                    href = (link.get_attribute("href") or "")[:60]
                    if name:
                        lines.append(f"  [Link] {name[:50]} → {href}")
                except Exception:
                    pass
        except Exception:
            pass
        if not lines:
            return "  (no interactive elements detected)"
        return "\n".join(lines)

    def list_input_elements(self, page: Any = None) -> str:
        """List input elements on a page."""
        p = page or self._page
        if p is None:
            return "(no page)"
        lines: list[str] = []
        try:
            inputs = p.locator("input, textarea, [role='textbox'], [role='searchbox']").all()
            for inp in inputs[:15]:
                try:
                    name = (inp.get_attribute("name") or inp.get_attribute("placeholder")
                            or inp.get_attribute("aria-label") or inp.get_attribute("id") or "").strip()
                    inp_type = (inp.get_attribute("type") or "text").strip()
                    if name:
                        lines.append(f"  [{inp_type}] {name[:60]}")
                except Exception:
                    pass
        except Exception:
            pass
        if not lines:
            return "  (no input elements detected)"
        return "\n".join(lines)

    @property
    def page(self) -> Any:
        return self._page

    @property
    def is_started(self) -> bool:
        return self._started


# ---- Module-level singleton ----
_session: BrowserSession | None = None
_session_lock = threading.Lock()


def get_session() -> BrowserSession:
    """Get or create the module-level BrowserSession singleton."""
    global _session
    if _session is None:
        with _session_lock:
            if _session is None:
                _session = BrowserSession()
                atexit.register(_session.cleanup_atexit)
    return _session


def reset_session() -> None:
    """Reset the singleton session (useful for testing)."""
    global _session
    with _session_lock:
        if _session is not None:
            _session.close()
        _session = None


# ---- Tool functions (thin wrappers around BrowserSession) ----


def _fmt_size(size: int) -> str:
    if size < 1024:
        return f"{size}B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f}KB"
    return f"{size / 1024 / 1024:.1f}MB"


@register
@tool
def browser_navigate(url: str) -> str:
    """Open a webpage and extract its content.

    This is the first step in browser interaction. The page content is
    automatically extracted after navigation. Use browser_click, browser_type
    etc. for further interaction.

    Args:
        url: URL to visit (https:// auto-prepended if missing), e.g. "baidu.com"
    """
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    session = get_session()
    with session._lock:
        page, error = session.ensure_page()
        if error:
            return f"❌ {error}"

        try:
            logger.info("[Browser] Navigating to: %s", url)
            response = page.goto(url, wait_until="domcontentloaded", timeout=_NAVIGATION_TIMEOUT)

            try:
                page.wait_for_load_state("networkidle", timeout=10_000)
            except Exception:
                pass

            status = response.status if response else "unknown"
            title = page.title()
            url_current = page.url
            text = session.extract_page_text()

            header = (
                f"📄 {title}\n"
                f"🔗 {url_current}\n"
                f"📊 HTTP {status}\n"
            )
            if not text.strip():
                header += "\n⚠️ Page content is empty or could not be extracted (JS-rendered page?)."
            else:
                header += "\n--- Page Content ---\n"

            return (header + text)[:_MAX_SNAPSHOT_CHARS + 500]

        except Exception as exc:
            logger.error("[Browser] Navigation failed: %s", exc)
            return f"❌ Navigation failed: {exc}"


@register
@tool
def browser_click(selector: str) -> str:
    """Click an element on the current page.

    Supports CSS selectors ("#submit", ".btn", "button"),
    text matching ("text=Login", "text=Submit"),
    and role selectors ("role=button[name='Search']").

    Args:
        selector: Element selector, e.g. "#login-btn", "text=Confirm", "role=button"
    """
    session = get_session()
    with session._lock:
        page, error = session.ensure_page()
        if error:
            return f"❌ {error}"

        try:
            logger.info("[Browser] Click: %s", selector)

            if not selector.startswith(("text=", "role=", "#", ".", "//", "xpath=")):
                text_selector = f"text={selector}"
                if page.locator(text_selector).count() > 0:
                    selector = text_selector

            element = page.locator(selector).first
            element.wait_for(state="attached", timeout=_ELEMENT_TIMEOUT)

            tag = element.evaluate("el => el.tagName.toLowerCase()") or "element"
            text = (element.inner_text() or "")[:50]

            if element.is_visible():
                element.click(timeout=_ELEMENT_TIMEOUT)
            else:
                logger.info("[Browser] Element not visible, using dispatchEvent")
                element.evaluate("el => el.dispatchEvent(new Event('click', {bubbles: true}))")

            try:
                page.wait_for_load_state("networkidle", timeout=8_000)
            except Exception:
                pass
            time.sleep(0.5)

            snapshot = session.extract_page_text()
            return (
                f"✅ Clicked <{tag}>: \"{text}\"\n"
                f"📄 Current page: {page.title()}\n"
                f"🔗 {page.url}\n\n"
                f"{snapshot}"
            )[:_MAX_SNAPSHOT_CHARS + 200]

        except Exception as exc:
            logger.error("[Browser] Click failed: %s", exc)
            hint = session.list_interactive_elements(page)
            return f"❌ Click failed: {exc}\n\nInteractive elements on page:\n{hint}"


@register
@tool
def browser_type(selector: str, text: str) -> str:
    """Type text into an input field.

    Clears the field first, then types the text. Handles both visible and
    hidden (React/Vue controlled) inputs automatically.

    Args:
        selector: Input field selector, e.g. "#search-input", "input[name='q']"
        text: Text to type
    """
    session = get_session()
    with session._lock:
        page, error = session.ensure_page()
        if error:
            return f"❌ {error}"

        try:
            logger.info("[Browser] Type: '%s' → %s", text, selector)

            element = page.locator(selector).first
            element.wait_for(state="attached", timeout=_ELEMENT_TIMEOUT)

            if not element.is_visible():
                logger.info("[Browser] Input not visible, using native setter + events")
                element.evaluate("""
                    (el, text) => {
                        const nativeSetter = Object.getOwnPropertyDescriptor(
                            window.HTMLInputElement.prototype, 'value'
                        ).set;
                        nativeSetter.call(el, '');
                        el.dispatchEvent(new Event('input', { bubbles: true }));
                        nativeSetter.call(el, text);
                        el.dispatchEvent(new Event('input', { bubbles: true }));
                        el.dispatchEvent(new Event('change', { bubbles: true }));
                    }
                """, text)
            else:
                element.click()
                try:
                    element.fill("", timeout=3_000)
                except Exception:
                    element.click()
                    element.press("Control+a")
                element.type(text, delay=20)

            return (
                f"✅ Typed \"{text}\" into {selector}\n"
                f"💡 To submit, use browser_click on the search/submit button, "
                f"or browser_press_key('{selector}', 'Enter')"
            )

        except Exception as exc:
            logger.error("[Browser] Type failed: %s", exc)
            hint = session.list_input_elements(page)
            return f"❌ Type failed: {exc}\n\nInput elements on page:\n{hint}"


@register
@tool
def browser_press_key(selector: str, key: str) -> str:
    """Press a keyboard key on an element.

    Use for submitting forms (Enter), canceling (Escape), etc.

    Args:
        selector: Target element selector
        key: Key name, e.g. "Enter", "Escape", "Tab", "ArrowDown"
    """
    session = get_session()
    with session._lock:
        page, error = session.ensure_page()
        if error:
            return f"❌ {error}"

        try:
            element = page.locator(selector).first
            element.wait_for(state="attached", timeout=_ELEMENT_TIMEOUT)

            if element.is_visible():
                element.press(key)
            else:
                if key == "Enter":
                    element.evaluate("""
                        el => {
                            const form = el.closest('form');
                            if (form && typeof form.requestSubmit === 'function') {
                                form.requestSubmit();
                            } else if (form) {
                                form.submit();
                            } else {
                                el.dispatchEvent(new KeyboardEvent('keydown', {key:'Enter',bubbles:true}));
                                el.dispatchEvent(new KeyboardEvent('keyup', {key:'Enter',bubbles:true}));
                            }
                        }
                    """)
                else:
                    element.evaluate("el => el.focus()")
                    page.keyboard.press(key)
            time.sleep(0.5)

            try:
                page.wait_for_load_state("networkidle", timeout=5_000)
            except Exception:
                pass

            snapshot = session.extract_page_text()
            return (
                f"✅ Pressed {key} on {selector}\n"
                f"📄 Current page: {page.title()}\n\n"
                f"{snapshot}"
            )[:_MAX_SNAPSHOT_CHARS + 200]

        except Exception as exc:
            logger.error("[Browser] Press key failed: %s", exc)
            return f"❌ Press key failed: {exc}"


@register
@tool
def browser_snapshot() -> str:
    """Get a text snapshot of the current page.

    Use to re-read page content after interactions, or check current state.
    Extracts page title, URL, and available text content.
    """
    session = get_session()
    with session._lock:
        page, error = session.ensure_page()
        if error:
            return f"❌ {error}"

        try:
            title = page.title()
            url = page.url
            text = session.extract_page_text()
            return f"📄 {title}\n🔗 {url}\n\n{text}"
        except Exception as exc:
            return f"❌ Snapshot failed: {exc}"


@register
@tool
def browser_list_interactive() -> str:
    """List all clickable and typable elements on the current page.

    Use when click or type operations fail, to find the correct selector.
    """
    session = get_session()
    with session._lock:
        page, error = session.ensure_page()
        if error:
            return f"❌ {error}"

        interactives = session.list_interactive_elements(page)
        inputs = session.list_input_elements(page)
        return (
            f"📄 {page.title()}\n\n"
            f"### Interactive Elements\n{interactives}\n\n"
            f"### Input Fields\n{inputs}"
        )


@register
@tool
def browser_screenshot(filename: str = "screenshot.png") -> str:
    """Save a screenshot of the current page.

    Screenshots are saved to the agent_workspace directory.

    Args:
        filename: Screenshot filename, default "screenshot.png". Supports .png/.jpg.
    """
    session = get_session()
    with session._lock:
        page, error = session.ensure_page()
        if error:
            return f"❌ {error}"

        try:
            safe = safe_resolve(filename, settings.FILE_READER_ROOT)
        except ValueError as exc:
            return f"Path error: {exc}"

        if not safe.lower().endswith((".png", ".jpg", ".jpeg")):
            safe += ".png"

        try:
            parent = os.path.dirname(safe)
            if parent:
                os.makedirs(parent, exist_ok=True)
            page.screenshot(path=safe, full_page=False)
            size = os.path.getsize(safe)
            return f"✅ Screenshot saved: {filename} ({_fmt_size(size)})"
        except Exception as exc:
            return f"❌ Screenshot failed: {exc}"


@register
@tool
def browser_close() -> str:
    """Close the browser session and free resources.

    Call this when browser tasks are complete.
    Subsequent browser_navigate will auto-restart the browser.
    """
    return get_session().close()