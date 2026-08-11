"""共享 JSON 提取器 —— 从 LLM 响应中稳健提取 JSON。

支持多种格式：
1. Markdown 代码块 ```json ... ```
2. 普通 ``` ... ``` 代码块
3. 直接嵌入的 JSON 对象
"""

import json
import re
from typing import Any


def extract_json(text: str) -> dict[str, Any] | list[Any]:
    """从 LLM 响应文本中提取并反序列化 JSON。

    Args:
        text: LLM 原始响应文本

    Returns:
        解析后的 dict 或 list

    Raises:
        ValueError: 无法提取 JSON 或 JSON 无效
    """
    if not text or not text.strip():
        raise ValueError("响应为空")

    # 策略 1：Markdown json 代码块
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if m:
        json_str = m.group(1).strip()
        return _try_load(json_str, text)

    # 策略 2：查找最外层 JSON 对象或数组
    m = re.search(r"\{[\s\S]*\}|\[[\s\S]*\]", text)
    if m:
        json_str = m.group(0)
        return _try_load(json_str, text)

    raise ValueError(f"无法从响应中提取 JSON: {text[:200]}")


def extract_json_safe(text: str, default: Any = None) -> dict[str, Any]:
    """安全版本：解析失败时返回默认值而非抛异常。"""
    try:
        result = extract_json(text)
        if isinstance(result, dict):
            return result
        return {"result": result}
    except Exception:
        return default if default is not None else {}


def _try_load(json_str: str, original_text: str) -> Any:
    """尝试加载 JSON，失败时抛出带上下文的错误。"""
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"JSON 解析失败: {exc}\n"
            f"尝试解析的内容: {json_str[:200]}\n"
            f"原始响应: {original_text[:200]}"
        ) from exc