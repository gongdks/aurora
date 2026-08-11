"""日期时间查询工具。

纯标准库，无外部依赖。
"""

import re
from datetime import datetime, timedelta

from langchain.tools import tool

from agent.tools.registry import register


@register
@tool
def datetime_query(question: str) -> str:
    """查询当前日期、时间、星期或计算日期间隔。

    常见用法：
    - "今天几号" → 返回当前日期和星期
    - "现在是几点" → 返回当前时间
    - "3天后是几号" → 返回计算后的日期
    - "从2026-01-01到现在过了多少天" → 返回天数

    Args:
        question: 关于日期时间的自然语言问题
    """
    try:
        now = datetime.now()
        today_str = now.strftime("%Y-%m-%d")
        weekday_cn = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"][now.weekday()]

        base = (
            f"当前时间：{now.strftime('%Y-%m-%d %H:%M:%S')}，{weekday_cn}\n"
            f"当前日期：{today_str}\n"
            f"ISO 周数：第 {now.isocalendar()[1]} 周\n"
        )

        # 尝试解析问题中的常见日期间隔

        # "N天后" / "N天后是几号"
        match = re.search(r"(\d+)\s*天后", question)
        if match:
            days = int(match.group(1))
            future = now + timedelta(days=days)
            return (
                base
                + f"计算：{days} 天后是 {future.strftime('%Y-%m-%d')}，"
                + ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"][future.weekday()]
            )

        # "N天前"
        match = re.search(r"(\d+)\s*天前", question)
        if match:
            days = int(match.group(1))
            past = now - timedelta(days=days)
            return (
                base
                + f"计算：{days} 天前是 {past.strftime('%Y-%m-%d')}，"
                + ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"][past.weekday()]
            )

        return base
    except (ValueError, OverflowError) as exc:
        return f"日期计算失败：{exc}。请提供合法的日期描述。"
