"""日期时间查询工具。

纯标准库，无外部依赖。
"""

import re
from datetime import datetime, timedelta

from langchain.tools import tool

from agent.tools.registry import register


@register(tags={"core"})
@tool
def datetime_query(question: str) -> str:
    """查询当前日期/时间或计算日期间隔。"""
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