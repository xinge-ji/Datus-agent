"""
字符串处理工具函数

从 datus/cli/import_view.py 迁移和整合的通用字符串处理功能。
"""

import re
from typing import Any


def to_snake(name: str) -> str:
    """
    将字符串转换为蛇形命名 (snake_case)。

    从 import_view.py._to_snake 迁移而来。

    Args:
        name: 输入字符串，可以是驼峰命名、带空格或连字符

    Returns:
        转换后的蛇形命名字符串

    Examples:
        >>> to_snake("UserName")
        'user_name'
        >>> to_snake("user-name")
        'user_name'
        >>> to_snake("User Name")
        'user_name'
    """
    if not name:
        return ""

    out = []
    prev_lower = False
    for ch in name:
        if ch.isupper() and prev_lower:
            out.append("_")
        if ch in "- ":
            out.append("_")
            prev_lower = False
            continue
        out.append(ch.lower())
        prev_lower = ch.islower()
    return "".join(out).strip("_")


def strip_ansi(text: str) -> str:
    """
    移除字符串中的ANSI转义字符。

    从 import_view.py._strip_ansi 迁移而来。

    Args:
        text: 包含ANSI转义字符的字符串

    Returns:
        移除ANSI转义字符后的纯净字符串

    Examples:
        >>> strip_ansi("Hello \\x1b[31mWorld\\x1b[0m")
        'Hello World'
    """
    if not text:
        return ""

    ansi_re = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]", re.IGNORECASE)
    return ansi_re.sub("", text)