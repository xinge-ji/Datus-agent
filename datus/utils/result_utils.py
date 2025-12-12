"""
数据库结果处理工具函数

从 datus/cli/import_view.py 迁移和整合的结果处理功能。
"""

import csv
import json
from io import StringIO
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import pyarrow as pa
except ImportError:
    pa = None


def get_row_value(row: Any, keys: List[str], idx: Optional[int] = None) -> Any:
    """
    读取查询结果行的值，优先字段名，其次col{idx}，最后按位置索引。

    从 import_view.py._get_row_value 迁移而来。

    Args:
        row: 查询结果行，可能是dict、tuple、list等
        keys: 字段名列表，按优先级顺序查找
        idx: 可选的位置索引，作为后备方案

    Returns:
        找到的值，如果未找到则返回None

    Examples:
        >>> row = {"name": "Alice", "age": 25}
        >>> get_row_value(row, ["name", "username"])
        'Alice'
        >>> get_row_value(row, ["email"])
        None
    """
    if not isinstance(keys, (list, tuple)):
        keys = [keys]

    if isinstance(row, dict):
        # 精确匹配
        for k in keys:
            if k in row and row.get(k) is not None:
                return row.get(k)

        # 不区分大小写匹配
        lower_map = {str(k).lower(): v for k, v in row.items()}
        for k in keys:
            lk = str(k).lower()
            if lk in lower_map and lower_map[lk] is not None:
                return lower_map[lk]

        if idx is not None:
            col_key = f"col{idx}"
            if col_key in row:
                return row.get(col_key)

    if idx is not None:
        try:
            return row[idx]
        except Exception:
            return None

    return None


def parse_query_result(result: Any) -> List[Dict[str, Any]]:
    """
    解析数据库查询结果为统一的字典列表格式。

    从 import_view.py._rows_from_result 迁移而来。

    Args:
        result: 数据库查询结果，可能是多种格式

    Returns:
        统一的字典列表格式

    Examples:
        >>> result = ExecuteSQLResult(success=True, sql_return=[{"name": "Alice"}, {"name": "Bob"}])
        >>> parse_query_result(result)
        [{'name': 'Alice'}, {'name': 'Bob'}]
    """
    if not result or not getattr(result, "success", False):
        return []

    data = getattr(result, "sql_return", None)
    if data is None:
        return []

    # bytes/bytearray 先解码
    if isinstance(data, (bytes, bytearray)):
        try:
            data = data.decode("utf-8")
        except Exception:
            data = data.decode("utf-8", errors="ignore")

    # list[dict]/list[tuple] 直接兜底
    if isinstance(data, list):
        if data and isinstance(data[0], dict):
            return data
        if data and isinstance(data[0], (list, tuple)):
            return [{f"col{i}": v for i, v in enumerate(row)} for row in data]
        return []

    # pyarrow.Table
    try:
        if pa and hasattr(data, "to_pylist") and hasattr(data, "column_names"):
            rows_arrow = data.to_pylist()
            if rows_arrow:
                if isinstance(rows_arrow[0], dict):
                    return rows_arrow
                cols = list(getattr(data, "column_names") or [])
                if cols:
                    converted: List[Dict[str, Any]] = []
                    for row in rows_arrow:
                        if isinstance(row, (list, tuple)):
                            converted.append(dict(zip(cols, row)))
                        else:
                            converted.append({cols[0]: row} if cols else {"col0": row})
                    return converted
    except Exception:
        pass

    # pandas.DataFrame
    try:
        if pd and isinstance(data, pd.DataFrame):
            # 替换 NaN 为 None，转为 list[dict]
            return data.where(pd.notnull(data), None).to_dict('records')
    except Exception:
        pass

    # tuple 单行兜底
    if isinstance(data, tuple):
        return [{f"col{i}": v for i, v in enumerate(data)}]

    # 字符串解析
    if isinstance(data, str):
        text = data.lstrip("\ufeff").strip()
        if not text:
            return []

        # 优先尝试 JSON 解析
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                if parsed and isinstance(parsed[0], dict):
                    return parsed
                return []
            if isinstance(parsed, dict):
                return [parsed]
        except Exception:
            pass

        # 尝试 python literal 解析
        try:
            from ast import literal_eval
            parsed = literal_eval(text)
            if isinstance(parsed, list):
                if parsed and isinstance(parsed[0], dict):
                    return parsed
                return []
            if isinstance(parsed, dict):
                return [parsed]
        except Exception:
            pass

        # CSV 解析
        def _detect_delimiter(sample_text: str) -> str:
            try:
                sample = "\n".join([ln for ln in sample_text.splitlines() if ln.strip()][:5])
                return csv.Sniffer().sniff(sample).delimiter or ","
            except Exception:
                candidates = [",", "\t", "|", ";"]
                counts = {sep: sample_text.count(sep) for sep in candidates}
                return max(counts, key=counts.get) if counts else ","

        def _csv_to_rows(csv_text: str, delimiter: str) -> List[Dict[str, Any]]:
            reader = csv.DictReader(StringIO(csv_text, newline=""), delimiter=delimiter)
            return [dict(row) for row in reader if row]

        delimiter = _detect_delimiter(text)
        try:
            rows_csv = _csv_to_rows(text, delimiter)
            if rows_csv:
                return rows_csv
        except Exception:
            pass

        # 兜底：手动拆分
        lines = [ln for ln in text.splitlines() if ln.strip()]
        if len(lines) >= 2:
            try:
                headers = [p.strip() for p in lines[0].split(delimiter)]
                manual_rows: List[Dict[str, Any]] = []
                for ln in lines[1:]:
                    cols = [p.strip() for p in ln.split(delimiter)]
                    if len(cols) == len(headers):
                        manual_rows.append(dict(zip(headers, cols)))
                if manual_rows:
                    return manual_rows
            except Exception:
                pass

    return []