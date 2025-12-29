from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from sqlglot import parse_one
except Exception:  # pragma: no cover
    parse_one = None

from datus.configuration.agent_config import AgentConfig
from datus.tools.db_tools.db_manager import DBManager
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


def _escape(val: Optional[str]) -> str:
    return (val or "").replace("'", "''")


def _rows_from_result(result: Any) -> List[Dict[str, Any]]:
    if not result:
        return []
    data = getattr(result, "data", None)
    if data is None:
        data = getattr(result, "raw", None)
    if isinstance(data, list):
        if data and isinstance(data[0], dict):
            return data
        if data and isinstance(data[0], (tuple, list)):
            headers = [f"col{i}" for i in range(len(data[0]))]
            return [dict(zip(headers, row)) for row in data]
    if isinstance(data, dict):
        return [data]
    return []


def _resolve_meta_conn(agent_config: AgentConfig, db_manager: DBManager, args) -> Tuple[Any, str]:
    logic_db = getattr(args, "database", "") or agent_config.current_database or ""
    conn = db_manager.get_conn(agent_config.current_namespace, logic_db)
    return conn, logic_db


def _layer_from_name(name: str) -> str:
    low = name.lower()
    if low.startswith("dim_") or low.startswith("dim."):
        return "DIM"
    if low.startswith("dwd_") or low.startswith("dwd."):
        return "DWD"
    if low.startswith("dws_") or low.startswith("dws."):
        return "DWS"
    return "DWD"


def _find_existing_model(conn: Any, table_name: str) -> Optional[int]:
    sql = (
        "SELECT model_id FROM dw_meta.dw_model "
        f"WHERE table_name = '{_escape(table_name)}' "
        "ORDER BY model_id DESC LIMIT 1"
    )
    res = conn.execute_query(sql)
    if not getattr(res, "success", True):
        return None
    rows = _rows_from_result(res)
    if rows:
        return int(rows[0].get("model_id"))
    return None


def _upsert_model(conn: Any, db_name: str, table_name: str, layer: str, origin: str, sync_direction: str) -> int:
    now = "CURRENT_TIMESTAMP"
    existing = _find_existing_model(conn, table_name)
    if existing:
        sql = (
            "UPDATE dw_meta.dw_model SET "
            f"db_name='{_escape(db_name)}', table_name='{_escape(table_name)}', model_name='{_escape(table_name)}', "
            f"layer='{_escape(layer)}', origin='{origin}', sync_direction='{sync_direction}', updated_at={now} "
            f"WHERE model_id={existing}"
        )
        res = conn.execute_update(sql)
        if not getattr(res, "success", True):
            logger.warning(f"更新 dw_model 失败 model_id={existing}, err={getattr(res, 'error', '')}")
        return existing
    sql = (
        "INSERT INTO dw_meta.dw_model "
        "(model_name, db_name, table_name, layer, origin, sync_direction, status, created_at, updated_at) "
        f"VALUES ('{_escape(table_name)}','{_escape(db_name)}','{_escape(table_name)}','{_escape(layer)}',"
        f"'{origin}','{sync_direction}','DRAFT',{now},{now})"
    )
    res = conn.execute_insert(sql)
    if not getattr(res, "success", True):
        raise RuntimeError(f"写 dw_model 失败 {table_name}, err={getattr(res, 'error', '')}")
    if getattr(res, "lastrowid", None):
        try:
            return int(res.lastrowid)
        except Exception:
            pass
    return _find_existing_model(conn, table_name) or 0


def _insert_columns(conn: Any, model_id: int, columns: List[Dict[str, Any]]) -> int:
    if not columns:
        return 0
    now = "CURRENT_TIMESTAMP"
    # 先删旧列
    conn.execute_delete(f"DELETE FROM dw_meta.dw_model_column WHERE model_id = {model_id}")
    inserted = 0
    for idx, col in enumerate(columns):
        col_name = col.get("column_name") or f"col_{idx}"
        expr = col.get("expression_sql") or col_name
        sql = (
            "INSERT INTO dw_meta.dw_model_column "
            "(model_id, column_name, column_order, std_field_id, expression_sql, "
            "is_primary_key, is_partition_key, is_distributed_key, not_null, comment, is_active, created_at, updated_at) "
            f"VALUES ({model_id}, '{_escape(col_name.lower())}', {idx}, NULL, '{_escape(expr)}', "
            f"0, 0, 0, 0, NULL, 1, {now}, {now})"
        )
        res = conn.execute_insert(sql)
        if getattr(res, "success", False):
            inserted += 1
    return inserted


def _collect_sql_files(path: Path) -> List[Path]:
    if path.is_file() and path.suffix.lower() == ".sql":
        return [path]
    files: List[Path] = []
    for root, _, filenames in os.walk(path):
        for name in filenames:
            if name.lower().endswith(".sql"):
                files.append(Path(root) / name)
    return files


def _parse_columns(sql_text: str) -> List[Dict[str, Any]]:
    if not parse_one:
        return []
    try:
        expr = parse_one(sql_text)
    except Exception:
        return []
    selects = expr.find_all("select")
    cols: List[Dict[str, Any]] = []
    for sel in selects:
        try:
            for proj in sel.expressions:
                alias = proj.alias_or_name
                cols.append({"column_name": alias or str(proj), "expression_sql": str(proj)})
        except Exception:
            continue
        break
    return cols


def run_import_sqlmesh(agent_config: AgentConfig, db_manager: DBManager, args) -> Dict[str, Any]:
    """
    解析 sqlmesh 模型文件并同步元数据（轻量占位实现，未解析 SELECT 字段）。
    """
    conn, _ = _resolve_meta_conn(agent_config, db_manager, args)
    path = Path(getattr(args, "path", ""))
    if not path.exists():
        return {"status": "error", "message": f"path not found: {path}"}
    files = _collect_sql_files(path)
    if not files:
        return {"status": "success", "message": "no sql files discovered"}
    apply = bool(getattr(args, "apply", False))
    layer_hint = getattr(args, "layer", None)

    imported = []
    for file in files:
        table_name = file.stem.replace("__", ".")
        layer = layer_hint or _layer_from_name(table_name)
        info = {"file": str(file), "model": table_name, "layer": layer}
        imported.append(info)
        if not apply:
            continue
        try:
            model_id = _upsert_model(conn, "", table_name, layer, origin="MANUAL", sync_direction="CODE_TO_META")
            cols = _parse_columns(file.read_text(encoding="utf-8"))
            if cols:
                _insert_columns(conn, model_id, cols)
        except Exception as exc:
            logger.warning(f"导入 {file} 失败: {exc}")

    msg = "dry-run" if not apply else "applied"
    logger.info(f"import-sqlmesh 完成，文件数={len(files)}，模式={msg}")
    return {"status": "success", "mode": msg, "models": imported}


if __name__ == "__main__":  # pragma: no cover
    print("请通过 datus-agent import-sqlmesh 调用本模块")
