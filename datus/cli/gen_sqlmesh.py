from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


def _load_models(conn: Any, include_draft: bool, model_ids: List[str], model_names: List[str]) -> List[Dict[str, Any]]:
    filters = ["1=1"]
    status_filter = "status IN ('ACTIVE','APPROVED')" if not include_draft else "status IN ('ACTIVE','APPROVED','DRAFT')"
    filters.append(status_filter)
    if model_ids:
        id_sql = ",".join(model_ids)
        filters.append(f"model_id IN ({id_sql})")
    if model_names:
        name_sql = ",".join(f\"'{_escape(n)}'\" for n in model_names)
        filters.append(f"model_name IN ({name_sql})")
    where_sql = " AND ".join(filters)
    sql = (
        "SELECT model_id, model_name, db_name, table_name, layer, default_filter, primary_keys, source_node_id "
        "FROM dw_meta.dw_model "
        f"WHERE {where_sql}"
    )
    res = conn.execute_query(sql)
    if not getattr(res, "success", True):
        logger.error(f"查询 dw_model 失败: {getattr(res, 'error', '')}")
        return []
    rows = _rows_from_result(res)
    # 预先把 source_ref 占位，后面填充
    for row in rows:
        row["source_ref"] = None
    return rows


def _load_columns(conn: Any, model_id: int) -> List[Dict[str, Any]]:
    sql = (
        "SELECT column_name, expression_sql, column_order "
        "FROM dw_meta.dw_model_column "
        f"WHERE model_id = {model_id} AND is_active = 1 "
        "ORDER BY column_order"
    )
    res = conn.execute_query(sql)
    if not getattr(res, "success", True):
        logger.error(f"查询 dw_model_column 失败 model_id={model_id}: {getattr(res, 'error', '')}")
        return []
    return _rows_from_result(res)


def _render_model_sql(model: Dict[str, Any], columns: List[Dict[str, Any]]) -> str:
    name = model.get("model_name") or model.get("table_name")
    db_name = model.get("db_name") or ""
    full_name = f"{db_name}.{name}" if db_name else name
    layer = (model.get("layer") or "").upper()
    if layer == "DIM":
        kind = "FULL"
    elif layer == "DWD":
        kind = "INCREMENTAL BY TIME RANGE"
    else:
        kind = "VIEW"
    pk = (model.get("primary_keys") or "").split(",") if model.get("primary_keys") else []
    pk_clause = f"grain ({', '.join(pk)})" if pk else ""
    header_lines = [f"MODEL (", f"    name '{full_name}',", f"    kind {kind},"]
    if pk_clause:
        header_lines.append(f"    {pk_clause},")
    header_lines.append(");")
    header = "\n".join(header_lines)
    select_lines = []
    for col in columns:
        expr = col.get("expression_sql") or col.get("column_name")
        col_name = col.get("column_name") or ""
        select_lines.append(f"    {expr} AS {col_name}")
    if not select_lines:
        select_lines.append("    *")
    source_ref = model.get("source_ref") or "/* TODO: 填写上游引用 */ source_table"
    body = "SELECT\n" + ",\n".join(select_lines) + f"\nFROM {source_ref}"
    default_filter = model.get("default_filter")
    if default_filter:
        body += f"\nWHERE {default_filter}"
    body += ";\n"
    return f"{header}\n\n{body}"


def _ensure_dir(path: Path) -> None:
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)


def _populate_source_refs(conn: Any, models: List[Dict[str, Any]]) -> None:
    node_ids = [str(m.get("source_node_id")) for m in models if m.get("source_node_id")]
    if not node_ids:
        return
    sql = (
        "SELECT dn.node_id, ts.table_name, ts.source_system "
        "FROM dw_meta.dw_node dn "
        "JOIN dw_meta.table_source ts ON dn.source_table_id = ts.table_id "
        f"WHERE dn.node_id IN ({','.join(node_ids)})"
    )
    res = conn.execute_query(sql)
    if not getattr(res, "success", True):
        logger.debug(f"查询 source_ref 失败: {getattr(res, 'error', '')}")
        return
    rows = _rows_from_result(res)
    ref_map = {int(r.get("node_id")): r for r in rows if r.get("node_id") is not None}
    for m in models:
        node_id = m.get("source_node_id")
        if node_id and node_id in ref_map:
            tname = ref_map[node_id].get("table_name") or ""
            m["source_ref"] = tname


def run_gen_sqlmesh(agent_config: AgentConfig, db_manager: DBManager, args) -> Dict[str, Any]:
    """
    从 dw_model 生成简易 sqlmesh 模型文件（骨架），默认只生成 ACTIVE/APPROVED。
    """
    conn, _ = _resolve_meta_conn(agent_config, db_manager, args)
    model_ids = []
    if getattr(args, "model_ids", None):
        model_ids = [mid.strip() for mid in str(args.model_ids).replace(",", " ").split() if mid.strip().isdigit()]
    model_names = []
    if getattr(args, "model_names", None):
        model_names = [m.strip() for m in str(args.model_names).replace(",", " ").split() if m.strip()]
    include_draft = bool(getattr(args, "include_draft", False))
    dry_run = bool(getattr(args, "dry_run", False))
    output_dir = Path(getattr(args, "output_dir", "models"))

    models = _load_models(conn, include_draft, model_ids, model_names)
    if not models:
        return {"status": "success", "message": "no models"}

    _populate_source_refs(conn, models)
    rendered: Dict[str, str] = {}
    for m in models:
        cols = _load_columns(conn, int(m.get("model_id")))
        sql = _render_model_sql(m, cols)
        fname = f"{m.get('db_name') or 'default'}__{m.get('table_name') or m.get('model_name')}.sql"
        rendered[fname] = sql

    if dry_run:
        logger.info(f"即将生成 {len(rendered)} 个模型（dry-run），示例文件：{list(rendered.keys())[:3]}")
        return {"status": "success", "generated": list(rendered.keys())}

    _ensure_dir(output_dir)
    for fname, content in rendered.items():
        path = output_dir / fname
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
    logger.info(f"生成 sqlmesh 模型完成，共 {len(rendered)} 个，目录 {output_dir}")
    return {"status": "success", "generated": [str(output_dir / k) for k in rendered.keys()]}


if __name__ == "__main__":  # pragma: no cover
    print("请通过 datus-agent gen-sqlmesh 调用本模块")
