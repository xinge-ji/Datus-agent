from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

from datus.configuration.agent_config import AgentConfig
from datus.tools.db_tools.db_manager import DBManager
from datus.utils.loggings import get_logger

try:
    from datus.models.base import LLMBaseModel
except Exception:  # pragma: no cover - 兜底
    LLMBaseModel = None

logger = get_logger(__name__)


@dataclass
class ViewCandidate:
    node_id: int
    source_table_id: Optional[int]
    table_id: Optional[int]
    source_system: str
    source_table_name: str
    view_name: str
    layer: str
    feature_json: str


def _escape(val: Optional[str]) -> str:
    return (val or "").replace("'", "''")


def _rows_from_result(result: Any) -> List[Dict[str, Any]]:
    if not result:
        return []
    # ExecuteSQLResult 场景
    data = getattr(result, "data", None)
    if data is None:
        data = getattr(result, "raw", None)
    if isinstance(data, list):
        if data and isinstance(data[0], dict):
            return data
        # sqlite/duckdb 可能返回 list[tuple]
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


def _parse_node_ids(raw: Optional[str]) -> List[int]:
    if not raw:
        return []
    ids: List[int] = []
    parts = raw.replace(",", " ").split()
    for part in parts:
        try:
            ids.append(int(part.strip()))
        except Exception:
            continue
    return ids


def _load_candidates(conn: Any, layer: Optional[str], node_ids: List[int], include_dws: bool) -> List[ViewCandidate]:
    filters: List[str] = [
        "ts.parse_status = 'PARSED'",
        "dn.node_type = 'SRC_VIEW'",
        "dn.human_layer_final IN ('DIM','DWD','DWS')",
    ]
    if layer:
        filters.append(f"dn.human_layer_final = '{_escape(layer)}'")
    if node_ids:
        id_list = ",".join(str(i) for i in node_ids)
        filters.append(f"dn.node_id IN ({id_list})")
    where_sql = " AND ".join(filters)
    sql = (
        "SELECT dn.node_id, dn.source_table_id, ts.table_id, ts.table_name AS view_name, "
        "ts.source_system, ts.table_name AS source_table_name, "
        "dn.human_layer_final AS layer, COALESCE(af.feature_json, '') AS feature_json "
        "FROM dw_meta.dw_node dn "
        "JOIN dw_meta.table_source ts ON dn.source_table_id = ts.table_id "
        "LEFT JOIN dw_meta.ai_view_feature af ON af.table_id = ts.table_id "
        f"WHERE {where_sql}"
    )
    result = conn.execute_query(sql)
    if not getattr(result, "success", True):
        logger.error(f"加载候选视图失败: {getattr(result, 'error', '')}")
        return []
    rows = _rows_from_result(result)
    candidates: List[ViewCandidate] = []
    for row in rows:
        layer_val = (row.get("layer") or "").upper()
        if layer_val == "DWS" and not include_dws:
            continue
        candidates.append(
            ViewCandidate(
                node_id=int(row.get("node_id")),
                source_table_id=row.get("source_table_id"),
                table_id=row.get("table_id"),
                source_system=(row.get("source_system") or "").lower(),
                source_table_name=(row.get("source_table_name") or "").lower(),
                view_name=row.get("view_name") or "",
                layer=layer_val,
                feature_json=row.get("feature_json") or "",
            )
        )
    return candidates


def _parse_feature(feature_json: str) -> Dict[str, Any]:
    try:
        return json.loads(feature_json) if feature_json else {}
    except Exception:
        logger.debug("feature_json 解析失败，使用空对象兜底")
        return {}


def _is_usable(feature: Dict[str, Any], strict: bool) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    cols = feature.get("columns") or []
    group_by = feature.get("group") or feature.get("group_by") or []
    case_when_count = feature.get("case_when_count") or 0
    coverage = 1.0 if cols else 0.0
    col_count = len(cols)
    gb_count = len(group_by)

    if strict:
        if coverage < 0.6:
            reasons.append("字段标准覆盖率不足（<0.6，宽松估计）")
        if gb_count > 12:
            reasons.append("GROUP BY 字段过多（>12）")
        if case_when_count and case_when_count > 3:
            reasons.append("CASE WHEN 复杂度过高（>3）")
        if col_count > 100:
            reasons.append("列数过多（>100）")
    usable = not reasons
    return usable, reasons


def _update_migration_status(conn: Any, node_id: int, status: str, msg: str = "") -> None:
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    status_esc = _escape(status)
    sql = (
        "UPDATE dw_meta.dw_node "
        f"SET migration_status = '{status_esc}', updated_at = '{now}' "
        f"WHERE node_id = {node_id}"
    )
    res = conn.execute_update(sql)
    if not getattr(res, "success", True):
        logger.warning(f"更新 migration_status 失败 node_id={node_id}, err={getattr(res, 'error', '')}, msg={msg}")


def _insert_feedback(conn: Any, object_key: str, suggestion_type: str, ai_value: str) -> None:
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    sql = (
        "INSERT INTO dw_meta.ai_feedback (object_type, object_key, suggestion_type, ai_value, created_at) "
        f"VALUES ('VIEW', '{_escape(object_key)}', '{_escape(suggestion_type)}', '{_escape(ai_value)}', '{now}')"
    )
    res = conn.execute_insert(sql)
    if not getattr(res, "success", True):
        logger.warning(f"写 ai_feedback 失败 object_key={object_key}, err={getattr(res, 'error', '')}")


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


def _upsert_model(
    conn: Any,
    view: ViewCandidate,
    db_name: str,
    primary_keys: Optional[List[str]] = None,
    partition_key: Optional[str] = None,
) -> int:
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    model_name = view.view_name.lower()
    existing = _find_existing_model(conn, model_name)
    layer = view.layer
    origin = "AUTO"
    sync_direction = "META_TO_CODE"
    status = "DRAFT"
    sqlmesh_model_name = f"{db_name}.{model_name}" if db_name else model_name

    pk_sql = "NULL" if not primary_keys else f"'{_escape(','.join(primary_keys))}'"
    part_sql = "NULL" if not partition_key else f"'{_escape(partition_key)}'"

    if existing:
        sql = (
            "UPDATE dw_meta.dw_model SET "
            f"model_name='{_escape(model_name)}', "
            f"db_name='{_escape(db_name)}', "
            f"table_name='{_escape(model_name)}', "
            f"layer='{_escape(layer)}', "
            f"sqlmesh_model_name='{_escape(sqlmesh_model_name)}', "
            f"origin='{origin}', "
            f"sync_direction='{sync_direction}', "
            f"source_node_id={view.node_id if view.node_id else 'NULL'}, "
            f"source_table_id={view.source_table_id if view.source_table_id else 'NULL'}, "
            f"primary_keys={pk_sql}, "
            f"partition_key={part_sql}, "
            f"status='{status}', "
            f"updated_at='{now}' "
            f"WHERE model_id={existing}"
        )
        res = conn.execute_update(sql)
        if not getattr(res, "success", True):
            logger.warning(f"更新 dw_model 失败 model_id={existing}, err={getattr(res, 'error', '')}")
        return existing

    sql = (
        "INSERT INTO dw_meta.dw_model "
        "(model_name, db_name, table_name, layer, sqlmesh_model_name, origin, sync_direction, "
        "biz_domain_code, biz_entity_code, grain_desc, primary_keys, partition_key, distributed_key, "
        "incremental_strategy, default_filter, source_node_id, source_table_id, status, created_at, updated_at) "
        f"VALUES ('{_escape(model_name)}','{_escape(db_name)}','{_escape(model_name)}','{_escape(layer)}',"
        f"'{_escape(sqlmesh_model_name)}','{origin}','{sync_direction}',"
        f"NULL,NULL,NULL,{pk_sql},{part_sql},"
        "NULL,NULL,NULL,"
        f"{view.node_id if view.node_id else 'NULL'},{view.source_table_id if view.source_table_id else 'NULL'},"
        f"'{status}','{now}','{now}')"
    )
    res = conn.execute_insert(sql)
    if not getattr(res, "success", True):
        raise RuntimeError(f"写入 dw_model 失败 view={view.view_name}, err={getattr(res, 'error', '')}")
    # 许多连接器不返回自增 id，兜底再查一次
    if getattr(res, "lastrowid", None):
        try:
            return int(res.lastrowid)
        except Exception:
            pass
    return _find_existing_model(conn, model_name) or 0


def _load_std_mapping(conn: Any, source_system: str, source_table: str) -> Dict[str, Dict[str, Any]]:
    sql = (
        "SELECT source_table, source_column, std_field_id, transform_expr, "
        "is_primary_key, is_partition_key, is_business_key "
        "FROM dw_meta.std_field_mapping "
        f"WHERE LOWER(source_system) = '{_escape(source_system.lower())}' "
        f"AND LOWER(source_table) = '{_escape(source_table.lower())}' "
        "AND is_active = 1"
    )
    res = conn.execute_query(sql)
    if not getattr(res, "success", True):
        logger.debug(f"查询 std_field_mapping 失败: {getattr(res, 'error', '')}")
        return {}
    mapping: Dict[str, Dict[str, Any]] = {}
    for row in _rows_from_result(res):
        key = (row.get("source_column") or "").lower()
        mapping[key] = row
    return mapping


def _guess_field_roles(cols: List[Dict[str, Any]]) -> Dict[str, str]:
    roles: Dict[str, str] = {}
    for col in cols:
        name = col.get("name") or col.get("alias") or ""
        lower_name = name.lower()
        role = "dimension"
        if any(tok in lower_name for tok in ["amount", "amt", "qty", "quantity", "num", "price", "total", "cnt"]):
            role = "measure"
        if any(tok in lower_name for tok in ["id", "code", "key"]) and not any(tok in lower_name for tok in ["status"]):
            role = "key"
        if any(tok in lower_name for tok in ["date", "time", "dt"]):
            role = "date"
        roles[lower_name] = role
    return roles


def _field_partition_llm_stub(feature: Dict[str, Any], mapping: Dict[str, Dict[str, Any]], target_layer: str) -> Dict[str, Any]:
    """
    轻量字段分层：用 std_mapping + 简单规则替代 LLM，返回 dwd_fields/dim_fields/pk/partition。
    """
    cols = feature.get("columns") or feature.get("selects") or []
    norm_cols = []
    for c in cols:
        if isinstance(c, str):
            norm_cols.append({"name": c})
        elif isinstance(c, dict):
            norm_cols.append(c)
    roles = _guess_field_roles(norm_cols)
    dwd_fields: List[Dict[str, Any]] = []
    dim_fields: List[Dict[str, Any]] = []
    pks: List[str] = []
    part_key: Optional[str] = None

    for col in norm_cols:
        name = col.get("name") or col.get("alias") or ""
        expr = col.get("expression") or name
        if not name:
            continue
        lower_name = name.lower()
        m = mapping.get(lower_name, {})
        role = roles.get(lower_name, "dimension")
        is_pk = int(m.get("is_primary_key") or 0)
        is_part = int(m.get("is_partition_key") or 0)
        if is_pk:
            pks.append(lower_name)
        if is_part or (not part_key and role == "date"):
            part_key = lower_name

        field_item = {
            "name": lower_name,
            "expr": m.get("transform_expr") or expr,
            "std_field_id": m.get("std_field_id"),
            "role": role,
            "is_pk": is_pk,
            "is_partition": is_part,
        }

        if target_layer == "DIM":
            dim_fields.append(field_item)
        else:  # DWD
            if role == "dimension" and not is_pk:
                dim_fields.append(field_item)
            else:
                dwd_fields.append(field_item)

    return {"dwd_fields": dwd_fields, "dim_fields": dim_fields, "pk": pks, "partition_key": part_key}


def _field_partition_llm(
    feature: Dict[str, Any],
    mapping: Dict[str, Dict[str, Any]],
    target_layer: str,
    llm: Optional["LLMBaseModel"],
) -> Optional[Dict[str, Any]]:
    if not llm:
        return None
    cols = feature.get("columns") or feature.get("selects") or []
    col_entries = []
    for c in cols:
        if isinstance(c, str):
            col_entries.append({"name": c})
        elif isinstance(c, dict):
            col_entries.append({"name": c.get("name") or c.get("alias") or "", "expression": c.get("expression")})
    map_entries = []
    for k, v in mapping.items():
        map_entries.append(
            {
                "source_column": k,
                "std_field_id": v.get("std_field_id"),
                "transform_expr": v.get("transform_expr"),
                "is_pk": int(v.get("is_primary_key") or 0),
                "is_partition": int(v.get("is_partition_key") or 0),
            }
        )
    prompt = {
        "task": "decide field partition for data warehouse modeling",
        "target_layer": target_layer,
        "columns": col_entries,
        "std_field_mapping": map_entries,
        "rules": [
            "DWD 保留主键/外键/事件度量/业务日期",
            "维属性应归 DIM，必要时新增或补充 DIM",
            "选择 pk 列并给出 partition_key（优先日期）",
        ],
        "output_format": {
            "dwd_fields": [{"name": "string", "expr": "string", "std_field_id": "int|null", "is_pk": "0/1", "is_partition": "0/1"}],
            "dim_fields": [{"name": "string", "expr": "string", "std_field_id": "int|null"}],
            "pk": ["string"],
            "partition_key": "string|null",
            "confidence": "0-1",
        },
    }
    try:
        resp = llm.generate_with_json_output(prompt)
        if not isinstance(resp, dict):
            return None
        return resp
    except Exception:
        logger.debug("LLM 分层调用失败，降级规则", exc_info=True)
        return None


def _insert_columns_from_feature(
    conn: Any, model_id: int, field_plan: Dict[str, Any], layer: str
) -> int:
    cols = field_plan.get("dwd_fields") or field_plan.get("dim_fields") or []
    if not cols:
        return 0
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    inserted = 0
    for idx, col in enumerate(cols):
        name = col.get("name") if isinstance(col, dict) else str(col)
        expr = col.get("expr") if isinstance(col, dict) else str(col)
        if not name:
            continue
        lower_name = name.lower()
        std_id = col.get("std_field_id") if isinstance(col, dict) else None
        is_pk = int(col.get("is_pk")) if isinstance(col, dict) and col.get("is_pk") is not None else 0
        is_part = int(col.get("is_partition")) if isinstance(col, dict) and col.get("is_partition") is not None else 0
        sql = (
            "INSERT INTO dw_meta.dw_model_column "
            "(model_id, column_name, column_order, std_field_id, expression_sql, "
            "is_primary_key, is_partition_key, is_distributed_key, not_null, comment, is_active, created_at, updated_at) "
            f"VALUES ({model_id}, '{_escape(lower_name)}', {idx}, "
            f"{std_id if std_id else 'NULL'}, '{_escape(expr)}', "
            f"{is_pk}, {is_part}, 0, 0, NULL, 1, '{now}', '{now}')"
        )
        res = conn.execute_insert(sql)
        if getattr(res, "success", False):
            inserted += 1
        else:
            logger.debug(f"插入 dw_model_column 失败 model_id={model_id}, col={name}, err={getattr(res, 'error', '')}")
    return inserted


def _upsert_dim_model_and_fields(conn: Any, view: ViewCandidate, logic_db: str, field_plan: Dict[str, Any]) -> Optional[int]:
    dim_name = f"dim_{view.source_table_name}".replace(".", "_")
    dim_view = ViewCandidate(
        node_id=view.node_id,
        source_table_id=view.source_table_id,
        table_id=view.table_id,
        source_system=view.source_system,
        source_table_name=view.source_table_name,
        view_name=dim_name,
        layer="DIM",
        feature_json="",
    )
    pk_list = field_plan.get("pk") or []
    model_id = _upsert_model(conn, dim_view, logic_db, primary_keys=pk_list, partition_key=None)
    dim_fields_plan = {"dim_fields": field_plan.get("dim_fields") or []}
    _insert_columns_from_feature(conn, model_id, dim_fields_plan, "DIM")
    return model_id


def run_propose_model(agent_config: AgentConfig, db_manager: DBManager, args) -> Dict[str, Any]:
    """
    入口：基于 AST 成功的视图生成模型草稿或记录 DWS 设计。
    """
    conn, logic_db = _resolve_meta_conn(agent_config, db_manager, args)
    node_ids = _parse_node_ids(getattr(args, "node_ids", None))
    include_dws = bool(getattr(args, "include_dws", False))
    layer_filter = getattr(args, "layer", None)
    strict = bool(getattr(args, "strict", False))
    llm_model_name = getattr(args, "llm_model", "") or ""

    candidates = _load_candidates(conn, layer_filter, node_ids, include_dws)
    if not candidates:
        logger.info("未找到符合条件的视图，退出。")
        return {"status": "success", "message": "no candidates", "stats": {"processed": 0}}

    stats = {"processed": 0, "proposed": 0, "skipped": 0, "dws_logged": 0, "columns_added": 0}
    llm_instance = None
    if llm_model_name and LLMBaseModel:
        try:
            llm_instance = LLMBaseModel.create_model(model_name=llm_model_name, agent_config=agent_config)
        except Exception as exc:  # pragma: no cover
            logger.warning(f"LLM 创建失败，降级规则模式: {exc}")
            llm_instance = None

    for view in candidates:
        stats["processed"] += 1
        feature = _parse_feature(view.feature_json)
        if view.layer == "DWS":
            _insert_feedback(conn, f"view:{view.view_name}", "DWS_DESIGN", json.dumps(feature)[:800])
            stats["dws_logged"] += 1
            continue

        usable, reasons = _is_usable(feature, strict)
        if not usable:
            reason_msg = "; ".join(reasons) or "规则未通过"
            _update_migration_status(conn, view.node_id, "SKIPPED", reason_msg)
            _insert_feedback(conn, f"view:{view.view_name}", "USABILITY", reason_msg)
            stats["skipped"] += 1
            continue

        std_mapping = _load_std_mapping(conn, view.source_system, view.source_table_name)
        field_plan = None
        if llm_instance:
            field_plan = _field_partition_llm(feature, std_mapping, view.layer, llm_instance)
            if field_plan and isinstance(field_plan, dict):
                # 记录 LLM 置信度
                conf = field_plan.get("confidence")
                try:
                    llm_feedback = {
                        "model": view.view_name,
                        "layer": view.layer,
                        "note": "LLM field partition",
                        "confidence": conf,
                    }
                    _insert_feedback(conn, f"view:{view.view_name}", "FIELD_PARTITION_LLM", json.dumps(llm_feedback))
                except Exception:
                    logger.debug("LLM 反馈记录失败", exc_info=True)
        if not field_plan:
            field_plan = _field_partition_llm_stub(feature, std_mapping, view.layer)

        pks = field_plan.get("pk") or [k for k, v in std_mapping.items() if int(v.get("is_primary_key") or 0) == 1]
        partition_key = field_plan.get("partition_key")
        model_id = _upsert_model(conn, view, logic_db, primary_keys=pks, partition_key=partition_key)
        added_cols = _insert_columns_from_feature(conn, model_id, field_plan, view.layer)
        stats["columns_added"] += added_cols

        # 如果有 DIM 补充字段，生成/更新一个 DIM 模型
        if field_plan.get("dim_fields"):
            try:
                dim_model_id = _upsert_dim_model_and_fields(conn, view, logic_db, field_plan)
                if dim_model_id:
                    _insert_feedback(conn, f"model:{dim_model_id}", "DIM_UPDATE", "auto_add_dim_fields_from_dwd")
            except Exception:
                logger.debug("补充 DIM 字段失败", exc_info=True)

        _update_migration_status(conn, view.node_id, "PROPOSED", "")
        _insert_feedback(conn, f"view:{view.view_name}", "FIELD_PARTITION", "generated_by_rules_llm_stub")
        stats["proposed"] += 1

    logger.info(f"propose-model 完成: {stats}")
    return {"status": "success", "stats": stats}


if __name__ == "__main__":  # pragma: no cover
    print("请通过 datus-agent propose-model 调用本模块")
