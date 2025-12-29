from __future__ import annotations

from datetime import datetime
import json
from typing import Any, Dict, Optional, Tuple

from datus.configuration.agent_config import AgentConfig
from datus.tools.db_tools.db_manager import DBManager
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


def _escape(val: Optional[str]) -> str:
    return (val or "").replace("'", "''")


def _resolve_meta_conn(agent_config: AgentConfig, db_manager: DBManager, args) -> Tuple[Any, str]:
    logic_db = getattr(args, "database", "") or agent_config.current_database or ""
    conn = db_manager.get_conn(agent_config.current_namespace, logic_db)
    return conn, logic_db


def _resolve_view(conn: Any, view_key: str) -> Optional[int]:
    if not view_key:
        return None
    if view_key.isdigit():
        return int(view_key)
    sql = (
        "SELECT node_id FROM dw_meta.dw_node "
        f"WHERE LOWER(table_name) = LOWER('{_escape(view_key)}') "
        "ORDER BY node_id DESC LIMIT 1"
    )
    res = conn.execute_query(sql)
    if not getattr(res, "success", True):
        return None
    rows = getattr(res, "data", None) or getattr(res, "raw", None) or []
    if rows and isinstance(rows, list):
        row = rows[0]
        if isinstance(row, dict):
            return int(row.get("node_id"))
        if isinstance(row, (list, tuple)) and row:
            try:
                return int(row[0])
            except Exception:
                return None
    return None


def _resolve_target_model(conn: Any, node_id: Optional[int]) -> Optional[int]:
    if not node_id:
        return None
    sql = (
        "SELECT model_id FROM dw_meta.dw_model "
        f"WHERE source_node_id = {node_id} "
        "ORDER BY model_id DESC LIMIT 1"
    )
    res = conn.execute_query(sql)
    if not getattr(res, "success", True):
        return None
    rows = getattr(res, "data", None) or getattr(res, "raw", None) or []
    if rows and isinstance(rows, list):
        row = rows[0]
        if isinstance(row, dict):
            return int(row.get("model_id"))
        if isinstance(row, (list, tuple)) and row:
            try:
                return int(row[0])
            except Exception:
                return None
    return None


def run_compare_view(agent_config: AgentConfig, db_manager: DBManager, args) -> Dict[str, Any]:
    """
    从标准模型重建目标表/视图，与源视图做 count/简单指标抽样对比，记录 view_compare_result。
    """
    meta_conn, _ = _resolve_meta_conn(agent_config, db_manager, args)
    view_key = getattr(args, "view", "")
    sample = int(getattr(args, "sample", 1000) or 1000)
    metrics_raw = getattr(args, "metrics", "") or ""
    metric_list = [m.strip() for m in metrics_raw.replace(",", " ").split() if m.strip()]

    source_view_id = _resolve_view(meta_conn, str(view_key))
    target_model_id = _resolve_target_model(meta_conn, source_view_id)

    # 查源视图物理名、目标表名
    src_name = None
    tgt_full = None
    try:
        if source_view_id:
            sql = (
                "SELECT ts.table_name FROM dw_meta.dw_node dn "
                "JOIN dw_meta.table_source ts ON dn.source_table_id = ts.table_id "
                f"WHERE dn.node_id = {source_view_id} LIMIT 1"
            )
            res = meta_conn.execute_query(sql)
            if getattr(res, "success", True):
                rows = getattr(res, "data", None) or getattr(res, "raw", None) or []
                if rows:
                    src_name = rows[0].get("table_name") if isinstance(rows[0], dict) else rows[0][0]
        if target_model_id:
            sql = (
                "SELECT db_name, table_name FROM dw_meta.dw_model "
                f"WHERE model_id = {target_model_id} LIMIT 1"
            )
            res = meta_conn.execute_query(sql)
            if getattr(res, "success", True):
                rows = getattr(res, "data", None) or getattr(res, "raw", None) or []
                if rows:
                    row = rows[0]
                    dbn = row.get("db_name") if isinstance(row, dict) else row[0]
                    tn = row.get("table_name") if isinstance(row, dict) else row[1]
                    tgt_full = f"{dbn}.{tn}" if dbn else tn
    except Exception:
        logger.debug("解析源/目标名称失败", exc_info=True)

    data_conn = meta_conn  # 默认复用当前连接；实际可按 namespace/db_name 选择

    def _scalar(sql_text: str) -> Optional[int]:
        try:
            res = data_conn.execute_query(sql_text)
            if not getattr(res, "success", True):
                return None
            rows = getattr(res, "data", None) or getattr(res, "raw", None) or []
            if rows:
                val = rows[0]
                if isinstance(val, dict):
                    return list(val.values())[0]
                if isinstance(val, (list, tuple)):
                    return val[0]
            return None
        except Exception:
            logger.debug(f"执行失败: {sql_text}", exc_info=True)
            return None

    count_diff = None
    metric_diff = {}
    status = "PENDING"
    message = ""

    if src_name and tgt_full:
        src_cnt = _scalar(f"SELECT COUNT(*) FROM {src_name}")
        tgt_cnt = _scalar(f"SELECT COUNT(*) FROM {tgt_full}")
        if src_cnt is not None and tgt_cnt is not None:
            count_diff = tgt_cnt - src_cnt
            status = "PASS" if count_diff == 0 else "FAIL"
            message = f"count src={src_cnt}, tgt={tgt_cnt}"
        else:
            status = "FAIL"
            message = "count compare failed"

        # 简单指标对比：sum(metric)
        for m in metric_list:
            src_sum = _scalar(f"SELECT SUM({m}) FROM {src_name} LIMIT {sample}")
            tgt_sum = _scalar(f"SELECT SUM({m}) FROM {tgt_full} LIMIT {sample}")
            metric_diff[m] = {"src": src_sum, "tgt": tgt_sum, "diff": None if None in (src_sum, tgt_sum) else tgt_sum - src_sum}

    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    metric_json = json.dumps(metric_diff, ensure_ascii=False) if metric_diff else None
    sql_ins = (
        "INSERT INTO dw_meta.view_compare_result "
        "(source_view_id, target_model_id, sample_size, row_count_diff, metric_diff, status, message, created_at, updated_at) "
        f"VALUES ({source_view_id if source_view_id else 'NULL'}, "
        f"{target_model_id if target_model_id else 'NULL'}, "
        f"{sample}, "
        f"{count_diff if count_diff is not None else 'NULL'}, "
        f"{'NULL' if metric_json is None else f'\\''{_escape(metric_json)}\\'''}, "
        f"'{status}', '{_escape(message)}', '{now}', '{now}')"
    )
    res = meta_conn.execute_insert(sql_ins)
    if not getattr(res, "success", True):
        return {"status": "error", "message": getattr(res, "error", "unknown error")}
    return {
        "status": status.lower(),
        "message": message,
        "row_count_diff": count_diff,
        "metric_diff": metric_diff,
        "source_view_id": source_view_id,
        "target_model_id": target_model_id,
    }


if __name__ == "__main__":  # pragma: no cover
    print("请通过 datus-agent compare-view 调用本模块")
