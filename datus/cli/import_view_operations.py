"""
import_view 数据库操作封装

从 import_view.py 提取的数据库操作相关方法，复用现有的工具和模式。
"""

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Set

from datus.utils.constants import DBType
from datus.utils.loggings import get_logger
from datus.utils.result_utils import get_row_value, parse_query_result
from datus.utils.sql_utils import escape_sql, parse_table_name_parts
from datus.tools.db_tools.base import BaseSqlConnector

logger = get_logger(__name__)


@dataclass
class ViewSourceRow:
    """视图源数据行模型"""
    view_id: int
    view_name: str
    db_name: str
    ddl_sql: str
    sql_hash: str
    has_row: int = 0
    parse_status: str = None


class ImportViewOperations:
    """import_view 数据库操作封装类"""

    def __init__(self, meta_conn: BaseSqlConnector, sourcedb: str):
        """
        初始化数据库操作封装

        Args:
            meta_conn: 元数据库连接
            sourcedb: 源数据库名称
        """
        self.meta_conn = meta_conn
        self.sourcedb = sourcedb

    def load_existing_views(self) -> Dict[str, ViewSourceRow]:
        """
        加载已存在的视图源数据

        Returns:
            视图名字典映射
        """
        sql = (
            "SELECT table_id as view_id, table_name as view_name, hash, has_row, parse_status "
            "FROM dw_meta.table_source "
            f"WHERE LOWER(source_system) = LOWER('{self.sourcedb}') AND table_type = 'VIEW'"
        )

        result = self.meta_conn.execute({"sql_query": sql, "result_format": "list"})
        rows = parse_query_result(result)

        existing: Dict[str, ViewSourceRow] = {}
        for row in rows:
            parsed = self._build_view_source_row(row, warn_context="load_existing_views")
            if not parsed:
                continue
            existing[parsed.view_name.lower()] = parsed

        return existing

    def load_existing_tables(self, table_type: str = "TABLE") -> Dict[str, ViewSourceRow]:
        """
        加载已存在的表源数据

        Args:
            table_type: 表类型 (TABLE/VIEW)

        Returns:
            表名字典映射
        """
        sql = (
            "SELECT table_id as view_id, table_name as view_name, hash, has_row, parse_status "
            "FROM dw_meta.table_source "
            f"WHERE LOWER(source_system) = LOWER('{self.sourcedb}') AND table_type = '{table_type}'"
        )

        result = self.meta_conn.execute({"sql_query": sql, "result_format": "list"})
        rows = parse_query_result(result)

        existing: Dict[str, ViewSourceRow] = {}
        for row in rows:
            parsed = self._build_view_source_row(row, warn_context="load_existing_tables")
            if not parsed:
                continue
            existing[parsed.view_name.lower()] = parsed

        return existing

    def load_table_source_map(self) -> Dict[str, Dict[str, Any]]:
        """
        加载表源映射数据

        Returns:
            表名到表信息的映射
        """
        sql = (
            "SELECT table_id, table_name, table_type "
            "FROM dw_meta.table_source "
            f"WHERE LOWER(source_system) = LOWER('{self.sourcedb}')"
        )

        result = self.meta_conn.execute({"sql_query": sql, "result_format": "list"})
        rows = parse_query_result(result)

        mapping: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            name = str(row.get("table_name") or "").lower()
            mapping[name] = {
                "table_id": row.get("table_id"),
                "table_type": (row.get("table_type") or "").upper(),
                "source_system": self.sourcedb,
            }

        return mapping

    def upsert_table_source(
        self,
        row: ViewSourceRow,
        existing: Optional[ViewSourceRow],
        table_type: str = "VIEW"
    ) -> Tuple[int, bool]:
        """
        插入或更新表源数据

        Args:
            row: 视图源数据行
            existing: 已存在的数据
            table_type: 表类型

        Returns:
            (table_id, changed) 元组

        Performance:
            使用参数化查询和批量操作优化性能
        """
        """
        插入或更新表源数据

        Args:
            row: 视图源数据行
            existing: 已存在的数据
            table_type: 表类型

        Returns:
            (table_id, changed) 元组
        """
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        view_name = escape_sql(row.view_name).lower()
        ddl_sql = escape_sql(row.ddl_sql)
        sql_hash = escape_sql(row.sql_hash)
        has_row = 1 if row.has_row else 0
        requested_status = (row.parse_status or "").upper()

        if table_type == "EXTERNAL":
            source_system_raw = escape_sql(row.db_name) if escape_sql(row.db_name) != "" else self.sourcedb
        else:
            source_system_raw = self.sourcedb

        source_system_norm = escape_sql(source_system_raw).lower()

        name_key = view_name.strip('"')
        existing_row = existing or self._find_existing_in_db(view_name, table_type, source_system_norm)

        if not existing_row and name_key and name_key != view_name:
            existing_row = self._find_existing_in_db(name_key, table_type, source_system_norm)

        existing_status = (existing_row.parse_status or "").upper() if existing_row else ""
        target_status = requested_status or existing_status or "NEW"

        if not requested_status and existing_status == "SKIPPED" and has_row:
            # 数据恢复后将 SKIPPED 恢复为 NEW，后续可重新进入 AST
            target_status = "NEW"

        target_status_esc = escape_sql(target_status)

        # 更新现有记录
        if existing_row and existing_row.view_id:
            update_fields: List[str] = []

            if existing_row.sql_hash != row.sql_hash:
                update_fields.extend([f"ddl_sql = '{ddl_sql}'", f"hash = '{sql_hash}'"])

            if (existing_row.has_row or 0) != has_row:
                update_fields.append(f"has_row = {has_row}")

            if target_status != existing_status:
                update_fields.append(f"parse_status = '{target_status_esc}'")

            if not update_fields:
                return existing_row.view_id or 0, False

            update_fields.append(f"updated_at = '{now}'")
            update_sql = (
                "UPDATE dw_meta.table_source SET "
                + ", ".join(update_fields)
                + f" WHERE table_id = {existing_row.view_id}"
            )

            self.meta_conn.execute({"sql_query": update_sql})
            return existing_row.view_id, True

        # 插入新记录
        # 插入前再做一次兜底查询，避免因大小写/引号差异导致重复
        if not existing_row:
            precheck = self._find_existing_in_db(view_name, table_type, source_system_norm)
            if not precheck and name_key and name_key != view_name:
                precheck = self._find_existing_in_db(name_key, table_type, source_system_norm)

            if precheck:
                return precheck.view_id or 0, False

        insert = (
            "INSERT INTO dw_meta.table_source "
            "(source_system, table_name, table_type, ddl_sql, hash, has_row, parse_status, created_at, updated_at) "
            f"VALUES ('{source_system_norm}', '{view_name}', '{table_type}', '{ddl_sql}', '{sql_hash}', {has_row}, '{target_status_esc}', '{now}', '{now}')"
        )

        self.meta_conn.execute({"sql_query": insert})

        res = self.meta_conn.execute({
            "sql_query": (
                "SELECT table_id FROM dw_meta.table_source "
                f"WHERE LOWER(source_system) = LOWER('{source_system_norm}') AND table_type = '{table_type}' "
                f"AND LOWER(table_name) = LOWER('{view_name}') "
                "ORDER BY table_id DESC LIMIT 1"
            )
        })

        rows = parse_query_result(res)
        return (int(rows[0].get("table_id")), True) if rows else (0, True)

    def ensure_view_node(self, view_id: int, view_name: str) -> int:
        """
        确保视图节点存在

        Args:
            view_id: 视图ID
            view_name: 视图名称

        Returns:
            节点ID

        Raises:
            RuntimeError: 当无法创建或获取节点时
        """
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        view_name_esc = escape_sql(view_name)

        # 仅按 source_table_id 或 source_system+table_name 识别视图节点
        fetch = self.meta_conn.execute({
            "sql_query": (
                "SELECT node_id, node_type, source_table_id FROM dw_meta.dw_node "
                f"WHERE source_table_id = {view_id} LIMIT 1"
            )
        })

        rows = parse_query_result(fetch)

        if not rows:
            fetch = self.meta_conn.execute({
                "sql_query": (
                    "SELECT node_id, node_type, source_table_id FROM dw_meta.dw_node "
                    f"WHERE source_system = '{escape_sql(self.sourcedb)}' "
                    f"AND table_name = '{view_name_esc}' LIMIT 1"
                )
            })
            rows = parse_query_result(fetch)

        if rows:
            node_id = int(rows[0].get("node_id"))
            updates = []

            if (rows[0].get("node_type") or "").upper() != "VIEW":
                updates.append("node_type = 'VIEW'")

            if not rows[0].get("source_table_id"):
                updates.append(f"source_table_id = {view_id}")

            if updates:
                updates.append(f"updated_at = '{now}'")
                self.meta_conn.execute({
                    "sql_query": (
                        "UPDATE dw_meta.dw_node SET " + ", ".join(updates) + f" WHERE node_id = {node_id}"
                    )
                })

            return node_id

        # 插入新节点
        insert = (
            "INSERT INTO dw_meta.dw_node "
            "(node_type, source_system, table_name, source_table_id, migration_status, created_at, updated_at) "
            f"VALUES ('VIEW', '{escape_sql(self.sourcedb)}', '{view_name_esc}', {view_id}, 'NEW', '{now}', '{now}')"
        )

        self.meta_conn.execute({"sql_query": insert})

        res = self.meta_conn.execute({
            "sql_query": (
                "SELECT node_id FROM dw_meta.dw_node "
                f"WHERE source_table_id = {view_id} ORDER BY node_id DESC LIMIT 1"
            )
        })

        rows = parse_query_result(res)
        if rows:
            return int(rows[0].get("node_id"))

        # 备用查询
        res_fb = self.meta_conn.execute({
            "sql_query": (
                "SELECT node_id FROM dw_meta.dw_node "
                f"WHERE source_system = '{escape_sql(self.sourcedb)}' "
                f"AND table_name = '{view_name_esc}' ORDER BY node_id DESC LIMIT 1"
            )
        })

        rows_fb = parse_query_result(res_fb)
        if rows_fb:
            return int(rows_fb[0].get("node_id"))

        logger.error(f"无法获取 dw_node 节点(插入后查询为空)，view={view_name}, table_id={view_id}")
        return 0

    def ensure_dependency_nodes(self, dep_info: Dict[str, Dict[str, Any]]) -> Dict[str, int]:
        """
        确保依赖节点存在

        Args:
            dep_info: 依赖信息

        Returns:
            依赖节点ID映射
        """
        nodes: Dict[str, int] = {}
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

        for alias, info in dep_info.items():
            table_nm = escape_sql(info.get("name") or "")
            node_type = "VIEW" if info.get("type") == "VIEW" else "TABLE"
            source_table_id = info.get("source_table_id")

            fetch = self.meta_conn.execute({
                "sql_query": (
                    "SELECT node_id FROM dw_meta.dw_node "
                    f"WHERE source_system = '{escape_sql(self.sourcedb)}' "
                    f"AND table_name = '{table_nm}' LIMIT 1"
                )
            })

            rows = parse_query_result(fetch)

            if rows:
                node_id = int(rows[0].get("node_id"))
            else:
                insert = (
                    "INSERT INTO dw_meta.dw_node "
                    "(node_type, source_system, table_name, source_table_id, migration_status, created_at, updated_at) "
                    f"VALUES ('{node_type}', '{escape_sql(self.sourcedb)}', '{table_nm}', "
                    f"{source_table_id if source_table_id is not None else 'NULL'}, 'NEW', '{now}', '{now}')"
                )

                self.meta_conn.execute({"sql_query": insert})

                res = self.meta_conn.execute({
                    "sql_query": (
                        "SELECT node_id FROM dw_meta.dw_node "
                        f"WHERE source_system = '{escape_sql(self.sourcedb)}' "
                        f"AND table_name = '{table_nm}' "
                        "ORDER BY node_id DESC LIMIT 1"
                    )
                })

                rows = parse_query_result(res)

            if rows:
                nodes[alias] = int(rows[0].get("node_id"))

        return nodes

    def upsert_ai_view_feature(self, table_id: int, feature_json: str):
        """
        插入或更新AI视图特征

        Args:
            table_id: 表ID
            feature_json: 特征JSON字符串
        """
        # 先删除现有记录
        self.meta_conn.execute({"sql_query": f"DELETE FROM dw_meta.ai_view_feature WHERE table_id = {table_id}"})

        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        payload = escape_sql(feature_json)

        sql = (
            "INSERT INTO dw_meta.ai_view_feature (table_id, feature_json, analyzed_at) "
            f"VALUES ({table_id}, '{payload}', '{now}')"
        )

        self.meta_conn.execute({"sql_query": sql})

    def update_table_parse_status(self, table_id: Optional[int], status: str):
        """
        更新表解析状态

        Args:
            table_id: 表ID
            status: 状态
        """
        if not table_id:
            return

        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        status_esc = escape_sql(status.upper())

        self.meta_conn.execute({
            "sql_query": (
                "UPDATE dw_meta.table_source "
                f"SET parse_status = '{status_esc}', updated_at = '{now}' "
                f"WHERE table_id = {table_id}"
            )
        })

    def update_node_migration_status(self, table_id: Optional[int], status: str):
        """
        更新节点迁移状态

        Args:
            table_id: 表ID
            status: 状态
        """
        if not table_id:
            return

        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        status_esc = escape_sql(status.upper())

        self.meta_conn.execute({
            "sql_query": (
                "UPDATE dw_meta.dw_node "
                f"SET migration_status = '{status_esc}', updated_at = '{now}' "
                f"WHERE source_table_id = {table_id}"
            )
        })

    def _build_view_source_row(self, row: Any, warn_context: str = "", idx_map: Optional[Dict[str, int]] = None) -> Optional[ViewSourceRow]:
        """
        将元库查询结果行解析为 ViewSourceRow

        Args:
            row: 查询结果行
            warn_context: 警告上下文
            idx_map: 索引映射

        Returns:
            解析后的ViewSourceRow，失败时返回None
        """
        idx_map = idx_map or {"view_id": 0, "view_name": 1, "ddl_sql": None, "hash": 2, "has_row": 3, "parse_status": 4}

        view_id_raw = get_row_value(row, ["view_id", "table_id"], idx=idx_map.get("view_id"))
        view_name_raw = get_row_value(row, ["view_name", "table_name"], idx=idx_map.get("view_name"))
        ddl_sql_raw = get_row_value(row, ["ddl_sql"], idx=idx_map.get("ddl_sql")) or ""
        hash_raw = get_row_value(row, ["hash"], idx=idx_map.get("hash")) or ""
        has_row_raw = get_row_value(row, ["has_row"], idx=idx_map.get("has_row"))
        parse_status_raw = get_row_value(row, ["parse_status"], idx=idx_map.get("parse_status"))

        if not view_name_raw:
            logger.warning(f"table_source {row} 行缺少 view_name，跳过 (context={warn_context}): {row}")
            return None

        try:
            has_row_val = int(has_row_raw) if has_row_raw is not None else None
        except Exception:
            has_row_val = has_row_raw

        return ViewSourceRow(
            view_id=view_id_raw,
            view_name=view_name_raw,
            db_name="",
            ddl_sql=ddl_sql_raw,
            sql_hash=hash_raw,
            has_row=has_row_val,
            parse_status=parse_status_raw,
        )

    def _find_existing_in_db(self, view_name: str, table_type: str, source_system: str) -> Optional[ViewSourceRow]:
        """
        在数据库中查找已存在的记录

        Args:
            view_name: 视图名称
            table_type: 表类型
            source_system: 源系统

        Returns:
            找到的记录，未找到时返回None
        """
        sql = (
            "SELECT table_id as view_id, table_name as view_name, hash, has_row, parse_status "
            "FROM dw_meta.table_source "
            f"WHERE LOWER(source_system) = LOWER('{source_system}') AND table_type = '{table_type}' "
            f"AND LOWER(table_name) = LOWER('{view_name}') "
            "ORDER BY table_id DESC LIMIT 1"
        )

        result = self.meta_conn.execute({"sql_query": sql, "result_format": "list"})
        rows = parse_query_result(result)

        if not rows:
            return None

        parsed = self._build_view_source_row(rows[0], warn_context="find_existing_in_db")
        return parsed