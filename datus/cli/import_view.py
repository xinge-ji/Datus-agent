from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from io import StringIO
from typing import Any, Dict, List, Optional, Set, Tuple

from datus.configuration.agent_config import AgentConfig
from datus.models.base import LLMBaseModel
from datus.tools.db_tools.db_manager import DBManager
from datus.utils.ast_analyzer import AstAnalyzer
from datus.utils.constants import DBType
from datus.utils.loggings import get_logger
from datus.utils.sql_utils import normalize_sql, parse_table_name_parts

logger = get_logger(__name__)


@dataclass
class ViewSourceRow:
    view_id: Optional[int]
    view_name: str
    db_name: str
    ddl_sql: str
    sql_hash: str


class ImportViewRunner:
    """视图导入与AST分析执行器。"""

    def __init__(self, agent_config: AgentConfig, db_manager: DBManager, namespace: str, sourcedb: str, strategy: str):
        self.agent_config = agent_config
        self.db_manager = db_manager
        self.namespace = namespace
        self.sourcedb = sourcedb
        self.strategy = strategy
        # 源库连接：优先从 sourcedb 配置创建独立连接，否则使用 namespace 下的逻辑库
        src_db_config = None
        try:
            src_db_config = agent_config.source_db_config(sourcedb)
        except Exception:
            src_db_config = None

        if src_db_config:
            from datus.tools.db_tools.db_manager import DBManager as SrcDBManager

            self._src_db_manager = SrcDBManager({sourcedb: {src_db_config.logic_name or sourcedb: src_db_config}})
            self.source_conn = self._src_db_manager.get_conn(sourcedb, src_db_config.logic_name or sourcedb)
            self.source_db_name = src_db_config.database or ""
            self.source_schema = src_db_config.schema or ""
        else:
            self._src_db_manager = None
            self.source_conn = self.db_manager.get_conn(namespace, sourcedb)
            self.source_db_name = getattr(self.source_conn, "database_name", "") or ""
            self.source_schema = getattr(self.source_conn, "schema_name", "") or ""

        # 元库连接：使用 namespace 默认/当前数据库（init-meta 所在）
        meta_logic_db = getattr(agent_config, "current_database", None) or agent_config.current_database or ""
        self.meta_conn = self.db_manager.get_conn(namespace, meta_logic_db)
        self.ast = AstAnalyzer(dialect="oracle")
        self.llm: Optional[LLMBaseModel] = None
        # 跨库前缀与 source_system 映射，例如 lyerp.table -> source_system=erp
        self.schema_system_map = {
            "lyerp": "erp",
            "lywms": "wms",
        }

    def sync_table_and_views(self) -> Dict[str, int]:
        """仅同步表/视图 DDL 到 table_source，不做 AST。"""
        all_tables = self._load_tables()
        all_views = self._load_views()
        table_existing = self._load_existing_table_source(table_type="TABLE")
        view_existing = self._load_existing_view_source()

        if self.strategy == "overwrite":
            self._cleanup_downstream(list(view_existing.keys()))

        for tbl in all_tables:
            row = self._normalize_view(tbl)
            key = row.view_name.lower()
            table_id = self._upsert_table_source(row, table_existing.get(key), table_type="TABLE")
            row.view_id = table_id
            table_existing[key] = row

        for view_meta in all_views:
            row = self._normalize_view(view_meta)
            key = row.view_name.lower()
            new_table_id = self._upsert_table_source(row, view_existing.get(key), table_type="VIEW")
            if new_table_id:
                row.view_id = new_table_id
                view_existing[key] = row

        return {"tables": len(all_tables), "views": len(all_views)}

    def analyze_views(self) -> Dict[str, Any]:
        """解析视图 AST、依赖、标准字段，假设表/视图已同步。"""
        all_views = self._load_views()
        view_existing = self._load_existing_view_source()

        to_process: List[ViewSourceRow] = []
        reused_views: List[ViewSourceRow] = []
        new_view_ids: List[int] = []

        for view_meta in all_views:
            row = self._normalize_view(view_meta)
            key = row.view_name.lower()
            new_table_id = self._upsert_table_source(row, view_existing.get(key), table_type="VIEW")
            if new_table_id:
                row.view_id = new_table_id
                view_existing[key] = row
            latest = view_existing.get(key)
            if latest and not row.view_id:
                row.view_id = latest.view_id
            if self.strategy == "incremental" and latest and latest.sql_hash == row.sql_hash:
                if self._can_skip(latest.view_id):
                    reused_views.append(latest)
                    continue
                reused_views.append(latest)
                to_process.append(latest)
                continue

            if not row.view_id:
                row.view_id = latest.view_id if latest else 0
            to_process.append(row)
            if row.view_id:
                new_view_ids.append(row.view_id)

        view_records = to_process + reused_views
        if not view_records:
            return {"new_views": len(new_view_ids), "processed": 0, "failed": 0, "reused": len(reused_views)}

        table_source_map = self._load_table_source_map()

        processed, failed = 0, 0
        analysis_results: List[Dict[str, Any]] = []
        for row in view_records:
            if not row.view_id:
                logger.warning(f"视图 {row.view_name} 缺少 view_id，跳过解析")
                continue
            try:
                ddl_sql = row.ddl_sql or ""
                logger.debug(f"待解析视图 {row.view_name} DDL: {ddl_sql}")
                feature = self.ast.analyze_view(ddl_sql, row.view_name)
                deps = self._resolve_dependencies(feature, table_source_map, row.db_name)
                if deps["unresolved"]:
                    logger.warning(f"视图 {row.view_name} 依赖未解析: {deps['unresolved']}")
                feature["status"] = "OK"
                feature["view_dependencies"] = sorted(deps["view_dependencies"])
                feature["table_dependencies"] = sorted(deps["table_dependencies"])
                feature["unresolved_dependencies"] = sorted(deps["unresolved"])
                feature_json = json.dumps(feature, ensure_ascii=True)
                self._upsert_ai_view_feature(row.view_id, feature_json)
                self._update_table_parse_status(row.view_id, "PARSED")
                self._update_node_migration_status(row.view_id, "ANALYZED")
                view_node_id = self._ensure_view_node(row.view_id, row.view_name)
                dependency_nodes = self._ensure_dependency_nodes(deps["dep_info"])
                self._upsert_relations(view_node_id, dependency_nodes, feature, deps["dep_info"])
                alias_map = {
                    (t.get("alias") or t.get("resolved_name") or t.get("name")): t for t in feature.get("tables", [])
                }
                analysis_results.append(
                    {
                        "row": row,
                        "feature": feature,
                        "alias_map": alias_map,
                    }
                )
                processed += 1
            except Exception as exc:  # pragma: no cover
                logger.error(f"视图解析失败 {row.view_name}: {exc}")
                error_json = json.dumps({"status": "ERROR", "error": str(exc)}, ensure_ascii=True)
                self._upsert_ai_view_feature(row.view_id, error_json)
                self._update_table_parse_status(row.view_id, "FAILED")
                self._update_node_migration_status(row.view_id, "AST_FAILED")
                failed += 1

        dep_graph = self._build_view_dep_graph(analysis_results)
        topo = self._topo_sort(dep_graph)
        analysis_map = {r["row"].view_name.lower(): r for r in analysis_results}
        for view_name in topo:
            result = analysis_map.get(view_name)
            if not result:
                continue
            std_items = self._prepare_std_items(
                result["feature"], result["alias_map"], result["row"].view_name, result["row"].db_name
            )
            self._persist_std_and_feedback(result["row"].view_name, result["row"].db_name, std_items)

        return {"new_views": len(new_view_ids), "processed": processed, "failed": failed, "reused": len(reused_views)}

    def run(self) -> Dict[str, Any]:
        """兼容旧入口：单库同步后直接解析。"""
        self.sync_table_and_views()
        return self.analyze_views()

    # ---------- 视图与 hash ---------- #
    def _load_views(self) -> List[Dict[str, str]]:
        views = []
        if hasattr(self.source_conn, "get_views_with_ddl"):
            db_name = self.source_db_name or getattr(self.source_conn, "database_name", "") or ""
            schema_name = self.source_schema or getattr(self.source_conn, "schema_name", "") or ""
            logger.info(f"准备拉取视图 DDL，db={db_name} schema={schema_name} connector={type(self.source_conn).__name__}")
            try:
                views = self.source_conn.get_views_with_ddl(database_name=db_name, schema_name=schema_name)
            except TypeError:
                views = self.source_conn.get_views_with_ddl()
            logger.info(f"已从源库获取视图 {len(views)} 个")
        else:
            logger.warning("连接器不支持 get_views_with_ddl，无法导入视图")
        return views

    def _load_tables(self) -> List[Dict[str, str]]:
        tables = []
        if hasattr(self.source_conn, "get_tables_with_ddl"):
            db_name = self.source_db_name or getattr(self.source_conn, "database_name", "") or ""
            schema_name = self.source_schema or getattr(self.source_conn, "schema_name", "") or ""
            logger.info(f"准备拉取表 DDL，db={db_name} schema={schema_name} connector={type(self.source_conn).__name__}")
            try:
                tables = self.source_conn.get_tables_with_ddl(database_name=db_name, schema_name=schema_name)
            except Exception as exc:
                logger.warning(f"获取表 DDL 失败: {exc}")
            logger.info(f"已从源库获取表 {len(tables)} 个")
        return tables

    def _normalize_view(self, view_meta: Dict[str, str]) -> ViewSourceRow:
        ddl_sql = view_meta.get("definition") or view_meta.get("ddl") or ""
        ddl_sql = self._strip_ansi(ddl_sql)
        normalized = normalize_sql(ddl_sql).lower()
        sql_hash = hashlib.md5(normalized.encode("utf-8")).hexdigest()
        return ViewSourceRow(
            view_id=None,
            view_name=view_meta.get("table_name") or view_meta.get("view_name") or view_meta.get("name") or "",
            db_name=view_meta.get("database_name") or self.sourcedb,
            ddl_sql=ddl_sql,
            sql_hash=sql_hash,
        )

    def _load_existing_view_source(self) -> Dict[str, ViewSourceRow]:
        sql = (
            "SELECT table_id as view_id, table_name as view_name, '' as db_name, ddl_sql, hash "
            "FROM dw_meta.table_source "
            f"WHERE source_system = '{self.sourcedb}' AND table_type = 'VIEW'"
        )
        result = self.meta_conn.execute({"sql_query": sql, "result_format": "list"})
        rows = self._rows_from_result(result)
        existing: Dict[str, ViewSourceRow] = {}
        for row in rows:
            key = str(row.get("view_name", "")).lower()
            existing[key] = ViewSourceRow(
                view_id=row.get("view_id"),
                view_name=row.get("view_name", ""),
                db_name="",
                ddl_sql=row.get("ddl_sql", ""),
                sql_hash=row.get("hash", "") or "",
            )
        return existing

    def _load_table_source_map(self) -> Dict[str, Dict[str, Any]]:
        sql = (
            "SELECT table_id, table_name, table_type "
            "FROM dw_meta.table_source "
            f"WHERE source_system = '{self.sourcedb}'"
        )
        result = self.meta_conn.execute({"sql_query": sql, "result_format": "list"})
        rows = self._rows_from_result(result)
        mapping: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            name = str(row.get("table_name") or "").lower()
            mapping[name] = {
                "table_id": row.get("table_id"),
                "table_type": (row.get("table_type") or "").upper(),
            }
        return mapping

    def _load_existing_table_source(self, table_type: str = "VIEW") -> Dict[str, ViewSourceRow]:
        sql = (
            "SELECT table_id as view_id, table_name as view_name, '' as db_name, ddl_sql, hash "
            "FROM dw_meta.table_source "
            f"WHERE source_system = '{self.sourcedb}' AND table_type = '{table_type}'"
        )
        result = self.meta_conn.execute({"sql_query": sql, "result_format": "list"})
        rows = self._rows_from_result(result)
        existing: Dict[str, ViewSourceRow] = {}
        for row in rows:
            key = str(row.get("view_name", "")).lower()
            existing[key] = ViewSourceRow(
                view_id=row.get("view_id"),
                view_name=row.get("view_name", ""),
                db_name="",
                ddl_sql=row.get("ddl_sql", ""),
                sql_hash=row.get("hash", "") or "",
            )
        return existing

    def _upsert_table_source(
        self, row: ViewSourceRow, existing: Optional[ViewSourceRow], table_type: str = "VIEW"
    ) -> int:
        if existing and existing.sql_hash == row.sql_hash:
            return existing.view_id or 0
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        view_name = self._escape(row.view_name)
        ddl_sql = self._escape(row.ddl_sql)
        sql_hash = self._escape(row.sql_hash)
        insert = (
            "INSERT INTO dw_meta.table_source "
            "(source_system, table_name, table_type, ddl_sql, hash, created_at, updated_at) "
            f"VALUES ('{self.sourcedb}', '{view_name}', '{table_type}', '{ddl_sql}', '{sql_hash}', '{now}', '{now}')"
        )
        self.meta_conn.execute({"sql_query": insert})
        res = self.meta_conn.execute(
            {
                "sql_query": (
                    "SELECT table_id FROM dw_meta.table_source "
                    f"WHERE source_system = '{self.sourcedb}' AND table_name = '{view_name}' "
                    "ORDER BY table_id DESC LIMIT 1"
                )
            }
        )
        rows = self._rows_from_result(res)
        return int(rows[0].get("table_id")) if rows else 0

    def _cleanup_downstream(self, view_names: List[str]):
        if not view_names:
            return
        names_sql = ",".join(f"'{self._escape(v)}'" for v in view_names)
        view_ids_sql = (
            "SELECT table_id as view_id FROM dw_meta.table_source "
            f"WHERE source_system = '{self.sourcedb}' AND table_type = 'VIEW' AND table_name IN ({names_sql})"
        )
        res = self.meta_conn.execute({"sql_query": view_ids_sql, "result_format": "list"})
        view_ids = [str(r.get("view_id")) for r in self._rows_from_result(res) if r.get("view_id")]
        if not view_ids:
            return
        id_list = ",".join(view_ids)
        self.meta_conn.execute({"sql_query": f"DELETE FROM dw_meta.ai_view_feature WHERE table_id IN ({id_list})"})
        self.meta_conn.execute(
            {"sql_query": f"DELETE FROM dw_meta.std_field_mapping WHERE source_system = '{self.sourcedb}'"}
        )
        self.meta_conn.execute(
            {
                "sql_query": (
                    "DELETE FROM dw_meta.dw_node_relation WHERE from_node_id IN "
                    f"(SELECT node_id FROM dw_meta.dw_node WHERE source_table_id IN ({id_list})) "
                    "OR to_node_id IN (SELECT node_id FROM dw_meta.dw_node WHERE source_table_id IN ({id_list}))"
                )
            }
        )
        self.meta_conn.execute({"sql_query": f"DELETE FROM dw_meta.dw_node WHERE source_table_id IN ({id_list})"})
        self.meta_conn.execute(
            {"sql_query": f"DELETE FROM dw_meta.ai_feedback WHERE object_type = 'VIEW' AND object_key IN ({names_sql})"}
        )

    def _can_skip(self, view_id: Optional[int]) -> bool:
        if not view_id:
            return False
        sql = (
            "SELECT migration_status FROM dw_meta.dw_node "
            f"WHERE source_table_id = {view_id} "
            "ORDER BY node_id DESC LIMIT 1"
        )
        res = self.meta_conn.execute({"sql_query": sql, "result_format": "list"})
        rows = self._rows_from_result(res)
        if not rows:
            return False
        status = (rows[0].get("migration_status") or "").upper()
        return status in {"REVIEWED", "IMPLEMENTED", "SKIPPED"}

    # ---------- 依赖解析与 DAG ---------- #
    def _get_or_create_external_dependency(self, raw_name: str, db_prefix: Optional[str]) -> Optional[Dict[str, Any]]:
        """
        处理跨库/DBLink 依赖：映射前缀到目标系统，查询元数据，必要时创建虚拟表。
        """
        target_system = self.sourcedb
        target_table_name = raw_name
        is_virtual = False

        # 显式 schema/db 前缀 (如 lyerp.table)
        if db_prefix:
            clean_prefix = str(db_prefix).lower()
            if clean_prefix in self.schema_system_map:
                target_system = self.schema_system_map[clean_prefix]

        # DBLink 风格 (如 table@iwcs)
        if "@" in raw_name:
            real_table_name, dblink_name = raw_name.split("@", 1)
            dblink_name = dblink_name.lower()
            if dblink_name in self.schema_system_map:
                target_system = self.schema_system_map[dblink_name]
                target_table_name = real_table_name
            else:
                # 未知 DBLink，当前系统内生成虚拟表，例如 iwcs_table
                target_system = self.sourcedb
                target_table_name = f"{dblink_name}_{real_table_name}"
                is_virtual = True

        # 优先查已有元数据
        sql = (
            "SELECT table_id, table_type FROM dw_meta.table_source "
            f"WHERE source_system = '{self._escape(target_system)}' "
            f"AND table_name = '{self._escape(target_table_name)}' LIMIT 1"
        )
        res = self.meta_conn.execute({"sql_query": sql, "result_format": "list"})
        rows = self._rows_from_result(res)
        if rows:
            return {
                "table_id": rows[0].get("table_id"),
                "table_type": (rows[0].get("table_type") or "").upper(),
                "resolved_name": target_table_name,
                "db_name": "",
            }

        # 未找到且需要虚拟表时，插入占位 EXTERNAL 节点，保证血缘不断裂
        if is_virtual:
            logger.info(f"创建虚拟表依赖: system={target_system}, table={target_table_name}")
            virtual_row = ViewSourceRow(
                view_id=None,
                view_name=target_table_name,
                db_name="",
                ddl_sql="-- Virtual table created by dependency resolution",
                sql_hash="virtual",
            )
            new_id = self._upsert_table_source(virtual_row, None, table_type="EXTERNAL")
            return {
                "table_id": new_id,
                "table_type": "EXTERNAL",
                "resolved_name": target_table_name,
                "db_name": "",
            }

        return None

    def _resolve_dependencies(
        self, feature: Dict[str, Any], table_source_map: Dict[str, Dict[str, Any]], default_db: str
    ) -> Dict[str, Any]:
        """
        基于 table_source 判定表/视图依赖类型，并构建节点所需信息。
        """
        view_deps: Set[str] = set()
        table_deps: Set[str] = set()
        unresolved: Set[str] = set()
        dep_info: Dict[str, Dict[str, Any]] = {}
        tables = feature.get("tables") or []
        for t in tables:
            alias = t.get("alias") or t.get("name") or ""
            raw_name = t.get("name") or ""
            parsed = parse_table_name_parts(raw_name, dialect=DBType.ORACLE)
            resolved_name = parsed.get("table_name") or raw_name
            dep_key = resolved_name.lower()
            db_prefix = t.get("db")
            db_name = db_prefix or default_db
            info: Optional[Dict[str, Any]] = None
            # 同库缓存优先（无前缀且非 DBLink）
            if not db_prefix and "@" not in raw_name:
                info = table_source_map.get(dep_key)
            # 跨库或缓存未命中时尝试外部解析/虚拟表
            if not info:
                external_info = self._get_or_create_external_dependency(raw_name, db_prefix)
                if external_info:
                    info = external_info
                    resolved_name = external_info.get("resolved_name") or resolved_name
                    dep_key = resolved_name.lower()
            alias_key = alias or resolved_name
            if info:
                t_type = (info.get("table_type") or "").upper()
                dep_type = "VIEW" if t_type == "VIEW" else "TABLE"
                t["source_type"] = dep_type
                t["resolved_name"] = resolved_name
                dep_info[alias_key] = {
                    "name": resolved_name,
                    "db_name": info.get("db_name", db_name),
                    "type": dep_type,
                    "source_table_id": info.get("table_id"),
                }
                if dep_type == "VIEW":
                    view_deps.add(dep_key)
                else:
                    table_deps.add(dep_key)
            else:
                unresolved.add(resolved_name)
        return {
            "view_dependencies": view_deps,
            "table_dependencies": table_deps,
            "unresolved": unresolved,
            "dep_info": dep_info,
        }

    def _build_view_dep_graph(self, analysis_results: List[Dict[str, Any]]) -> Dict[str, Set[str]]:
        view_set = {r["row"].view_name.lower() for r in analysis_results}
        graph: Dict[str, Set[str]] = {}
        for result in analysis_results:
            view_nm = result["row"].view_name.lower()
            deps = set(result["feature"].get("view_dependencies") or [])
            graph[view_nm] = {d for d in deps if d in view_set}
        return graph

    def _topo_sort(self, graph: Dict[str, Set[str]]) -> List[str]:
        indeg: Dict[str, int] = {k: 0 for k in graph}
        for deps in graph.values():
            for dep in deps:
                if dep in indeg:
                    indeg[dep] += 1
        queue = [k for k, v in indeg.items() if v == 0]
        order: List[str] = []
        while queue:
            node = queue.pop(0)
            order.append(node)
            for dep in graph.get(node, []):
                indeg[dep] -= 1
                if indeg[dep] == 0:
                    queue.append(dep)
        if len(order) < len(graph):
            logger.warning(f"检测到循环依赖，剩余未排序节点: {set(graph) - set(order)}")
            for k in graph:
                if k not in order:
                    order.append(k)
        return order

    def _find_row(self, view_name: str, views: List[ViewSourceRow]) -> Optional[ViewSourceRow]:
        for v in views:
            if v.view_name.lower() == view_name:
                return v
        return None

    # ---------- AST 落库 ---------- #
    def _upsert_ai_view_feature(self, table_id: int, feature_json: str):
        self.meta_conn.execute({"sql_query": f"DELETE FROM dw_meta.ai_view_feature WHERE table_id = {table_id}"})
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        payload = self._escape(feature_json)
        sql = (
            "INSERT INTO dw_meta.ai_view_feature (table_id, feature_json, analyzed_at) "
            f"VALUES ({table_id}, '{payload}', '{now}')"
        )
        self.meta_conn.execute({"sql_query": sql})

    def _update_table_parse_status(self, table_id: Optional[int], status: str):
        if not table_id:
            return
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        status_esc = self._escape(status.upper())
        self.meta_conn.execute(
            {
                "sql_query": (
                    "UPDATE dw_meta.table_source "
                    f"SET parse_status = '{status_esc}', updated_at = '{now}' "
                    f"WHERE table_id = {table_id}"
                )
            }
        )

    def _update_node_migration_status(self, table_id: Optional[int], status: str):
        if not table_id:
            return
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        status_esc = self._escape(status.upper())
        self.meta_conn.execute(
            {
                "sql_query": (
                    "UPDATE dw_meta.dw_node "
                    f"SET migration_status = '{status_esc}', updated_at = '{now}' "
                    f"WHERE source_table_id = {table_id}"
                )
            }
        )

    def _ensure_view_node(self, view_id: int, view_name: str) -> int:
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        view_name_esc = self._escape(view_name)
        # 仅按 source_table_id 或 source_system+table_name 识别视图节点
        fetch = self.meta_conn.execute(
            {
                "sql_query": (
                    "SELECT node_id, node_type, source_table_id FROM dw_meta.dw_node "
                    f"WHERE source_table_id = {view_id} LIMIT 1"
                )
            }
        )
        rows = self._rows_from_result(fetch)
        if not rows:
            fetch = self.meta_conn.execute(
                {
                    "sql_query": (
                        "SELECT node_id, node_type, source_table_id FROM dw_meta.dw_node "
                        f"WHERE source_system = '{self._escape(self.sourcedb)}' "
                        f"AND table_name = '{view_name_esc}' LIMIT 1"
                    )
                }
            )
            rows = self._rows_from_result(fetch)

        if rows:
            node_id = int(rows[0].get("node_id"))
            updates = []
            if (rows[0].get("node_type") or "").upper() != "VIEW":
                updates.append("node_type = 'VIEW'")
            if not rows[0].get("source_table_id"):
                updates.append(f"source_table_id = {view_id}")
            if updates:
                updates.append(f"updated_at = '{now}'")
                self.meta_conn.execute(
                    {
                        "sql_query": (
                            "UPDATE dw_meta.dw_node SET " + ", ".join(updates) + f" WHERE node_id = {node_id}"
                        )
                    }
                )
            return node_id

        insert = (
            "INSERT INTO dw_meta.dw_node "
            "(node_type, source_system, table_name, source_table_id, migration_status, created_at, updated_at) "
            f"VALUES ('VIEW', '{self._escape(self.sourcedb)}', '{view_name_esc}', {view_id}, 'NEW', '{now}', '{now}')"
        )
        self.meta_conn.execute({"sql_query": insert})
        res = self.meta_conn.execute(
            {
                "sql_query": (
                    "SELECT node_id FROM dw_meta.dw_node "
                    f"WHERE source_table_id = {view_id} ORDER BY node_id DESC LIMIT 1"
                )
            }
        )
        rows2 = self._rows_from_result(res)
        if rows2:
            return int(rows2[0].get("node_id"))
        res_fb = self.meta_conn.execute(
            {
                "sql_query": (
                    "SELECT node_id FROM dw_meta.dw_node "
                    f"WHERE source_system = '{self._escape(self.sourcedb)}' "
                    f"AND table_name = '{view_name_esc}' ORDER BY node_id DESC LIMIT 1"
                )
            }
        )
        rows_fb = self._rows_from_result(res_fb)
        if rows_fb:
            return int(rows_fb[0].get("node_id"))
        logger.error(f"无法获取 dw_node 节点(插入后查询为空)，view={view_name}, table_id={view_id}")
        return 0

    def _ensure_dependency_nodes(self, dep_info: Dict[str, Dict[str, Any]]) -> Dict[str, int]:
        nodes: Dict[str, int] = {}
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        for alias, info in dep_info.items():
            table_nm = self._escape(info.get("name") or "")
            node_type = "VIEW" if info.get("type") == "VIEW" else "TABLE"
            source_table_id = info.get("source_table_id")
            fetch = self.meta_conn.execute(
                {
                    "sql_query": (
                        "SELECT node_id FROM dw_meta.dw_node "
                        f"WHERE source_system = '{self._escape(self.sourcedb)}' "
                        f"AND table_name = '{table_nm}' LIMIT 1"
                    )
                }
            )
            rows = self._rows_from_result(fetch)
            if rows:
                node_id = int(rows[0].get("node_id"))
            else:
                insert = (
                    "INSERT INTO dw_meta.dw_node "
                    "(node_type, source_system, table_name, source_table_id, migration_status, created_at, updated_at) "
                    f"VALUES ('{node_type}', '{self._escape(self.sourcedb)}', '{table_nm}', "
                    f"{source_table_id if source_table_id is not None else 'NULL'}, 'NEW', '{now}', '{now}')"
                )
                self.meta_conn.execute({"sql_query": insert})
                res = self.meta_conn.execute(
                    {
                        "sql_query": (
                            "SELECT node_id FROM dw_meta.dw_node "
                            f"WHERE source_system = '{self._escape(self.sourcedb)}' "
                            f"AND table_name = '{table_nm}' "
                            "ORDER BY node_id DESC LIMIT 1"
                        )
                    }
                )
                rows = self._rows_from_result(res)
            if rows:
                nodes[alias] = int(rows[0].get("node_id"))
        return nodes

    def _upsert_relations(
        self, view_node_id: int, dependency_nodes: Dict[str, int], feature: Dict[str, Any], dep_info: Dict[str, Dict]
    ):
        for alias, node_id in dependency_nodes.items():
            info = dep_info.get(alias) or {}
            detail = json.dumps(
                {"alias": alias, "dependency_type": info.get("type"), "table_name": info.get("name")},
                ensure_ascii=True,
            )
            self._insert_relation(view_node_id, node_id, "VIEW_DEP", detail)

        for join in feature.get("joins") or []:
            left_alias = join.get("left_alias") or ""
            right_alias = join.get("right_alias") or ""
            left_id = dependency_nodes.get(left_alias)
            right_id = dependency_nodes.get(right_alias)
            if left_id and right_id:
                detail = json.dumps(join, ensure_ascii=True)
                self._insert_relation(left_id, right_id, "JOIN", detail)

    def _insert_relation(self, from_id: int, to_id: int, relation_type: str, detail: str = ""):
        self.meta_conn.execute(
            {
                "sql_query": (
                    "DELETE FROM dw_meta.dw_node_relation "
                    f"WHERE from_node_id = {from_id} AND to_node_id = {to_id} AND relation_type = '{relation_type}'"
                )
            }
        )
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        detail_safe = self._escape(detail)
        sql = (
            "INSERT INTO dw_meta.dw_node_relation "
            "(from_node_id, to_node_id, relation_type, relation_detail, created_at, updated_at) "
            f"VALUES ({from_id}, {to_id}, '{relation_type}', '{detail_safe}', '{now}', '{now}')"
        )
        self.meta_conn.execute({"sql_query": sql})

    # ---------- 标准字段与反馈 ---------- #
    def _prepare_std_items(
        self,
        feature: Dict[str, Any],
        alias_map: Dict[str, Dict],
        view_name: str,
        db_name: str,
    ) -> List[Dict[str, str]]:
        items: List[Dict[str, str]] = []
        for col in feature.get("columns") or []:
            src_alias = col.get("source_table_alias")
            src_col = col.get("source_column")
            table_name = view_name
            if src_alias and src_alias in alias_map:
                table_meta = alias_map[src_alias]
                table_name = table_meta.get("resolved_name") or table_meta.get("name") or view_name
            items.append(
                {
                    "std_field_name": self._to_snake(col.get("output_name") or ""),
                    "std_field_name_cn": col.get("output_name") or "",
                    "data_type_std": "string",
                    "source_table": table_name,
                    "source_column": src_col or (col.get("output_name") or ""),
                    "source_db": db_name,
                    "expression_sql": col.get("expression_sql") or "",
                    "ai_note": "auto-generated, 待确认",
                }
            )
        return items

    def _persist_std_and_feedback(self, view_name: str, db_name: str, items: List[Dict[str, str]]):
        if not items:
            return
        if self.llm is None:
            try:
                self.llm = LLMBaseModel.create_model(model_name="default", agent_config=self.agent_config)
            except Exception as exc:  # pragma: no cover
                logger.warning(f"初始化 LLM 失败，改为人工确认模式: {exc}")
                self.llm = None

        for item in items:
            std_id = self._get_or_create_std_field(item)
            self._upsert_std_mapping(std_id, item)
            ai_value = json.dumps(
                {
                    "std_field_name": item["std_field_name"],
                    "std_field_name_cn": item["std_field_name_cn"],
                    "data_type_std": item["data_type_std"],
                },
                ensure_ascii=True,
            )
            human_value = self._interactive_confirm(item)
            ai_value_esc = self._escape(ai_value)
            human_value_esc = self._escape(human_value)
            expr_esc = self._escape(item.get("expression_sql") or "")
            feedback_sql = (
                "INSERT INTO dw_meta.ai_feedback "
                "(object_type, object_key, suggestion_type, ai_value, human_value, context_feature, created_at) "
                f"VALUES ('VIEW', '{self._escape(view_name)}', 'STD_FIELD', '{ai_value_esc}', "
                f"'{human_value_esc}', '{expr_esc}', '{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}')"
            )
            self.meta_conn.execute({"sql_query": feedback_sql})

    def _get_or_create_std_field(self, item: Dict[str, str]) -> int:
        fetch = self.meta_conn.execute(
            {
                "sql_query": (
                    "SELECT std_field_id FROM dw_meta.std_field "
                    f"WHERE std_field_name = '{self._escape(item['std_field_name'])}' LIMIT 1"
                )
            }
        )
        rows = self._rows_from_result(fetch)
        if rows:
            return int(rows[0].get("std_field_id"))
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        std_name = self._escape(item["std_field_name"])
        std_name_cn = self._escape(item["std_field_name_cn"])
        data_type_std = self._escape(item["data_type_std"])
        insert = (
            "INSERT INTO dw_meta.std_field "
            "(std_field_name, std_field_name_cn, data_type_std, default_agg, is_active, created_at, updated_at) "
            f"VALUES ('{std_name}', '{std_name_cn}', '{data_type_std}', "
            f"'none', 1, '{now}', '{now}')"
        )
        self.meta_conn.execute({"sql_query": insert})
        res = self.meta_conn.execute(
            {
                "sql_query": (
                    "SELECT std_field_id FROM dw_meta.std_field "
                    f"WHERE std_field_name = '{std_name}' ORDER BY std_field_id DESC LIMIT 1"
                )
            }
        )
        rows2 = self._rows_from_result(res)
        if rows2:
            return int(rows2[0].get("std_field_id"))
        raise RuntimeError(f"无法获取 std_field_id: {item['std_field_name']}")

    def _upsert_std_mapping(self, std_field_id: int, item: Dict[str, str]):
        delete = (
            "DELETE FROM dw_meta.std_field_mapping "
            f"WHERE source_system = '{self.sourcedb}' "
            f"AND source_db = '{self._escape(item['source_db'])}' "
            f"AND source_table = '{self._escape(item['source_table'])}' "
            f"AND source_column = '{self._escape(item['source_column'])}'"
        )
        self.meta_conn.execute({"sql_query": delete})
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        expr = self._escape(item.get("expression_sql") or "")
        insert = (
            "INSERT INTO dw_meta.std_field_mapping "
            "(source_system, source_db, source_table, source_column, source_column_comment, source_data_type, "
            "std_field_id, transform_expr, is_primary_key, is_business_key, is_partition_key, is_active, remark, "
            "created_at, updated_at) "
            f"VALUES ('{self.sourcedb}', '{self._escape(item['source_db'])}', '{self._escape(item['source_table'])}', "
            f"'{self._escape(item['source_column'])}', '', NULL, {std_field_id}, '{expr}', "
            "0, 0, 0, 1, 'auto-generated', "
            f"'{now}', '{now}')"
        )
        self.meta_conn.execute({"sql_query": insert})

    def _interactive_confirm(self, item: Dict[str, str]) -> str:
        prompt = (
            f"视图字段 [{item['source_table']}.{item['source_column']}] -> std_field "
            f"{item['std_field_name']} (默认中文名: {item['std_field_name_cn']})\n"
            "请输入确认/修改后的中文名，直接回车表示接受当前值: "
        )
        try:
            human = input(prompt)
        except Exception:
            return ""
        return human or ""

    def _to_snake(self, name: str) -> str:
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

    def _escape(self, value: str) -> str:
        return (value or "").replace("'", "''")

    def _rows_from_result(self, result: Any) -> List[Dict[str, Any]]:
        if not result or not getattr(result, "success", False):
            return []
        data = getattr(result, "sql_return", None)
        if isinstance(data, list):
            if data and isinstance(data[0], dict):
                return data
            return []
        if isinstance(data, str):
            try:
                reader = csv.DictReader(StringIO(data))
                return [dict(row) for row in reader if row]
            except Exception:
                return []
        return []

    def _strip_ansi(self, text: str) -> str:
        ansi_re = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]", re.IGNORECASE)
        return ansi_re.sub("", text)

def run_import_view(agent_config: AgentConfig, db_manager: DBManager, args) -> Dict[str, Any]:
    sourcedb_configs = agent_config.source_db_configs()
    if not sourcedb_configs:
        raise ValueError("未配置任何 sourcedb，请检查 agent.yml")

    results: Dict[str, Dict[str, Any]] = {}
    names = list(sourcedb_configs.keys())

    # 先全量同步表/视图 DDL
    for name in names:
        logger.info(f"开始同步 sourcedb={name}")
        runner = ImportViewRunner(
            agent_config=agent_config,
            db_manager=db_manager,
            namespace=args.namespace,
            sourcedb=name,
            strategy=args.update_strategy,
        )
        results[name] = {"sync": runner.sync_table_and_views()}

    # 再统一做 AST 解析
    for name in names:
        logger.info(f"开始解析 sourcedb={name}")
        runner = ImportViewRunner(
            agent_config=agent_config,
            db_manager=db_manager,
            namespace=args.namespace,
            sourcedb=name,
            strategy=args.update_strategy,
        )
        parse_result = runner.analyze_views()
        results[name]["analyze"] = parse_result

    summary = {
        "total_new_views": sum(r["analyze"].get("new_views", 0) for r in results.values()),
        "total_processed": sum(r["analyze"].get("processed", 0) for r in results.values()),
        "total_failed": sum(r["analyze"].get("failed", 0) for r in results.values()),
        "total_reused": sum(r["analyze"].get("reused", 0) for r in results.values()),
    }
    return {"status": "success", "results": results, "summary": summary}


if __name__ == "__main__":
    print("请通过 datus-agent import-view 调用本模块")
    sys.exit(1)
