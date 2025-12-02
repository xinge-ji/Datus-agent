from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

from datus.configuration.agent_config import AgentConfig
from datus.models.base import LLMBaseModel
from datus.tools.db_tools.db_manager import DBManager
from datus.utils.ast_analyzer import AstAnalyzer
from datus.utils.constants import DBType
from datus.utils.loggings import get_logger
from datus.utils.sql_utils import extract_table_names, normalize_sql, parse_table_name_parts

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
        self.source_conn = self.db_manager.get_conn(namespace, sourcedb)
        # 元数据写 dw_meta，使用当前 database 连接即可，SQL 中全量限定 dw_meta.*
        meta_logic_db = getattr(agent_config, "current_database", None) or sourcedb
        self.meta_conn = self.db_manager.get_conn(namespace, meta_logic_db)
        self.ast = AstAnalyzer(dialect="oracle")
        self.llm: Optional[LLMBaseModel] = None

    # ---------- 对外入口 ---------- #
    def run(self) -> Dict[str, Any]:
        """执行全流程，返回简单统计。"""
        all_views = self._load_views()
        existing = self._load_existing_view_source()

        if self.strategy == "overwrite":
            self._cleanup_downstream(list(existing.keys()))

        # 1) 计算 hash & 决定待处理队列
        to_process: List[ViewSourceRow] = []
        reused_views: List[ViewSourceRow] = []
        new_view_ids: List[int] = []

        for view_meta in all_views:
            row = self._normalize_view(view_meta)
            key = row.view_name.lower()
            latest = existing.get(key)
            if self.strategy == "incremental" and latest and latest.sql_hash == row.sql_hash:
                # 未变更，检查 dw_node 状态
                if self._can_skip(latest.view_id):
                    reused_views.append(latest)
                    continue
                reused_views.append(latest)
                to_process.append(latest)
                continue

            # 变更或 overwrite，插入新 view_source
            new_id = self._insert_view_source(row)
            row.view_id = new_id
            to_process.append(row)
            new_view_ids.append(new_id)

        # 2) DAG 排序（仅视图间依赖）
        view_name_set = {v.view_name.lower() for v in to_process + reused_views}
        dep_graph, table_deps = self._build_dependency_graph(to_process + reused_views, view_name_set)
        topo = self._topo_sort(dep_graph)

        # 3) AST 分析 + 落表
        processed, failed = 0, 0
        for view_name in topo:
            row = self._find_row(view_name, to_process + reused_views)
            if not row or not row.view_id:
                continue
            try:
                feature = self.ast.analyze_view(row.ddl_sql, row.view_name)
                feature["status"] = "OK"
                feature["table_dependencies"] = sorted(table_deps.get(view_name.lower(), []))
                feature_json = json.dumps(feature, ensure_ascii=True)
                self._upsert_ai_view_feature(row.view_id, feature_json)
                node_id = self._ensure_view_node(row.view_id, row.view_name, row.db_name)
                alias_map = {t["alias"]: t for t in feature.get("tables", [])}
                table_nodes = self._ensure_table_nodes(feature, row.db_name)
                self._upsert_relations(node_id, table_nodes, feature, alias_map)
                std_items = self._prepare_std_items(feature, alias_map, row.view_name, row.db_name)
                self._persist_std_and_feedback(row.view_name, row.db_name, std_items)
                processed += 1
            except Exception as exc:  # pragma: no cover - 运行时错误写入 feature
                logger.error("视图解析失败 %s: %s", row.view_name, exc)
                error_json = json.dumps({"status": "ERROR", "error": str(exc)}, ensure_ascii=True)
                self._upsert_ai_view_feature(row.view_id, error_json)
                failed += 1

        return {
            "new_views": len(new_view_ids),
            "processed": processed,
            "failed": failed,
            "reused": len(reused_views),
        }

    # ---------- 视图与 hash ---------- #
    def _load_views(self) -> List[Dict[str, str]]:
        """从源库读取视图 DDL。"""
        views = []
        if hasattr(self.source_conn, "get_views_with_ddl"):
            views = self.source_conn.get_views_with_ddl()
        else:
            logger.warning("连接器不支持 get_views_with_ddl，无法导入视图")
        return views

    def _normalize_view(self, view_meta: Dict[str, str]) -> ViewSourceRow:
        ddl_sql = view_meta.get("definition") or view_meta.get("ddl") or ""
        normalized = normalize_sql(ddl_sql).lower()
        sql_hash = hashlib.md5(normalized.encode("utf-8")).hexdigest()
        return ViewSourceRow(
            view_id=None,
            view_name=view_meta.get("table_name") or view_meta.get("view_name") or "",
            db_name=view_meta.get("database_name") or self.sourcedb,
            ddl_sql=ddl_sql,
            sql_hash=sql_hash,
        )

    def _load_existing_view_source(self) -> Dict[str, ViewSourceRow]:
        sql = (
            "SELECT view_id, view_name, db_name, ddl_sql, hash "
            "FROM dw_meta.view_source "
            f"WHERE source_system = '{self.sourcedb}'"
        )
        result = self.meta_conn.execute({"sql_query": sql, "result_format": "list"})
        rows = result.sql_return if result and result.success else []
        existing: Dict[str, ViewSourceRow] = {}
        for row in rows:
            key = str(row.get("view_name", "")).lower()
            existing[key] = ViewSourceRow(
                view_id=row.get("view_id"),
                view_name=row.get("view_name", ""),
                db_name=row.get("db_name", ""),
                ddl_sql=row.get("ddl_sql", ""),
                sql_hash=row.get("hash", "") or "",
            )
        return existing

    def _insert_view_source(self, row: ViewSourceRow) -> int:
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        view_name = self._escape(row.view_name)
        db_name = self._escape(row.db_name)
        ddl_sql = self._escape(row.ddl_sql)
        sql_hash = self._escape(row.sql_hash)
        sql = (
            "INSERT INTO dw_meta.view_source "
            "(source_system, view_name, db_name, ddl_sql, hash, created_at, updated_at) "
            f"VALUES ('{self.sourcedb}', '{view_name}', '{db_name}', "
            f"'{ddl_sql}', '{sql_hash}', '{now}', '{now}')"
        )
        self.meta_conn.execute({"sql_query": sql})
        fetch_sql = (
            "SELECT view_id FROM dw_meta.view_source "
            f"WHERE source_system = '{self.sourcedb}' AND view_name = '{view_name}' "
            "ORDER BY view_id DESC LIMIT 1"
        )
        res = self.meta_conn.execute({"sql_query": fetch_sql, "result_format": "list"})
        if res and res.success and res.sql_return:
            return int(res.sql_return[0].get("view_id"))
        raise RuntimeError(f"无法获取 view_id: {row.view_name}")

    def _cleanup_downstream(self, view_names: List[str]):
        if not view_names:
            return
        names_sql = ",".join(f"'{self._escape(v)}'" for v in view_names)
        view_ids_sql = (
            "SELECT view_id FROM dw_meta.view_source "
            f"WHERE source_system = '{self.sourcedb}' AND view_name IN ({names_sql})"
        )
        res = self.meta_conn.execute({"sql_query": view_ids_sql, "result_format": "list"})
        view_ids = [str(r.get("view_id")) for r in (res.sql_return if res and res.success else []) if r.get("view_id")]
        if not view_ids:
            return
        id_list = ",".join(view_ids)
        # 清理 ai_view_feature
        self.meta_conn.execute({"sql_query": f"DELETE FROM dw_meta.ai_view_feature WHERE view_id IN ({id_list})"})
        # 清理 mapping/节点/关系
        self.meta_conn.execute(
            {"sql_query": f"DELETE FROM dw_meta.std_field_mapping WHERE source_system = '{self.sourcedb}'"}
        )
        self.meta_conn.execute(
            {
                "sql_query": (
                    "DELETE FROM dw_meta.dw_node_relation WHERE from_node_id IN "
                    f"(SELECT node_id FROM dw_meta.dw_node WHERE source_view_id IN ({id_list})) "
                    "OR to_node_id IN (SELECT node_id FROM dw_meta.dw_node WHERE source_view_id IN ({id_list}))"
                )
            }
        )
        self.meta_conn.execute({"sql_query": f"DELETE FROM dw_meta.dw_node WHERE source_view_id IN ({id_list})"})
        self.meta_conn.execute(
            {"sql_query": f"DELETE FROM dw_meta.ai_feedback WHERE object_type = 'VIEW' AND object_key IN ({names_sql})"}
        )

    def _can_skip(self, view_id: Optional[int]) -> bool:
        if not view_id:
            return False
        sql = (
            "SELECT migration_status FROM dw_meta.dw_node "
            f"WHERE source_view_id = {view_id} "
            "ORDER BY node_id DESC LIMIT 1"
        )
        res = self.meta_conn.execute({"sql_query": sql, "result_format": "list"})
        if not res or not res.success or not res.sql_return:
            return False
        status = (res.sql_return[0].get("migration_status") or "").upper()
        return status in {"REVIEWED", "IMPLEMENTED", "SKIPPED"}

    # ---------- DAG ---------- #
    def _build_dependency_graph(
        self, views: List[ViewSourceRow], view_name_set: Set[str]
    ) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
        graph: Dict[str, Set[str]] = {}
        table_deps: Dict[str, Set[str]] = {}
        for row in views:
            deps = extract_table_names(row.ddl_sql, dialect=DBType.ORACLE)
            view_deps = set()
            physical = set()
            for dep in deps:
                dep_norm = dep.lower()
                if dep_norm in view_name_set:
                    view_deps.add(dep_norm)
                else:
                    physical.add(dep_norm)
            graph[row.view_name.lower()] = view_deps
            table_deps[row.view_name.lower()] = physical
        return graph, table_deps

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
            logger.warning("检测到循环依赖，剩余未排序节点: %s", set(graph) - set(order))
            # 将剩余节点按名称补上
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
    def _upsert_ai_view_feature(self, view_id: int, feature_json: str):
        self.meta_conn.execute({"sql_query": f"DELETE FROM dw_meta.ai_view_feature WHERE view_id = {view_id}"})
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        payload = self._escape(feature_json)
        sql = (
            "INSERT INTO dw_meta.ai_view_feature (view_id, feature_json, analyzed_at) "
            f"VALUES ({view_id}, '{payload}', '{now}')"
        )
        self.meta_conn.execute({"sql_query": sql})

    def _ensure_view_node(self, view_id: int, view_name: str, db_name: str) -> int:
        fetch = self.meta_conn.execute(
            {"sql_query": f"SELECT node_id FROM dw_meta.dw_node WHERE source_view_id = {view_id} LIMIT 1"}
        )
        if fetch and fetch.success and fetch.sql_return:
            return int(fetch.sql_return[0].get("node_id"))
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        view_name_esc = self._escape(view_name)
        db_name_esc = self._escape(db_name)
        insert = (
            "INSERT INTO dw_meta.dw_node "
            "(node_type, db_name, table_name, source_view_id, migration_status, created_at, updated_at) "
            f"VALUES ('SRC_VIEW', '{db_name_esc}', '{view_name_esc}', {view_id}, 'NEW', '{now}', '{now}')"
        )
        self.meta_conn.execute({"sql_query": insert})
        res = self.meta_conn.execute(
            {
                "sql_query": (
                    "SELECT node_id FROM dw_meta.dw_node "
                    f"WHERE source_view_id = {view_id} ORDER BY node_id DESC LIMIT 1"
                )
            }
        )
        if res and res.success and res.sql_return:
            return int(res.sql_return[0].get("node_id"))
        raise RuntimeError(f"无法获取 dw_node 节点: {view_name}")

    def _ensure_table_nodes(self, feature: Dict[str, Any], db_name: str) -> Dict[str, int]:
        nodes: Dict[str, int] = {}
        tables = feature.get("tables") or []
        for t in tables:
            table_name = t.get("name") or ""
            alias = t.get("alias") or table_name
            parsed = parse_table_name_parts(table_name, dialect=DBType.ORACLE)
            table_db = self._escape(parsed.get("database_name") or db_name)
            table_nm = self._escape(parsed.get("table_name") or "")
            fetch = self.meta_conn.execute(
                {
                    "sql_query": (
                        "SELECT node_id FROM dw_meta.dw_node "
                        f"WHERE db_name = '{table_db}' AND table_name = '{table_nm}' LIMIT 1"
                    )
                }
            )
            node_id = None
            if fetch and fetch.success and fetch.sql_return:
                node_id = int(fetch.sql_return[0].get("node_id"))
            else:
                now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                insert = (
                    "INSERT INTO dw_meta.dw_node "
                    "(node_type, db_name, table_name, migration_status, created_at, updated_at) "
                    f"VALUES ('ODS_TABLE', '{table_db}', '{table_nm}', 'NEW', '{now}', '{now}')"
                )
                self.meta_conn.execute({"sql_query": insert})
                res = self.meta_conn.execute(
                    {
                        "sql_query": (
                            "SELECT node_id FROM dw_meta.dw_node "
                            f"WHERE db_name = '{table_db}' AND table_name = '{table_nm}' "
                            "ORDER BY node_id DESC LIMIT 1"
                        )
                    }
                )
                if res and res.success and res.sql_return:
                    node_id = int(res.sql_return[0].get("node_id"))
            if node_id is not None:
                nodes[alias] = node_id
        return nodes

    def _upsert_relations(
        self, view_node_id: int, table_nodes: Dict[str, int], feature: Dict[str, Any], alias_map: Dict[str, Dict]
    ):
        # 引用关系
        for alias, node_id in table_nodes.items():
            self._insert_relation(view_node_id, node_id, "VIEW_DEP", json.dumps({"alias": alias}))

        # join 关系
        for join in feature.get("joins") or []:
            left_alias = join.get("left_alias") or ""
            right_alias = join.get("right_alias") or ""
            left_id = table_nodes.get(left_alias)
            right_id = table_nodes.get(right_alias)
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
                table_name = alias_map[src_alias].get("name") or view_name
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
            except Exception as exc:  # pragma: no cover - 无网络情况下跳过
                logger.warning("初始化 LLM 失败，改为人工确认模式: %s", exc)
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
        if fetch and fetch.success and fetch.sql_return:
            return int(fetch.sql_return[0].get("std_field_id"))
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
        if res and res.success and res.sql_return:
            return int(res.sql_return[0].get("std_field_id"))
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

    # ---------- 工具 ---------- #
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


def run_import_view(agent_config: AgentConfig, db_manager: DBManager, args) -> Dict[str, Any]:
    runner = ImportViewRunner(
        agent_config=agent_config,
        db_manager=db_manager,
        namespace=args.namespace,
        sourcedb=args.sourcedb,
        strategy=args.update_strategy,
    )
    return runner.run()


if __name__ == "__main__":
    print("请通过 datus-agent import-view 调用本模块")
    sys.exit(1)
