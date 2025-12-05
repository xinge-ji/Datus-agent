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
from datus.tools.llms_tools.classify_layer import classify_view_layer
from rich.prompt import Prompt
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

    def __init__(
        self, agent_config: AgentConfig, db_manager: DBManager, namespace: str, sourcedb: str, strategy: str, step: str = "all"
    ):
        self.agent_config = agent_config
        self.db_manager = db_manager
        self.namespace = namespace
        self.sourcedb = sourcedb
        self.strategy = strategy
        self.step = step
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
            "cosmic_pro_secd": "yunuopg_secd",
            "cosmic_pro_lycus": "yunuopg_lycus"
        }

    def run(self) -> Dict[str, Any]:
        """根据 step 参数调度执行。"""
        stats: Dict[str, Any] = {"step": self.step, "details": {}}

        if self.step in {"all", "import"}:
            stats["details"]["import"] = self.run_import_ddl()

        if self.step in {"all", "analyze"}:
            stats["details"]["analyze"] = self.run_ast_analysis()

        if self.step in {"all", "classify"}:
            stats["details"]["classify"] = self.run_layer_classification()

        if self.step in {"all", "naming"}:
            stats["details"]["naming"] = self.run_naming()

        return stats

    def run_import_ddl(self) -> Dict[str, int]:
        """阶段1：导入表/视图 DDL。"""
        all_tables = self._load_tables()
        all_views = self._load_views()
        table_existing = self._load_existing_table_source(table_type="TABLE")
        view_existing = self._load_existing_view_source()

        if self.strategy == "overwrite":
            self._cleanup_downstream(list(view_existing.keys()))

        added = updated = skipped = 0

        for tbl in all_tables:
            row = self._normalize_view(tbl)
            key = row.view_name.lower()
            existing = table_existing.get(key)
            table_id, changed = self._upsert_table_source(row, existing, table_type="TABLE")
            row.view_id = table_id
            table_existing[key] = row
            if existing:
                if changed:
                    updated += 1
                else:
                    skipped += 1
            else:
                added += 1

        for view_meta in all_views:
            row = self._normalize_view(view_meta)
            key = row.view_name.lower()
            existing = view_existing.get(key)
            view_id, changed = self._upsert_table_source(row, existing, table_type="VIEW")
            row.view_id = view_id
            view_existing[key] = row
            if existing:
                if changed:
                    updated += 1
                else:
                    skipped += 1
            else:
                added += 1

        logger.info(f"DDL 导入完成: 新增 {added}, 更新 {updated}, 跳过 {skipped}")
        return {"added": added, "updated": updated, "skipped": skipped}

    def run_ast_analysis(self) -> Dict[str, int]:
        """阶段2：AST 分析与血缘落库。"""
        logger.info(">>> 阶段 2: 开始 AST 分析 ...")
        views_to_process = self._load_views_from_meta_for_analysis()
        if not views_to_process:
            return {"success": 0, "failed": 0, "skipped": 0}

        table_source_map = self._load_table_source_map()
        success = failed = skipped = 0

        for row in views_to_process:
            view_id = row.get("view_id")
            view_name = row.get("view_name") or ""
            ddl_sql = row.get("ddl_sql") or ""
            parse_status = (row.get("parse_status") or "").upper()
            current_hash = row.get("hash") or ""
            prev_hash = self._get_feature_hash(view_id)

            if (
                self.strategy == "incremental"
                and parse_status == "PARSED"
                and prev_hash
                and prev_hash == current_hash
            ):
                skipped += 1
                continue

            try:
                feature = self.ast.analyze_view(ddl_sql, view_name)
                deps = self._resolve_dependencies(feature, table_source_map, row.get("db_name") or self.sourcedb)
                feature["status"] = "OK"
                feature["source_hash"] = current_hash
                feature["view_dependencies"] = sorted(deps["view_dependencies"])
                feature["table_dependencies"] = sorted(deps["table_dependencies"])
                feature["unresolved_dependencies"] = sorted(deps["unresolved"])

                feature_json = json.dumps(feature, ensure_ascii=True)
                self._upsert_ai_view_feature(view_id, feature_json)
                self._update_table_parse_status(view_id, "PARSED")
                self._update_node_migration_status(view_id, "ANALYZED")

                view_node_id = self._ensure_view_node(view_id, view_name)
                dependency_nodes = self._ensure_dependency_nodes(deps["dep_info"])
                self._upsert_relations(view_node_id, dependency_nodes, feature, deps["dep_info"])
                success += 1
            except Exception as exc:  # pragma: no cover
                logger.error(f"视图 {view_name} 解析失败: {exc}")
                error_json = json.dumps({"status": "ERROR", "error": str(exc)}, ensure_ascii=True)
                self._upsert_ai_view_feature(view_id, error_json)
                self._update_table_parse_status(view_id, "FAILED")
                self._update_node_migration_status(view_id, "AST_FAILED")
                failed += 1

        logger.info(f"AST 分析完成: 成功 {success}, 失败 {failed}, 跳过 {skipped}")
        return {"success": success, "failed": failed, "skipped": skipped}

    def run_layer_classification(self) -> Dict[str, int]:
        """阶段3：AI 分层 + 人工确认。"""
        logger.info(">>> 阶段 3: 开始 AI 分层确认 ...")

        if self.llm is None:
            try:
                self.llm = LLMBaseModel.create_model(model_name="default", agent_config=self.agent_config)
            except Exception as exc:
                logger.warning(f"LLM 初始化失败: {exc}")
                return {"error": "LLM init failed"}

        nodes_data = self._load_nodes_for_classification()
        if not nodes_data:
            return {"processed": 0, "skipped": 0}

        dep_graph = self._build_graph_from_nodes(nodes_data)
        topo_order = self._topo_sort(dep_graph)
        nodes_map = {n["view_name"].lower(): n for n in nodes_data}
        
        logger.info("\n".join(topo_order))

        layer_cache = {n["view_name"].lower(): n["human_layer_final"] for n in nodes_data if n.get("human_layer_final")}
        processed = skipped = 0

        for view_name_lower in topo_order:
            node = nodes_map.get(view_name_lower)
            if not node:
                continue

            if (
                self.strategy == "incremental"
                and node.get("migration_status") in ["REVIEWED", "IMPLEMENTED"]
            ):
                skipped += 1
                continue

            view_name = node["view_name"]
            view_id = node["source_table_id"]
            feature = json.loads(node["feature_json"]) if node.get("feature_json") else {}

            dependencies_ctx = self._build_dependencies_ctx(
                feature,
                {},
                {},
                layer_cache,
            )

            print(f"\n 正在分析视图: [cyan]{view_name}[/cyan] ...")
            ai_result = classify_view_layer(
                model=self.llm,
                view_name=view_name,
                feature=feature,
                dependencies=dependencies_ctx,
                ddl_sql=node.get("ddl_sql", ""),
            )

            print(f"\n视图: [bold]{view_name}[/bold]")
            dep_names = ", ".join([d.get("name") for d in dependencies_ctx]) if dependencies_ctx else "无"
            print(f"依赖: {dep_names}")
            print(
                f"AI 建议: [green]{ai_result.get('layer', 'OTHER')}[/green] "
                f"(置信度: {ai_result.get('confidence', 0.0)})"
            )
            print(f"AI 描述: {ai_result.get('description', '')}")

            human_layer = self._interactive_confirm_layer(view_name, ai_result.get("layer", "OTHER"))
            layer_cache[view_name_lower] = human_layer

            self._update_dw_node_layer_info(
                view_id=view_id,
                ai_suggest=ai_result.get("layer", "OTHER"),
                ai_desc=ai_result.get("description", ""),
                ai_conf=ai_result.get("confidence", 0.0),
                human_final=human_layer,
            )
            processed += 1

        logger.info(f"分层确认完成: 已处理 {processed}, 跳过 {skipped}")
        return {"processed": processed, "skipped": skipped}

    def run_naming(self) -> Dict[str, int]:
        """阶段4：AI+人工确认标准化字段命名。"""
        logger.info(">>> 阶段 4: 开始字段命名 ...")
        if self.llm is None:
            try:
                self.llm = LLMBaseModel.create_model(model_name="default", agent_config=self.agent_config)
            except Exception as exc:
                logger.warning(f"LLM 初始化失败: {exc}")
                return {"error": "LLM init failed"}

        views = self._load_views_for_naming()
        if not views:
            return {"processed": 0, "skipped": 0, "mapped": 0}

        processed = skipped = mapped = 0
        existing_std_names = self._load_existing_std_names()

        for row in views:
            view_name = row.get("view_name") or ""
            view_hash = row.get("hash") or ""
            feature_json = row.get("feature_json") or ""
            feature = json.loads(feature_json) if feature_json else {}

            if not feature.get("columns"):
                logger.info(f"视图 {view_name} 缺少字段信息，跳过命名")
                skipped += 1
                continue

            if self.strategy == "incremental":
                has_mapping = self._has_existing_mapping(view_name)
                feature_hash = feature.get("source_hash")
                if has_mapping and feature_hash and feature_hash == view_hash:
                    skipped += 1
                    continue

            if self.strategy == "overwrite":
                self._delete_std_mapping(view_name)

            alias_map = {(t.get("alias") or t.get("resolved_name") or t.get("name")): t for t in feature.get("tables", [])}

            for col in feature.get("columns") or []:
                std_en, std_cn = self._suggest_std_field_names(
                    view_name=view_name,
                    column=col,
                    banned_names=existing_std_names,
                )
                std_en, std_cn = self._interactive_confirm_naming(view_name, col.get("output_name") or col.get("source_column") or "", std_en, std_cn)
                existing_std_names.add(std_en)
                std_id = self._get_or_create_std_field(
                    {
                        "std_field_name": std_en,
                        "std_field_name_cn": std_cn,
                        "data_type_std": "string",
                    }
                )
                source_alias = col.get("source_table_alias")
                source_table = view_name
                if source_alias and source_alias in alias_map:
                    table_meta = alias_map[source_alias]
                    source_table = table_meta.get("resolved_name") or table_meta.get("name") or view_name
                item = {
                    "source_db": row.get("db_name") or self.sourcedb,
                    "source_table": source_table,
                    "source_column": col.get("source_column") or (col.get("output_name") or ""),
                    "expression_sql": col.get("expression_sql") or "",
                }
                self._upsert_std_mapping(std_id, item)
                mapped += 1

            processed += 1

        logger.info(f"字段命名完成: 已处理视图 {processed}, 跳过 {skipped}, 新增/更新映射 {mapped}")
        return {"processed": processed, "skipped": skipped, "mapped": mapped}

    def _build_dependencies_ctx(
        self,
        feature: Dict[str, Any],
        dep_nodes_info: Dict[str, Dict[str, Any]],
        dep_features_cache: Dict[str, str],
        node_layer_cache: Dict[str, str],
    ) -> List[Dict[str, Any]]:
        """构建 AI 所需的依赖上下文。"""
        deps = (feature.get("view_dependencies") or []) + (feature.get("table_dependencies") or [])
        ctx: List[Dict[str, Any]] = []
        for dep_name in deps:
            dep_key = dep_name.lower()
            node_info = dep_nodes_info.get(dep_key, {})
            known_layer = (
                node_layer_cache.get(dep_key)
                or node_info.get("human_layer_final")
                or node_info.get("ai_layer_suggest")
                or "UNKNOWN"
            )

            ctx_item: Dict[str, Any] = {
                "name": dep_name,
                "type": node_info.get("node_type")
                or ("VIEW" if dep_name in (feature.get("view_dependencies") or []) else "TABLE"),
                "layer": known_layer,
                "node_type": node_info.get("node_type") or "UNKNOWN",
                "ai_description": node_info.get("ai_description") or "",
            }

            feat_raw = dep_features_cache.get(dep_key)
            if feat_raw:
                # 避免 prompt 过长，截断到 3000 字符
                ctx_item["feature_json"] = feat_raw[:3000]

            ctx.append(ctx_item)

        return ctx

    def _interactive_confirm_layer(self, view_name: str, ai_suggest: str) -> str:
        choices = ["DIM", "DWD", "DWS", "OTHER"]
        default_choice = ai_suggest if ai_suggest in choices else "OTHER"
        prompt_text = f"请确认 [cyan]{view_name}[/cyan] 的数仓层级"
        user_input = Prompt.ask(prompt_text, choices=choices, default=default_choice, show_choices=True)
        return user_input

    def _update_dw_node_layer_info(
        self, view_id: int, ai_suggest: str, ai_desc: str, ai_conf: float, human_final: str
    ):
        if not view_id:
            return
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        ai_desc_esc = self._escape(ai_desc)
        sql = (
            "UPDATE dw_meta.dw_node SET "
            f"ai_layer_suggest = '{ai_suggest}', "
            f"ai_description = '{ai_desc_esc}', "
            f"ai_confidence = {ai_conf}, "
            f"human_layer_final = '{human_final}', "
            "migration_status = 'REVIEWED', "
            f"updated_at = '{now}' "
            f"WHERE source_table_id = {view_id}"
        )
        try:
            self.meta_conn.execute({"sql_query": sql})
        except Exception as e:  # pragma: no cover
            logger.error(f"更新节点层级信息失败 ID={view_id}: {e}")

    def _load_dep_nodes_info(self, analysis_results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """批量查询依赖的 dw_node 信息（node_type/human_layer_final/ai_description/ai_layer_suggest）。"""
        dep_names: Set[str] = set()
        for r in analysis_results:
            feature = r.get("feature") or {}
            dep_names.update([d.lower() for d in feature.get("view_dependencies") or []])
            dep_names.update([d.lower() for d in feature.get("table_dependencies") or []])
        if not dep_names:
            return {}
        names_sql = ",".join(f"'{self._escape(n)}'" for n in dep_names)
        sql = (
            "SELECT table_name, node_type, human_layer_final, ai_description, ai_layer_suggest "
            "FROM dw_meta.dw_node "
            f"WHERE source_system = '{self._escape(self.sourcedb)}' AND table_name IN ({names_sql})"
        )
        res = self.meta_conn.execute({"sql_query": sql, "result_format": "list"})
        rows = self._rows_from_result(res)
        info: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            key = str(row.get("table_name") or "").lower()
            info[key] = {
                "node_type": (row.get("node_type") or "").upper(),
                "human_layer_final": row.get("human_layer_final") or "",
                "ai_description": row.get("ai_description") or "",
                "ai_layer_suggest": row.get("ai_layer_suggest") or "",
            }
        return info

    def _load_dep_features(self, dep_keys: List[str]) -> Dict[str, str]:
        """批量获取依赖节点的 feature_json（如存在）。"""
        if not dep_keys:
            return {}
        names_sql = ",".join(f"'{self._escape(n)}'" for n in dep_keys)
        sql = (
            "SELECT ts.table_name, af.feature_json "
            "FROM dw_meta.table_source ts "
            "JOIN dw_meta.ai_view_feature af ON ts.table_id = af.table_id "
            f"WHERE ts.source_system = '{self._escape(self.sourcedb)}' AND ts.table_name IN ({names_sql})"
        )
        res = self.meta_conn.execute({"sql_query": sql, "result_format": "list"})
        rows = self._rows_from_result(res)
        cache: Dict[str, str] = {}
        for row in rows:
            key = str(row.get("table_name") or "").lower()
            cache[key] = row.get("feature_json") or ""
        return cache

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
    ) -> Tuple[int, bool]:
        """
        返回 (table_id, changed)，changed 表示是否发生了插入/更新。
        """
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        view_name = self._escape(row.view_name).lower()
        ddl_sql = self._escape(row.ddl_sql)
        sql_hash = self._escape(row.sql_hash)
        if table_type == "EXTERNAL":
            source_system = self._escape(row.db_name) if self._escape(row.db_name) != "" else self.sourcedb
        else:
            source_system = self.sourcedb

        # 已存在且 hash 未变
        if existing and existing.sql_hash == row.sql_hash:
            return existing.view_id or 0, False

        # 已存在但 hash 变化 -> 更新
        if existing and existing.view_id:
            update_sql = (
                "UPDATE dw_meta.table_source SET "
                f"ddl_sql = '{ddl_sql}', hash = '{sql_hash}', updated_at = '{now}' "
                f"WHERE table_id = {existing.view_id}"
            )
            self.meta_conn.execute({"sql_query": update_sql})
            return existing.view_id, True

        # 新增
        insert = (
            "INSERT INTO dw_meta.table_source "
            "(source_system, table_name, table_type, ddl_sql, hash, created_at, updated_at) "
            f"VALUES ('{source_system}', '{view_name}', '{table_type}', '{ddl_sql}', '{sql_hash}', '{now}', '{now}')"
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
        return (int(rows[0].get("table_id")), True) if rows else (0, True)

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
    def _get_or_create_external_dependency(self, view_name: str, raw_name: str, db_prefix: Optional[str]) -> Optional[Dict[str, Any]]:
        """
        处理跨库/DBLink 依赖：映射前缀到目标系统，查询元数据，必要时创建虚拟表。
        """
        target_system = self.sourcedb
        target_table_name = raw_name.lower()
        is_virtual = False

        # 显式 schema/db 前缀 (如 lyerp.table)
        if db_prefix:
            clean_prefix = str(db_prefix).lower()
            if clean_prefix in self.schema_system_map:
                target_system = self.schema_system_map[clean_prefix]
            else:
                # 未知 db_prefix，当前系统内生成虚拟表，例如 iwcs_table
                target_system = self.sourcedb
                target_table_name = f"{clean_prefix}_{target_table_name}"
                is_virtual = True

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
        else:
            is_virtual = True

        # 未找到且需要虚拟表时，插入占位 EXTERNAL 节点，保证血缘不断裂
        if is_virtual:
            logger.info(f"为 {view_name} 创建虚拟表依赖: system={target_system}, table={target_table_name}")
            virtual_row = ViewSourceRow(
                view_id=None,
                view_name=target_table_name,
                db_name=target_system,
                ddl_sql="-- Virtual table created by dependency resolution",
                sql_hash="virtual",
            )
            new_id, _ = self._upsert_table_source(virtual_row, None, table_type="EXTERNAL")
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
        view_name = feature.get("view_name")
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
                external_info = self._get_or_create_external_dependency(view_name, raw_name, db_prefix)
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

    # ---------- 新增阶段辅助 ---------- #
    def _load_views_from_meta_for_analysis(self) -> List[Dict[str, Any]]:
        sql = (
            "SELECT ts.table_id as view_id, ts.table_name as view_name, ts.ddl_sql, ts.parse_status, ts.hash "
            "FROM dw_meta.table_source ts "
            f"WHERE ts.source_system = '{self.sourcedb}' AND ts.table_type = 'VIEW'"
        )
        res = self.meta_conn.execute({"sql_query": sql, "result_format": "list"})
        rows = self._rows_from_result(res)
        if not rows:
            raw = getattr(res, "sql_return", None)
            raw_preview = ""
            if raw is not None:
                raw_preview = str(raw)
                if len(raw_preview) > 500:
                    raw_preview = raw_preview[:500] + "...(truncated)"
            logger.warning(
                f"AST 筛选加载结果为空，source_system={self.sourcedb}，success={getattr(res, 'success', None)}, "
                f"return_type={type(raw).__name__ if raw is not None else 'None'}, preview={raw_preview}"
            )
        if self.strategy == "overwrite":
            return rows

        filtered: List[Dict[str, Any]] = []
        not_parsed_names: List[str] = []
        hash_mismatch_names: List[str] = []
        for row in rows:
            status = (row.get("parse_status") or "").upper()
            # current_hash = row.get("hash") or ""
            # feature_json = row.get("feature_json") or ""
            # prev_hash = ""
            # if feature_json:
            #     try:
            #         prev_hash = (json.loads(feature_json) or {}).get("source_hash") or ""
            #     except Exception:
            #         prev_hash = ""

            if status != "PARSED":
                not_parsed_names.append(row.get("view_name") or "")
                filtered.append(row)
                continue
            # if not prev_hash or prev_hash != current_hash:
            #     hash_mismatch_names.append(row.get("view_name") or "")
            #     filtered.append(row)
        if rows:
            logger.info(
                f"AST 筛选 (strategy=incremental): 总计={len(rows)}, parse_status!=PARSED={len(not_parsed_names)}, "
                f"hash 变更/缺失={len(hash_mismatch_names)}, 待处理={len(filtered)}"
            )
            sample_np = ", ".join([n for n in not_parsed_names[:10] if n])
            sample_hash = ", ".join([n for n in hash_mismatch_names[:10] if n])
            if sample_np:
                logger.debug(f"待解析视图样例(parse_status!=PARSED): {sample_np}")
            if sample_hash:
                logger.debug(f"哈希变更/缺失样例: {sample_hash}")
        return filtered

    def _load_nodes_for_classification(self) -> List[Dict[str, Any]]:
        sql = (
            "SELECT n.node_id, n.table_name as view_name, n.source_table_id, "
            "n.human_layer_final, n.migration_status, "
            "f.feature_json, t.ddl_sql, t.hash "
            "FROM dw_meta.dw_node n "
            "LEFT JOIN dw_meta.ai_view_feature f ON n.source_table_id = f.table_id "
            "LEFT JOIN dw_meta.table_source t ON n.source_table_id = t.table_id "
            f"WHERE n.source_system = '{self.sourcedb}' AND n.node_type = 'VIEW'"
        )
        res = self.meta_conn.execute({"sql_query": sql, "result_format": "list"})
        rows = self._rows_from_result(res)
        if self.strategy == "overwrite":
            return rows

        filtered: List[Dict[str, Any]] = []
        success_status = {"REVIEWED", "IMPLEMENTED", "SKIPPED"}
        for row in rows:
            status = (row.get("migration_status") or "").upper()
            current_hash = row.get("hash") or ""
            feature_json = row.get("feature_json") or ""
            prev_hash = ""
            if feature_json:
                try:
                    prev_hash = (json.loads(feature_json) or {}).get("source_hash") or ""
                except Exception:
                    prev_hash = ""

            if status not in success_status:
                filtered.append(row)
                continue

            # 成功但内容有更新/缺少特征时需要重跑
            if not prev_hash or (current_hash and prev_hash != current_hash):
                filtered.append(row)

        return filtered

    def _build_graph_from_nodes(self, nodes: List[Dict[str, Any]]) -> Dict[str, Set[str]]:
        graph: Dict[str, Set[str]] = {}
        for n in nodes:
            name = n["view_name"].lower()
            try:
                feature = json.loads(n["feature_json"]) if n.get("feature_json") else {}
            except Exception:
                val = n["feature_json"]  # 在报错前插桩或 REPL 读取
                print(repr(val[360:430]))
                print([hex(ord(c)) for c in val[360:410]])
                logger.info(f"{name} has bad json: {n['feature_json']}")
                raise Exception
            deps = set([d.lower() for d in feature.get("view_dependencies", [])])
            graph[name] = deps
        return graph

    def _get_feature_hash(self, view_id: Optional[int]) -> str:
        if not view_id:
            return ""
        res = self.meta_conn.execute(
            {"sql_query": f"SELECT feature_json FROM dw_meta.ai_view_feature WHERE table_id = {view_id} LIMIT 1"}
        )
        rows = self._rows_from_result(res)
        if not rows:
            return ""
        try:
            feature = json.loads(rows[0].get("feature_json") or "{}")
            return feature.get("source_hash") or ""
        except Exception:
            return ""

    def _load_views_for_naming(self) -> List[Dict[str, Any]]:
        sql = (
            "SELECT ts.table_id as view_id, ts.table_name as view_name, ts.ddl_sql, ts.hash, "
            "'' as db_name, af.feature_json "
            "FROM dw_meta.table_source ts "
            "LEFT JOIN dw_meta.ai_view_feature af ON ts.table_id = af.table_id "
            f"WHERE ts.source_system = '{self.sourcedb}' AND ts.table_type = 'VIEW'"
        )
        res = self.meta_conn.execute({"sql_query": sql, "result_format": "list"})
        rows = self._rows_from_result(res)
        if self.strategy == "overwrite":
            return rows

        filtered: List[Dict[str, Any]] = []
        for row in rows:
            view_name = row.get("view_name") or ""
            current_hash = row.get("hash") or ""
            feature_json = row.get("feature_json") or ""
            feature_hash = ""
            if feature_json:
                try:
                    feature_hash = (json.loads(feature_json) or {}).get("source_hash") or ""
                except Exception:
                    feature_hash = ""

            has_mapping = self._has_existing_mapping(view_name)
            if not has_mapping:
                filtered.append(row)
                continue
            if not feature_hash or feature_hash != current_hash:
                filtered.append(row)

        return filtered

    def _has_existing_mapping(self, view_name: str) -> bool:
        sql = (
            "SELECT COUNT(1) as cnt FROM dw_meta.std_field_mapping "
            f"WHERE source_system = '{self._escape(self.sourcedb)}' "
            f"AND source_table = '{self._escape(view_name)}'"
        )
        res = self.meta_conn.execute({"sql_query": sql, "result_format": "list"})
        rows = self._rows_from_result(res)
        if not rows:
            return False
        try:
            return int(rows[0].get("cnt") or 0) > 0
        except Exception:
            return False

    def _delete_std_mapping(self, view_name: str):
        sql = (
            "DELETE FROM dw_meta.std_field_mapping "
            f"WHERE source_system = '{self._escape(self.sourcedb)}' "
            f"AND source_table = '{self._escape(view_name)}'"
        )
        self.meta_conn.execute({"sql_query": sql})

    def _load_existing_std_names(self) -> Set[str]:
        res = self.meta_conn.execute({"sql_query": "SELECT std_field_name FROM dw_meta.std_field"})
        rows = self._rows_from_result(res)
        return {str(r.get("std_field_name")).lower() for r in rows if r.get("std_field_name")}

    def _suggest_std_field_names(self, view_name: str, column: Dict[str, Any], banned_names: Set[str]) -> Tuple[str, str]:
        base_en = self._to_snake(column.get("output_name") or column.get("source_column") or "")
        base_cn = column.get("output_name") or column.get("source_column") or ""
        if not self.llm:
            return base_en, base_cn

        prompt = (
            "你是数仓标准字段命名助手，请根据字段含义给出英文蛇形命名和中文命名。\n"
            f"视图: {view_name}\n"
            f"字段: {column}\n"
            f"禁止使用的英文名: {', '.join(list(banned_names)[:20])}\n"
            "输出 JSON，键为 std_field_name, std_field_name_cn，不要包含多余文本。"
        )

        for _ in range(3):
            resp = self.llm.generate(prompt)
            parsed = self._extract_json_dict(resp)
            std_en = self._to_snake(parsed.get("std_field_name") or base_en)
            std_cn = parsed.get("std_field_name_cn") or base_cn
            if std_en and std_en not in banned_names:
                return std_en, std_cn
            prompt += f"\n请重新生成，避免使用: {std_en}"
        return base_en, base_cn

    def _interactive_confirm_naming(self, view_name: str, source_column: str, suggest_en: str, suggest_cn: str) -> Tuple[str, str]:
        prompt_text = (
            f"确认字段命名 {view_name}.{source_column}\n"
            f"默认英文: {suggest_en}, 默认中文: {suggest_cn}\n"
            "如需修改，请输入 英文,中文（逗号分隔），直接回车接受默认: "
        )
        try:
            user_input = input(prompt_text)
        except Exception:
            return suggest_en, suggest_cn
        if not user_input:
            return suggest_en, suggest_cn
        parts = [p.strip() for p in user_input.split(",") if p.strip()]
        if len(parts) == 1:
            return self._to_snake(parts[0]), suggest_cn
        if len(parts) >= 2:
            return self._to_snake(parts[0]), parts[1]
        return suggest_en, suggest_cn

    def _extract_json_dict(self, text: str) -> Dict[str, Any]:
        try:
            return json.loads(text)
        except Exception:
            pass
        if "{" in text and "}" in text:
            try:
                raw = text[text.index("{") : text.rindex("}") + 1]
                return json.loads(raw)
            except Exception:
                return {}
        return {}

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
            if hasattr(data, "to_pylist") and hasattr(data, "column_names"):
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
            logger.debug("_rows_from_result arrow parse failed", exc_info=True)

        # pandas.DataFrame
        try:
            import pandas as pd
            if isinstance(data, pd.DataFrame):
                # 替换 NaN 为 None，转为 list[dict]
                return data.where(pd.notnull(data), None).to_dict('records')
        except Exception:
            logger.debug("_rows_from_result pandas parse failed", exc_info=True)

        # tuple 单行兜底
        if isinstance(data, tuple):
            return [{f"col{i}": v for i, v in enumerate(data)}]

        if isinstance(data, str):
            text = data.lstrip("\ufeff").strip()
            if not text:
                return []

            # 优先尝试 JSON 解析；失败再尝试 python literal，再按 CSV 解析
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
            except Exception as exc:
                logger.debug(f"_rows_from_result csv parse error ({exc}), raw length={len(text)}")

            # 兜底：去空行/手动拆分
            lines = [ln for ln in text.splitlines() if ln.strip()]
            if len(lines) >= 2:
                try:
                    rows_csv2 = _csv_to_rows("\n".join(lines), delimiter)
                    if rows_csv2:
                        return rows_csv2
                except Exception as exc2:
                    logger.debug(f"_rows_from_result csv line parse error ({exc2}), raw length={len(text)}")

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
                    logger.debug(f"_rows_from_result manual csv parse failed, raw length={len(text)}")
            logger.debug(f"_rows_from_result csv parse empty, raw length={len(text)}")
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

    for name in names:
        logger.info(f"开始执行 import-view，sourcedb={name}，step={getattr(args, 'step', 'all')}")
        runner = ImportViewRunner(
            agent_config=agent_config,
            db_manager=db_manager,
            namespace=args.namespace,
            sourcedb=name,
            strategy=args.update_strategy,
            step=getattr(args, "step", "all"),
        )
        results[name] = runner.run()

    return {"status": "success", "results": results}


if __name__ == "__main__":
    print("请通过 datus-agent import-view 调用本模块")
    sys.exit(1)
