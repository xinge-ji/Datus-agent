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
from datus.utils.field_equivalence import FieldNode, FieldResolver, UnionFind
from datus.utils.constants import DBType
from datus.utils.loggings import get_logger
from datus.utils.sql_utils import normalize_sql, parse_table_name_parts

logger = get_logger(__name__)


@dataclass
class ViewSourceRow:
    view_id: int
    view_name: str
    db_name: str
    ddl_sql: str
    sql_hash: str
    has_row: int = 0
    parse_status: str = None


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
            # incremental 模式复用已有 has_row，避免重复探测
            has_row = existing.has_row if self.strategy == "incremental" and existing else None
            status_override = None
            if has_row is None:
                has_row, status_override = self._check_has_data(tbl)
            row.has_row = has_row
            if status_override:
                row.parse_status = status_override
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
            # incremental 模式复用已有 has_row，避免重复探测
            has_row = existing.has_row if self.strategy == "incremental" and existing else None
            status_override = None
            if has_row is None:
                has_row, status_override = self._check_has_data(view_meta)
            row.has_row = has_row
            if status_override:
                row.parse_status = status_override
            elif not (self.strategy == "incremental" and existing):
                row.parse_status = "SKIPPED" if not row.has_row else None
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

            if parse_status == "SKIPPED":
                skipped += 1
                logger.info(f"跳过 AST（parse_status=SKIPPED）: {view_name}")
                continue

            logger.info(f"Processing {view_name}")

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

        table_type_map = {k: v.get("table_type") for k, v in self._load_table_source_map().items()}
        dep_graph, priority_map = self._build_graph_from_nodes(nodes_data, table_type_map)
        topo_order = self._topo_sort(dep_graph, priority_map)
        nodes_map = {n["view_name"].lower(): n for n in nodes_data}
        
        logger.info("\n".join(topo_order))

        layer_cache = {n["view_name"].lower(): n["human_layer_final"] for n in nodes_data if n.get("human_layer_final")}
        processed = skipped = 0

        for view_name_lower in topo_order:
            node = nodes_map.get(view_name_lower)
            if not node:
                continue

            if self.strategy == "incremental" and not node.get("_need_process", True):
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
        """阶段4：基于字段等价关系的标准化命名。"""
        logger.info(">>> 阶段 4: 开始字段命名 ...")
        if self.llm is None:
            try:
                self.llm = LLMBaseModel.create_model(model_name="default", agent_config=self.agent_config)
            except Exception as exc:
                logger.warning(f"LLM 初始化失败: {exc}")
                return {"error": "LLM init failed"}

        view_features = self._load_view_features_for_naming()
        if not view_features:
            return {"processed": 0, "skipped": 0, "mapped": 0}

        feature_map: Dict[str, Dict[str, Any]] = {}
        for row in view_features:
            view_name = (row.get("view_name") or "").lower()
            feature_json = row.get("feature_json") or ""
            try:
                feature = json.loads(feature_json) if feature_json else {}
            except Exception as exc:
                logger.warning(f"解析视图特征失败 {view_name}: {exc}")
                continue
            if not feature:
                continue
            feature["view_name"] = view_name
            feature_map[view_name] = feature

        if not feature_map:
            return {"processed": 0, "skipped": 0, "mapped": 0}

        table_source_index = self._load_table_source_index()
        resolver = FieldResolver(feature_map=feature_map, table_source_index=table_source_index, default_source_system=self.sourcedb)
        uf = UnionFind()
        self._build_field_graph(feature_map, resolver, uf)

        groups = uf.groups()
        if not groups:
            return {"processed": 0, "skipped": 0, "mapped": 0}

        existing_mappings = self._load_existing_std_mappings()

        processed = 0
        mapped = 0
        skipped = 0

        for _, node_keys in groups.items():
            if not node_keys:
                continue

            std_id, conflict_ids = self._select_existing_std_id(node_keys, existing_mappings)
            if conflict_ids:
                logger.warning(f"分组存在多个 std_field_id，优先复用 {std_id}，其它={conflict_ids}")

            if not std_id:
                suggest_en, suggest_cn = self._suggest_group_std_field_name(node_keys, set())
                suggest_en, suggest_cn = self._interactive_confirm_group_naming(node_keys, suggest_en, suggest_cn)
                std_id = self._get_or_create_std_field(
                    {
                        "std_field_name": suggest_en,
                        "std_field_name_cn": suggest_cn,
                        "data_type_std": "string",
                    }
                )

            for node_key in node_keys:
                node = FieldNode.from_key(node_key)
                item = {
                    "source_system": node.source_system,
                    "source_db": "",
                    "source_table": node.table,
                    "source_column": node.column,
                    "expression_sql": "",
                }
                self._upsert_std_mapping(std_id, item)
                mapped += 1

            processed += 1

        logger.info(f"字段命名完成: 分组 {processed}, 跳过 {skipped}, 新增/更新映射 {mapped}")
        return {"processed": processed, "skipped": skipped, "mapped": mapped}

    def _build_field_graph(self, feature_map: Dict[str, Dict[str, Any]], resolver: FieldResolver, uf: UnionFind):
        for view_name, feature in feature_map.items():
            # 收集列使用的源节点
            for col in feature.get("columns") or []:
                nodes = resolver.resolve_column_feature(view_name, feature, col)
                for node in nodes:
                    uf.add(node.key)

            # 收集 JOIN/WHERE 等值条件带来的等价边
            for join in feature.get("joins") or []:
                for cond in join.get("conditions") or []:
                    left_nodes = resolver.resolve_alias(
                        view_name, feature, cond.get("left_table_alias"), cond.get("left_column")
                    )
                    right_nodes = resolver.resolve_alias(
                        view_name, feature, cond.get("right_table_alias"), cond.get("right_column")
                    )
                    for node in left_nodes + right_nodes:
                        uf.add(node.key)
                    for ln in left_nodes:
                        for rn in right_nodes:
                            uf.union(ln.key, rn.key)

    def _load_view_features_for_naming(self) -> List[Dict[str, Any]]:
        """
        加载所有已解析视图的特征，用于跨视图字段等价图构建。
        """
        sql = (
            "SELECT ts.table_id as view_id, ts.table_name as view_name, ts.hash, af.feature_json "
            "FROM dw_meta.table_source ts "
            "LEFT JOIN dw_meta.ai_view_feature af ON ts.table_id = af.table_id "
            f"WHERE ts.source_system = '{self._escape(self.sourcedb)}' "
            "AND ts.table_type = 'VIEW' AND ts.parse_status = 'PARSED'"
        )
        res = self.meta_conn.execute({"sql_query": sql, "result_format": "list"})
        rows = self._rows_from_result(res)
        return rows

    def _load_table_source_index(self) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """
        返回 (source_system, table_name)->table 信息的索引，全部小写。
        """
        sql = "SELECT table_id, table_name, table_type, source_system FROM dw_meta.table_source"
        res = self.meta_conn.execute({"sql_query": sql, "result_format": "list"})
        rows = self._rows_from_result(res)
        mapping: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for row in rows:
            ss = str(row.get("source_system") or "").lower()
            name = str(row.get("table_name") or "").lower()
            mapping[(ss, name)] = {
                "table_id": row.get("table_id"),
                "table_name": name,
                "table_type": (row.get("table_type") or "").upper(),
                "source_system": ss,
            }
        return mapping

    def _load_existing_std_mappings(self) -> Dict[str, int]:
        """
        返回节点键 -> std_field_id 映射，节点键格式 source_system.table.column（均小写）。
        """
        sql = "SELECT source_system, source_table, source_column, std_field_id FROM dw_meta.std_field_mapping WHERE is_active = 1"
        res = self.meta_conn.execute({"sql_query": sql, "result_format": "list"})
        rows = self._rows_from_result(res)
        mapping: Dict[str, int] = {}
        for row in rows:
            ss = str(row.get("source_system") or "").lower()
            tbl = str(row.get("source_table") or "").lower()
            col = str(row.get("source_column") or "").lower()
            key = f"{ss}.{tbl}.{col}"
            try:
                mapping[key] = int(row.get("std_field_id"))
            except Exception:
                continue
        return mapping

    def _select_existing_std_id(self, node_keys: Set[str], existing_mappings: Dict[str, int]) -> Tuple[Optional[int], Set[int]]:
        ids: List[int] = []
        for key in node_keys:
            sid = existing_mappings.get(key)
            if sid:
                ids.append(sid)
        if not ids:
            return None, set()
        main = ids[0]
        return main, set(ids[1:])

    def _suggest_group_std_field_name(self, node_keys: Set[str], banned_names: Set[str]) -> Tuple[str, str]:
        sample_nodes = sorted(list(node_keys))[:10]
        default_en = node_keys and FieldNode.from_key(next(iter(node_keys))).column or "field"
        default_cn = default_en
        if not self.llm:
            return default_en, default_cn

        prompt = (
            "你是数仓标准字段命名助手，以下字段在业务上等价，请给出统一的英文蛇形命名和中文名。\n"
            "字段列表（source_system.table.column）：\n"
            + "\n".join(sample_nodes)
            + "\n禁止使用的英文名: "
            + ", ".join(list(banned_names)[:20])
            + "\n输出 JSON，键为 std_field_name, std_field_name_cn，不要包含其他文本。"
        )

        for _ in range(3):
            resp = self.llm.generate(prompt)
            parsed = self._extract_json_dict(resp)
            std_en = self._to_snake(parsed.get("std_field_name") or default_en)
            std_cn = parsed.get("std_field_name_cn") or default_cn
            if std_en and std_en not in banned_names:
                return std_en, std_cn
            prompt += f"\n请重新生成，避免使用: {std_en}"
        return default_en, default_cn

    def _interactive_confirm_group_naming(self, node_keys: Set[str], suggest_en: str, suggest_cn: str) -> Tuple[str, str]:
        sample = ", ".join(sorted(list(node_keys)))
        prompt_text = (
            f"确认分组命名（示例字段: {sample}）\n"
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

    def _get_source_dialect(self) -> str:
        return str(getattr(self.source_conn, "dialect", "") or getattr(self.agent_config, "db_type", "") or "").lower()

    def _compose_full_table_name(self, table_meta: Dict[str, str]) -> str:
        raw_name = table_meta.get("table_name") or table_meta.get("view_name") or table_meta.get("name") or ""
        if not raw_name:
            return ""
        if "." in raw_name:
            return raw_name
        db_name = table_meta.get("database_name") or table_meta.get("db_name") or self.source_db_name or ""
        schema_name = (
            table_meta.get("schema_name")
            or table_meta.get("schema")
            or table_meta.get("owner")
            or self.source_schema
            or ""
        )
        parts = [f'"{p}"' for p in [schema_name, raw_name] if p]
        return ".".join(parts) if parts else raw_name

    def _build_probe_sql(self, full_table_name: str) -> str:
        dialect = self._get_source_dialect()
        if "oracle" in dialect:
            return f"SELECT 1 FROM {full_table_name} WHERE ROWNUM < 2"
        if "sqlserver" in dialect or "mssql" in dialect:
            return f"SELECT TOP 1 1 FROM {full_table_name}"
        return f"SELECT 1 FROM {full_table_name} LIMIT 1"

    def _check_has_data(self, table_meta: Dict[str, str]) -> Tuple[int, Optional[str]]:
        full_table_name = self._compose_full_table_name(table_meta)
        if not full_table_name:
            raise ValueError(f"源对象缺少名称，无法检查是否有数据: {table_meta}")
        sql = self._build_probe_sql(full_table_name)
        if not hasattr(self.source_conn, "execute"):
            raise RuntimeError("源连接不支持 execute，无法检查是否有数据")
        try:
            res = self.source_conn.execute({"sql_query": sql, "result_format": "list"})
        except Exception as exc:
            if "ORA" in str(exc).upper():
                logger.warning(
                    f"检测 {full_table_name} 遇到 ORA-ERROR，跳过探测，默认 has_row=1 parse_status=NEW，SQL=[{sql}]"
                )
                return 1, "NEW"
            raise RuntimeError(f"检测 {full_table_name} 是否有数据失败, SQL=[{sql}]: {exc}") from exc

        if not res or not getattr(res, "success", False):
            err = getattr(res, "error", "未知错误")
            err_upper = str(err).upper()
            if "ORA" in err_upper:
                logger.warning(
                    f"检测 {full_table_name} 遇到 ORA-ERROR，跳过探测，默认 has_row=1 parse_status=NEW，SQL=[{sql}]"
                )
                return 1, "NEW"
            raise RuntimeError(f"检测 {full_table_name} 是否有数据失败, SQL=[{sql}]: {err}")
        err_detail = getattr(res, "error", None)
        if err_detail:
            err_upper = str(err_detail).upper()
            if "ORA" in err_upper:
                logger.warning(
                    f"检测 {full_table_name} 遇到 ORA-ERROR，跳过探测，默认 has_row=1 parse_status=NEW，SQL=[{sql}]"
                )
                return 1, "NEW"
            raise RuntimeError(f"检测 {full_table_name} 是否有数据失败, SQL=[{sql}]: {err_detail}")
        rows = self._rows_from_result(res)
        return (1 if rows else 0), None

    def _load_existing_view_source(self) -> Dict[str, ViewSourceRow]:
        sql = (
            "SELECT table_id as view_id, table_name as view_name, '' as db_name, ddl_sql, hash, has_row, parse_status "
            "FROM dw_meta.table_source "
            f"WHERE LOWER(source_system) = LOWER('{self.sourcedb}') AND table_type = 'VIEW'"
        )
        result = self.meta_conn.execute({"sql_query": sql, "result_format": "list"})
        rows = self._rows_from_result(result)
        existing: Dict[str, ViewSourceRow] = {}
        for row in rows:
            parsed = self._build_view_source_row(row, warn_context="load_existing_view_source")
            if not parsed:
                continue
            existing[parsed.view_name.lower()] = parsed
        return existing

    def _load_table_source_map(self) -> Dict[str, Dict[str, Any]]:
        sql = (
            "SELECT table_id, table_name, table_type "
            "FROM dw_meta.table_source "
            f"WHERE LOWER(source_system) = LOWER('{self.sourcedb}')"
        )
        result = self.meta_conn.execute({"sql_query": sql, "result_format": "list"})
        rows = self._rows_from_result(result)
        mapping: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            name = str(row.get("table_name") or "").lower()
            mapping[name] = {
                "table_id": row.get("table_id"),
                "table_type": (row.get("table_type") or "").upper(),
                "source_system": self.sourcedb,
            }
        return mapping

    def _load_existing_table_source(self, table_type: str = "VIEW") -> Dict[str, ViewSourceRow]:
        sql = (
            "SELECT table_id as view_id, table_name as view_name, '' as db_name, ddl_sql, hash, has_row, parse_status "
            "FROM dw_meta.table_source "
            f"WHERE LOWER(source_system) = LOWER('{self.sourcedb}') AND table_type = '{table_type}'"
        )
        result = self.meta_conn.execute({"sql_query": sql, "result_format": "list"})
        rows = self._rows_from_result(result)
        existing: Dict[str, ViewSourceRow] = {}
        for row in rows:
            parsed = self._build_view_source_row(row, warn_context="load_existing_table_source")
            if not parsed:
                continue
            existing[parsed.view_name.lower()] = parsed
        return existing

    def _find_existing_in_db(self, view_name: str, table_type: str, source_system: str) -> Optional[ViewSourceRow]:
        sql = (
            "SELECT table_id as view_id, table_name as view_name, '' as db_name, ddl_sql, hash, has_row, parse_status "
            "FROM dw_meta.table_source "
            f"WHERE LOWER(source_system) = LOWER('{source_system}') AND table_type = '{table_type}' "
            f"AND LOWER(table_name) = LOWER('{view_name}') "
            "ORDER BY table_id DESC LIMIT 1"
        )
        res = self.meta_conn.execute({"sql_query": sql, "result_format": "list"})
        rows = self._rows_from_result(res)
        if not rows:
            return None
        parsed = self._build_view_source_row(rows[0], warn_context="find_existing_in_db")
        return parsed

    def _get_row_value(self, row: Any, keys: List[str], idx: Optional[int]) -> Any:
        """
        读取查询结果行的值，优先字段名，其次 col{idx}，最后按位置索引，兼容 tuple/list。
        """
        if not isinstance(keys, (list, tuple)):
            keys = [keys]
        if isinstance(row, dict):
            for k in keys:
                if k in row and row.get(k) is not None:
                    return row.get(k)
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

    def _build_view_source_row(self, row: Any, warn_context: str = "") -> Optional[ViewSourceRow]:
        """
        将元库查询结果行解析为 ViewSourceRow，容错字段名缺失或返回 tuple/list。
        """
        view_id_raw = self._get_row_value(row, ["view_id", "table_id"], idx=0)
        view_name_raw = self._get_row_value(row, ["view_name", "table_name"], idx=1)
        ddl_sql_raw = self._get_row_value(row, ["ddl_sql"], idx=3) or ""
        hash_raw = self._get_row_value(row, ["hash"], idx=4) or ""
        has_row_raw = self._get_row_value(row, ["has_row"], idx=5)
        parse_status_raw = self._get_row_value(row, ["parse_status"], idx=6)

        if not view_name_raw:
            logger.warning(f"table_source 行缺少 view_name，跳过 (context={warn_context}): {row}")
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
        has_row = 1 if row.has_row else 0
        requested_status = (row.parse_status or "").upper()
        if table_type == "EXTERNAL":
            source_system_raw = self._escape(row.db_name) if self._escape(row.db_name) != "" else self.sourcedb
        else:
            source_system_raw = self.sourcedb
        source_system_norm = self._escape(source_system_raw).lower()

        name_key = view_name.strip('"')
        existing_row = existing or self._find_existing_in_db(view_name, table_type, source_system_norm)
        if not existing_row and name_key and name_key != view_name:
            existing_row = self._find_existing_in_db(name_key, table_type, source_system_norm)
        existing_status = (existing_row.parse_status or "").upper() if existing_row else ""
        target_status = requested_status or existing_status or "NEW"
        if not requested_status and existing_status == "SKIPPED" and has_row:
            # 数据恢复后将 SKIPPED 恢复为 NEW，后续可重新进入 AST
            target_status = "NEW"
        target_status_esc = self._escape(target_status)

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

        # 新增
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
        res = self.meta_conn.execute(
            {
                "sql_query": (
                    "SELECT table_id FROM dw_meta.table_source "
                    f"WHERE LOWER(source_system) = LOWER('{source_system_norm}') AND table_type = '{table_type}' "
                    f"AND LOWER(table_name) = LOWER('{view_name}') "
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
                "source_system": target_system,
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
                "source_system": target_system,
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
                t["source_system"] = info.get("source_system", self.sourcedb)
                dep_info[alias_key] = {
                    "name": resolved_name,
                    "db_name": info.get("db_name", db_name),
                    "type": dep_type,
                    "source_table_id": info.get("table_id"),
                    "source_system": info.get("source_system", self.sourcedb),
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

    def _topo_sort(self, graph: Dict[str, Set[str]], priority_map: Optional[Dict[str, int]] = None) -> List[str]:
        indeg: Dict[str, int] = {k: 0 for k in graph}
        for deps in graph.values():
            for dep in deps:
                if dep in indeg:
                    indeg[dep] += 1
        queue = [k for k, v in indeg.items() if v == 0]
        if priority_map:
            queue.sort(key=lambda x: priority_map.get(x, 2))
        order: List[str] = []
        while queue:
            node = queue.pop(0)
            order.append(node)
            for dep in graph.get(node, []):
                if dep not in indeg:
                    logger.debug(f"拓扑排序忽略未知依赖: {node} -> {dep}")
                    continue
                indeg[dep] -= 1
                if indeg[dep] == 0:
                    queue.append(dep)
                    if priority_map:
                        queue.sort(key=lambda x: priority_map.get(x, 2))
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
            f"WHERE n.source_system = '{self.sourcedb}' AND n.node_type = 'VIEW' "
            "AND t.table_type = 'VIEW' AND t.parse_status = 'PARSED'"
        )
        res = self.meta_conn.execute({"sql_query": sql, "result_format": "list"})
        rows = self._rows_from_result(res)
        if self.strategy == "overwrite":
            return rows

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

            need_process = True
            if self.strategy == "incremental" and status in success_status and prev_hash and current_hash and prev_hash == current_hash:
                need_process = False

            row["_need_process"] = need_process
        return rows

    def _build_graph_from_nodes(self, nodes: List[Dict[str, Any]], type_map: Dict[str, Optional[str]]) -> Tuple[Dict[str, Set[str]], Dict[str, int]]:
        node_set = {n["view_name"].lower() for n in nodes}
        graph: Dict[str, Set[str]] = {}
        priority: Dict[str, int] = {}
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
            missing = {d for d in deps if d not in node_set}
            if missing:
                logger.debug(f"忽略未纳入分层的依赖: {name} -> {missing}")
            graph[name] = {d for d in deps if d in node_set}
            priority[name] = self._calc_dep_priority(feature, type_map, missing)
        return graph, priority

    def _calc_dep_priority(self, feature: Dict[str, Any], type_map: Dict[str, Optional[str]], missing_view_deps: Set[str]) -> int:
        deps_view = {d.lower() for d in feature.get("view_dependencies", [])}
        deps_table = {d.lower() for d in feature.get("table_dependencies", [])}
        unresolved = {d.lower() for d in feature.get("unresolved_dependencies", [])}

        contains_view = False
        contains_external = False

        for v in deps_view:
            t_type = (type_map.get(v) or "").upper()
            if t_type == "VIEW":
                contains_view = True
            elif t_type == "EXTERNAL" or not t_type:
                contains_external = True

        for t in deps_table:
            t_type = (type_map.get(t) or "").upper()
            if t_type == "EXTERNAL" or not t_type:
                contains_external = True

        for mv in missing_view_deps:
            t_type = (type_map.get(mv) or "").upper()
            if t_type == "VIEW":
                contains_view = True
            elif t_type == "EXTERNAL" or not t_type:
                contains_external = True

        if unresolved:
            contains_external = True

        if contains_external:
            return 3
        if contains_view:
            return 2
        return 1

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
        std_name_raw = self._to_snake(item["std_field_name"])
        std_name = self._escape(std_name_raw)
        std_name_cn = self._escape(item["std_field_name_cn"])
        source_system = self._escape(item.get("source_system") or self.sourcedb)
        select_sql = (
            "SELECT std_field_id FROM dw_meta.std_field "
            f"WHERE LOWER(std_field_name) = LOWER('{std_name}') "
            f"AND source_system = '{source_system}' "
            "ORDER BY std_field_id DESC LIMIT 1"
        )

        logger.info(f"[STD] 查找 std_field: name={std_name_raw}, source_system={source_system}")
        insert = (
            "INSERT INTO dw_meta.std_field "
            "(std_field_name, std_field_name_cn, source_system, semantic_type) "
            f"VALUES ('{std_name}', '{std_name_cn}', '{source_system}', NULL)"
        )
        logger.info(f"[STD] 插入 std_field: name={std_name_raw}, cn={item['std_field_name_cn']}, source_system={source_system}")
        insert_res = self.meta_conn.execute({"sql_query": insert})
        logger.info(
            f"[STD] 插入结果: success={getattr(insert_res, 'success', None)}, return={getattr(insert_res, 'sql_return', None)}"
        )
        if not insert_res or not getattr(insert_res, "success", False):
            raw_ret = getattr(insert_res, "sql_return", "")
            raise RuntimeError(f"std_field 插入失败: {std_name_raw}, 原始返回={raw_ret}")

        logger.info(f"[STD] 插入后再次查询 std_field: name={std_name_raw}")
        res = self.meta_conn.execute({"sql_query": select_sql, "result_format": "list"})
        rows2 = self._rows_from_result(res)
        if rows2:
            return int(rows2[0].get("std_field_id"))
        # 再做一次兜底查询，避免大小写或事务延迟问题
        res_fb = self.meta_conn.execute(
            {
                "sql_query": (
                    "SELECT std_field_id FROM dw_meta.std_field "
                    f"WHERE source_system = '{source_system}' "
                    "ORDER BY std_field_id DESC LIMIT 1"
                ),
                "result_format": "list",
            }
        )
        rows_fb = self._rows_from_result(res_fb)
        if rows_fb:
            return int(rows_fb[0].get("std_field_id"))
        # 追加诊断信息
        res_all = self.meta_conn.execute(
            {
                "sql_query": (
                    "SELECT std_field_id, std_field_name, source_system "
                    "FROM dw_meta.std_field ORDER BY std_field_id DESC LIMIT 5"
                ),
                "result_format": "list",
            }
        )
        rows_all = self._rows_from_result(res_all)
        count_res = self.meta_conn.execute(
            {
                "sql_query": (
                    "SELECT COUNT(1) AS cnt FROM dw_meta.std_field "
                    f"WHERE source_system = '{source_system}'"
                ),
                "result_format": "list",
            }
        )
        cnt_rows = self._rows_from_result(count_res)
        cnt_val = cnt_rows[0].get("cnt") if cnt_rows else "unknown"
        raw_select = getattr(res, "sql_return", None)
        raw_insert = getattr(insert_res, "sql_return", None)
        raise RuntimeError(
            f"无法获取 std_field_id: {std_name_raw}, source_system={source_system}, count={cnt_val}, "
            f"select_raw={raw_select}, insert_raw={raw_insert}, rows_all={rows_all}"
        )

    def _upsert_std_mapping(self, std_field_id: int, item: Dict[str, str]):
        source_system = item.get("source_system") or self.sourcedb
        delete = (
            "DELETE FROM dw_meta.std_field_mapping "
            f"WHERE source_system = '{self._escape(source_system)}' "
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
            f"VALUES ('{self._escape(source_system)}', '{self._escape(item['source_db'])}', '{self._escape(item['source_table'])}', "
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
