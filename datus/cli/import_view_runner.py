"""
import_view 核心业务逻辑

从原始 import_view.py 提取的核心业务逻辑，复用现有工具和模式。
"""

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

from rich.prompt import Prompt

from datus.configuration.agent_config import AgentConfig
from datus.models.base import LLMBaseModel
from datus.tools.db_tools.db_manager import DBManager
from datus.tools.llms_tools.classify_layer import classify_view_layer
from datus.utils.ast_analyzer import AstAnalyzer
from datus.utils.field_equivalence import FieldNode, FieldResolver, UnionFind
from datus.utils.constants import DBType
from datus.utils.json_utils import safe_extract_json
from datus.utils.loggings import get_logger
from datus.utils.result_utils import parse_query_result
from datus.utils.sql_utils import normalize_sql, parse_table_name_parts, build_probe_sql, escape_sql
from datus.utils.string_utils import to_snake, strip_ansi
from datus.cli.import_view_operations import ImportViewOperations, ViewSourceRow

logger = get_logger(__name__)


class ImportViewRunner:
    """视图导入与AST分析执行器"""

    def __init__(
        self,
        agent_config: AgentConfig,
        db_manager: DBManager,
        namespace: str,
        sourcedb: str,
        strategy: str,
        step: str = "all",
        interactive: bool = False,
        dry_run: bool = False
    ):
        """
        初始化导入执行器

        Args:
            agent_config: 代理配置
            db_manager: 数据库管理器
            namespace: 命名空间
            sourcedb: 源数据库名
            strategy: 更新策略
            step: 执行步骤
            interactive: 是否交互模式
            dry_run: 是否预览模式
        """
        self.agent_config = agent_config
        self.db_manager = db_manager
        self.namespace = namespace
        self.sourcedb = sourcedb
        self.strategy = strategy
        self.step = step
        self.interactive = interactive
        self.dry_run = dry_run

        # 源库连接：优先从 sourcedb 配置创建独立连接，否则使用 namespace 下的逻辑库
        self._setup_source_connection()

        # 元库连接：使用 namespace 默认/当前数据库（init-meta 所在）
        meta_logic_db = getattr(agent_config, "current_database", None) or agent_config.current_database or ""
        self.meta_conn = self.db_manager.get_conn(namespace, meta_logic_db)

        # 初始化工具
        self.ast = AstAnalyzer(dialect="oracle")
        self.operations = ImportViewOperations(self.meta_conn, sourcedb)
        self.llm: Optional[LLMBaseModel] = None

        # 跨库前缀与 source_system 映射，例如 lyerp.table -> source_system=erp
        self.schema_system_map = {
            "lyerp": "erp",
            "lywms": "wms",
            "cosmic_pro_secd": "yunuopg_secd",
            "cosmic_pro_lycus": "yunuopg_lycus"
        }

    def _setup_source_connection(self):
        """设置源库连接"""
        src_db_config = None
        try:
            src_db_config = self.agent_config.source_db_config(self.sourcedb)
        except Exception:
            src_db_config = None

        if src_db_config:
            from datus.tools.db_tools.db_manager import DBManager as SrcDBManager
            self._src_db_manager = SrcDBManager({self.sourcedb: {src_db_config.logic_name or self.sourcedb: src_db_config}})
            self.source_conn = self._src_db_manager.get_conn(self.sourcedb, src_db_config.logic_name or self.sourcedb)
            self.source_db_name = src_db_config.database or ""
            self.source_schema = src_db_config.schema or ""
        else:
            self._src_db_manager = None
            self.source_conn = self.db_manager.get_conn(self.namespace, self.sourcedb)
            self.source_db_name = getattr(self.source_conn, "database_name", "") or ""
            self.source_schema = getattr(self.source_conn, "schema_name", "") or ""

    def run(self) -> Dict[str, Any]:
        """
        根据步骤参数调度执行

        Returns:
            执行结果统计
        """
        if self.dry_run:
            logger.info("运行在预览模式，不会实际修改数据")
            return self._run_dry_run()

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

    def _run_dry_run(self) -> Dict[str, Any]:
        """
        预览模式运行

        Returns:
            预览结果
        """
        stats: Dict[str, Any] = {"step": self.step, "details": {}, "dry_run": True}

        # 预览要导入的表和视图
        all_tables = self._load_tables()
        all_views = self._load_views()

        stats["preview"] = {
            "tables_to_import": len(all_tables),
            "views_to_import": len(all_views),
            "existing_views": len(self.operations.load_existing_views()),
            "existing_tables": len(self.operations.load_existing_tables("TABLE")),
            "source_database": self.sourcedb,
            "strategy": self.strategy,
            "steps": self.step
        }

        logger.info(f"预览模式: 将导入 {len(all_tables)} 个表, {len(all_views)} 个视图")
        return stats

    def run_import_ddl(self) -> Dict[str, int]:
        """
        阶段1：导入表/视图 DDL

        Returns:
            导入统计结果
        """
        all_tables = self._load_tables()
        all_views = self._load_views()
        table_existing = self.operations.load_existing_tables("TABLE")
        view_existing = self.operations.load_existing_views()

        if self.strategy == "overwrite":
            self._cleanup_downstream(list(view_existing.keys()))

        added = updated = skipped = 0

        # 处理表
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

            table_id, changed = self.operations.upsert_table_source(row, existing, table_type="TABLE")
            row.view_id = table_id
            table_existing[key] = row

            if existing:
                if changed:
                    updated += 1
                else:
                    skipped += 1
            else:
                added += 1

        # 处理视图
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

            view_id, changed = self.operations.upsert_table_source(row, existing, table_type="VIEW")
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
        """
        阶段2：AST 分析与血缘落库

        Returns:
            分析统计结果
        """
        logger.info(">>> 阶段 2: 开始 AST 分析 ...")
        views_to_process = self._load_views_from_meta_for_analysis()

        if not views_to_process:
            return {"success": 0, "failed": 0, "skipped": 0}

        table_source_map = self.operations.load_table_source_map()
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
                self.operations.upsert_ai_view_feature(view_id, feature_json)
                self.operations.update_table_parse_status(view_id, "PARSED")
                self.operations.update_node_migration_status(view_id, "ANALYZED")

                view_node_id = self.operations.ensure_view_node(view_id, view_name)
                dependency_nodes = self.operations.ensure_dependency_nodes(deps["dep_info"])
                self._upsert_relations(view_node_id, dependency_nodes, feature, deps["dep_info"])
                success += 1

            except Exception as exc:
                logger.error(f"视图 {view_name} 解析失败: {exc}")
                error_json = json.dumps({"status": "ERROR", "error": str(exc)}, ensure_ascii=True)
                self.operations.upsert_ai_view_feature(view_id, error_json)
                self.operations.update_table_parse_status(view_id, "FAILED")
                self.operations.update_node_migration_status(view_id, "AST_FAILED")
                failed += 1

        logger.info(f"AST 分析完成: 成功 {success}, 失败 {failed}, 跳过 {skipped}")
        return {"success": success, "failed": failed, "skipped": skipped}

    def run_layer_classification(self) -> Dict[str, int]:
        """
        阶段3：AI 分层 + 人工确认

        Returns:
            分层统计结果
        """
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

        table_type_map = {k: v.get("table_type") for k, v in self.operations.load_table_source_map().items()}
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
        """
        阶段4：基于字段等价关系的标准化命名

        Returns:
            命名统计结果
        """
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
                std_id = self._get_or_create_std_field({
                    "std_field_name": suggest_en,
                    "std_field_name_cn": suggest_cn,
                    "data_type_std": "string",
                })

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

    # 以下是完整的业务方法实现

    def _load_views(self) -> List[Dict[str, str]]:
        """加载视图列表"""
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
        """加载表列表"""
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
        """标准化视图元数据"""
        ddl_sql = view_meta.get("definition") or view_meta.get("ddl") or ""
        ddl_sql = strip_ansi(ddl_sql)
        normalized = normalize_sql(ddl_sql).lower()
        sql_hash = hashlib.md5(normalized.encode("utf-8")).hexdigest()

        return ViewSourceRow(
            view_id=None,
            view_name=view_meta.get("table_name") or view_meta.get("view_name") or view_meta.get("name") or "",
            db_name=view_meta.get("database_name") or self.sourcedb,
            ddl_sql=ddl_sql,
            sql_hash=sql_hash,
        )

    def _check_has_data(self, table_meta: Dict[str, str]) -> Tuple[int, Optional[str]]:
        """检查表是否有数据"""
        full_table_name = self._compose_full_table_name(table_meta)
        if not full_table_name:
            raise ValueError(f"源对象缺少名称，无法检查是否有数据: {table_meta}")

        sql = build_probe_sql(full_table_name, self._get_source_dialect())
        if not hasattr(self.source_conn, "execute"):
            raise RuntimeError("源连接不支持 execute，无法检查是否有数据")

        try:
            res = self.source_conn.execute({"sql_query": sql, "result_format": "list"})
        except Exception as exc:
            if "ORA" in str(exc).upper():
                logger.warning(
                    f"检测 {full_table_name} 遇到 ORA-ERROR，跳过探测，默认 has_row=0 parse_status=SKIPPED，SQL=[{sql}]"
                )
                return 0, "SKIPPED"
            raise RuntimeError(f"检测 {full_table_name} 是否有数据失败, SQL=[{sql}]: {exc}") from exc

        if not res or not getattr(res, "success", False):
            err = getattr(res, "error", "未知错误")
            err_upper = str(err).upper()
            if "ORA" in err_upper:
                logger.warning(
                    f"检测 {full_table_name} 遇到 ORA-ERROR，跳过探测，默认 has_row=0 parse_status=SKIPPED，SQL=[{sql}]"
                )
                return 0, "SKIPPED"
            raise RuntimeError(f"检测 {full_table_name} 是否有数据失败, SQL=[{sql}]: {err}")

        rows = parse_query_result(res)
        return (1 if rows else 0), None

    def _get_source_dialect(self) -> str:
        """获取源数据库方言"""
        return str(getattr(self.source_conn, "dialect", "") or getattr(self.agent_config, "db_type", "") or "").lower()

    def _compose_full_table_name(self, table_meta: Dict[str, str]) -> str:
        """组合完整表名"""
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

    def _cleanup_downstream(self, view_names: List[str]):
        """清理下游数据"""
        if not view_names:
            return

        names_sql = ",".join(f"'{escape_sql(v)}'" for v in view_names)
        view_ids_sql = (
            "SELECT table_id as view_id FROM dw_meta.table_source "
            f"WHERE source_system = '{self.sourcedb}' AND table_type = 'VIEW' AND table_name IN ({names_sql})"
        )

        res = self.meta_conn.execute({"sql_query": view_ids_sql, "result_format": "list"})
        rows = parse_query_result(res)
        view_ids = [str(r.get("view_id")) for r in rows if r.get("view_id")]

        if not view_ids:
            return

        id_list = ",".join(view_ids)

        # 批量删除相关数据
        operations = [
            f"DELETE FROM dw_meta.ai_view_feature WHERE table_id IN ({id_list})",
            f"DELETE FROM dw_meta.std_field_mapping WHERE source_system = '{self.sourcedb}'",
            f"DELETE FROM dw_meta.dw_node_relation WHERE from_node_id IN (SELECT node_id FROM dw_meta.dw_node WHERE source_table_id IN ({id_list})) OR to_node_id IN (SELECT node_id FROM dw_meta.dw_node WHERE source_table_id IN ({id_list}))",
            f"DELETE FROM dw_meta.dw_node WHERE source_table_id IN ({id_list})",
            f"DELETE FROM dw_meta.ai_feedback WHERE object_type = 'VIEW' AND object_key IN ({names_sql})"
        ]

        for sql in operations:
            self.meta_conn.execute({"sql_query": sql})

    def _load_views_from_meta_for_analysis(self) -> List[Dict[str, Any]]:
        """从元数据加载需要分析的视图"""
        sql = (
            "SELECT ts.table_id as view_id, ts.table_name as view_name, ts.ddl_sql, ts.parse_status, ts.hash "
            "FROM dw_meta.table_source ts "
            f"WHERE ts.source_system = '{self.sourcedb}' AND ts.table_type = 'VIEW'"
        )

        res = self.meta_conn.execute({"sql_query": sql, "result_format": "list"})
        rows = parse_query_result(res)

        if self.strategy == "overwrite":
            return rows

        filtered: List[Dict[str, Any]] = []
        not_parsed_names: List[str] = []

        for row in rows:
            status = (row.get("parse_status") or "").upper()
            if status != "PARSED":
                not_parsed_names.append(row.get("view_name") or "")
                filtered.append(row)

        if rows:
            logger.info(
                f"AST 筛选 (strategy=incremental): 总计={len(rows)}, parse_status!=PARSED={len(not_parsed_names)}, "
                f"待处理={len(filtered)}"
            )

        return filtered

    def _get_feature_hash(self, view_id: Optional[int]) -> str:
        """获取特征哈希"""
        if not view_id:
            return ""

        res = self.meta_conn.execute({"sql_query": f"SELECT feature_json FROM dw_meta.ai_view_feature WHERE table_id = {view_id} LIMIT 1"})
        rows = parse_query_result(res)

        if not rows:
            return ""

        try:
            feature = json.loads(rows[0].get("feature_json") or "{}")
            return feature.get("source_hash") or ""
        except Exception:
            return ""

    def _resolve_dependencies(
        self, feature: Dict[str, Any], table_source_map: Dict[str, Dict[str, Any]], default_db: str
    ) -> Dict[str, Any]:
        """解析依赖关系"""
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

            # 同库缓存优先
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

    def _get_or_create_external_dependency(self, view_name: str, raw_name: str, db_prefix: Optional[str]) -> Optional[Dict[str, Any]]:
        """获取或创建外部依赖"""
        target_system = self.sourcedb
        target_table_name = raw_name.lower()
        is_virtual = False

        # 显式 schema/db 前缀处理
        if db_prefix:
            clean_prefix = str(db_prefix).lower()
            if clean_prefix in self.schema_system_map:
                target_system = self.schema_system_map[clean_prefix]
            else:
                target_system = self.sourcedb
                target_table_name = f"{clean_prefix}_{target_table_name}"
                is_virtual = True

        # DBLink 风格处理
        if "@" in raw_name:
            real_table_name, dblink_name = raw_name.split("@", 1)
            dblink_name = dblink_name.lower()
            if dblink_name in self.schema_system_map:
                target_system = self.schema_system_map[dblink_name]
                target_table_name = real_table_name
            else:
                target_system = self.sourcedb
                target_table_name = f"{dblink_name}_{real_table_name}"
                is_virtual = True

        # 查询已有元数据
        sql = (
            "SELECT table_id, table_type FROM dw_meta.table_source "
            f"WHERE source_system = '{escape_sql(target_system)}' "
            f"AND table_name = '{escape_sql(target_table_name)}' LIMIT 1"
        )

        res = self.meta_conn.execute({"sql_query": sql, "result_format": "list"})
        rows = parse_query_result(res)

        if rows:
            return {
                "table_id": rows[0].get("table_id"),
                "table_type": (rows[0].get("table_type") or "").upper(),
                "resolved_name": target_table_name,
                "db_name": "",
                "source_system": target_system,
            }

        # 创建虚拟表
        if is_virtual:
            logger.info(f"为 {view_name} 创建虚拟表依赖: system={target_system}, table={target_table_name}")
            virtual_row = ViewSourceRow(
                view_id=None,
                view_name=target_table_name,
                db_name=target_system,
                ddl_sql="-- Virtual table created by dependency resolution",
                sql_hash="virtual",
            )
            ops = ImportViewOperations(self.meta_conn, self.sourcedb)
            new_id, _ = ops.upsert_table_source(virtual_row, None, table_type="EXTERNAL")
            return {
                "table_id": new_id,
                "table_type": "EXTERNAL",
                "resolved_name": target_table_name,
                "db_name": "",
                "source_system": target_system,
            }

        return None

    def _upsert_relations(
        self, view_node_id: int, dependency_nodes: Dict[str, int], feature: Dict[str, Any], dep_info: Dict[str, Dict]
    ):
        """插入关系数据"""
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
        """插入关系"""
        self.meta_conn.execute({
            "sql_query": (
                "DELETE FROM dw_meta.dw_node_relation "
                f"WHERE from_node_id = {from_id} AND to_node_id = {to_id} AND relation_type = '{relation_type}'"
            )
        })

        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        detail_safe = escape_sql(detail)
        sql = (
            "INSERT INTO dw_meta.dw_node_relation "
            "(from_node_id, to_node_id, relation_type, relation_detail, created_at, updated_at) "
            f"VALUES ({from_id}, {to_id}, '{relation_type}', '{detail_safe}', '{now}', '{now}')"
        )
        self.meta_conn.execute({"sql_query": sql})

    def _load_nodes_for_classification(self) -> List[Dict[str, Any]]:
        """加载需要分类的节点"""
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
        rows = parse_query_result(res)

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
        """从节点构建依赖图"""
        node_set = {n["view_name"].lower() for n in nodes}
        graph: Dict[str, Set[str]] = {}
        priority: Dict[str, int] = {}

        for n in nodes:
            name = n["view_name"].lower()
            try:
                feature = json.loads(n["feature_json"]) if n.get("feature_json") else {}
            except Exception:
                logger.warning(f"无法解析 feature_json for {name}")
                feature = {}

            deps = set([d.lower() for d in feature.get("view_dependencies", [])])
            missing = {d for d in deps if d not in node_set}
            if missing:
                logger.debug(f"忽略未纳入分层的依赖: {name} -> {missing}")
            graph[name] = {d for d in deps if d in node_set}
            priority[name] = self._calc_dep_priority(feature, type_map, missing)

        return graph, priority

    def _calc_dep_priority(self, feature: Dict[str, Any], type_map: Dict[str, Optional[str]], missing_view_deps: Set[str]) -> int:
        """计算依赖优先级"""
        deps_view = {d.lower() for d in feature.get("view_dependencies", [])}
        deps_table = {d.lower() for d in feature.get("table_dependencies", [])}
        unresolved = {d.lower() for d in feature.get("unresolved_dependencies", [])}

        contains_view = False
        contains_external = False

        # 检查视图依赖
        for v in deps_view:
            t_type = (type_map.get(v) or "").upper()
            if t_type == "VIEW":
                contains_view = True
            elif t_type == "EXTERNAL" or not t_type:
                contains_external = True

        # 检查表依赖
        for t in deps_table:
            t_type = (type_map.get(t) or "").upper()
            if t_type == "EXTERNAL" or not t_type:
                contains_external = True

        # 检查缺失的视图依赖
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

    def _topo_sort(self, graph: Dict[str, Set[str]], priority_map: Optional[Dict[str, int]] = None) -> List[str]:
        """拓扑排序"""
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

    def _build_dependencies_ctx(
        self,
        feature: Dict[str, Any],
        dep_nodes_info: Dict[str, Dict[str, Any]],
        dep_features_cache: Dict[str, str],
        node_layer_cache: Dict[str, str],
    ) -> List[Dict[str, Any]]:
        """构建 AI 所需的依赖上下文"""
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
                "type": node_info.get("node_type") or ("VIEW" if dep_name in (feature.get("view_dependencies") or []) else "TABLE"),
                "layer": known_layer,
                "node_type": node_info.get("node_type") or "UNKNOWN",
                "ai_description": node_info.get("ai_description") or "",
            }

            feat_raw = dep_features_cache.get(dep_key)
            if feat_raw:
                ctx_item["feature_json"] = feat_raw[:3000]

            ctx.append(ctx_item)

        return ctx

    def _interactive_confirm_layer(self, view_name: str, ai_suggest: str) -> str:
        """交互式确认分层"""
        choices = ["DIM", "DWD", "DWS", "OTHER"]
        default_choice = ai_suggest if ai_suggest in choices else "OTHER"
        prompt_text = f"请确认 [cyan]{view_name}[/cyan] 的数仓层级"
        user_input = Prompt.ask(prompt_text, choices=choices, default=default_choice, show_choices=True)
        return user_input

    def _update_dw_node_layer_info(
        self, view_id: int, ai_suggest: str, ai_desc: str, ai_conf: float, human_final: str
    ):
        """更新节点层级信息"""
        if not view_id:
            return

        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        ai_desc_esc = escape_sql(ai_desc)
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
        except Exception as e:
            logger.error(f"更新节点层级信息失败 ID={view_id}: {e}")

    def _load_view_features_for_naming(self) -> List[Dict[str, Any]]:
        """加载用于命名的视图特征"""
        sql = (
            "SELECT ts.table_id as view_id, ts.table_name as view_name, ts.hash, "
            "af.feature_json "
            "FROM dw_meta.table_source ts "
            "LEFT JOIN dw_meta.ai_view_feature af ON ts.table_id = af.table_id "
            f"WHERE ts.source_system = '{self.sourcedb}' AND ts.table_type = 'VIEW' AND ts.parse_status = 'PARSED'"
        )

        res = self.meta_conn.execute({"sql_query": sql, "result_format": "list"})
        return parse_query_result(res)

    def _load_table_source_index(self) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """加载表源索引"""
        sql = "SELECT table_id, table_name, table_type, source_system FROM dw_meta.table_source"
        res = self.meta_conn.execute({"sql_query": sql, "result_format": "list"})
        rows = parse_query_result(res)

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

    def _build_field_graph(self, feature_map: Dict[str, Dict[str, Any]], resolver: FieldResolver, uf: UnionFind):
        """构建字段等价图"""
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

    def _load_existing_std_mappings(self) -> Dict[str, int]:
        """加载已有标准字段映射"""
        sql = "SELECT source_system, source_table, source_column, std_field_id FROM dw_meta.std_field_mapping WHERE is_active = 1"
        res = self.meta_conn.execute({"sql_query": sql, "result_format": "list"})
        rows = parse_query_result(res)

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
        """选择已有的标准字段ID"""
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
        """建议分组标准字段名"""
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
            parsed = safe_extract_json(resp)
            std_en = to_snake(parsed.get("std_field_name") or default_en)
            std_cn = parsed.get("std_field_name_cn") or default_cn
            if std_en and std_en not in banned_names:
                return std_en, std_cn
            prompt += f"\n请重新生成，避免使用: {std_en}"

        return default_en, default_cn

    def _interactive_confirm_group_naming(self, node_keys: Set[str], suggest_en: str, suggest_cn: str) -> Tuple[str, str]:
        """交互式确认分组命名"""
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
            return to_snake(parts[0]), suggest_cn
        if len(parts) >= 2:
            return to_snake(parts[0]), parts[1]

        return suggest_en, suggest_cn

    def _get_or_create_std_field(self, item: Dict[str, str]) -> int:
        """获取或创建标准字段"""
        std_name_raw = to_snake(item["std_field_name"])
        std_name = escape_sql(std_name_raw)
        std_name_cn = escape_sql(item["std_field_name_cn"])
        source_system = escape_sql(item.get("source_system") or self.sourcedb)

        select_sql = (
            "SELECT std_field_id FROM dw_meta.std_field "
            f"WHERE LOWER(std_field_name) = LOWER('{std_name}') "
            f"AND source_system = '{source_system}' "
            "ORDER BY std_field_id DESC LIMIT 1"
        )

        # 先查询是否存在
        res = self.meta_conn.execute({"sql_query": select_sql, "result_format": "list"})
        rows = parse_query_result(res)
        if rows:
            return int(rows[0].get("std_field_id"))

        # 不存在则创建
        insert = (
            "INSERT INTO dw_meta.std_field "
            "(std_field_name, std_field_name_cn, source_system, semantic_type) "
            f"VALUES ('{std_name}', '{std_name_cn}', '{source_system}', NULL)"
        )

        self.meta_conn.execute({"sql_query": insert})

        # 再次查询获取ID
        res = self.meta_conn.execute({"sql_query": select_sql, "result_format": "list"})
        rows = parse_query_result(res)
        if rows:
            return int(rows[0].get("std_field_id"))

        raise RuntimeError(f"无法获取 std_field_id: {std_name_raw}")

    def _upsert_std_mapping(self, std_field_id: int, item: Dict[str, str]):
        """插入或更新标准字段映射"""
        source_system = item.get("source_system") or self.sourcedb

        delete = (
            "DELETE FROM dw_meta.std_field_mapping "
            f"WHERE source_system = '{escape_sql(source_system)}' "
            f"AND source_db = '{escape_sql(item['source_db'])}' "
            f"AND source_table = '{escape_sql(item['source_table'])}' "
            f"AND source_column = '{escape_sql(item['source_column'])}'"
        )

        self.meta_conn.execute({"sql_query": delete})

        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        expr = escape_sql(item.get("expression_sql") or "")

        insert = (
            "INSERT INTO dw_meta.std_field_mapping "
            "(source_system, source_db, source_table, source_column, source_column_comment, source_data_type, "
            "std_field_id, transform_expr, is_primary_key, is_business_key, is_partition_key, is_active, remark, "
            "created_at, updated_at) "
            f"VALUES ('{escape_sql(source_system)}', '{escape_sql(item['source_db'])}', '{escape_sql(item['source_table'])}', "
            f"'{escape_sql(item['source_column'])}', '', NULL, {std_field_id}, '{expr}', "
            "0, 0, 0, 1, 'auto-generated', "
            f"'{now}', '{now}')"
        )

        self.meta_conn.execute({"sql_query": insert})