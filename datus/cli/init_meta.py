#!/usr/bin/env python3

# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Bootstrap the dw_meta metadata schema inside a namespace database."""

from __future__ import annotations

from dataclasses import dataclass
from textwrap import dedent, indent
from typing import List

from sqlglot import parse_one
from sqlglot.errors import SqlglotError

from datus.configuration.agent_config import AgentConfig
from datus.configuration.agent_config_loader import load_agent_config
from datus.tools.db_tools.db_manager import DBManager
from datus.utils.constants import DBType
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

DEFAULT_BUCKETS = 10
DEFAULT_PROPERTIES = dedent(
    """\
"replication_allocation" = "tag.location.default: 3",
"in_memory" = "false",
"storage_format" = "V2",
"disable_auto_compaction" = "false"
"""
).strip()


SQLGLOT_DIALECT_MAP = {
    DBType.DORIS.value: "doris",
    DBType.STARROCKS.value: "starrocks",
    DBType.MYSQL.value: "mysql",
    DBType.POSTGRES.value: "postgres",
    DBType.POSTGRESQL.value: "postgres",
    DBType.SNOWFLAKE.value: "snowflake",
    DBType.SQLITE.value: "sqlite",
    DBType.DUCKDB.value: "duckdb",
    DBType.BIGQUERY.value: "bigquery",
    DBType.CLICKHOUSE.value: "clickhouse",
    DBType.MSSQL.value: "tsql",
    DBType.SQLSERVER.value: "tsql",
    DBType.ORACLE.value: "oracle",
    DBType.HIVE.value: "hive",
}


@dataclass(frozen=True)
class TableDefinition:
    name: str
    columns: str
    buckets: int = DEFAULT_BUCKETS

    def _primary_key_columns(self) -> List[str]:
        raw_lines = [line.strip().rstrip(",") for line in dedent(self.columns).splitlines() if line.strip()]
        for line in raw_lines:
            upper = line.upper()
            if upper.startswith("PRIMARY KEY"):
                start = line.find("(")
                end = line.rfind(")")
                if start != -1 and end != -1 and end > start:
                    cols_segment = line[start + 1 : end]
                    cols = [col.strip(" `") for col in cols_segment.split(",") if col.strip()]
                    if cols:
                        return cols
        return []

    def _clean_columns(self) -> List[str]:
        raw_lines = [line.strip() for line in dedent(self.columns).splitlines() if line.strip()]
        filtered: List[str] = [line for line in raw_lines if not line.upper().startswith(("PRIMARY KEY", "UNIQUE KEY"))]
        if filtered and filtered[-1].endswith(","):
            filtered[-1] = filtered[-1].rstrip(",")
        return filtered

    def _unique_key_column(self) -> str:
        pk_cols = self._primary_key_columns()
        if pk_cols:
            return ", ".join(pk_cols)
        cleaned = self._clean_columns()
        if not cleaned:
            raise ValueError(f"表 {self.name} 缺少字段定义，无法生成 UNIQUE KEY")
        return cleaned[0].split()[0]

    def doris_statements(self) -> List[str]:
        clean_columns = self._clean_columns()
        unique_key = self._unique_key_column()
        column_sql = indent("\n".join(clean_columns), "    ")
        properties = indent(DEFAULT_PROPERTIES, "    ")
        drop_stmt = f"DROP TABLE IF EXISTS dw_meta.{self.name};"
        create_stmt = (
            f"CREATE TABLE dw_meta.{self.name} (\n"
            f"{column_sql}\n"
            f") UNIQUE KEY ({unique_key}) DISTRIBUTED BY HASH ({unique_key}) BUCKETS {self.buckets} PROPERTIES (\n"
            f"{properties}\n"
            ");"
        )
        return [drop_stmt, create_stmt]


DW_META_TABLES: tuple[TableDefinition, ...] = (
    TableDefinition(
        name="table_source",
        columns="""
table_id       BIGINT NOT NULL AUTO_INCREMENT,
source_system  VARCHAR(64) NOT NULL COMMENT 'ERP/CRM/..',
table_name     VARCHAR(256) NOT NULL COMMENT '源表或视图名称',
table_type     VARCHAR(32) NOT NULL COMMENT 'TABLE/VIEW/MATERIALIZED_VIEW',
ddl_sql        STRING       NOT NULL COMMENT '表或视图定义SQL',
hash           VARCHAR(64)  NULL COMMENT 'SQL hash，避免重复处理',
parse_status   VARCHAR(32) DEFAULT 'NEW' COMMENT 'NEW/PARSED/FAILED',
created_at     DATETIME,
updated_at     DATETIME,
PRIMARY KEY (table_id)
""",
    ),
    TableDefinition(
        name="dw_node",
        columns="""
node_id           BIGINT NOT NULL AUTO_INCREMENT,
node_type         VARCHAR(32) NOT NULL COMMENT 'ODS_TABLE/SRC_VIEW/DIM_TABLE/DWD_TABLE/DWS_TABLE',
source_system     VARCHAR(64) NOT NULL COMMENT 'ERP/CRM/..',
table_name        VARCHAR(256) NOT NULL,
source_table_id   BIGINT NULL COMMENT '如果是由某个表或视图迁移而来',
ai_layer_suggest  VARCHAR(16) NULL COMMENT 'AI建议层：DIM/DWD/DWS/OTHER',
ai_description    STRING NULL COMMENT 'AI描述',
ai_confidence     DECIMAL(5,4) NULL,
human_layer_final VARCHAR(16) NULL COMMENT '人确认后的层级',
migration_status  VARCHAR(32) DEFAULT 'NEW' COMMENT 'NEW/ANALYZED/PROPOSED/REVIEWED/IMPLEMENTED/SKIPPED/AST_FAILED',
created_at        DATETIME,
updated_at        DATETIME,
PRIMARY KEY (node_id),
""",
    ),
    TableDefinition(
        name="dw_node_relation",
        columns="""
from_node_id    BIGINT NOT NULL,
to_node_id      BIGINT NOT NULL,
relation_type   VARCHAR(32) NOT NULL COMMENT 'LINEAGE/JOIN/AGGREGATE/VIEW_DEP',
relation_detail STRING NULL COMMENT '可存JSON: join条件/聚合维度等',
created_at      DATETIME,
updated_at      DATETIME,
PRIMARY KEY (from_node_id, to_node_id, relation_type)
""",
    ),
    TableDefinition(
        name="std_field",
        columns="""
std_field_id       BIGINT NOT NULL AUTO_INCREMENT,
std_field_name     VARCHAR(128) NOT NULL COMMENT 'snake_case，如 order_id',
std_field_name_cn  VARCHAR(128) NOT NULL COMMENT '中文名',
biz_domain_code    VARCHAR(64) NULL COMMENT '业务域，如 SALES',
biz_entity_code    VARCHAR(64) NULL COMMENT '业务实体，如 ORDER',
data_type_std      VARCHAR(32) NOT NULL COMMENT 'bigint/string/decimal/date/..',
semantic_type      VARCHAR(32) NULL COMMENT 'id/code/name/status/amount/..',
description        STRING NULL,
unit               VARCHAR(64) NULL,
default_agg        VARCHAR(16) DEFAULT 'none',
is_active          TINYINT DEFAULT 1,
created_at         DATETIME,
updated_at         DATETIME,
PRIMARY KEY (std_field_id),
""",
    ),
    TableDefinition(
        name="std_field_mapping",
        columns="""
mapping_id            BIGINT NOT NULL AUTO_INCREMENT,
source_system         VARCHAR(64) NOT NULL,
source_db             VARCHAR(128) NOT NULL,
source_table          VARCHAR(256) NOT NULL,
source_column         VARCHAR(256) NOT NULL,
source_column_comment STRING NULL,
source_data_type      VARCHAR(64) NULL,
std_field_id          BIGINT NULL,
transform_expr        STRING NULL COMMENT '从源字段到标准字段的转换表达式',
is_primary_key        TINYINT DEFAULT 0,
is_business_key       TINYINT DEFAULT 0,
is_partition_key      TINYINT DEFAULT 0,
is_active             TINYINT DEFAULT 1,
remark                STRING NULL,
created_at            DATETIME,
updated_at            DATETIME,
PRIMARY KEY (mapping_id)
""",
    ),
    TableDefinition(
        name="dw_model",
        columns="""
model_id             BIGINT NOT NULL AUTO_INCREMENT,
model_name           VARCHAR(128) NOT NULL COMMENT '如 dwd_sales_detail',
db_name              VARCHAR(128) NOT NULL COMMENT '如 dwd_erp',
table_name           VARCHAR(128) NOT NULL COMMENT '物理表名，通常=模型名',
layer                VARCHAR(16)  NOT NULL COMMENT 'DIM/DWD/DWS',
sqlmesh_model_name   VARCHAR(256) NULL COMMENT 'sqlmesh 模型名，如 dwd_erp.dwd_sales_detail',
biz_domain_code      VARCHAR(64)  NULL,
biz_entity_code      VARCHAR(64)  NULL,
grain_desc           STRING NULL COMMENT '粒度描述',
primary_keys         STRING NULL COMMENT '逗号分隔',
partition_key        VARCHAR(128) NULL,
distributed_key      VARCHAR(128) NULL,
incremental_strategy VARCHAR(64)  NULL,
default_filter       STRING NULL COMMENT '如 is_active = 1',
source_table_id      BIGINT NULL COMMENT '原始表或视图ID',
status               VARCHAR(16) DEFAULT 'DRAFT' COMMENT 'DRAFT/ACTIVE/DEPRECATED',
created_at           DATETIME,
updated_at           DATETIME,
PRIMARY KEY (model_id),
""",
    ),
    TableDefinition(
        name="dw_model_column",
        columns="""
model_id           BIGINT NOT NULL,
column_name        VARCHAR(128) NOT NULL COMMENT 'snake_case',
column_order       INT NOT NULL,
std_field_id       BIGINT NULL,
expression_sql     STRING NULL COMMENT '相对于上游别名的表达式',
is_primary_key     TINYINT DEFAULT 0,
is_partition_key   TINYINT DEFAULT 0,
is_distributed_key TINYINT DEFAULT 0,
not_null           TINYINT DEFAULT 0,
comment            STRING NULL,
is_active          TINYINT DEFAULT 1,
created_at         DATETIME,
updated_at         DATETIME,
PRIMARY KEY (model_id, column_name)
""",
    ),
    TableDefinition(
        name="ai_view_feature",
        columns="""
table_id     BIGINT NOT NULL,
feature_json STRING NOT NULL COMMENT 'sqlglot 抽取的特征 JSON',
analyzed_at  DATETIME,
PRIMARY KEY (table_id)
""",
    ),
    TableDefinition(
        name="ai_feedback",
        columns="""
feedback_id     BIGINT NOT NULL AUTO_INCREMENT,
object_type     VARCHAR(32) NOT NULL COMMENT 'VIEW/MODEL/COLUMN',
object_key      VARCHAR(256) NOT NULL COMMENT '如 view:V_SA..., model:dwd_sales_detail, column:model_id:column_name',
suggestion_type VARCHAR(64) NOT NULL COMMENT 'LAYER/PK/STD_FIELD/MODEL_NAME/...',
ai_value        STRING NULL,
human_value     STRING NULL,
context_feature STRING NULL,
created_by      VARCHAR(64) NULL,
created_at      DATETIME,
PRIMARY KEY (feedback_id)
""",
    ),
    TableDefinition(
        name="ai_rule",
        columns="""
rule_id                   BIGINT NOT NULL AUTO_INCREMENT,
rule_type                 VARCHAR(32) NOT NULL COMMENT 'VIEW_LAYER/STD_FIELD_MAPPING/NAME_PATTERN',
pattern                   STRING NOT NULL COMMENT '匹配条件描述(文本/简单DSL)',
action                    STRING NOT NULL COMMENT '执行动作描述，如 layer=DWD,biz_domain=SALES',
weight                    INT DEFAULT 1,
is_active                 TINYINT DEFAULT 1,
created_from_feedback_ids STRING NULL,
created_at                DATETIME,
PRIMARY KEY (rule_id)
""",
    ),
)


def _base_statements() -> List[str]:
    statements = [
        dedent(
            """\
CREATE DATABASE IF NOT EXISTS dw_meta;
"""
        ).strip(),
        "USE dw_meta;",
    ]
    for table in DW_META_TABLES:
        statements.extend(table.doris_statements())
    return statements


def _normalize_dialect(db_type: str) -> str:
    normalized = (db_type or "").lower()
    return SQLGLOT_DIALECT_MAP.get(normalized, normalized)


def compile_dw_meta_statements(target_dialect: str) -> List[str]:
    statements: List[str] = []
    try:
        base_statements = _base_statements()
    except ValueError as exc:
        raise RuntimeError(f"生成 dw_meta DDL 失败: {exc}") from exc

    for raw_sql in base_statements:
        try:
            expr = parse_one(raw_sql, read="doris")
        except SqlglotError as exc:  # pragma: no cover - parse errors handled upstream
            raise RuntimeError(f"Failed to parse Doris DDL: {exc}") from exc
        statements.append(expr.sql(dialect=target_dialect))
        logger.debug(f"编译 SQL 成功（方言={target_dialect}）: {statements[-1]}")
    return statements


class SqlMeshMetaInitializer:
    """Execute dw_meta schema DDL against a namespace/database connection."""

    def __init__(self, namespace: str, config_path: str = "", database: str = ""):
        self.config_path = config_path or ""
        self.namespace = namespace or ""
        self.database = database or ""

    def run(self) -> int:
        if not self.namespace:
            logger.error("Namespace is required. Use --namespace <name>.")
            return 1

        agent_config = load_agent_config(
            config=self.config_path,
            namespace=self.namespace,
            database=self.database,
            action="init-meta",
        )

        logic_db = self._resolve_database(agent_config)
        if not logic_db:
            return 1

        db_manager = DBManager(agent_config.namespaces)
        connector = db_manager.get_conn(agent_config.current_namespace, logic_db)
        dialect = _normalize_dialect(connector.dialect or agent_config.db_type)

        if not dialect:
            logger.error(f"Unable to determine SQL dialect for namespace {self.namespace}.")
            return 1

        try:
            statements = compile_dw_meta_statements(dialect)
        except (RuntimeError, SqlglotError) as exc:
            logger.error(f"Failed to compile dw_meta schema for dialect '{dialect}': {exc}")
            return 1

        logger.info(
            f"准备在命名空间 '{self.namespace}' 的数据库 '{logic_db}' 执行 {len(statements)} 条建表语句，方言={dialect}",
        )

        for statement in statements:
            logger.debug(f"执行SQL: {statement}")
            result = connector.execute({"sql_query": statement})
            if not result.success or result.error:
                error_msg = result.error or "Unknown error"
                logger.error(f"SQL execution failed: {error_msg}\nSQL: {statement}")
                return 1

        logger.info(
            f"dw_meta metadata schema initialized on namespace '{self.namespace}' (database '{logic_db}')",
        )
        return 0

    def _resolve_database(self, agent_config: AgentConfig) -> str | None:
        db_configs = agent_config.current_db_configs()
        logic_db = self.database or agent_config.current_database
        if logic_db:
            if logic_db not in db_configs:
                logger.error(
                    f"Database '{logic_db}' not found under namespace '{self.namespace}'. Use --database to pick one of: {', '.join(sorted(db_configs.keys()))}",
                )
                return None
            agent_config.current_database = logic_db
            return logic_db

        if len(db_configs) == 1:
            single = list(db_configs.keys())[0]
            agent_config.current_database = single
            return single

        logger.error(
            f"Namespace '{self.namespace}' has multiple databases ({', '.join(sorted(db_configs.keys()))}). Please provide --database <logic_name>.",
        )
        return None
