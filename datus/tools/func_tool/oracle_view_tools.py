# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Oracle view analysis helpers for Doris data warehouse migrations.

This tool parses Oracle `CREATE VIEW` statements with SQLGlot, extracts lineage
and transformation details, and suggests DWD/DWS/ADS designs plus Doris-ready
SQL skeletons (including sqlmesh model stubs).
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

import sqlglot
from agents import Tool
from sqlglot import exp

from datus.tools.func_tool.base import FuncToolResult, trans_to_function_tool
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


def _safe_sql(expression: exp.Expression, dialect: str = "oracle") -> str:
    """Render an expression as SQL with best-effort error handling."""

    try:
        return expression.sql(dialect=dialect)
    except Exception:
        return str(expression)


def _rewrite_to_doris(sql_fragment: str) -> str:
    """Rewrite Oracle SQL fragment to a Doris/MySQL-compatible expression."""

    try:
        return sqlglot.transpile(sql_fragment, read="oracle", write="mysql", pretty=False)[0]
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.warning(f"Failed to rewrite expression to Doris: {exc}")
        return sql_fragment


def _collect_projection_columns(select_expr: exp.Select) -> List[Dict[str, str]]:
    """Collect projected column metadata (name + source expression)."""

    columns: List[Dict[str, str]] = []
    for proj in select_expr.expressions:
        base_expr = proj.this if isinstance(proj, exp.Alias) else proj
        alias = proj.alias_or_name if isinstance(proj, exp.Alias) else proj.alias_or_name
        column_name = alias or getattr(base_expr, "name", None) or _safe_sql(base_expr)
        columns.append(
            {
                "name": column_name,
                "expression": _safe_sql(base_expr),
                "doris_expression": _rewrite_to_doris(_safe_sql(base_expr)),
                "is_aggregate": bool(base_expr.find(exp.Aggregate)),
                "is_simple_column": isinstance(base_expr, exp.Column),
            }
        )
    return columns


def _extract_group_bys(select_expr: exp.Select) -> List[str]:
    group_clause = select_expr.args.get("group")
    if not group_clause:
        return []
    return [_safe_sql(expr) for expr in group_clause.expressions]


def _table_identifier(table_expr: exp.Table) -> Dict[str, str]:
    return {
        "name": table_expr.name,
        "schema": table_expr.db or "",
        "catalog": table_expr.catalog or "",
        "alias": table_expr.alias or "",
    }


def _dedupe_dicts(items: List[Dict[str, str]]) -> List[Dict[str, str]]:
    seen = set()
    unique_items: List[Dict[str, str]] = []
    for item in items:
        marker = tuple(item.values())
        if marker in seen:
            continue
        seen.add(marker)
        unique_items.append(item)
    return unique_items


def _pick_partition_key(column_names: List[str]) -> str:
    """Pick a partition key, preferring date-like columns."""

    for name in column_names:
        if re.search(r"(date|dt|day|time)$", name, re.IGNORECASE):
            return name
    return "biz_date"


def _preferred_primary_keys(column_names: List[str]) -> List[str]:
    candidates = [name for name in column_names if name.lower().endswith("id")]
    return candidates or column_names[:1]


def _build_doris_ddl(
    table_name: str,
    columns: List[Dict[str, str]],
    primary_keys: Optional[List[str]],
    partition_key: str,
) -> str:
    """Generate a Doris CREATE TABLE skeleton."""

    column_defs: List[str] = []
    for col in columns:
        definition = f"  `{col['name']}` STRING"
        expression = col.get("doris_expression") or col.get("expression")
        if expression:
            definition += f" COMMENT 'derived from: {expression}'"
        column_defs.append(definition)

    if partition_key not in [c["name"] for c in columns]:
        column_defs.append(f"  `{partition_key}` DATE COMMENT 'partition key'")

    pk_clause = ", ".join(f"`{pk}`" for pk in (primary_keys or []))
    hash_key = (primary_keys or [columns[0]["name"] if columns else partition_key])[0]

    ddl_lines = [
        f"CREATE TABLE IF NOT EXISTS `{table_name}` (",
        ",\n".join(column_defs),
        f"\n) DUPLICATE KEY ({pk_clause})" if pk_clause else "\n) DUPLICATE KEY()",
        f"DISTRIBUTED BY HASH(`{hash_key}`) BUCKETS 10",
        f"PARTITION BY RANGE(`{partition_key}`) (\n  START (CURRENT_DATE - INTERVAL 3 YEAR)\n  END (CURRENT_DATE + INTERVAL 1 YEAR)\n  EVERY (INTERVAL 1 DAY)\n)",
        "PROPERTIES (\n  'replication_num' = '3',\n  'light_schema_change' = 'true'\n);",
    ]
    return "\n".join(ddl_lines)


def _clean_name(name: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", name.lower())


def _collect_function_calls(select_expr: exp.Select) -> List[str]:
    """Collect unique function names referenced in the SELECT tree."""

    names: List[str] = []
    for func in select_expr.find_all(exp.Func):
        if isinstance(func, exp.Anonymous):
            func_name = func.name or ""
        else:
            func_name = func.key or getattr(func, "this", None) or ""
        func_name = str(func_name)
        if not func_name:
            continue
        normalized = func_name.lower()
        if normalized not in names:
            names.append(normalized)
    return names


class OracleViewTools:
    """Function tools for analyzing Oracle views and proposing Doris layers."""

    def available_tools(self) -> List[Tool]:
        return [trans_to_function_tool(self.analyze_oracle_view)]

    def analyze_oracle_view(
        self, view_sql: str, default_domain: str = "", oracle_functions: Optional[Dict[str, str]] = None
    ) -> FuncToolResult:
        """
        Parse an Oracle CREATE VIEW statement and propose Doris layering.

        Args:
            view_sql: Full CREATE OR REPLACE VIEW statement (Oracle dialect).
            default_domain: Optional domain prefix for naming when schema is missing.
            oracle_functions: Optional mapping of function name to source DDL/definition
                used by the view. Provide this to keep function lineage with the view
                analysis and make Oracle→Doris rewrites explicit.

        Returns:
            Structured analysis including lineage, grouping, Doris DDL, and sqlmesh stubs.
        """

        try:
            parsed = sqlglot.parse_one(view_sql, read="oracle", error_level=sqlglot.ErrorLevel.IGNORE)
        except Exception as exc:
            logger.error(f"Failed to parse Oracle view SQL: {exc}")
            return FuncToolResult(success=0, error=f"Failed to parse Oracle view SQL: {exc}")

        if parsed is None:
            return FuncToolResult(success=0, error="Empty or invalid SQL provided")

        view_name = ""
        select_expr: Optional[exp.Select] = None

        if isinstance(parsed, exp.Create):
            if isinstance(parsed.this, exp.Table):
                view_name = parsed.this.name
            select_expr = parsed.find(exp.Select)
        elif isinstance(parsed, exp.Select):
            select_expr = parsed
        else:
            select_expr = parsed.find(exp.Select)

        if not select_expr:
            return FuncToolResult(success=0, error="No SELECT statement found in view definition")

        # Identify CTE names to avoid treating them as base sources
        cte_names = {
            cte.alias_or_name.lower()
            for cte in select_expr.find_all(exp.CTE)
            if cte.alias_or_name
        }

        function_definitions = oracle_functions or {}
        function_definitions_lower = {k.lower(): v for k, v in function_definitions.items()}
        functions_used = _collect_function_calls(select_expr)
        functions_with_defs = [
            {
                "name": fn,
                "provided": fn in function_definitions_lower,
                "definition": function_definitions.get(fn)
                or function_definitions.get(fn.upper())
                or function_definitions_lower.get(fn),
            }
            for fn in functions_used
        ]

        source_tables = _dedupe_dicts(
            [
                _table_identifier(table)
                for table in select_expr.find_all(exp.Table)
                if table.name.lower() not in cte_names
            ]
        )

        join_conditions = [
            {
                "type": join.args.get("kind", ""),
                "source": _table_identifier(join.this) if isinstance(join.this, exp.Table) else {},
                "condition": _safe_sql(join.args.get("on")) if join.args.get("on") else "",
            }
            for join in select_expr.find_all(exp.Join)
        ]

        filters = [
            _safe_sql(where.this)
            for where in select_expr.find_all(exp.Where)
            if where.this is not None
        ]

        group_by_fields = _extract_group_bys(select_expr)
        projection_columns = _collect_projection_columns(select_expr)
        column_names = [col["name"] for col in projection_columns]
        partition_key = _pick_partition_key(column_names)
        primary_keys = _preferred_primary_keys(column_names)

        has_aggregates = any(col["is_aggregate"] for col in projection_columns) or bool(group_by_fields)
        domain = default_domain or (source_tables[0]["schema"] if source_tables else "core")
        clean_view_name = _clean_name(view_name or "view")
        clean_domain = _clean_name(domain or "core")

        dwd_table = f"dwd_{clean_domain}_{clean_view_name}"
        dws_table = f"dws_{clean_domain}_{clean_view_name}_agg"
        ads_table = f"ads_{clean_domain}_{clean_view_name}_report"

        try:
            rewritten_select = sqlglot.transpile(
                select_expr.sql(dialect="oracle"), read="oracle", write="mysql", pretty=True
            )[0]
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning(f"Failed to rewrite SELECT for Doris: {exc}")
            rewritten_select = select_expr.sql(dialect="oracle")

        aggregate_projections = [col for col in projection_columns if col["is_aggregate"]]
        group_projections = [col for col in projection_columns if not col["is_aggregate"]]

        dws_select_lines: List[str] = []
        if has_aggregates:
            if group_projections:
                dws_select_lines.append(
                    ",\n  ".join(col["doris_expression"] for col in group_projections)
                )
            if aggregate_projections:
                dws_select_lines.append(
                    ",\n  ".join(
                        f"{col['doris_expression']} AS {col['name']}" for col in aggregate_projections
                    )
                )
        else:
            dws_select_lines.append("*")

        dws_select_sql = (
            "SELECT\n  "
            + "\n  ".join([line for line in dws_select_lines if line])
            + f"\nFROM {dwd_table}"
            + (f"\nGROUP BY {', '.join(group_by_fields)}" if group_by_fields else "")
        )

        ads_select_sql = f"SELECT * FROM {dws_table}"

        doris_ddl = {
            "dwd": _build_doris_ddl(dwd_table, projection_columns, primary_keys, partition_key),
            "dws": _build_doris_ddl(dws_table, projection_columns, group_by_fields or [partition_key], partition_key),
            "ads": _build_doris_ddl(ads_table, projection_columns, group_by_fields or primary_keys, partition_key),
        }

        sqlmesh_models = {
            "dwd": (
                f"MODEL (\n  name {dwd_table}\n  kind INCREMENTAL_BY_TIME_RANGE (time_column '{partition_key}')\n  cron '@daily'\n);\n\n"
                f"SELECT\n{rewritten_select}\n"
            ),
            "dws": (
                f"MODEL (\n  name {dws_table}\n  kind INCREMENTAL_BY_TIME_RANGE (time_column '{partition_key}')\n  cron '@daily'\n);\n\n{dws_select_sql}\n"
            ),
            "ads": (
                f"MODEL (\n  name {ads_table}\n  kind VIEW\n);\n\n{ads_select_sql}\n"
            ),
        }

        layering_strategy = {
            "playbook": {
                "ods": "Only light cleaning; mirror Oracle objects if needed.",
                "dwd": "Normalize joins and retain business keys without aggregation.",
                "dws": "Aggregate by time/dimensions; reusable subject summaries.",
                "ads": "Wide tables or report-facing outputs derived from DWS.",
            },
            "recommendation": {
                "dwd": {
                    "table": dwd_table,
                    "reason": "Joins/filters/derivations should land in the detail fact layer before aggregation.",
                    "primary_keys": primary_keys,
                    "partition_key": partition_key,
                },
                "dws": {
                    "table": dws_table,
                    "reason": "Group BY or aggregate expressions detected; consolidate into subject-level summaries.",
                    "grain": group_by_fields,
                }
                if has_aggregates
                else {},
                "ads": {
                    "table": ads_table,
                    "reason": "Report-facing layout built on top of DWS outputs.",
                },
            },
        }

        return FuncToolResult(
            result={
                "view_name": view_name,
                "source_tables": source_tables,
                "join_conditions": join_conditions,
                "filters": filters,
                "group_by": group_by_fields,
                "computed_columns": projection_columns,
                "functions_used": functions_with_defs,
                "missing_functions": [f for f in functions_used if f not in function_definitions_lower],
                "rewritten_select": rewritten_select,
                "layering_strategy": layering_strategy,
                "doris_ddl": doris_ddl,
                "dws_sql": dws_select_sql,
                "ads_sql": ads_select_sql,
                "sqlmesh_models": sqlmesh_models,
            }
        )
