"""
基于 sqlglot 的视图 AST 解析器。
仅支持 oracle 方言（可按需扩展），输出结构化特征用于元数据入库。
"""

from __future__ import annotations

import re

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from sqlglot import exp, parse_one
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


@dataclass
class TableInfo:
    alias: str
    db: Optional[str]
    name: str
    source_type: Optional[str]


@dataclass
class ColumnInfo:
    output_name: str
    expression_sql: str
    source_table_alias: Optional[str]
    source_column: Optional[str]
    is_aggregated: bool
    is_group_key: bool
    is_window: bool = False


@dataclass
class JoinCondition:
    left: str
    right: str
    op: str
    left_table_alias: Optional[str] = None
    left_column: Optional[str] = None
    right_table_alias: Optional[str] = None
    right_column: Optional[str] = None


@dataclass
class JoinInfo:
    left_alias: str
    right_alias: str
    join_type: str
    conditions: List[JoinCondition]


@dataclass
class ViewFeatures:
    view_name: Optional[str]
    has_group_by: bool
    has_window: bool
    aggregates: List[str]
    tables: List[TableInfo]
    columns: List[ColumnInfo]
    joins: List[JoinInfo]


class AstAnalyzer:
    """单条视图 SQL 的 AST 分析器。"""

    def __init__(self, dialect: str = "oracle"):
        self.dialect = dialect

    def analyze_view(self, sql_text: str, view_name: Optional[str] = None) -> Dict[str, Any]:
        """
        :param sql_text: 视图定义 SQL
        :param view_name: 可选，视图名
        :return: 可 JSON 序列化的 dict
        """
        # 先尝试使用配置的方言（默认 Oracle），如果失败则回退到 MySQL
        select_sql = self._prepare_select_sql(sql_text)
        try:
            root: exp.Expression = parse_one(select_sql, read=self.dialect)
            logger.debug(f"Successfully parsed SQL for view {view_name or ''} using {self.dialect} dialect")
        except Exception as oracle_exc:
            logger.debug(
                f"Failed to parse SQL for view {view_name or ''} using {self.dialect} dialect: {oracle_exc}. "
                f"Falling back to MySQL dialect."
            )
            try:
                root: exp.Expression = parse_one(select_sql, read="mysql")
                logger.debug(f"Successfully parsed SQL for view {view_name or ''} using MySQL dialect (fallback)")
            except Exception as mysql_exc:
                raise ValueError(
                    f"Failed to parse SQL for view {view_name or ''} with both {self.dialect} and MySQL dialects. "
                    f"SQL: {select_sql}"
                ) from mysql_exc

        tables = self._collect_tables(root)
        columns = self._collect_columns(root)
        joins = self._collect_joins(root)
        has_group_by, has_window, aggregates = self._collect_global_features(root)

        vf = ViewFeatures(
            view_name=view_name,
            has_group_by=has_group_by,
            has_window=has_window,
            aggregates=aggregates,
            tables=tables,
            columns=columns,
            joins=joins,
        )
        return asdict(vf)

    def _prepare_select_sql(self, sql_text: str) -> str:
        """从视图 DDL 中提取可解析的 SELECT，并去掉顶层包裹括号。"""
        ddl_body = self._strip_block_comments(sql_text)
        ddl_body = self._normalize_commas_and_parentheses(ddl_body)
        ddl_body = self._normalize_dblink_spacing(ddl_body)
        ddl_body = self._strip_ly_domain_in_dblink(ddl_body)
        ddl_body = self._replace_cdiwcs_domain(ddl_body)
        ddl_body = self._rewrite_dblink_to_db_table(ddl_body)
        ddl_body = self._normalize_comparison_operators(ddl_body)
        ddl_body = self._strip_extra_closing_parentheses(ddl_body)
        ddl_body = ddl_body.strip()
        ddl_body = ddl_body.rstrip(";")

        as_match = re.search(r"\bAS\b", ddl_body, flags=re.IGNORECASE)
        if as_match:
            ddl_body = ddl_body[as_match.end() :]

        ddl_body = ddl_body.strip()
        ddl_body = self._strip_oracle_plus(ddl_body)
        ddl_body = self._strip_outer_select_parentheses(ddl_body)
        ddl_body = self._rewrite_nvl_new_column(ddl_body)

        # 如果前缀被误切掉，尝试定位第一个 SELECT 重新对齐（保留 WITH 开头的场景）。
        if not re.match(r"(?is)^(select|with)\b", ddl_body):
            select_match = re.search(r"(?is)\bselect\b", ddl_body)
            if select_match:
                ddl_body = ddl_body[select_match.start() :].lstrip()

        return ddl_body

    def _normalize_commas_and_parentheses(self, sql: str) -> str:
        """仅替换引号外的中文逗号和括号为英文逗号和括号，避免误改字符串字面量。"""
        if "，" not in sql and "（" not in sql and "）" not in sql:
            return sql

        result: List[str] = []
        in_single = False
        in_double = False
        i = 0
        length = len(sql)

        while i < length:
            ch = sql[i]
            if ch == "'" and not in_double:
                # 单引号内采用 '' 转义，遇到成对单引号直接跳过一位
                if in_single and i + 1 < length and sql[i + 1] == "'":
                    result.append("''")
                    i += 2
                    continue
                in_single = not in_single
                result.append(ch)
            elif ch == '"' and not in_single:
                # 双引号内采用 "" 转义
                if in_double and i + 1 < length and sql[i + 1] == '"':
                    result.append('""')
                    i += 2
                    continue
                in_double = not in_double
                result.append(ch)
            else:
                if ch == "，" and not in_single and not in_double:
                    result.append(",")
                elif ch == "（" and not in_single and not in_double:
                    result.append("(")
                elif ch == "）" and not in_single and not in_double:
                    result.append(")")
                else:
                    result.append(ch)
            i += 1

        return "".join(result)

    def _strip_block_comments(self, sql: str) -> str:
        """删除 /* */ 包围的块注释。"""
        return re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)

    def _normalize_comparison_operators(self, sql: str) -> str:
        """将 > =、< =、< > 之间的空格去掉。"""
        sql = re.sub(r">\s+=", ">=", sql)
        sql = re.sub(r"<\s+=", "<=", sql)
        sql = re.sub(r"<\s+>", "<>", sql)
        return sql

    def _strip_ly_domain_in_dblink(self, sql: str) -> str:
        """仅在跨库表引用中去掉 .ly.com 域名（形如 table@xxx.ly.com 或 "table"@xxx.ly.com -> table@xxx）。"""
        # 支持 table 和 "table" 两种形式
        return re.sub(r"\.ly\.com", "", sql)

    def _rewrite_dblink_to_db_table(self, sql: str) -> str:
        """
        将 table@db 或 "table"@db 改写为 db.table / db."table"，仅处理引号与行注释外的片段，避免误改字面量。
        """
        result: List[str] = []
        in_single = False
        in_line_comment = False
        i = 0
        length = len(sql)

        while i < length:
            ch = sql[i]

            if in_line_comment:
                result.append(ch)
                if ch == "\n":
                    in_line_comment = False
                i += 1
                continue

            if not in_single and ch == "-" and i + 1 < length and sql[i + 1] == "-":
                in_line_comment = True
                result.append(ch)
                result.append(sql[i + 1])
                i += 2
                continue

            if ch == "'":
                if in_single and i + 1 < length and sql[i + 1] == "'":
                    result.append("''")
                    i += 2
                    continue
                in_single = not in_single
                result.append(ch)
                i += 1
                continue

            if in_single:
                result.append(ch)
                i += 1
                continue

            ident, end_idx = self._match_identifier_at(sql, i)
            if ident is not None:
                j = end_idx
                while j < length and sql[j].isspace():
                    j += 1
                if j < length and sql[j] == "@":
                    k = j + 1
                    while k < length and sql[k].isspace():
                        k += 1
                    db_ident, db_end_idx = self._match_identifier_at(sql, k)
                    if db_ident is not None:
                        result.append(f"{db_ident}.{ident}")
                        i = db_end_idx
                        continue
                result.append(sql[i:end_idx])
                i = end_idx
                continue

            result.append(ch)
            i += 1

        return "".join(result)

    def _match_identifier_at(self, sql: str, index: int) -> tuple[Optional[str], int]:
        """
        从指定位置尝试匹配标识符（支持双引号包裹的标识符），返回 (匹配文本, 终止位置)；未匹配返回 (None, index)。
        """
        if index >= len(sql):
            return None, index

        ch = sql[index]
        if ch == '"':
            i = index + 1
            while i < len(sql):
                if sql[i] == '"':
                    if i + 1 < len(sql) and sql[i + 1] == '"':
                        i += 2
                        continue
                    return sql[index : i + 1], i + 1
                i += 1
            return None, index

        if ch.isalpha() or ch == "_":
            i = index + 1
            while i < len(sql) and (sql[i].isalnum() or sql[i] in "_$#"):
                i += 1
            return sql[index:i], i

        return None, index

    def _strip_extra_closing_parentheses(self, sql: str) -> str:
        """删除多余的右括号（默认出现在末端），忽略字符串和行注释内的括号。"""
        opens, closes = self._collect_paren_positions(sql)
        extra = len(closes) - len(opens)
        if extra <= 0:
            return sql

        remove_indices = set(closes[-extra:])
        return "".join(ch for idx, ch in enumerate(sql) if idx not in remove_indices)

    def _collect_paren_positions(self, sql: str) -> tuple[List[int], List[int]]:
        """收集引号与行注释外的括号位置，用于平衡计算。"""
        opens: List[int] = []
        closes: List[int] = []
        in_single = False
        in_double = False
        in_line_comment = False
        i = 0
        length = len(sql)

        while i < length:
            ch = sql[i]

            if in_line_comment:
                if ch == "\n":
                    in_line_comment = False
                i += 1
                continue

            if not in_single and not in_double and ch == "-" and i + 1 < length and sql[i + 1] == "-":
                in_line_comment = True
                i += 2
                continue

            if ch == "'" and not in_double:
                if in_single and i + 1 < length and sql[i + 1] == "'":
                    i += 2
                    continue
                in_single = not in_single
                i += 1
                continue

            if ch == '"' and not in_single:
                if in_double and i + 1 < length and sql[i + 1] == '"':
                    i += 2
                    continue
                in_double = not in_double
                i += 1
                continue

            if in_single or in_double:
                i += 1
                continue

            if ch == "(":
                opens.append(i)
            elif ch == ")":
                closes.append(i)

            i += 1

        return opens, closes

    def _strip_outer_select_parentheses(self, sql: str) -> str:
        """仅在顶层 SELECT 被一对括号整体包裹时去掉括号。"""
        trimmed = sql.strip()
        # 末尾存在形如 ") -- xxx" 的注释时，先移除尾部注释再判断括号
        tail_comment = re.search(r"\)\s*--.*$", trimmed)
        if tail_comment:
            trimmed = trimmed[: tail_comment.start() + 1].rstrip()

        if not (trimmed.startswith("(") and trimmed.endswith(")")):
            return trimmed

        depth = 0
        for idx, ch in enumerate(trimmed):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0 and idx != len(trimmed) - 1:
                    return trimmed

        inner = trimmed[1:-1].lstrip()
        # 去掉内层开头的行注释与块注释，再判断是否 select/with 开头
        while True:
            if inner.startswith("--"):
                newline = inner.find("\n")
                if newline == -1:
                    inner = ""
                    break
                inner = inner[newline + 1 :].lstrip()
                continue
            if inner.startswith("/*"):
                end = inner.find("*/")
                if end == -1:
                    inner = ""
                    break
                inner = inner[end + 2 :].lstrip()
                continue
            break

        if re.match(r"(?is)^select\b", inner):
            return inner
        return trimmed

    def _strip_oracle_plus(self, sql: str) -> str:
        """
        将 oracle 中的 (+) 去掉
        """
        pattern = re.compile(r"\(\+\)")
        return pattern.sub("", sql)

    def _rewrite_nvl_new_column(self, sql: str) -> str:
        """
        将 NVL(x, default) x 这种“新建字段”写法改写为 default AS x，避免不存在的列导致解析报错。
        仅匹配未加引号的标识符，default 假定为简单字面量/常量。
        """
        pattern = re.compile(r"(?is)nvl\\s*\\(\\s*([A-Za-z_][\\w$#]*)\\s*,\\s*([^)]+?)\\s*\\)\\s+\\1(?![\\w$#])")
        return pattern.sub(r"\\2 AS \\1", sql)

    def _replace_cdiwcs_domain(self, sql: str) -> str:
        """
        将 cdiwcs 不区分大小写地替换为 iwcs。
        """
        sql = re.sub(r"@cdiwcs", "@iwcs", sql, flags=re.IGNORECASE)
        return sql

    def _normalize_dblink_spacing(self, sql: str) -> str:
        """
        将跨库/DBLink 表名中的 @ 周围空格去掉，仅处理标识符@标识符的模式，避免误伤邮箱等字符串。
        """
        pattern = re.compile(r"(?i)([A-Za-z_][\w$#]*)\s*@\s*([A-Za-z_][\w$#]*)")
        return pattern.sub(lambda m: f"{m.group(1)}@{m.group(2)}", sql)

    def _collect_tables(self, root: exp.Expression) -> List[TableInfo]:
        tables: Dict[str, TableInfo] = {}

        for table in root.find_all(exp.Table):
            name = table.name
            db = table.args.get("db")
            db_name = db.name if isinstance(db, exp.Identifier) else None

            alias_expr = table.args.get("alias")
            if isinstance(alias_expr, exp.TableAlias):
                alias_name = alias_expr.this.name
            else:
                alias_name = name

            key = alias_name
            if key not in tables:
                tables[key] = TableInfo(
                    alias=alias_name,
                    db=db_name,
                    name=name,
                    source_type=None,
                )

        return list(tables.values())

    def _collect_columns(self, root: exp.Expression) -> List[ColumnInfo]:
        columns: List[ColumnInfo] = []

        for select in root.find_all(exp.Select):
            group_keys = set()
            group = select.args.get("group")
            if isinstance(group, exp.Group):
                for g_exp in group.expressions:
                    group_keys.add(g_exp.sql(dialect=self.dialect))

            agg_funcs: List[exp.Func] = [fn for fn in select.find_all(exp.Func) if getattr(fn, "is_aggregate", False)]
            agg_expr_sqls = {fn.sql(dialect=self.dialect) for fn in agg_funcs}
            window_exprs = list(select.find_all(exp.Window))

            for proj in select.expressions:
                if isinstance(proj, exp.Alias):
                    output_name = proj.alias
                    expr = proj.this
                else:
                    output_name = proj.alias_or_name or proj.sql(dialect=self.dialect)
                    expr = proj

                expr_sql = expr.sql(dialect=self.dialect)
                source_alias, source_col = self._extract_source_column(expr)
                is_group_key = expr_sql in group_keys
                is_aggregated = False
                if isinstance(expr, exp.Func) and getattr(expr, "is_aggregate", False):
                    is_aggregated = True
                elif expr_sql in agg_expr_sqls:
                    is_aggregated = True

                is_window = any(expr in w.args.get("this").walk() for w in window_exprs) if window_exprs else False

                columns.append(
                    ColumnInfo(
                        output_name=output_name,
                        expression_sql=expr_sql,
                        source_table_alias=source_alias,
                        source_column=source_col,
                        is_aggregated=is_aggregated,
                        is_group_key=is_group_key,
                        is_window=is_window,
                    )
                )

        return columns

    def _extract_source_column(self, expr: exp.Expression) -> tuple[Optional[str], Optional[str]]:
        if isinstance(expr, exp.Column):
            return expr.table, expr.name

        cols = list(expr.find_all(exp.Column))
        if len(cols) == 1:
            c = cols[0]
            return c.table, c.name
        return None, None

    def _collect_joins(self, root: exp.Expression) -> List[JoinInfo]:
        joins: List[JoinInfo] = []

        for select in root.find_all(exp.Select):
            from_expr = select.args.get("from")
            if not isinstance(from_expr, exp.From):
                continue

            join_list = (select.args.get("joins") or []) + (from_expr.args.get("joins") or [])
            left_alias = self._extract_table_alias_from_from_this(from_expr.this)
            current_left_alias = left_alias
            seen_aliases: List[str] = [alias for alias in [left_alias] if alias]
            where_join_eqs = self._extract_where_join_equalities(select.args.get("where"))

            for join in join_list:
                if not isinstance(join, exp.Join):
                    continue

                right_alias = self._extract_table_alias_from_from_this(join.this)
                join_type = self._resolve_join_type(join)

                conds: List[JoinCondition] = []
                on_expr = join.args.get("on")
                if on_expr is not None:
                    conds.extend(self._extract_join_conditions_from_on(on_expr))
                else:
                    conds.extend(self._match_implicit_join_conditions(right_alias, seen_aliases, where_join_eqs))

                join_left_alias = current_left_alias
                if join_type == "IMPLICIT" and conds:
                    other_aliases = [
                        alias
                        for cond in conds
                        for alias in [cond.left_table_alias, cond.right_table_alias]
                        if alias and alias != right_alias
                    ]
                    if other_aliases:
                        join_left_alias = other_aliases[0]

                joins.append(
                    JoinInfo(
                        left_alias=join_left_alias or "",
                        right_alias=right_alias or "",
                        join_type=join_type,
                        conditions=conds,
                    )
                )

                if right_alias:
                    seen_aliases.append(right_alias)
                current_left_alias = right_alias

        return joins

    def _extract_table_alias_from_from_this(self, node: exp.Expression) -> Optional[str]:
        if isinstance(node, exp.Table):
            alias_expr = node.args.get("alias")
            if isinstance(alias_expr, exp.TableAlias):
                return alias_expr.this.name
            return node.name

        if isinstance(node, exp.Subquery):
            alias_expr = node.args.get("alias")
            if isinstance(alias_expr, exp.TableAlias):
                return alias_expr.this.name
            return None
        return None

    def _resolve_join_type(self, join: exp.Join) -> str:
        kind = join.args.get("kind")
        side = join.args.get("side")
        if kind:
            return str(kind).upper()
        if side:
            return str(side).upper()
        if join.args.get("on") is None:
            return "IMPLICIT"
        return "INNER"

    def _extract_join_conditions_from_on(self, on_expr: exp.Expression) -> List[JoinCondition]:
        conds: List[JoinCondition] = []
        for eq in on_expr.find_all(exp.EQ):
            if not (isinstance(eq.left, exp.Column) and isinstance(eq.right, exp.Column)):
                continue
            conds.append(
                JoinCondition(
                    left=eq.left.sql(dialect=self.dialect),
                    right=eq.right.sql(dialect=self.dialect),
                    op="=",
                    left_table_alias=eq.left.table,
                    left_column=eq.left.name,
                    right_table_alias=eq.right.table,
                    right_column=eq.right.name,
                )
            )
        return conds

    def _extract_where_join_equalities(self, where_expr: Optional[exp.Expression]) -> List[JoinCondition]:
        """
        抽取最外层 WHERE 中的等值条件，跳过子查询内的条件。
        """
        if where_expr is None:
            return []

        conds: List[JoinCondition] = []
        for eq in self._iter_eq_expressions(where_expr):
            if not (isinstance(eq.left, exp.Column) and isinstance(eq.right, exp.Column)):
                continue
            conds.append(
                JoinCondition(
                    left=eq.left.sql(dialect=self.dialect),
                    right=eq.right.sql(dialect=self.dialect),
                    op="=",
                    left_table_alias=eq.left.table,
                    left_column=eq.left.name,
                    right_table_alias=eq.right.table,
                    right_column=eq.right.name,
                )
            )
        return conds

    def _match_implicit_join_conditions(
        self, right_alias: Optional[str], seen_aliases: List[str], where_eqs: List[JoinCondition]
    ) -> List[JoinCondition]:
        if not right_alias:
            return []

        conds: List[JoinCondition] = []
        for cond in where_eqs:
            if not cond.left_table_alias or not cond.right_table_alias:
                continue

            pair = {cond.left_table_alias, cond.right_table_alias}
            if right_alias not in pair:
                continue

            other_alias = (pair - {right_alias}).pop() if len(pair) == 2 else None
            if other_alias and other_alias in seen_aliases:
                conds.append(cond)

        return conds

    def _iter_eq_expressions(self, expr: exp.Expression):
        """
        遍历最外层布尔表达式中的等值比较，忽略子查询，避免误把子查询里的筛选当作主查询 join。
        """
        stack = [expr]
        while stack:
            node = stack.pop()
            if isinstance(node, exp.Subquery):
                continue
            if isinstance(node, exp.Select) and node is not expr:
                continue
            if isinstance(node, exp.EQ):
                yield node
            for child in node.iter_expressions():
                stack.append(child)

    def _collect_global_features(self, root: exp.Expression) -> tuple[bool, bool, List[str]]:
        main_select: Optional[exp.Select]
        if isinstance(root, exp.Select):
            main_select = root
        else:
            main_select = next(root.find_all(exp.Select), None)

        group_expr = main_select.args.get("group") if main_select else None
        has_group_by = isinstance(group_expr, exp.Group)
        has_window = any(isinstance(node, exp.Window) for node in root.walk())

        agg_funcs = set()
        for func in root.find_all(exp.Func):
            if getattr(func, "is_aggregate", False):
                agg_funcs.add(func.name.upper())

        return has_group_by, has_window, sorted(agg_funcs)

    if __name__ == "__main__":
        from datus.utils.ast_analyzer import AstAnalyzer

        analyzer = AstAnalyzer()
        sql = """
CREATE OR REPLACE FORCE VIEW "LYERP"."GSP_COMPANY_LC_DTL_V" ("LCDTLID", "LCDOCID", "LICENSETYPEID", "LICENSENAME", "FACTORYFLAG", "CUSTOMFLAG", "SUPPLYFLAG", "RANGEFLAG", "REMINDMETHOD", "EARLYWARNDAYS", "SUPERDAYS") AS 
  (
-- 企业证照管控细单视图
select
a.lcdtlid,
a.lcdocid,
a.licensetypeid,
b.licensename ,
b.factoryflag,
b.customflag,
b.supplyflag,
b.rangeflag,
a.remindmethod,
a.earlywarndays,
a.superdays

from gsp_license_control_dtl a ,
gsp_license_type b
where a.licensetypeid = b.licensetypeid)
        """
        features = analyzer.analyze_view(sql)
        print(features)
