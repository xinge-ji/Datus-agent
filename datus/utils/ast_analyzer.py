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
        ddl_body = sql_text.strip()
        ddl_body = ddl_body.rstrip(";")

        as_match = re.search(r"\bAS\b", ddl_body, flags=re.IGNORECASE)
        if as_match:
            ddl_body = ddl_body[as_match.end() :]

        ddl_body = ddl_body.strip()
        ddl_body = self._strip_outer_select_parentheses(ddl_body)
        ddl_body = self._rewrite_nvl_new_column(ddl_body)

        # 如果前缀被误切掉，尝试定位第一个 SELECT 重新对齐（保留 WITH 开头的场景）。
        if not re.match(r"(?is)^(select|with)\b", ddl_body):
            select_match = re.search(r"(?is)\bselect\b", ddl_body)
            if select_match:
                ddl_body = ddl_body[select_match.start() :].lstrip()
        return ddl_body

    def _strip_outer_select_parentheses(self, sql: str) -> str:
        """仅在顶层 SELECT 被一对括号整体包裹时去掉括号。"""
        trimmed = sql.strip()
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

        inner = trimmed[1:-1].strip()
        if re.match(r"(?is)^select\b", inner):
            return inner
        return trimmed

    def _rewrite_nvl_new_column(self, sql: str) -> str:
        """
        将 NVL(x, default) x 这种“新建字段”写法改写为 default AS x，避免不存在的列导致解析报错。
        仅匹配未加引号的标识符，default 假定为简单字面量/常量。
        """
        pattern = re.compile(
            r"(?is)nvl\\s*\\(\\s*([A-Za-z_][\\w$#]*)\\s*,\\s*([^)]+?)\\s*\\)\\s+\\1(?![\\w$#])"
        )
        return pattern.sub(r"\\2 AS \\1", sql)

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

            left_alias = self._extract_table_alias_from_from_this(from_expr.this)
            join_list = from_expr.args.get("joins") or []
            current_left_alias = left_alias

            for join in join_list:
                if not isinstance(join, exp.Join):
                    continue

                right_alias = self._extract_table_alias_from_from_this(join.this)
                join_type = (join.args.get("kind") or "INNER").upper()

                conds: List[JoinCondition] = []
                on_expr = join.args.get("on")
                if on_expr is not None:
                    for eq in on_expr.find_all(exp.EQ):
                        left = eq.left
                        right = eq.right
                        left_sql = left.sql(dialect=self.dialect)
                        right_sql = right.sql(dialect=self.dialect)
                        jc = JoinCondition(left=left_sql, right=right_sql, op="=")
                        if isinstance(left, exp.Column):
                            jc.left_table_alias = left.table
                            jc.left_column = left.name
                        if isinstance(right, exp.Column):
                            jc.right_table_alias = right.table
                            jc.right_column = right.name
                        conds.append(jc)

                joins.append(
                    JoinInfo(
                        left_alias=current_left_alias or "",
                        right_alias=right_alias or "",
                        join_type=join_type,
                        conditions=conds,
                    )
                )
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

    def _collect_global_features(self, root: exp.Expression) -> tuple[bool, bool, List[str]]:
        has_group_by = any(isinstance(node, exp.Group) for node in root.walk())
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
        CREATE OR REPLACE FORCE VIEW "LYWMS"."TPLA_EDI_T400_V" ("GOODSOWNERID", "TRADEID", "TRADEDATE", "OPERATIONTYPE", "DTLLINES", "USESTATUS", "TRADEDTLID", "OWNERGOODSID", "FROMCOMPANYID", "FROMSTORAGEID", "FROMGOODSTATUS", "FROMQUANSTATUS", "FROMBATCHNO", "FROMLOTNO", "FROMPRODDATE", "FROMVALIDDATE", "FROMAPPROVENO", "TOCOMPANYID", "TOSTORAGEID", "TOGOODSTATUS", "TOQUNSTATUS", "TOBATCHNO", "TOLOTNO", "TOPRODDATE", "TOVALIDDATE", "TOAPPROVENO", "TRADEGOODSQTY", "TRADEGOODSPACK", "GOODSPACKID", "QTY", "GSTORAGEID", "PACKNAME", "SRCDTLID", "PACKSIZE", "WAREHID", "PRINTMONTHFLAG", "PRINTEXPIREFLAG", "FROMVALIDDATE2", "TOVALIDDATE2", "TOLOTNOFACTNAME", "PRODMONTHFLAG", "TOPACKSIZE", "TOPACKNAME", "MAH", "TOMAH") AS 
  select a.Goodsownerid, -- 1 :1:2ID
       a.tradeid, --  2 :3:4:5:6:7ID
       a.Tradedate,
       '00' Operationtype, -- 4 :8:9:10:11
       a.Dtllines, -- 5 :12:13:14:15:16:17:18:19:20
       a.Usestatus, --  6 :21:22:23:24
       b.Tradedtlid, -- 7 :25:26:27:28:29:30ID
       d.goodsownid ownerGoodsid, --  8 :31:32:33:34:35ID
       0 Fromcompanyid, -- 9 调整前货源单位ID
       '' Fromstorageid, --  10  调整前仓别
       ab.gostatusid FromGoodstatus, -- 11  调整前货品状态
       aa.goquantityid Fromquanstatus, -- 12  :36:37:38:39:40:41:42
       '0' Frombatchno, --  13  :43:44:45:46:47
       b.Fromlotno, --  14  :48:49:50:51:52
       b.Fromproddate Fromproddate,
       b.FromValiddate FromValiddate,
       b.FROMAPPROVEDOCNO FromApproveno, -- 17  :53:54:55:56:57:58:59
       0 Tocompanyid,
       '' Tostorageid, --  19  调整后仓别
       ad.gostatusid ToGoodstatus, -- 20  :60:61:62:63:64:65:66
       ac.goquantityid TOQUNSTATUS,
       '0' Tobatchno, --  22  :67:68:69:70:71
       b.Tolotno, --  23  :72:73:74:75:76
       b.Toproddate Toproddate,
       b.ToValiddate ToValiddate,
       b.TOAPPROVEDOCNO ToApproveno, -- 26  :77:78:79:80:81:82:83
       b.Tradegoodsqty, --  27  :84:85:86:87:88:89:90:91
       b.Tradegoodspack, -- 28  :92:93:94:95:96:97
       null Goodspackid, -- 29  :98:99
       b.Qty, --  30  :100:101
       '' gstorageid,
       b.packname,
       a.srcdtlid,
       b.packsize,
       a.warehid, --add by xyue 2016-03-09 新增中间表物流中心字段，用于区分同一核算单元下不同物流中心保管帐
       e.printmonthflag,
       e.printexpireflag,
       e.validdate2 fromvaliddate2,
       e.validdate2 tovaliddate2,
       nvl(e.factname,d.factname) as tolotnofactname,
       e.prodmonthflag,
       xyue_get_trade_topacksize(b.tradedtlid) as topacksize,
       xyue_get_trade_topackname(b.tradedtlid) as topackname,
       e.mah,
       e.mah as tomah
  From Tpl_trade_order           a,
       Tpl_trade_dtl             b,
       tpl_goods                 d,
       tpl_pub_goods_packs       d1,
       tpl_edi_queue             h,
       tpl_owner_quanstatus_def  aa,
       tpl_owner_goodsstatus_def ab,
       tpl_owner_quanstatus_def  ac,
       tpl_owner_goodsstatus_def ad,
       wms_goods_lot             e
 Where a.tradeid = b.tradeid
   and b.ownergoodsid = d.ownergoodsid
   and b.goodspackid = d1.goodspackid
   and a.usestatus = 3
   and h.srcid = a.tradeid
   and a.goodsownerid = aa.goodsownerid
   and a.goodsownerid = ab.goodsownerid
   and b.fromgoodsstatusid = ab.goodsstatusid
   and b.fromquanstatus = aa.quantityid
   and a.goodsownerid = ac.goodsownerid
   and a.goodsownerid = ad.goodsownerid
   and b.togoodsstatusid = ad.goodsstatusid
   and b.toquanstatus = ac.quantityid
   and nvl(h.expflag, 0) = 0
   and h.queuetype = 3
   and nvl(b.qty, 0) <> 0
   and b.togoodsstatusid not in (-4, -3) --change by xyue 2020-08-19 货位丢失、在库丢失不反馈
   and b.tolotid = e.lotid(+)
 Order by a.tradeid
        """
        features = analyzer.analyze_view(sql)
        print(features)
