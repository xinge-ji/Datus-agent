import pytest

from datus.utils.ast_analyzer import AstAnalyzer


def test_prepare_cleans_dblink_domain_and_operators_and_block_comment():
    sql_text = """
    CREATE OR REPLACE VIEW V AS /* 删除的块注释 */
    (
        select *
        from foo@bar.ly.com t
        where t.a > = 1
          and t.b < > 2
          and t.c < = 3
    )
    """
    analyzer = AstAnalyzer()
    prepared = analyzer._prepare_select_sql(sql_text)

    assert prepared.lower().startswith("select")
    assert "foo@bar.ly.com" not in prepared
    assert "foo@bar" in prepared
    assert ">=" in prepared and "<>" in prepared and "<=" in prepared
    assert "/*" not in prepared and "*/" not in prepared


def test_prepare_strips_extra_closing_parenthesis_with_line_comment():
    sql_text = "create or replace view v as ( --comment before select (\nselect 1 from dual))"
    analyzer = AstAnalyzer()

    prepared = analyzer._prepare_select_sql(sql_text)

    assert prepared == "select 1 from dual"
