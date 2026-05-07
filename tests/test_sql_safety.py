"""tests/test_sql_safety.py"""
import pytest
from src.sql_safety import validate_sql, SQLValidationError


def test_select_allowed():
    sql = "SELECT EQP_ID, COUNT(*) FROM TEST_TABLE GROUP BY EQP_ID"
    result = validate_sql(sql, allowed_statements=["SELECT", "WITH"],
                          blocked_keywords=["INSERT", "DELETE", "DROP"],
                          table_whitelist=["TEST_TABLE"], max_rows=10000)
    assert "TOP" in result


def test_cte_allowed():
    sql = "WITH cte AS (SELECT * FROM TEST_TABLE) SELECT * FROM cte"
    result = validate_sql(sql, allowed_statements=["SELECT", "WITH"],
                          blocked_keywords=["DELETE"],
                          table_whitelist=["TEST_TABLE"], max_rows=10000)
    assert result is not None


def test_insert_blocked():
    sql = "INSERT INTO TEST_TABLE VALUES (1, 2)"
    with pytest.raises(SQLValidationError, match="금지 키워드"):
        validate_sql(sql, allowed_statements=["SELECT", "WITH"],
                      blocked_keywords=["INSERT", "DELETE"],
                      table_whitelist=["TEST_TABLE"], max_rows=10000)


def test_delete_blocked():
    sql = "DELETE FROM TEST_TABLE WHERE ID = 1"
    with pytest.raises(SQLValidationError, match="금지 키워드"):
        validate_sql(sql, allowed_statements=["SELECT", "WITH"],
                      blocked_keywords=["INSERT", "DELETE"],
                      table_whitelist=["TEST_TABLE"], max_rows=10000)


def test_drop_in_string_not_blocked():
    """문자열 안의 DROP은 차단하지 않는다."""
    sql = "SELECT * FROM TEST_TABLE WHERE COMMENT = 'drop this'"
    result = validate_sql(sql, allowed_statements=["SELECT", "WITH"],
                          blocked_keywords=["DROP"],
                          table_whitelist=["TEST_TABLE"], max_rows=10000)
    assert result is not None


def test_table_not_in_whitelist():
    sql = "SELECT * FROM SECRET_TABLE"
    with pytest.raises(SQLValidationError, match="허용되지 않는 테이블"):
        validate_sql(sql, allowed_statements=["SELECT", "WITH"],
                      blocked_keywords=[],
                      table_whitelist=["TEST_TABLE"], max_rows=10000)


def test_empty_whitelist_allows_all():
    """화이트리스트가 비어있으면 모든 테이블 허용 (초기 설정용)."""
    sql = "SELECT * FROM ANY_TABLE"
    result = validate_sql(sql, allowed_statements=["SELECT", "WITH"],
                          blocked_keywords=[],
                          table_whitelist=[], max_rows=10000)
    assert result is not None


def test_top_already_present():
    """이미 TOP이 있으면 추가하지 않는다."""
    sql = "SELECT TOP 100 * FROM TEST_TABLE"
    result = validate_sql(sql, allowed_statements=["SELECT", "WITH"],
                          blocked_keywords=[],
                          table_whitelist=["TEST_TABLE"], max_rows=10000)
    assert result.count("TOP") == 1


def test_top_auto_added():
    """TOP이 없으면 자동 추가한다."""
    sql = "SELECT EQP_ID FROM TEST_TABLE"
    result = validate_sql(sql, allowed_statements=["SELECT", "WITH"],
                          blocked_keywords=[],
                          table_whitelist=["TEST_TABLE"], max_rows=5000)
    assert "TOP 5000" in result
