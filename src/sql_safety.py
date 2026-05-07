"""src/sql_safety.py — LLM이 생성한 SQL의 안전성을 검증한다."""
import re
import sqlparse


class SQLValidationError(Exception):
    """SQL 검증 실패."""
    pass


def _extract_cte_names(sql: str) -> set[str]:
    """WITH 절에서 정의된 CTE 이름을 추출한다."""
    cte_names: set[str] = set()
    # Match CTE names: WITH name AS (...), name2 AS (...)
    pattern = r"(?i)\bWITH\b\s+([\s\S]+?)(?=\bSELECT\b)"
    m = re.search(pattern, sql)
    if m:
        cte_block = m.group(1)
        # Each CTE: name AS (
        for cte_match in re.finditer(r"(\w+)\s+AS\s*\(", cte_block, re.IGNORECASE):
            cte_names.add(cte_match.group(1).upper())
    return cte_names


def _extract_tables(sql: str) -> list[str]:
    """SQL에서 참조하는 테이블명을 추출한다 (CTE 이름은 제외)."""
    tables = []
    parsed = sqlparse.parse(sql)
    for stmt in parsed:
        _walk_tokens_for_tables(stmt.tokens, tables)
    cte_names = _extract_cte_names(sql)
    return [t.upper() for t in tables if t.upper() not in cte_names]


def _walk_tokens_for_tables(tokens, tables):
    """sqlparse 토큰 트리를 순회하며 FROM/JOIN 뒤의 테이블명을 찾는다."""
    from_seen = False
    for token in tokens:
        if token.ttype is sqlparse.tokens.Keyword and token.normalized in (
            "FROM", "JOIN", "INNER JOIN", "LEFT JOIN", "RIGHT JOIN",
            "FULL JOIN", "CROSS JOIN", "LEFT OUTER JOIN", "RIGHT OUTER JOIN",
        ):
            from_seen = True
            continue
        if from_seen:
            if hasattr(token, "tokens"):
                for sub in token.flatten():
                    if sub.ttype is sqlparse.tokens.Name:
                        tables.append(sub.value)
                        from_seen = False
                        break
            elif token.ttype is sqlparse.tokens.Name:
                tables.append(token.value)
                from_seen = False
            elif token.ttype not in (sqlparse.tokens.Whitespace, sqlparse.tokens.Newline,
                                      sqlparse.tokens.Punctuation):
                from_seen = False
        if hasattr(token, "tokens"):
            _walk_tokens_for_tables(token.tokens, tables)


def _strip_string_literals(sql: str) -> str:
    """SQL 문자열 리터럴('...')을 제거하여 키워드 검사 시 오탐을 방지한다."""
    return re.sub(r"'[^']*'", "''", sql)


def validate_sql(
    sql: str,
    allowed_statements: list[str],
    blocked_keywords: list[str],
    table_whitelist: list[str],
    max_rows: int,
) -> str:
    """
    SQL을 검증하고, 필요시 TOP을 추가하여 안전한 SQL을 반환한다.

    Raises:
        SQLValidationError: 검증 실패 시
    """
    sql = sql.strip().rstrip(";")
    sql_upper = sql.upper().strip()

    # 1) 금지 키워드 체크 (문자열 리터럴 제외) — 허용문 체크보다 먼저 수행
    sql_no_strings = _strip_string_literals(sql_upper)
    for kw in blocked_keywords:
        pattern = r"\b" + re.escape(kw.upper()) + r"\b"
        if re.search(pattern, sql_no_strings):
            raise SQLValidationError(
                f"금지 키워드 '{kw}'가 SQL에 포함되어 있습니다.\n"
                f"SQL: {sql[:100]}..."
            )

    # 2) 허용 문(statement) 체크
    allowed_upper = [s.upper() for s in allowed_statements]
    starts_ok = any(sql_upper.startswith(s) for s in allowed_upper)
    if not starts_ok:
        raise SQLValidationError(
            f"허용되지 않는 SQL 문입니다. 허용: {allowed_statements}\n"
            f"SQL 시작: {sql[:50]}..."
        )

    # 3) 테이블 화이트리스트 (비어있으면 모든 테이블 허용)
    if table_whitelist:
        wl_upper = [t.upper() for t in table_whitelist]
        referenced = _extract_tables(sql)
        for tbl in referenced:
            tbl_name = tbl.split(".")[-1] if "." in tbl else tbl
            if tbl_name not in wl_upper:
                raise SQLValidationError(
                    f"허용되지 않는 테이블 '{tbl}'입니다.\n"
                    f"허용 목록: {table_whitelist}"
                )

    # 4) TOP 자동 추가 (MSSQL 스타일)
    if "TOP" not in sql_upper:
        sql = re.sub(
            r"(?i)^SELECT\b",
            f"SELECT TOP {max_rows}",
            sql,
            count=1,
        )

    return sql
