# Bin Defect Analyzer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CLI tool that uses LightRAG domain knowledge + LLM Agent to trace Bin defect root-cause equipment in semiconductor package processes, executing real SQL against MSSQL.

**Architecture:** LightRAG provides domain knowledge (analysis flows, table schemas, judgment criteria). An LLM Agent generates SQL step-by-step, Python code validates/executes SQL and runs statistics, then the LLM interprets results. Phase 1 requires user approval before each SQL execution.

**Tech Stack:** Python 3.10, LightRAG (existing on-prem setup), ONNX Runtime embedding, pymssql, scipy, PyYAML. Reuses `lightrag_onprem_demo.py` patterns for LLM/embedding/tokenizer.

---

## File Map

| File | Responsibility |
|------|---------------|
| `src/__init__.py` | Package marker |
| `src/config.py` | Load config.yaml + .env.onprem, expose typed settings |
| `src/llm_client.py` | On-prem LLM wrapper (extracted from lightrag_onprem_demo.py) |
| `src/knowledge.py` | LightRAG initialization + domain knowledge query |
| `src/sql_safety.py` | SQL validation (SELECT-only, whitelist, TOP limit) |
| `src/db_client.py` | MSSQL connection + query execution via pymssql |
| `src/stats.py` | Statistical tests (chi-square, sigma calc) via scipy |
| `src/query_analyzer.py` | Extract process/bin/keywords from user query via LLM |
| `src/plan_generator.py` | Generate step-by-step analysis plan from LightRAG knowledge |
| `src/step_executor.py` | Execute analysis steps: SQL gen -> validate -> run -> interpret |
| `src/report.py` | Format final report from accumulated step results |
| `analyze.py` | CLI entry point |
| `config.yaml` | DB connection, safety rules, execution mode, LightRAG paths |
| `tests/test_sql_safety.py` | SQL safety validation tests |
| `tests/test_stats.py` | Statistics calculation tests |
| `tests/test_config.py` | Config loading tests |
| `docs/domain/analysis_flows/bin_defect_analysis.txt` | Bin defect analysis flow document |
| `docs/domain/table_schemas/package_tables.txt` | Placeholder for user-provided table schemas |
| `docs/domain/judgment_criteria/statistical_thresholds.txt` | Statistical judgment criteria |
| `docs/domain/equipment_knowledge/da_process.txt` | DA process equipment knowledge |

---

### Task 1: Add pymssql and scipy wheels + update requirements

**Files:**
- Modify: `requirements_onprem.txt`
- Modify: `wheels_cp310/` (add 2 wheels)

- [ ] **Step 1: Download pymssql and scipy wheels for cp310**

```bash
cd "E:\light-rag-test\light-rag-test"
.venv/Scripts/pip download --python-version 3.10 --platform win_amd64 --only-binary=:all: --no-deps -d wheels_cp310/ pymssql==2.3.13 scipy==1.15.3
```

Expected: 2 new .whl files in wheels_cp310/

- [ ] **Step 2: Update requirements_onprem.txt**

Add these lines after the `# --- ONNX 임베딩` section:

```
# --- MSSQL + 통계 (분석 시스템용) ---
pymssql==2.3.13
scipy==1.15.3
```

- [ ] **Step 3: Commit**

```bash
git add wheels_cp310/pymssql-*.whl wheels_cp310/scipy-*.whl requirements_onprem.txt
git commit -m "feat: add pymssql and scipy wheels for bin defect analyzer"
```

---

### Task 2: Create config.yaml and src/config.py

**Files:**
- Create: `config.yaml`
- Create: `src/__init__.py`
- Create: `src/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

Create `tests/__init__.py` (empty) and `tests/test_config.py`:

```python
"""tests/test_config.py"""
import os
import tempfile
from pathlib import Path


def test_load_config_from_yaml():
    """config.yaml을 읽어서 설정 객체를 반환한다."""
    from src.config import load_config

    # 임시 yaml 작성
    yaml_content = """
database:
  driver: mssql
  host: 10.0.0.1
  port: 1433
  database: TEST_DB
  username: testuser
  password: testpass

safety:
  allowed_statements: [SELECT, WITH]
  blocked_keywords: [INSERT, DELETE, DROP]
  max_rows: 5000
  table_whitelist: [TABLE_A, TABLE_B]

execution:
  mode: approval
  log_all_sql: true
  log_path: ./logs/test.log

lightrag:
  working_dir: ./rag_storage_analysis
  domain_docs_dir: ./docs/domain
  embed_model_path: ./models/bge-m3-onnx
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as f:
        f.write(yaml_content)
        tmp_path = f.name

    try:
        cfg = load_config(tmp_path)
        assert cfg.database.host == "10.0.0.1"
        assert cfg.database.port == 1433
        assert cfg.database.database == "TEST_DB"
        assert cfg.safety.max_rows == 5000
        assert "TABLE_A" in cfg.safety.table_whitelist
        assert cfg.execution.mode == "approval"
        assert cfg.lightrag.embed_model_path == "./models/bge-m3-onnx"
    finally:
        os.unlink(tmp_path)


def test_config_env_substitution():
    """${VAR} 형태의 환경변수 참조가 치환된다."""
    from src.config import load_config

    os.environ["TEST_DB_HOST"] = "192.168.1.100"
    os.environ["TEST_DB_PASS"] = "secret123"

    yaml_content = """
database:
  driver: mssql
  host: ${TEST_DB_HOST}
  port: 1433
  database: DB
  username: user
  password: ${TEST_DB_PASS}

safety:
  allowed_statements: [SELECT]
  blocked_keywords: [DELETE]
  max_rows: 10000
  table_whitelist: []

execution:
  mode: approval
  log_all_sql: false
  log_path: ./logs/sql.log

lightrag:
  working_dir: ./rag
  domain_docs_dir: ./docs
  embed_model_path: ./models/bge-m3-onnx
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as f:
        f.write(yaml_content)
        tmp_path = f.name

    try:
        cfg = load_config(tmp_path)
        assert cfg.database.host == "192.168.1.100"
        assert cfg.database.password == "secret123"
    finally:
        os.unlink(tmp_path)
        del os.environ["TEST_DB_HOST"]
        del os.environ["TEST_DB_PASS"]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd "E:\light-rag-test\light-rag-test"
.venv/Scripts/python -m pytest tests/test_config.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.config'`

- [ ] **Step 3: Write src/config.py**

Create `src/__init__.py` (empty file) and `src/config.py`:

```python
"""src/config.py — config.yaml 로딩 + 환경변수 치환."""
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class DatabaseConfig:
    driver: str
    host: str
    port: int
    database: str
    username: str
    password: str


@dataclass
class SafetyConfig:
    allowed_statements: list[str]
    blocked_keywords: list[str]
    max_rows: int
    table_whitelist: list[str]


@dataclass
class ExecutionConfig:
    mode: str  # "approval" or "auto"
    log_all_sql: bool
    log_path: str


@dataclass
class LightRAGConfig:
    working_dir: str
    domain_docs_dir: str
    embed_model_path: str


@dataclass
class AppConfig:
    database: DatabaseConfig
    safety: SafetyConfig
    execution: ExecutionConfig
    lightrag: LightRAGConfig


def _substitute_env_vars(value: str) -> str:
    """${VAR_NAME} 패턴을 환경변수 값으로 치환한다."""
    if not isinstance(value, str):
        return value
    return re.sub(
        r"\$\{(\w+)\}",
        lambda m: os.environ.get(m.group(1), m.group(0)),
        value,
    )


def _substitute_recursive(obj):
    """dict/list를 재귀적으로 순회하며 환경변수를 치환한다."""
    if isinstance(obj, str):
        return _substitute_env_vars(obj)
    if isinstance(obj, dict):
        return {k: _substitute_recursive(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_substitute_recursive(v) for v in obj]
    return obj


def load_config(path: str = "config.yaml") -> AppConfig:
    """YAML 파일을 읽어 AppConfig를 반환한다."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    raw = _substitute_recursive(raw)

    return AppConfig(
        database=DatabaseConfig(**raw["database"]),
        safety=SafetyConfig(**raw["safety"]),
        execution=ExecutionConfig(**raw["execution"]),
        lightrag=LightRAGConfig(**raw["lightrag"]),
    )
```

- [ ] **Step 4: Create config.yaml**

```yaml
# config.yaml — Bin 불량 분석 시스템 설정
# 환경변수는 ${VAR_NAME} 형태로 참조 (.env.onprem에서 로드)

database:
  driver: mssql
  host: ${MSSQL_HOST}
  port: 1433
  database: ${MSSQL_DATABASE}
  username: ${MSSQL_USER}
  password: ${MSSQL_PASSWORD}

safety:
  allowed_statements: [SELECT, WITH]
  blocked_keywords: [INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, EXEC, EXECUTE, MERGE, GRANT, REVOKE]
  max_rows: 10000
  # table_whitelist는 사용자가 테이블 스키마 제공 후 채울 것
  table_whitelist: []

execution:
  mode: approval    # approval | auto
  log_all_sql: true
  log_path: ./logs/sql_execution.log

lightrag:
  working_dir: ./rag_storage_analysis
  domain_docs_dir: ./docs/domain
  embed_model_path: ./models/bge-m3-onnx
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
.venv/Scripts/python -m pytest tests/test_config.py -v
```

Expected: 2 passed

- [ ] **Step 6: Commit**

```bash
git add src/__init__.py src/config.py config.yaml tests/__init__.py tests/test_config.py
git commit -m "feat: add config loading with env var substitution"
```

---

### Task 3: Create src/sql_safety.py

**Files:**
- Create: `src/sql_safety.py`
- Create: `tests/test_sql_safety.py`

- [ ] **Step 1: Write the failing tests**

```python
"""tests/test_sql_safety.py"""
import pytest
from src.sql_safety import validate_sql, SQLValidationError


def test_select_allowed():
    sql = "SELECT EQP_ID, COUNT(*) FROM TEST_TABLE GROUP BY EQP_ID"
    result = validate_sql(sql, allowed_statements=["SELECT", "WITH"],
                          blocked_keywords=["INSERT", "DELETE", "DROP"],
                          table_whitelist=["TEST_TABLE"], max_rows=10000)
    assert "TOP" in result  # TOP이 자동 추가됨


def test_cte_allowed():
    sql = "WITH cte AS (SELECT * FROM TEST_TABLE) SELECT * FROM cte"
    result = validate_sql(sql, allowed_statements=["SELECT", "WITH"],
                          blocked_keywords=["DELETE"],
                          table_whitelist=["TEST_TABLE"], max_rows=10000)
    assert result is not None


def test_insert_blocked():
    sql = "INSERT INTO TEST_TABLE VALUES (1, 2)"
    with pytest.raises(SQLValidationError, match="허용되지 않는 SQL"):
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/Scripts/python -m pytest tests/test_sql_safety.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.sql_safety'`

- [ ] **Step 3: Write src/sql_safety.py**

```python
"""src/sql_safety.py — LLM이 생성한 SQL의 안전성을 검증한다."""
import re
import sqlparse


class SQLValidationError(Exception):
    """SQL 검증 실패."""
    pass


def _extract_tables(sql: str) -> list[str]:
    """SQL에서 참조하는 테이블명을 추출한다."""
    tables = []
    parsed = sqlparse.parse(sql)
    for stmt in parsed:
        _walk_tokens_for_tables(stmt.tokens, tables)
    return [t.upper() for t in tables]


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
                # Identifier or IdentifierList
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

    # 1) 허용 문(statement) 체크
    allowed_upper = [s.upper() for s in allowed_statements]
    starts_ok = any(sql_upper.startswith(s) for s in allowed_upper)
    if not starts_ok:
        raise SQLValidationError(
            f"허용되지 않는 SQL 문입니다. 허용: {allowed_statements}\n"
            f"SQL 시작: {sql[:50]}..."
        )

    # 2) 금지 키워드 체크 (문자열 리터럴 제외)
    sql_no_strings = _strip_string_literals(sql_upper)
    for kw in blocked_keywords:
        # 단어 경계로 매치하여 컬럼명 내 부분 매치 방지
        pattern = r"\b" + re.escape(kw.upper()) + r"\b"
        if re.search(pattern, sql_no_strings):
            raise SQLValidationError(
                f"금지 키워드 '{kw}'가 SQL에 포함되어 있습니다.\n"
                f"SQL: {sql[:100]}..."
            )

    # 3) 테이블 화이트리스트 (비어있으면 모든 테이블 허용)
    if table_whitelist:
        wl_upper = [t.upper() for t in table_whitelist]
        referenced = _extract_tables(sql)
        for tbl in referenced:
            # 스키마.테이블 형태 처리 (dbo.TABLE -> TABLE)
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
```

- [ ] **Step 4: Run tests**

```bash
.venv/Scripts/python -m pytest tests/test_sql_safety.py -v
```

Expected: All passed. Note: `sqlparse` is needed — add to requirements (it's a pure-python wheel).

- [ ] **Step 5: Download sqlparse wheel and update requirements**

```bash
.venv/Scripts/pip download --python-version 3.10 --platform win_amd64 --only-binary=:all: --no-deps -d wheels_cp310/ sqlparse
```

Add to `requirements_onprem.txt`:

```
sqlparse==0.5.3
```

- [ ] **Step 6: Commit**

```bash
git add src/sql_safety.py tests/test_sql_safety.py wheels_cp310/sqlparse-*.whl requirements_onprem.txt
git commit -m "feat: add SQL safety validation with whitelist and TOP enforcement"
```

---

### Task 4: Create src/db_client.py

**Files:**
- Create: `src/db_client.py`

- [ ] **Step 1: Write src/db_client.py**

This module can't be unit-tested without a real MSSQL instance, so we write it with clear error messages for on-prem debugging.

```python
"""src/db_client.py — MSSQL 연결 및 쿼리 실행."""
import logging
from contextlib import contextmanager
from dataclasses import dataclass

import pymssql

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    columns: list[str]
    rows: list[tuple]
    row_count: int

    def to_dicts(self) -> list[dict]:
        return [dict(zip(self.columns, row)) for row in self.rows]


class DatabaseClient:
    def __init__(self, host: str, port: int, database: str,
                 username: str, password: str):
        self._host = host
        self._port = port
        self._database = database
        self._username = username
        self._password = password

    @contextmanager
    def _connect(self):
        conn = pymssql.connect(
            server=self._host,
            port=self._port,
            user=self._username,
            password=self._password,
            database=self._database,
            charset="utf8",
            login_timeout=10,
            timeout=120,
        )
        try:
            yield conn
        finally:
            conn.close()

    def execute(self, sql: str) -> QueryResult:
        """SELECT SQL을 실행하고 결과를 반환한다."""
        logger.info(f"Executing SQL: {sql[:200]}...")
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(sql)
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            rows = cursor.fetchall()
            logger.info(f"Result: {len(rows)} rows, {len(columns)} columns")
            return QueryResult(columns=columns, rows=rows, row_count=len(rows))

    def test_connection(self) -> bool:
        """연결 테스트. 성공하면 True."""
        try:
            with self._connect() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                return True
        except Exception as e:
            logger.error(f"DB 연결 실패: {e}")
            return False
```

- [ ] **Step 2: Commit**

```bash
git add src/db_client.py
git commit -m "feat: add MSSQL database client"
```

---

### Task 5: Create src/stats.py

**Files:**
- Create: `src/stats.py`
- Create: `tests/test_stats.py`

- [ ] **Step 1: Write the failing tests**

```python
"""tests/test_stats.py"""
from src.stats import calc_sigma_outliers, chi_square_test


def test_sigma_outliers_detects_anomaly():
    """평균 + 3sigma를 초과하는 값을 특이치로 선별한다."""
    data = {"DA-01": 2.0, "DA-02": 2.1, "DA-03": 12.5, "DA-04": 1.9, "DA-05": 2.2}
    outliers = calc_sigma_outliers(data, sigma_threshold=3.0)
    assert "DA-03" in outliers
    assert "DA-01" not in outliers
    assert outliers["DA-03"]["sigma"] > 3.0


def test_sigma_outliers_no_anomaly():
    """특이치가 없으면 빈 dict를 반환한다."""
    data = {"DA-01": 2.0, "DA-02": 2.1, "DA-03": 2.0}
    outliers = calc_sigma_outliers(data, sigma_threshold=3.0)
    assert len(outliers) == 0


def test_chi_square_test():
    """카이제곱 검정 결과에 p_value와 significant 판정이 포함된다."""
    # 혐의 설비: 불량 50, 양품 50 / 기타 설비: 불량 10, 양품 190
    result = chi_square_test(
        defect_in_suspect=50, total_in_suspect=100,
        defect_in_others=10, total_in_others=200,
        alpha=0.05,
    )
    assert "p_value" in result
    assert "significant" in result
    assert result["significant"] is True  # 매우 유의미한 차이
    assert result["p_value"] < 0.05
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/Scripts/python -m pytest tests/test_stats.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write src/stats.py**

```python
"""src/stats.py — 통계 검정 (숫자 계산은 코드가, 해석은 LLM이)."""
import numpy as np
from scipy import stats


def calc_sigma_outliers(
    data: dict[str, float],
    sigma_threshold: float = 3.0,
) -> dict[str, dict]:
    """
    설비별 수치에서 평균 + N*sigma를 초과하는 특이치를 찾는다.

    Args:
        data: {설비ID: 불량률} 형태
        sigma_threshold: 기준 시그마 (기본 3.0)

    Returns:
        {설비ID: {"value": float, "sigma": float, "mean": float, "std": float}}
    """
    values = np.array(list(data.values()))
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0

    if std == 0:
        return {}

    outliers = {}
    for eqp_id, val in data.items():
        sigma = (val - mean) / std
        if sigma > sigma_threshold:
            outliers[eqp_id] = {
                "value": val,
                "sigma": round(sigma, 2),
                "mean": round(mean, 4),
                "std": round(std, 4),
            }
    return outliers


def chi_square_test(
    defect_in_suspect: int,
    total_in_suspect: int,
    defect_in_others: int,
    total_in_others: int,
    alpha: float = 0.05,
) -> dict:
    """
    카이제곱 검정으로 혐의 설비의 불량률이 통계적으로 유의미하게 높은지 검증.

    Returns:
        {"chi2": float, "p_value": float, "significant": bool}
    """
    good_in_suspect = total_in_suspect - defect_in_suspect
    good_in_others = total_in_others - defect_in_others

    # 2x2 분할표: [[불량_혐의, 양품_혐의], [불량_기타, 양품_기타]]
    table = np.array([
        [defect_in_suspect, good_in_suspect],
        [defect_in_others, good_in_others],
    ])

    chi2, p_value, dof, expected = stats.chi2_contingency(table)

    return {
        "chi2": round(float(chi2), 4),
        "p_value": round(float(p_value), 6),
        "significant": p_value < alpha,
        "alpha": alpha,
    }
```

- [ ] **Step 4: Run tests**

```bash
.venv/Scripts/python -m pytest tests/test_stats.py -v
```

Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/stats.py tests/test_stats.py
git commit -m "feat: add sigma outlier detection and chi-square test"
```

---

### Task 6: Create src/llm_client.py (extracted from lightrag_onprem_demo.py)

**Files:**
- Create: `src/llm_client.py`

- [ ] **Step 1: Write src/llm_client.py**

Extract the LLM calling logic from `lightrag_onprem_demo.py` into a reusable module. This avoids duplicating the custom header auth code.

```python
"""src/llm_client.py — 사내 LLM API 래퍼 (lightrag_onprem_demo.py에서 추출)."""
import os
import uuid
import logging

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

# .env.onprem에서 로드됨
LLM_BASE_URL = os.getenv(
    "ONPREM_LLM_BASE_URL",
    "http://apigw-stg.shyum.net:8000/gpt-oss/1/gpt-oss-120b/v1",
)
LLM_MODEL = os.getenv("ONPREM_LLM_MODEL", "openai/gpt-oss-120b")
LLM_CREDENTIAL_KEY = os.getenv("ONPREM_LLM_CREDENTIAL_KEY", "")
LLM_SEND_SYSTEM_NAME = os.getenv("ONPREM_SEND_SYSTEM_NAME", "")
LLM_USER_ID = os.getenv("ONPREM_USER_ID", "")


def _build_headers() -> dict:
    """호출마다 고유한 Prompt-Msg-Id / Completion-Msg-Id를 생성."""
    return {
        "x-dep-ticket": LLM_CREDENTIAL_KEY,
        "Send-System-Name": LLM_SEND_SYSTEM_NAME,
        "User-Id": LLM_USER_ID,
        "User-Type": "AD_ID",
        "Prompt-Msg-Id": str(uuid.uuid4()),
        "Completion-Msg-Id": str(uuid.uuid4()),
    }


async def llm_complete(prompt: str, system_prompt: str = None, timeout: int = 600) -> str:
    """
    사내 LLM API에 프롬프트를 보내고 응답 텍스트를 반환한다.
    lightrag_onprem_demo.py의 onprem_llm_complete를 단순화한 버전.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    client = AsyncOpenAI(
        base_url=LLM_BASE_URL,
        api_key="unused",
        default_headers=_build_headers(),
    )
    completion = await client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        timeout=timeout,
    )
    return completion.choices[0].message.content or ""
```

- [ ] **Step 2: Commit**

```bash
git add src/llm_client.py
git commit -m "feat: extract LLM client from onprem demo"
```

---

### Task 7: Create src/knowledge.py (LightRAG wrapper)

**Files:**
- Create: `src/knowledge.py`

- [ ] **Step 1: Write src/knowledge.py**

Wraps LightRAG initialization (reusing patterns from `lightrag_onprem_demo.py`) and provides a simple query interface for the agent.

```python
"""src/knowledge.py — LightRAG 래핑: 도메인 지식 인덱싱 및 검색."""
import os
import logging
import importlib
import numpy as np
from pathlib import Path

from lightrag import LightRAG, QueryParam
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import EmbeddingFunc, Tokenizer, setup_logger

logger = logging.getLogger(__name__)


class OfflineCharTokenizer:
    """폐쇄망용 문자 기반 토크나이저."""
    def encode(self, content: str) -> list[int]:
        return [ord(ch) for ch in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(t) for t in tokens)


# ONNX 임베딩 싱글턴
_ort_session = None
_ort_tokenizer = None


def _get_ort_model(model_path: str):
    global _ort_session, _ort_tokenizer
    if _ort_session is None:
        import onnxruntime as ort
        from transformers import AutoTokenizer

        onnx_file = Path(model_path) / "model.onnx"
        if not onnx_file.exists():
            raise RuntimeError(f"ONNX 모델 없음: {onnx_file}")

        logger.info(f"ONNX 모델 로딩: {model_path}")
        _ort_session = ort.InferenceSession(str(onnx_file))
        _ort_tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), local_files_only=True
        )
    return _ort_session, _ort_tokenizer


async def _onnx_embed(texts: list[str], model_path: str) -> np.ndarray:
    session, tokenizer = _get_ort_model(model_path)
    input_names = [i.name for i in session.get_inputs()]
    all_embeddings = []
    for i in range(0, len(texts), 32):
        batch = texts[i:i + 32]
        inputs = tokenizer(batch, return_tensors="np", padding=True,
                           truncation=True, max_length=8192)
        feeds = {k: v for k, v in inputs.items() if k in input_names}
        outputs = session.run(None, feeds)
        all_embeddings.append(outputs[0][:, 0, :])
    embeddings = np.vstack(all_embeddings).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return embeddings / norms


class KnowledgeBase:
    """LightRAG 기반 도메인 지식 검색."""

    def __init__(self, working_dir: str, embed_model_path: str,
                 llm_model_func, llm_model_name: str):
        self._working_dir = working_dir
        self._embed_model_path = embed_model_path
        self._llm_model_func = llm_model_func
        self._llm_model_name = llm_model_name
        self._rag = None

    async def initialize(self):
        """LightRAG 인스턴스를 초기화한다."""
        Path(self._working_dir).mkdir(parents=True, exist_ok=True)

        embed_path = self._embed_model_path

        self._rag = LightRAG(
            working_dir=self._working_dir,
            tokenizer=Tokenizer(
                model_name="offline-char-tokenizer",
                tokenizer=OfflineCharTokenizer(),
            ),
            tiktoken_model_name="text-embedding-3-small",
            chunk_token_size=800,
            chunk_overlap_token_size=80,
            llm_model_func=self._llm_model_func,
            llm_model_name=self._llm_model_name,
            llm_model_kwargs={},
            embedding_func=EmbeddingFunc(
                embedding_dim=1024,
                max_token_size=8192,
                func=lambda texts: _onnx_embed(texts, embed_path),
            ),
        )
        await self._rag.initialize_storages()
        await initialize_pipeline_status()

    async def index_documents(self, docs_dir: str):
        """docs_dir 내의 모든 .txt 파일을 인덱싱한다."""
        docs_path = Path(docs_dir)
        txt_files = sorted(docs_path.rglob("*.txt"))
        if not txt_files:
            logger.warning(f"인덱싱할 .txt 파일 없음: {docs_dir}")
            return

        for f in txt_files:
            logger.info(f"인덱싱: {f}")
            text = f.read_text(encoding="utf-8")
            await self._rag.ainsert(text)

        logger.info(f"{len(txt_files)}개 문서 인덱싱 완료")

    async def query(self, question: str, mode: str = "hybrid") -> str:
        """도메인 지식을 검색하여 답변을 반환한다."""
        return await self._rag.aquery(question, param=QueryParam(mode=mode))

    async def finalize(self):
        if self._rag:
            await self._rag.finalize_storages()
```

- [ ] **Step 2: Commit**

```bash
git add src/knowledge.py
git commit -m "feat: add LightRAG knowledge base wrapper"
```

---

### Task 8: Create src/query_analyzer.py and src/plan_generator.py

**Files:**
- Create: `src/query_analyzer.py`
- Create: `src/plan_generator.py`

- [ ] **Step 1: Write src/query_analyzer.py**

```python
"""src/query_analyzer.py — 사용자 쿼리에서 분석 파라미터를 추출한다."""
import json
import logging

from src.llm_client import llm_complete

logger = logging.getLogger(__name__)

EXTRACT_PROMPT = """사용자의 반도체 수율 분석 질문에서 아래 파라미터를 추출하세요.
JSON 형태로만 응답하세요. 값이 없으면 null로 채우세요.

{{
  "process": "공정명 (예: Die Attach, Wire Bond, Molding, ...)",
  "bin_code": "Bin 코드 번호 (예: 3)",
  "equipment_id": "특정 설비 ID (언급된 경우)",
  "issue": "문제 유형 (예: 급증, 저하, 특이점)",
  "time_range": "분석 기간 (예: 최근 1개월)",
  "keywords": ["핵심 키워드 리스트"]
}}

사용자 질문: {query}"""


async def analyze_query(query: str) -> dict:
    """
    사용자 자연어 쿼리에서 분석 파라미터를 추출한다.

    Returns:
        {"process": str|None, "bin_code": int|None, "equipment_id": str|None,
         "issue": str|None, "time_range": str|None, "keywords": list[str]}
    """
    response = await llm_complete(
        prompt=EXTRACT_PROMPT.format(query=query),
        system_prompt="당신은 반도체 수율 분석 전문가입니다. JSON만 출력하세요.",
    )

    # LLM 응답에서 JSON 추출
    try:
        # ```json ... ``` 블록이 있으면 그 안의 내용만 추출
        if "```" in response:
            json_str = response.split("```")[1]
            if json_str.startswith("json"):
                json_str = json_str[4:]
            json_str = json_str.strip()
        else:
            json_str = response.strip()

        params = json.loads(json_str)
    except (json.JSONDecodeError, IndexError):
        logger.warning(f"LLM 응답에서 JSON 파싱 실패, 원문: {response[:200]}")
        params = {
            "process": None, "bin_code": None, "equipment_id": None,
            "issue": None, "time_range": None, "keywords": [],
        }

    # bin_code를 int로 변환
    if params.get("bin_code") is not None:
        try:
            params["bin_code"] = int(params["bin_code"])
        except (ValueError, TypeError):
            params["bin_code"] = None

    return params
```

- [ ] **Step 2: Write src/plan_generator.py**

```python
"""src/plan_generator.py — LightRAG 지식 기반 분석 계획 수립."""
import json
import logging

from src.llm_client import llm_complete

logger = logging.getLogger(__name__)

PLAN_PROMPT = """당신은 반도체 수율 분석 시스템입니다.
아래 도메인 지식과 사용자 요청을 바탕으로 분석 계획을 JSON으로 수립하세요.

## 도메인 지식 (LightRAG 검색 결과)
{knowledge}

## 사용자 요청 파라미터
{params}

## 출력 형식 (JSON만 출력)
{{
  "plan_name": "분석 계획 이름",
  "steps": [
    {{
      "step_number": 1,
      "name": "단계 이름",
      "purpose": "이 단계의 목적",
      "action": "sql_query | statistics | interpret",
      "description": "구체적으로 무엇을 해야 하는지",
      "tables": ["사용할 테이블명"],
      "output": "이 단계의 출력물"
    }}
  ]
}}"""


async def generate_plan(params: dict, knowledge: str) -> dict:
    """
    분석 파라미터와 도메인 지식으로 단계별 실행 계획을 생성한다.

    Returns:
        {"plan_name": str, "steps": [{"step_number": int, "name": str, ...}]}
    """
    response = await llm_complete(
        prompt=PLAN_PROMPT.format(
            knowledge=knowledge,
            params=json.dumps(params, ensure_ascii=False, indent=2),
        ),
        system_prompt="반도체 수율 분석 전문가. JSON만 출력.",
    )

    try:
        if "```" in response:
            json_str = response.split("```")[1]
            if json_str.startswith("json"):
                json_str = json_str[4:]
            json_str = json_str.strip()
        else:
            json_str = response.strip()
        plan = json.loads(json_str)
    except (json.JSONDecodeError, IndexError):
        logger.error(f"분석 계획 JSON 파싱 실패: {response[:300]}")
        raise RuntimeError("LLM이 유효한 분석 계획을 생성하지 못했습니다.")

    return plan
```

- [ ] **Step 3: Commit**

```bash
git add src/query_analyzer.py src/plan_generator.py
git commit -m "feat: add query analyzer and plan generator"
```

---

### Task 9: Create src/step_executor.py

**Files:**
- Create: `src/step_executor.py`

- [ ] **Step 1: Write src/step_executor.py**

This is the core loop — the largest module.

```python
"""src/step_executor.py — 분석 단계별 실행: SQL 생성 → 검증 → 실행 → 해석."""
import json
import logging
from datetime import datetime
from pathlib import Path

from src.llm_client import llm_complete
from src.sql_safety import validate_sql, SQLValidationError
from src.db_client import DatabaseClient, QueryResult
from src.config import SafetyConfig, ExecutionConfig

logger = logging.getLogger(__name__)

SQL_GEN_PROMPT = """당신은 MSSQL SQL 작성 전문가입니다.
아래 분석 단계의 요구사항에 맞는 SELECT SQL을 작성하세요.

## 분석 단계
{step_description}

## 테이블 스키마 정보
{schema_knowledge}

## 이전 단계 결과 (참고)
{previous_results}

## 규칙
- MSSQL 문법 사용 (TOP, DATEADD 등)
- SELECT 문만 사용 가능
- SQL만 출력하세요 (설명 없이)"""

INTERPRET_PROMPT = """당신은 반도체 수율 분석 전문가입니다.
아래 SQL 실행 결과를 분석하고 해석하세요.

## 분석 단계 목적
{purpose}

## SQL 실행 결과 (상위 {row_count}건)
{result_text}

## 해석 요청
- 특이점이 있는지 판단
- 다음 단계에서 집중해야 할 대상(설비, 시점 등) 명시
- 간결하게 한국어로 답변"""


class StepExecutor:
    """분석 단계별 실행 엔진."""

    def __init__(self, db: DatabaseClient, safety: SafetyConfig,
                 execution: ExecutionConfig):
        self._db = db
        self._safety = safety
        self._execution = execution
        self._log_path = Path(execution.log_path)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._step_results = []  # 각 단계 결과 누적

    async def execute_step(self, step: dict, schema_knowledge: str) -> dict:
        """
        분석 단계 1개를 실행한다.

        Args:
            step: {"step_number", "name", "purpose", "action", "description", "tables", "output"}
            schema_knowledge: 테이블 스키마 관련 LightRAG 검색 결과

        Returns:
            {"step_number": int, "name": str, "sql": str, "result": QueryResult or None,
             "interpretation": str, "status": "success"|"skipped"|"error"}
        """
        step_num = step["step_number"]
        print(f"\n{'='*60}")
        print(f"  Step {step_num}: {step['name']}")
        print(f"  목적: {step['purpose']}")
        print(f"{'='*60}")

        previous_summary = self._summarize_previous_results()

        if step.get("action") == "interpret":
            # SQL 없이 이전 결과만 해석
            interpretation = await self._interpret_results(
                step["purpose"], previous_summary
            )
            result_entry = {
                "step_number": step_num,
                "name": step["name"],
                "sql": None,
                "result": None,
                "interpretation": interpretation,
                "status": "success",
            }
            self._step_results.append(result_entry)
            print(f"\n[해석]\n{interpretation}")
            return result_entry

        # 1) SQL 생성
        sql = await self._generate_sql(step, schema_knowledge, previous_summary)
        print(f"\n[생성된 SQL]\n{sql}")

        # 2) SQL 검증
        try:
            validated_sql = validate_sql(
                sql,
                allowed_statements=self._safety.allowed_statements,
                blocked_keywords=self._safety.blocked_keywords,
                table_whitelist=self._safety.table_whitelist,
                max_rows=self._safety.max_rows,
            )
        except SQLValidationError as e:
            print(f"\n[SQL 검증 실패] {e}")
            result_entry = {
                "step_number": step_num, "name": step["name"],
                "sql": sql, "result": None,
                "interpretation": f"SQL 검증 실패: {e}",
                "status": "error",
            }
            self._step_results.append(result_entry)
            return result_entry

        # 3) 사용자 승인 (approval 모드)
        if self._execution.mode == "approval":
            print(f"\n[검증 완료 SQL]\n{validated_sql}")
            choice = input("\n실행? (y)실행 (e)수정 (s)스킵 (q)중단: ").strip().lower()
            if choice == "s":
                result_entry = {
                    "step_number": step_num, "name": step["name"],
                    "sql": validated_sql, "result": None,
                    "interpretation": "사용자가 스킵함",
                    "status": "skipped",
                }
                self._step_results.append(result_entry)
                return result_entry
            elif choice == "q":
                raise KeyboardInterrupt("사용자가 분석을 중단했습니다.")
            elif choice == "e":
                validated_sql = input("수정된 SQL 입력: ").strip()
            elif choice != "y":
                print("'y'로 간주합니다.")

        # 4) SQL 실행
        self._log_sql(validated_sql, step_num)
        try:
            query_result = self._db.execute(validated_sql)
        except Exception as e:
            print(f"\n[SQL 실행 오류] {e}")
            result_entry = {
                "step_number": step_num, "name": step["name"],
                "sql": validated_sql, "result": None,
                "interpretation": f"SQL 실행 오류: {e}",
                "status": "error",
            }
            self._step_results.append(result_entry)
            return result_entry

        # 5) 결과 표시
        self._print_result(query_result)

        # 6) 결과 해석 (LLM)
        result_text = self._format_result_for_llm(query_result)
        interpretation = await self._interpret_results(step["purpose"], result_text)
        print(f"\n[해석]\n{interpretation}")

        result_entry = {
            "step_number": step_num, "name": step["name"],
            "sql": validated_sql, "result": query_result,
            "interpretation": interpretation,
            "status": "success",
        }
        self._step_results.append(result_entry)
        return result_entry

    def get_all_results(self) -> list[dict]:
        return self._step_results

    async def _generate_sql(self, step: dict, schema_knowledge: str,
                            previous_results: str) -> str:
        response = await llm_complete(
            prompt=SQL_GEN_PROMPT.format(
                step_description=json.dumps(step, ensure_ascii=False, indent=2),
                schema_knowledge=schema_knowledge,
                previous_results=previous_results or "없음 (첫 단계)",
            ),
            system_prompt="MSSQL SELECT SQL만 출력. 설명 없이 SQL만.",
        )
        # ```sql ... ``` 블록 추출
        sql = response.strip()
        if "```" in sql:
            parts = sql.split("```")
            for part in parts:
                cleaned = part.strip()
                if cleaned.startswith("sql"):
                    cleaned = cleaned[3:].strip()
                if cleaned.upper().startswith("SELECT") or cleaned.upper().startswith("WITH"):
                    sql = cleaned
                    break
        return sql.strip()

    async def _interpret_results(self, purpose: str, result_text: str) -> str:
        return await llm_complete(
            prompt=INTERPRET_PROMPT.format(
                purpose=purpose,
                row_count="전체",
                result_text=result_text,
            ),
            system_prompt="반도체 수율 분석 전문가. 간결하게 한국어로 답변.",
        )

    def _summarize_previous_results(self) -> str:
        if not self._step_results:
            return ""
        lines = []
        for r in self._step_results:
            lines.append(f"Step {r['step_number']} ({r['name']}): {r['interpretation'][:200]}")
        return "\n".join(lines)

    def _format_result_for_llm(self, result: QueryResult, max_rows: int = 50) -> str:
        if not result.rows:
            return "(결과 없음)"
        header = " | ".join(result.columns)
        lines = [header, "-" * len(header)]
        for row in result.rows[:max_rows]:
            lines.append(" | ".join(str(v) for v in row))
        if result.row_count > max_rows:
            lines.append(f"... (총 {result.row_count}건 중 {max_rows}건 표시)")
        return "\n".join(lines)

    def _print_result(self, result: QueryResult):
        if not result.rows:
            print("\n(결과 없음)")
            return
        col_widths = [max(len(str(c)), max(len(str(row[i])) for row in result.rows[:20]))
                      for i, c in enumerate(result.columns)]
        header = "  ".join(str(c).ljust(w) for c, w in zip(result.columns, col_widths))
        print(f"\n[실행 결과] ({result.row_count}건)")
        print(f"  {header}")
        print(f"  {'  '.join('-' * w for w in col_widths)}")
        for row in result.rows[:20]:
            print(f"  {'  '.join(str(v).ljust(w) for v, w in zip(row, col_widths))}")
        if result.row_count > 20:
            print(f"  ... ({result.row_count - 20}건 더)")

    def _log_sql(self, sql: str, step_num: int):
        if self._execution.log_all_sql:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with self._log_path.open("a", encoding="utf-8") as f:
                f.write(f"\n[{ts}] Step {step_num}\n{sql}\n")
```

- [ ] **Step 2: Commit**

```bash
git add src/step_executor.py
git commit -m "feat: add step executor with SQL gen/validate/run/interpret loop"
```

---

### Task 10: Create src/report.py

**Files:**
- Create: `src/report.py`

- [ ] **Step 1: Write src/report.py**

```python
"""src/report.py — 분석 결과 최종 리포트 생성."""
import logging

from src.llm_client import llm_complete

logger = logging.getLogger(__name__)

REPORT_PROMPT = """당신은 반도체 수율 분석 전문가입니다.
아래 분석 단계별 결과를 종합하여 최종 리포트를 작성하세요.

## 분석 단계별 결과
{step_summaries}

## 리포트 형식
1. 혐의 설비 (설비ID + 챔버ID)
2. 불량률 수치 (전체 평균 대비 N sigma)
3. 통계 유의성 (p-value)
4. 불량 패턴 (Map 분석 기반)
5. 예상 원인
6. 권장 조치

한국어로 간결하게 작성하세요."""


async def generate_report(step_results: list[dict]) -> str:
    """
    누적된 분석 결과로 최종 리포트를 생성한다.
    """
    summaries = []
    for r in step_results:
        entry = f"### Step {r['step_number']}: {r['name']}\n"
        entry += f"상태: {r['status']}\n"
        if r.get("sql"):
            entry += f"SQL: {r['sql'][:200]}...\n"
        entry += f"해석: {r['interpretation']}\n"
        summaries.append(entry)

    response = await llm_complete(
        prompt=REPORT_PROMPT.format(step_summaries="\n".join(summaries)),
        system_prompt="반도체 수율 분석 전문가. 최종 리포트 작성.",
    )
    return response
```

- [ ] **Step 2: Commit**

```bash
git add src/report.py
git commit -m "feat: add final report generator"
```

---

### Task 11: Create analyze.py (CLI entry point)

**Files:**
- Create: `analyze.py`

- [ ] **Step 1: Write analyze.py**

```python
"""analyze.py — Bin 불량 분석 시스템 CLI 진입점."""
import asyncio
import sys
import logging
from pathlib import Path

# .env.onprem 로드
try:
    from dotenv import load_dotenv
    load_dotenv(".env.onprem")
except ImportError:
    pass

from src.config import load_config
from src.llm_client import llm_complete
from src.knowledge import KnowledgeBase
from src.query_analyzer import analyze_query
from src.plan_generator import generate_plan
from src.step_executor import StepExecutor
from src.db_client import DatabaseClient
from src.report import generate_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


async def main():
    # 1) 설정 로드
    cfg = load_config("config.yaml")
    print("[1/6] 설정 로드 완료")

    # 2) DB 연결 확인
    db = DatabaseClient(
        host=cfg.database.host,
        port=cfg.database.port,
        database=cfg.database.database,
        username=cfg.database.username,
        password=cfg.database.password,
    )
    if not db.test_connection():
        print("DB 연결 실패. config.yaml과 .env.onprem을 확인하세요.")
        sys.exit(1)
    print("[2/6] DB 연결 확인 완료")

    # 3) LightRAG 지식 베이스 초기화
    # lightrag_onprem_demo.py의 onprem_llm_complete 패턴을 재사용
    from lightrag_onprem_demo import onprem_llm_complete, LLM_MODEL

    kb = KnowledgeBase(
        working_dir=cfg.lightrag.working_dir,
        embed_model_path=cfg.lightrag.embed_model_path,
        llm_model_func=onprem_llm_complete,
        llm_model_name=LLM_MODEL,
    )

    rag_index = Path(cfg.lightrag.working_dir) / "vdb_chunks.json"
    if not rag_index.exists():
        print("[3/6] 도메인 지식 인덱싱 시작...")
        await kb.initialize()
        await kb.index_documents(cfg.lightrag.domain_docs_dir)
        print("[3/6] 인덱싱 완료")
    else:
        print("[3/6] 기존 인덱스 사용")
        await kb.initialize()

    # 4) 사용자 입력 루프
    print("\n" + "=" * 60)
    print("  Bin 불량 분석 시스템 (Phase 1: 승인 모드)")
    print("  종료: Ctrl+C 또는 'quit' 입력")
    print("=" * 60)

    while True:
        try:
            query = input("\n질문: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query or query.lower() in ("quit", "exit", "q"):
            break

        try:
            # 4a) 쿼리 분석
            print("\n[4/6] 쿼리 분석 중...")
            params = await analyze_query(query)
            print(f"  추출 파라미터: {params}")

            # 4b) 도메인 지식 검색
            print("[5/6] 도메인 지식 검색 중...")
            knowledge_query = f"{params.get('process', '')} {params.get('issue', '')} Bin{params.get('bin_code', '')} 분석 플로우 테이블 스키마"
            knowledge = await kb.query(knowledge_query.strip(), mode="hybrid")

            schema_query = f"테이블 스키마 조인 관계 {' '.join(params.get('keywords', []))}"
            schema_knowledge = await kb.query(schema_query.strip(), mode="local")

            # 4c) 분석 계획 수립
            print("[6/6] 분석 계획 수립 중...")
            plan = await generate_plan(params, knowledge)
            print(f"\n[분석 계획] {plan.get('plan_name', '분석')}")
            for step in plan.get("steps", []):
                print(f"  Step {step['step_number']}: {step['name']}")

            proceed = input("\n진행하시겠습니까? (y/n): ").strip().lower()
            if proceed != "y":
                continue

            # 4d) 단계별 실행
            executor = StepExecutor(db=db, safety=cfg.safety, execution=cfg.execution)
            for step in plan.get("steps", []):
                await executor.execute_step(step, schema_knowledge)

            # 4e) 최종 리포트
            print("\n" + "=" * 60)
            print("  최종 분석 리포트")
            print("=" * 60)
            report = await generate_report(executor.get_all_results())
            print(report)

        except KeyboardInterrupt:
            print("\n분석 중단")
        except Exception as e:
            logger.error(f"분석 오류: {e}", exc_info=True)
            print(f"\n오류 발생: {e}")

    await kb.finalize()
    print("\n종료.")


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 2: Commit**

```bash
git add analyze.py
git commit -m "feat: add CLI entry point for bin defect analyzer"
```

---

### Task 12: Create domain knowledge template documents

**Files:**
- Create: `docs/domain/analysis_flows/bin_defect_analysis.txt`
- Create: `docs/domain/table_schemas/package_tables.txt`
- Create: `docs/domain/judgment_criteria/statistical_thresholds.txt`
- Create: `docs/domain/equipment_knowledge/da_process.txt`

- [ ] **Step 1: Write bin_defect_analysis.txt**

```text
# Bin 불량 분석 모델

## 적용 조건
특정 Bin의 불량률이 급증했을 때 원인 설비를 추적하는 분석 모델이다.
반도체 패키지 공정(Die Attach, Wire Bond, Molding, Marking, Plating, Singulation 등) 전체에 적용 가능하다.

## 분석 플로우

### Step 1: 설비별 Bin 불량률 개요 (Overview)
- 목적: 전체 설비 중 특이 설비를 선별한다
- 조회 대상: 대상 공정의 모든 설비와 챔버
- 집계 방식: 설비+챔버별 GROUP BY, 불량률 = 해당 Bin 불량수 / 전체 검사수 * 100
- 특이 판단 기준: 불량률이 전체 평균 + 3시그마를 초과하면 특이 설비로 선별
- 기간: 사용자가 지정하지 않으면 최근 1개월

### Step 2: 특이 설비 시계열 Bin 불량률 확인 (Time Series)
- 목적: 특이 설비의 불량이 갑작스런 변화인지, 점진적 악화인지 판단
- 입력: Step 1에서 선별된 특이 설비 ID 목록
- 집계 방식: 일별 또는 주별 Bin 불량률 추이
- 판단: 특정 시점에서 급증했으면 해당 시점 전후 이벤트(PM, 레시피 변경, 자재 변경) 확인 대상

### Step 3: Wafer Map / Frame Map 패턴 확인 (Map Pattern)
- 목적: 불량의 공간적 분포 패턴을 파악하여 원인 범위를 축소
- 입력: 혐의 설비 + 시점 범위
- 조회 대상: 해당 설비에서 처리된 유닛의 Map 좌표별 Pass/Fail 데이터
- 패턴 유형과 의미:
  - Edge 불량: 가장자리 집중 → 설비 외곽부 접촉 불량, 온도 불균일
  - Center 불량: 중심부 집중 → 척(Chuck) 문제, 가스 흐름 불균일
  - Cluster 불량: 특정 영역 집중 → 이물질, 노즐 막힘
  - Random 불량: 무작위 분포 → 파티클 오염
  - Line/Scratch 패턴: 직선형 → 이송 과정 스크래치

### Step 4: 정밀 Bin 불량률 재계산 (Validation)
- 목적: 별도 테이블(더 정확한 데이터 소스)로 교차 검증하여 혐의를 확정
- 통계 검정: 카이제곱 검정으로 혐의 설비와 나머지 설비 간 불량률 차이 유의성 검증
- 유의 기준: p-value < 0.05이면 통계적으로 유의미한 차이
- 출력: 확정된 혐의 설비, 정밀 불량률, p-value

## 최종 리포트 포맷
1. 혐의 설비: 설비ID + 챔버ID (또는 위치)
2. 불량률: X% (전체 평균 Y% 대비 N시그마 이탈)
3. 통계 유의성: 카이제곱 검정 p-value
4. 불량 패턴: Map 분석 기반 패턴 유형
5. 예상 원인: 패턴 유형과 설비 특성 기반 추정
6. 권장 조치: PM, 부품 교체, 레시피 조정 등
```

- [ ] **Step 2: Write package_tables.txt (placeholder)**

```text
# 패키지 공정 테이블 스키마

## 안내
이 파일에 실제 사내 테이블 스키마를 작성해주세요.
아래는 작성 예시 형식입니다. 실제 테이블명과 컬럼으로 교체하세요.

## 예시 형식

### 테이블: EXAMPLE_BIN_RESULT
용도: 최종 테스트 Bin 결과 테이블

| 컬럼명 | 타입 | 설명 |
|--------|------|------|
| LOT_ID | VARCHAR | Lot 식별자 |
| EQP_ID | VARCHAR | 설비 식별자 |
| CHAMBER_ID | VARCHAR | 챔버/위치 식별자 |
| BIN_CODE | INT | Bin 코드 (1=양품, 2~=불량유형) |
| TEST_DATE | DATETIME | 테스트 일시 |
| PROCESS_ID | VARCHAR | 공정 코드 |

### 테이블 간 조인 관계
- [테이블A] ↔ [테이블B]: LOT_ID로 조인
- [테이블B] ↔ [테이블C]: LOT_ID + WAFER_ID로 조인

## 주의사항
- config.yaml의 safety.table_whitelist에 여기 적힌 테이블명을 등록해야 합니다
- 스키마가 정확할수록 LLM의 SQL 생성 품질이 높아집니다
```

- [ ] **Step 3: Write statistical_thresholds.txt**

```text
# 통계 판단 기준

## 시그마 기준 (특이 설비 선별)
- 3시그마 초과: 특이 설비로 선별 (기본값)
- 2시그마 초과: 주의 대상
- 6시그마 초과: 심각한 이상

## p-value 기준 (통계 유의성)
- p-value < 0.05: 통계적으로 유의미한 차이 (혐의 확정)
- p-value < 0.01: 매우 유의미한 차이
- p-value >= 0.05: 유의미하지 않음 (혐의 기각)

## 불량률 임계값
- 공정별로 다르지만, 일반적인 기준:
  - 정상 범위: 공정 평균 불량률의 1.5배 이내
  - 주의: 공정 평균의 1.5~3배
  - 이상: 공정 평균의 3배 초과

## 카이제곱 검정 적용 조건
- 기대 빈도가 5 미만인 셀이 있으면 Fisher's exact test 사용 권장
- 2x2 분할표: [[혐의설비_불량, 혐의설비_양품], [기타설비_불량, 기타설비_양품]]
```

- [ ] **Step 4: Write da_process.txt**

```text
# Die Attach 공정 설비 지식

## 공정 개요
Die Attach(DA)는 반도체 칩(다이)을 리드프레임이나 기판에 접착하는 공정이다.
접착 방식에 따라 에폭시 접착, 솔더 접착, 은 소결 등이 있다.

## 주요 설비 구성
- Die Attach 설비: 다이를 픽업하여 기판에 배치하고 접착
- 각 설비는 여러 헤드(Head) 또는 본딩 유닛을 가질 수 있음
- 설비 내 위치(유닛, 레인)별로 불량 경향이 다를 수 있음

## 주요 불량 모드
- Die Tilt: 다이 기울어짐 → 본딩 위치 정밀도 문제
- Void: 접착층 내 기포 → 에폭시 디스펜싱 문제
- Die Crack: 다이 깨짐 → 픽업 힘 과다, 이송 충격
- Missing Die: 다이 누락 → 픽업 실패
- Contamination: 이물질 → 클리닝 불량

## Map 패턴과 원인 추정
- Edge 불량: 설비 외곽 유닛의 접착 조건 불균일, 온도 프로필 차이
- 특정 유닛 집중: 해당 유닛의 노즐/헤드 마모, 캘리브레이션 이탈
- Random: 재료(에폭시, 솔더) 품질 문제, 환경(온습도) 변동
```

- [ ] **Step 5: Commit**

```bash
git add docs/domain/
git commit -m "feat: add domain knowledge templates for bin defect analysis"
```

---

### Task 13: Update .gitignore and .env.onprem.example

**Files:**
- Modify: `.gitignore`
- Modify: `.env.onprem.example`

- [ ] **Step 1: Add to .gitignore**

Append these lines:

```
# 분석 시스템
rag_storage_analysis/
logs/
```

- [ ] **Step 2: Add MSSQL env vars to .env.onprem.example**

Append after the Reranker section:

```
# --- MSSQL (분석 시스템용) ---
MSSQL_HOST=10.x.x.x
MSSQL_DATABASE=PACKAGE_DB
MSSQL_USER=여기에_DB_사용자명
MSSQL_PASSWORD=여기에_DB_비밀번호
```

- [ ] **Step 3: Commit**

```bash
git add .gitignore .env.onprem.example
git commit -m "feat: add MSSQL config and analysis gitignore entries"
```

---

### Task 14: Integration smoke test (CLI dry run)

- [ ] **Step 1: Run config load test**

```bash
cd "E:\light-rag-test\light-rag-test"
.venv/Scripts/python -c "
from src.config import load_config
cfg = load_config('config.yaml')
print(f'DB: {cfg.database.driver}')
print(f'Safety mode: {cfg.execution.mode}')
print(f'LightRAG dir: {cfg.lightrag.working_dir}')
print('Config OK')
"
```

Expected: Config values printed, no errors.

- [ ] **Step 2: Run all unit tests**

```bash
.venv/Scripts/python -m pytest tests/ -v
```

Expected: All tests pass (test_config, test_sql_safety, test_stats).

- [ ] **Step 3: Final commit with all tests green**

```bash
git add -A
git commit -m "feat: bin defect analyzer v0.1 - complete POC structure"
git push origin main
```
