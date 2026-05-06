"""tests/test_config.py"""
import os
import tempfile
from pathlib import Path


def test_load_config_from_yaml():
    """config.yaml을 읽어서 설정 객체를 반환한다."""
    from src.config import load_config

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
