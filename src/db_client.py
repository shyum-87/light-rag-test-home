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
