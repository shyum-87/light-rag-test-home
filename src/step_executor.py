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
        self._step_results = []

    async def execute_step(self, step: dict, schema_knowledge: str) -> dict:
        """
        분석 단계 1개를 실행한다.

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
