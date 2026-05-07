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
    """누적된 분석 결과로 최종 리포트를 생성한다."""
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
