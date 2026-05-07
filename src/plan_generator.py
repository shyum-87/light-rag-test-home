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
