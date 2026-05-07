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

    try:
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

    if params.get("bin_code") is not None:
        try:
            params["bin_code"] = int(params["bin_code"])
        except (ValueError, TypeError):
            params["bin_code"] = None

    return params
