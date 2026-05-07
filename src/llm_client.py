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
