"""
LightRAG 사내 폐쇄망 데모 스크립트
====================================
사내 커스텀 LLM API (OpenAI-compatible + 커스텀 헤더 인증) 및
사내 Reranker API를 사용하는 버전입니다.

실행 전 준비:
1. .env.onprem 파일을 열어 API 키/URL 등을 설정
2. python lightrag_onprem_demo.py
"""

import os
import uuid
import asyncio
import shutil
import aiohttp
import logging
from pathlib import Path
from functools import partial

# .env 파일 자동 로드
try:
    from dotenv import load_dotenv

    load_dotenv(".env.onprem")
except ImportError:
    pass

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import EmbeddingFunc, setup_logger

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# 사내 API 설정 (.env.onprem 에서 읽어옴)
# ------------------------------------------------------------------
LLM_BASE_URL = os.getenv(
    "ONPREM_LLM_BASE_URL",
    "http://apigw-stg.shyum.net:8000/gpt-oss/1/gpt-oss-120b/v1",
)
LLM_MODEL = os.getenv("ONPREM_LLM_MODEL", "openai/gpt-oss-120b")
LLM_CREDENTIAL_KEY = os.getenv("ONPREM_LLM_CREDENTIAL_KEY", "")
LLM_SEND_SYSTEM_NAME = os.getenv("ONPREM_SEND_SYSTEM_NAME", "test_api_1")
LLM_USER_ID = os.getenv("ONPREM_USER_ID", "")

# 임베딩 — 사내에 별도 임베딩 서버가 있으면 설정, 없으면 LLM과 같은 base_url 사용
EMBED_BASE_URL = os.getenv("ONPREM_EMBED_BASE_URL", LLM_BASE_URL)
EMBED_MODEL = os.getenv("ONPREM_EMBED_MODEL", "text-embedding-3-small")
EMBED_DIM = int(os.getenv("ONPREM_EMBED_DIM", "1536"))
EMBED_CREDENTIAL_KEY = os.getenv("ONPREM_EMBED_CREDENTIAL_KEY", LLM_CREDENTIAL_KEY)

# Reranker
RERANK_BASE_URL = os.getenv(
    "ONPREM_RERANK_BASE_URL",
    "http://apigw-stg.shyum.net:8000/reranker/1/v2/rerank",
)
RERANK_MODEL = os.getenv("ONPREM_RERANK_MODEL", "bge-reranker-v2-m3-ko")
RERANK_CREDENTIAL_KEY = os.getenv("ONPREM_RERANK_CREDENTIAL_KEY", "")
RERANK_TOP_N = int(os.getenv("ONPREM_RERANK_TOP_N", "5"))

# ------------------------------------------------------------------
# 사내 LLM 헤더 생성
# ------------------------------------------------------------------
WORKING_DIR = Path("./rag_storage_onprem")
DOC_PATH = Path("./sample_doc.txt")

setup_logger("lightrag", level="WARNING")


def _build_llm_headers() -> dict:
    """호출마다 고유한 Prompt-Msg-Id / Completion-Msg-Id를 생성"""
    return {
        "x-dep-ticket": LLM_CREDENTIAL_KEY,
        "Send-System-Name": LLM_SEND_SYSTEM_NAME,
        "User-Id": LLM_USER_ID,
        "User-Type": "AD_ID",
        "Prompt-Msg-Id": str(uuid.uuid4()),
        "Completion-Msg-Id": str(uuid.uuid4()),
    }


# ------------------------------------------------------------------
# 사내 Reranker 래퍼 (커스텀 헤더 지원)
# ------------------------------------------------------------------
async def onprem_rerank(
    query: str,
    documents: list[str],
    top_n: int = RERANK_TOP_N,
    **kwargs,
) -> list[dict]:
    """
    사내 Reranker API를 호출합니다.
    LightRAG의 rerank_model_func 시그니처에 맞추어
    [{"index": int, "relevance_score": float}, ...] 를 반환합니다.
    """
    headers = {
        "X-DEP-TICKET": RERANK_CREDENTIAL_KEY,
        "Content-Type": "application/json",
    }
    payload = {
        "model": RERANK_MODEL,
        "query": query,
        "documents": documents,
        "top_n": top_n,
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(
            RERANK_BASE_URL, headers=headers, json=payload
        ) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                logger.error(f"Rerank API error {resp.status}: {error_text}")
                raise RuntimeError(f"Rerank API error {resp.status}: {error_text}")

            data = await resp.json()

    # 사내 API 응답 포맷 -> LightRAG 기대 포맷
    results = []
    for item in data.get("results", data.get("data", [])):
        results.append(
            {
                "index": item.get("index", 0),
                "relevance_score": item.get("relevance_score", item.get("score", 0.0)),
            }
        )
    return results


# ------------------------------------------------------------------
# LightRAG 초기화
# ------------------------------------------------------------------
async def initialize_rag() -> LightRAG:
    if WORKING_DIR.exists():
        shutil.rmtree(WORKING_DIR)
    WORKING_DIR.mkdir(parents=True, exist_ok=True)

    # 사내 LLM: openai_complete_if_cache + openai_client_configs 로 커스텀 헤더 전달
    rag = LightRAG(
        working_dir=str(WORKING_DIR),
        # --- LLM 설정 ---
        llm_model_func=openai_complete_if_cache,
        llm_model_name=LLM_MODEL,
        llm_model_kwargs={
            "base_url": LLM_BASE_URL,
            "api_key": "unused",  # 커스텀 헤더 인증이므로 임의 값
            "openai_client_configs": {
                "default_headers": _build_llm_headers(),
            },
        },
        # --- 임베딩 설정 ---
        embedding_func=EmbeddingFunc(
            embedding_dim=EMBED_DIM,
            max_token_size=8192,
            func=partial(
                openai_embed.func,
                model=EMBED_MODEL,
                base_url=EMBED_BASE_URL,
                api_key="unused",
                client_configs={
                    "default_headers": {
                        "x-dep-ticket": EMBED_CREDENTIAL_KEY,
                        "Send-System-Name": LLM_SEND_SYSTEM_NAME,
                        "User-Id": LLM_USER_ID,
                        "User-Type": "AD_ID",
                    },
                },
            ),
        ),
        # --- Reranker 설정 (선택) ---
        rerank_model_func=onprem_rerank if RERANK_CREDENTIAL_KEY else None,
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()
    return rag


async def index_document(rag: LightRAG) -> None:
    print(f"문서 로딩: {DOC_PATH}")
    text = DOC_PATH.read_text(encoding="utf-8")
    print("인덱싱 시작 (LLM이 엔티티/관계 추출 중)... 몇 분 걸릴 수 있음")
    await rag.ainsert(text)
    print("인덱싱 완료\n")


async def run_query(rag: LightRAG, question: str, mode: str) -> None:
    print("=" * 70)
    print(f"[MODE = {mode.upper()}]  Q: {question}")
    print("-" * 70)
    try:
        answer = await rag.aquery(question, param=QueryParam(mode=mode))
        print(answer)
    except Exception as e:
        print(f"오류: {e}")
    print()


async def main() -> None:
    rag = await initialize_rag()
    await index_document(rag)

    question = "수율 특이점이 발생했을 때 원인 설비를 찾아내는 분석 프로세스와 사용하는 테이블, 통계 기법을 단계별로 설명해줘."
    for mode in ["naive", "local", "global", "hybrid", "mix"]:
        await run_query(rag, question, mode)

    await rag.finalize_storages()


if __name__ == "__main__":
    if not LLM_CREDENTIAL_KEY:
        raise SystemExit(
            "ONPREM_LLM_CREDENTIAL_KEY가 설정되지 않았습니다.\n"
            ".env.onprem 파일을 확인하세요."
        )
    asyncio.run(main())
