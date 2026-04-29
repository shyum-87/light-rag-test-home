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
from openai import AsyncOpenAI

# .env 파일 자동 로드
try:
    from dotenv import load_dotenv

    load_dotenv(".env.onprem")
except ImportError:
    pass

from lightrag import LightRAG, QueryParam
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import EmbeddingFunc, Tokenizer, setup_logger

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

# 임베딩 — 현재 사내 임베딩 API 미구축 상태를 고려해 로컬 Ollama 사용
OLLAMA_BASE_URL = os.getenv("ONPREM_OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_EMBED_MODEL = os.getenv("ONPREM_OLLAMA_EMBED_MODEL", "bge-m3:latest")
EMBED_DIM = int(os.getenv("ONPREM_EMBED_DIM", "1024"))
# tiktoken 모델명 고정 (폐쇄망에서 원격 tokenizer 다운로드 방지 목적)
# 집(인터넷 가능) 환경에서 검증한 값으로 ONPREM_TIKTOKEN_MODEL 을 맞춰 사용하세요.
TIKTOKEN_MODEL = os.getenv("ONPREM_TIKTOKEN_MODEL", "text-embedding-3-small")
USE_OFFLINE_CHAR_TOKENIZER = (
    os.getenv("ONPREM_USE_OFFLINE_CHAR_TOKENIZER", "true").lower() == "true"
)

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


async def onprem_llm_complete(*args, **kwargs) -> str:
    """
    LightRAG 내부 호출 경로별 인자 차이를 흡수하는 LLM 래퍼.
    """
    prompt = kwargs.pop("prompt", None)
    if prompt is None and len(args) >= 1:
        prompt = args[0]
    if prompt is None:
        prompt = kwargs.pop("query", None) or kwargs.pop("user_prompt", None)
    if prompt is None:
        raise RuntimeError(
            "LLM 호출에 prompt가 없어 요청을 처리할 수 없습니다. "
            "LightRAG 호출 인자(prompt/query)를 확인하세요."
        )

    system_prompt = kwargs.pop("system_prompt", None)
    history_messages = kwargs.pop("history_messages", None) or []

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    client = AsyncOpenAI(
        base_url=LLM_BASE_URL,
        api_key="unused",
        default_headers=_build_llm_headers(),
    )
    completion = await client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        timeout=kwargs.pop("timeout", None),
    )
    return completion.choices[0].message.content or ""


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


async def ollama_embed(texts: list[str]) -> list[list[float]]:
    """
    폐쇄망 PC의 로컬 Ollama 임베딩 모델(bge-m3:latest 등)을 호출합니다.
    """
    endpoint = f"{OLLAMA_BASE_URL.rstrip('/')}/api/embed"
    payload = {"model": OLLAMA_EMBED_MODEL, "input": texts}

    async with aiohttp.ClientSession() as session:
        async with session.post(endpoint, json=payload) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                logger.error(f"Ollama embed API error {resp.status}: {error_text}")
                raise RuntimeError(
                    f"Ollama embed API error {resp.status}: {error_text}"
                )
            data = await resp.json()

    embeddings = data.get("embeddings")
    if not embeddings:
        raise RuntimeError(f"Ollama embed 응답에 embeddings가 없습니다: {data}")
    return embeddings


async def ensure_ollama_embed_model() -> None:
    """
    시작 전에 로컬 Ollama 모델 존재 여부를 확인해 404를 사전에 방지합니다.
    """
    endpoint = f"{OLLAMA_BASE_URL.rstrip('/')}/api/tags"
    async with aiohttp.ClientSession() as session:
        async with session.get(endpoint) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(
                    f"Ollama 연결 실패({resp.status}): {body}\n"
                    f"- ONPREM_OLLAMA_BASE_URL 확인: {OLLAMA_BASE_URL}"
                )
            data = await resp.json()

    model_names = {m.get("name", "") for m in data.get("models", [])}
    if OLLAMA_EMBED_MODEL not in model_names:
        raise RuntimeError(
            f"Ollama 모델 '{OLLAMA_EMBED_MODEL}' 이(가) 없습니다.\n"
            f"사내 PC에서 먼저 실행하세요: ollama pull {OLLAMA_EMBED_MODEL}\n"
            f"현재 설치 모델: {sorted([n for n in model_names if n])}"
        )


class OfflineCharTokenizer:
    """
    tiktoken 다운로드가 불가능한 폐쇄망 환경용 단순 문자 기반 토크나이저.
    """

    def encode(self, content: str) -> list[int]:
        return [ord(ch) for ch in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(t) for t in tokens)


# ------------------------------------------------------------------
# LightRAG 초기화
# ------------------------------------------------------------------
async def initialize_rag() -> LightRAG:
    if WORKING_DIR.exists():
        shutil.rmtree(WORKING_DIR)
    WORKING_DIR.mkdir(parents=True, exist_ok=True)

    # 폐쇄망에서 tiktoken 원격 다운로드가 불가하면 문자 기반 토크나이저 사용
    tokenizer = None
    if USE_OFFLINE_CHAR_TOKENIZER:
        tokenizer = Tokenizer(
            model_name="offline-char-tokenizer",
            tokenizer=OfflineCharTokenizer(),
        )

    # 사내 LLM: openai_complete_if_cache + openai_client_configs 로 커스텀 헤더 전달
    rag = LightRAG(
        working_dir=str(WORKING_DIR),
        tokenizer=tokenizer,
        tiktoken_model_name=TIKTOKEN_MODEL,
        # --- LLM 설정 ---
        llm_model_func=onprem_llm_complete,
        llm_model_name=LLM_MODEL,
        llm_model_kwargs={},
        # --- 임베딩 설정 (로컬 Ollama) ---
        embedding_func=EmbeddingFunc(
            embedding_dim=EMBED_DIM,
            max_token_size=8192,
            func=ollama_embed,
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
    try:
        await rag.ainsert(text)
    except Exception:
        print("인덱싱 실패: 임베딩/LLM 설정을 확인하세요.\n")
        raise
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
    await ensure_ollama_embed_model()
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
