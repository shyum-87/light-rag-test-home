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
import numpy as np
import json
import importlib
from datetime import datetime
from collections import Counter
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
LLM_SEND_SYSTEM_NAME = os.getenv("ONPREM_SEND_SYSTEM_NAME", "")
LLM_USER_ID = os.getenv("ONPREM_USER_ID", "")

# 임베딩 — 현재 사내 임베딩 API 미구축 상태를 고려해 로컬 Ollama 사용
OLLAMA_BASE_URL = os.getenv("ONPREM_OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_EMBED_MODEL = os.getenv("ONPREM_OLLAMA_EMBED_MODEL", "bge-m3:latest")
EMBED_DIM = int(os.getenv("ONPREM_EMBED_DIM", "1024"))
CHUNK_TOKEN_SIZE = int(os.getenv("ONPREM_CHUNK_TOKEN_SIZE", "800"))
CHUNK_OVERLAP_TOKEN_SIZE = int(os.getenv("ONPREM_CHUNK_OVERLAP_TOKEN_SIZE", "80"))
MAX_EXTRACT_INPUT_TOKENS = int(os.getenv("ONPREM_MAX_EXTRACT_INPUT_TOKENS", "16000"))
# tiktoken 모델명 고정 (폐쇄망에서 원격 tokenizer 다운로드 방지 목적)
# 집(인터넷 가능) 환경에서 검증한 값으로 ONPREM_TIKTOKEN_MODEL 을 맞춰 사용하세요.
TIKTOKEN_MODEL = os.getenv("ONPREM_TIKTOKEN_MODEL", "text-embedding-3-small")
USE_OFFLINE_CHAR_TOKENIZER = (
    os.getenv("ONPREM_USE_OFFLINE_CHAR_TOKENIZER", "true").lower() == "true"
)
TOKENIZER_TYPE = os.getenv("ONPREM_TOKENIZER_TYPE", "char").lower()
XLMR_TOKENIZER_PATH = os.getenv("ONPREM_XLMR_TOKENIZER_PATH", "./tokenizer")

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
BAD_EMBED_LOG_PATH: Path | None = None

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
        timeout=kwargs.pop("timeout", 600),
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


MIN_EMBED_INPUT_LENGTH = int(os.getenv("ONPREM_MIN_EMBED_INPUT_LENGTH", "64"))

import re as _re

def _sanitize_embed_input(text: str) -> str:
    """임베딩 입력 전처리: 특수 구분자 정규화 + 최소 길이 보장 (NaN 방지)."""
    # 탭 → 공백, 연속 공백/개행 정리
    text = text.replace("\t", " ")
    text = _re.sub(r"\r\n|\r", "\n", text)
    text = _re.sub(r" {2,}", " ", text)
    text = text.strip()
    # 짧은 입력은 의미 있는 접두어를 붙여 최소 길이 확보
    # 단순 반복보다 다양한 토큰을 포함시켜 NaN 발생 확률을 낮춤
    if len(text) < MIN_EMBED_INPUT_LENGTH:
        text = f"The following is a description of: {text}. This term is relevant to semiconductor manufacturing yield analysis."
    return text


async def ollama_embed(texts: list[str]) -> np.ndarray:
    """
    폐쇄망 PC의 로컬 Ollama 임베딩 모델(bge-m3:latest 등)을 호출합니다.
    입력 텍스트에 전처리(특수문자 정규화, 최소 길이 패딩)를 적용합니다.
    """
    endpoint = f"{OLLAMA_BASE_URL.rstrip('/')}/api/embed"
    texts = [_sanitize_embed_input(t) for t in texts]

    async def _embed_batch(batch_texts: list[str]) -> list[list[float]]:
        payload = {"model": OLLAMA_EMBED_MODEL, "input": batch_texts}
        async with aiohttp.ClientSession() as session:
            async with session.post(endpoint, json=payload) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise RuntimeError(
                        f"Ollama embed API error {resp.status}: {error_text}"
                    )
                data = await resp.json()
        embeddings = data.get("embeddings")
        if not embeddings:
            raise RuntimeError(f"Ollama embed 응답에 embeddings가 없습니다: {data}")
        return embeddings

    try:
        arr = np.array(await _embed_batch(texts), dtype=np.float32)
        if not np.isfinite(arr).all():
            raise RuntimeError("Ollama 임베딩 결과에 NaN/Inf가 포함되었습니다.")
        return arr
    except Exception as e:
        msg = str(e)
        if "unsupported value: NaN" not in msg and "NaN" not in msg:
            logger.error(f"Ollama embed API error: {e}")
            raise

        logger.warning(
            "Ollama 배치 임베딩에서 NaN 오류 발생. 단건 임베딩 fallback을 시도합니다."
        )
        vectors: list[np.ndarray] = []
        bad_inputs: list[str] = []
        for t in texts:
            try:
                single_arr = np.array(await _embed_batch([t]), dtype=np.float32)
                vec = single_arr[0]
                if not np.isfinite(vec).all():
                    raise RuntimeError("single embedding contains NaN/Inf")
                vectors.append(vec)
            except Exception as single_e:
                logger.error(
                    f"단건 임베딩도 실패하여 zero-vector로 대체합니다: {single_e}"
                )
                bad_inputs.append(t)
                vectors.append(np.zeros(EMBED_DIM, dtype=np.float32))
        if bad_inputs:
            target_log = BAD_EMBED_LOG_PATH or Path("./bad_embed_inputs.log")
            counter = Counter(bad_inputs)
            with target_log.open("a", encoding="utf-8") as f:
                f.write("\n=== zero-vector fallback batch ===\n")
                for idx, bad in enumerate(bad_inputs, start=1):
                    f.write(f"[{idx}] {bad}\n")
                f.write("--- count summary ---\n")
                for text, cnt in counter.most_common():
                    f.write(f"{cnt}x | {text}\n")
            logger.warning(
                f"zero-vector로 대체된 입력 {len(bad_inputs)}건을 {target_log}에 기록했습니다."
            )
        return np.vstack(vectors)


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


async def ensure_llm_gateway_ready() -> None:
    """
    인덱싱 전에 LLM 게이트웨이 인증/헤더 설정이 유효한지 사전 점검합니다.
    """
    try:
        await onprem_llm_complete(
            prompt="ping",
            system_prompt="You are a helpful assistant. Reply with one word.",
        )
    except Exception as e:
        raise RuntimeError(
            "LLM 게이트웨이 사전 점검 실패. "
            "ONPREM_SEND_SYSTEM_NAME / ONPREM_USER_ID / ONPREM_LLM_CREDENTIAL_KEY 값을 확인하세요.\n"
            f"원인: {e}"
        ) from e


class OfflineCharTokenizer:
    """
    tiktoken 다운로드가 불가능한 폐쇄망 환경용 단순 문자 기반 토크나이저.
    """

    def encode(self, content: str) -> list[int]:
        return [ord(ch) for ch in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(t) for t in tokens)


class HuggingFaceTokenizer:
    """
    로컬 Hugging Face tokenizer(xlm-roberta 등) 래퍼.
    """

    def __init__(self, model_path: str):
        transformers = importlib.import_module("transformers")
        auto_tokenizer = getattr(transformers, "AutoTokenizer")
        self.tk = auto_tokenizer.from_pretrained(model_path, local_files_only=True)

    def encode(self, content: str) -> list[int]:
        return self.tk.encode(content, add_special_tokens=False)

    def decode(self, tokens: list[int]) -> str:
        return self.tk.decode(tokens, skip_special_tokens=True)


# ------------------------------------------------------------------
# LightRAG 초기화
# ------------------------------------------------------------------
async def initialize_rag() -> LightRAG:
    if WORKING_DIR.exists():
        shutil.rmtree(WORKING_DIR)
    WORKING_DIR.mkdir(parents=True, exist_ok=True)

    # 폐쇄망에서 tiktoken 원격 다운로드가 불가하면 문자 기반 토크나이저 사용
    tokenizer = None
    if TOKENIZER_TYPE == "xlmr":
        tokenizer = Tokenizer(
            model_name="xlm-roberta-local",
            tokenizer=HuggingFaceTokenizer(XLMR_TOKENIZER_PATH),
        )
    elif USE_OFFLINE_CHAR_TOKENIZER:
        tokenizer = Tokenizer(
            model_name="offline-char-tokenizer",
            tokenizer=OfflineCharTokenizer(),
        )

    # 사내 LLM: openai_complete_if_cache + openai_client_configs 로 커스텀 헤더 전달
    rag = LightRAG(
        working_dir=str(WORKING_DIR),
        tokenizer=tokenizer,
        tiktoken_model_name=TIKTOKEN_MODEL,
        chunk_token_size=CHUNK_TOKEN_SIZE,
        chunk_overlap_token_size=CHUNK_OVERLAP_TOKEN_SIZE,
        max_extract_input_tokens=MAX_EXTRACT_INPUT_TOKENS,
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
    chunks_file = WORKING_DIR / "vdb_chunks.json"
    if not chunks_file.exists():
        raise RuntimeError("인덱싱 결과 파일(vdb_chunks.json)이 생성되지 않았습니다.")
    try:
        data = json.loads(chunks_file.read_text(encoding="utf-8"))
        if not data.get("data"):
            raise RuntimeError(
                "인덱싱 결과 청크가 비어 있습니다. 상단 LLM/임베딩 오류 로그를 확인하세요."
            )
    except json.JSONDecodeError:
        raise RuntimeError("인덱싱 결과 파일(vdb_chunks.json) 파싱에 실패했습니다.")
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
    global BAD_EMBED_LOG_PATH
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    BAD_EMBED_LOG_PATH = Path(f"./bad_embed_inputs_{ts}.log")
    await ensure_ollama_embed_model()
    await ensure_llm_gateway_ready()
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
    if not LLM_SEND_SYSTEM_NAME:
        raise SystemExit(
            "ONPREM_SEND_SYSTEM_NAME가 설정되지 않았습니다.\n"
            "DS API HUB에 등록된 Send-System-Name 값을 .env.onprem에 입력하세요."
        )
    if not LLM_USER_ID:
        raise SystemExit(
            "ONPREM_USER_ID가 설정되지 않았습니다.\n"
            "사내 KNOX ID 값을 .env.onprem에 입력하세요."
        )
    asyncio.run(main())
