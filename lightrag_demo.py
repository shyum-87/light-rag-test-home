"""
LightRAG 데모 스크립트 (OpenAI 버전)
=====================================
한국어 문서 하나를 인덱싱하고, 네 가지 쿼리 모드(naive / local / global / hybrid / mix)로
질문을 던져 결과를 비교합니다.

실행 전 준비:
1. pip install -r requirements.txt
2. export OPENAI_API_KEY="sk-..."
3. python lightrag_demo.py
"""

import os
import asyncio
import shutil
from pathlib import Path

# .env 파일이 있으면 자동 로드 (python-dotenv가 설치되어 있을 때)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import setup_logger


async def openrouter_complete(
    prompt,
    system_prompt=None,
    history_messages=None,
    enable_cot: bool = False,
    keyword_extraction=False,
    **kwargs,
) -> str:
    if history_messages is None:
        history_messages = []
    return await openai_complete_if_cache(
        "openai/gpt-4o-mini",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        enable_cot=enable_cot,
        keyword_extraction=keyword_extraction,
        **kwargs,
    )

# ------------------------------------------------------------------
# 설정
# ------------------------------------------------------------------
WORKING_DIR = Path("./rag_storage")
DOC_PATH = Path("./sample_doc.txt")

# LightRAG 내부 로그 보고 싶으면 INFO로
setup_logger("lightrag", level="WARNING")


async def initialize_rag() -> LightRAG:
    """LightRAG 인스턴스를 만들고 storage/pipeline을 초기화한다."""
    # 이전 실행 결과가 남아있으면 깨끗이 지우고 시작 (처음 돌릴 때만 필요)
    if WORKING_DIR.exists():
        shutil.rmtree(WORKING_DIR)
    WORKING_DIR.mkdir(parents=True, exist_ok=True)

    rag = LightRAG(
        working_dir=str(WORKING_DIR),
        embedding_func=openai_embed,          # OpenAI text-embedding-3-small via OpenRouter
        llm_model_func=openrouter_complete,   # openai/gpt-4o-mini via OpenRouter
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()
    return rag


async def index_document(rag: LightRAG) -> None:
    """문서를 읽어 LightRAG에 투입한다.
    이 단계에서 LLM이 엔티티/관계를 추출해 지식 그래프를 만든다."""
    print(f"📄 문서 로딩: {DOC_PATH}")
    text = DOC_PATH.read_text(encoding="utf-8")

    print("🧠 인덱싱 시작 (LLM이 엔티티/관계 추출 중)... 몇 분 걸릴 수 있음")
    await rag.ainsert(text)
    print("✅ 인덱싱 완료\n")


async def run_query(rag: LightRAG, question: str, mode: str) -> None:
    """질문 하나를 지정된 모드로 물어보고 결과를 예쁘게 출력한다."""
    print("=" * 70)
    print(f"[MODE = {mode.upper()}]  Q: {question}")
    print("-" * 70)

    try:
        answer = await rag.aquery(question, param=QueryParam(mode=mode))
        print(answer)
    except Exception as e:
        print(f"❌ 오류: {e}")
    print()


async def main() -> None:
    # 1) 초기화 + 문서 인덱싱
    rag = await initialize_rag()
    await index_document(rag)

    # 2) 핵심 질문: 수율 특이점 원인 설비 추적 프로세스 (5가지 모드 비교)
    question = "수율 특이점이 발생했을 때 원인 설비를 찾아내는 분석 프로세스와 사용하는 테이블, 통계 기법을 단계별로 설명해줘."
    for mode in ["naive", "local", "global", "hybrid", "mix"]:
        await run_query(rag, question, mode)

    # 3) 실전 시나리오 질문 (hybrid 모드)
    q2 = "ETCH 설비의 특정 챔버에서 수율이 떨어질 때, 어떤 테이블을 조인해서 어떤 통계 검정으로 원인을 확정하는지 구체적으로 알려줘."
    print("\n" + "#" * 70)
    print("#  두 번째 질문: ETCH 설비 챔버 원인 분석 (hybrid)")
    print("#" * 70)
    await run_query(rag, q2, "hybrid")

    # 4) 엔티티 중심 질문 (local 모드) - 특정 테이블 정보
    q3 = "LOT_HISTORY 테이블과 EQP_SENSOR_DATA 테이블의 컬럼 정보와 조인 관계를 알려줘."
    print("\n" + "#" * 70)
    print("#  세 번째 질문: 테이블 스키마 및 조인 관계 (local)")
    print("#" * 70)
    await run_query(rag, q3, "local")

    # 5) 웨이퍼 맵 패턴 분석 (mix 모드)
    q4 = "웨이퍼 맵에서 Edge 불량 패턴이 나타났을 때, 어떤 공정과 설비를 의심해야 하며 Commonality Analysis는 어떻게 수행하나?"
    print("\n" + "#" * 70)
    print("#  네 번째 질문: 웨이퍼 맵 패턴 + Commonality 분석 (mix)")
    print("#" * 70)
    await run_query(rag, q4, "mix")

    # 정리
    await rag.finalize_storages()


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit(
            "❌ OPENAI_API_KEY 환경변수가 필요합니다.\n"
            '   예) export OPENAI_API_KEY="sk-..."'
        )
    asyncio.run(main())
