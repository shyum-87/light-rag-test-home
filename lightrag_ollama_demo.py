"""
LightRAG 데모 스크립트 (Ollama 로컬 버전)
==========================================
OpenAI API 키 없이 완전 로컬에서 돌리고 싶을 때 쓰는 버전.
Ollama가 실행 중이어야 하고, 모델을 미리 받아두어야 한다.

사전 준비:
1. https://ollama.com 에서 Ollama 설치
2. 모델 pull:
   ollama pull qwen2.5:7b               # LLM (엔티티/관계 추출 능력 필요 → 7B+ 권장)
   ollama pull bge-m3:latest            # 임베딩 (다국어)
3. pip install -r requirements.txt
4. python lightrag_ollama_demo.py
"""

import asyncio
import shutil
from pathlib import Path

from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc, setup_logger
from lightrag.kg.shared_storage import initialize_pipeline_status

WORKING_DIR = Path("./rag_storage_ollama")
DOC_PATH = Path("./sample_doc.txt")

# LLM/Embedding 설정 - 환경에 맞게 바꿀 것
LLM_MODEL = "qwen2.5:7b"       # 또는 llama3.1:8b, qwen2.5:14b (더 좋음)
EMBED_MODEL = "bge-m3:latest"  # 한국어 포함 다국어 임베딩
OLLAMA_HOST = "http://localhost:11434"

setup_logger("lightrag", level="WARNING")


async def initialize_rag() -> LightRAG:
    if WORKING_DIR.exists():
        shutil.rmtree(WORKING_DIR)
    WORKING_DIR.mkdir(parents=True, exist_ok=True)

    rag = LightRAG(
        working_dir=str(WORKING_DIR),
        llm_model_func=ollama_model_complete,
        llm_model_name=LLM_MODEL,
        llm_model_kwargs={
            "host": OLLAMA_HOST,
            "options": {"num_ctx": 32768},  # 컨텍스트 늘려주면 그래프 품질 ↑
        },
        embedding_func=EmbeddingFunc(
            embedding_dim=1024,      # bge-m3의 차원
            max_token_size=8192,
            func=lambda texts: ollama_embed(
                texts, embed_model=EMBED_MODEL, host=OLLAMA_HOST
            ),
        ),
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()
    return rag


async def main() -> None:
    print(f"🤖 LLM: {LLM_MODEL}   🧮 Embed: {EMBED_MODEL}")
    rag = await initialize_rag()

    print(f"📄 문서 로딩: {DOC_PATH}")
    text = DOC_PATH.read_text(encoding="utf-8")

    print("🧠 인덱싱 시작 (로컬 LLM이라 OpenAI보다 훨씬 오래 걸립니다)")
    await rag.ainsert(text)
    print("✅ 인덱싱 완료\n")

    question = "IBM과 구글은 양자 컴퓨팅 분야에서 각각 어떤 접근을 하고 있나?"
    for mode in ["naive", "local", "global", "hybrid", "mix"]:
        print("=" * 70)
        print(f"[MODE = {mode.upper()}]  Q: {question}")
        print("-" * 70)
        try:
            answer = await rag.aquery(question, param=QueryParam(mode=mode))
            print(answer)
        except Exception as e:
            print(f"❌ 오류: {e}")
        print()

    await rag.finalize_storages()


if __name__ == "__main__":
    asyncio.run(main())
