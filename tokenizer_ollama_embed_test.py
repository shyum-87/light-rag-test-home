"""
로컬 Ollama 임베딩 모델 동작 점검 스크립트

사용법:
  python tokenizer_ollama_embed_test.py
"""

import os
import asyncio
import aiohttp
import numpy as np
import argparse
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(".env.onprem")
except ImportError:
    pass


OLLAMA_BASE_URL = os.getenv("ONPREM_OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_EMBED_MODEL = os.getenv("ONPREM_OLLAMA_EMBED_MODEL", "bge-m3:latest")


async def check_model_exists() -> None:
    tags_url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/tags"
    async with aiohttp.ClientSession() as session:
        async with session.get(tags_url) as resp:
            if resp.status != 200:
                raise RuntimeError(f"Ollama tags 조회 실패({resp.status}): {await resp.text()}")
            data = await resp.json()
    names = {m.get("name", "") for m in data.get("models", [])}
    if OLLAMA_EMBED_MODEL not in names:
        raise RuntimeError(
            f"모델 '{OLLAMA_EMBED_MODEL}' 이(가) 없습니다. 먼저 `ollama pull {OLLAMA_EMBED_MODEL}` 실행하세요."
        )


async def test_embed() -> None:
    await check_model_exists()
    embed_url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/embed"
    sample_inputs = [
        "테스트 문장입니다.",
        "수율 특이점 원인 설비를 분석합니다.",
        "Bad Lot Group / Good Lot Group / Test End Date",
    ]

    payload = {"model": OLLAMA_EMBED_MODEL, "input": sample_inputs}
    async with aiohttp.ClientSession() as session:
        async with session.post(embed_url, json=payload) as resp:
            body = await resp.text()
            if resp.status != 200:
                raise RuntimeError(f"Ollama embed 실패({resp.status}): {body}")
            data = await resp.json()

    embeddings = data.get("embeddings")
    if not embeddings:
        raise RuntimeError(f"응답에 embeddings가 없습니다: {data}")

    arr = np.array(embeddings, dtype=np.float32)
    if arr.ndim != 2:
        raise RuntimeError(f"임베딩 shape 비정상: {arr.shape}")
    if not np.isfinite(arr).all():
        raise RuntimeError("임베딩 결과에 NaN/Inf 포함")
    if arr.shape[0] != len(sample_inputs):
        raise RuntimeError(f"입력/출력 개수 불일치: in={len(sample_inputs)} out={arr.shape[0]}")

    print("✅ Ollama 임베딩 모델 정상 응답")
    print(f"- model: {OLLAMA_EMBED_MODEL}")
    print(f"- base_url: {OLLAMA_BASE_URL}")
    print(f"- embedding shape: {arr.shape}")
    print(f"- dtype: {arr.dtype}")


async def test_embed_with_file(file_path: Path, max_samples: int = 50) -> None:
    await check_model_exists()
    if not file_path.exists():
        raise RuntimeError(f"파일이 없습니다: {file_path}")

    text = file_path.read_text(encoding="utf-8")
    chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
    if not chunks:
        raise RuntimeError(f"유효한 문단이 없습니다: {file_path}")
    sample_inputs = chunks[:max_samples]

    embed_url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/embed"
    payload = {"model": OLLAMA_EMBED_MODEL, "input": sample_inputs}
    async with aiohttp.ClientSession() as session:
        async with session.post(embed_url, json=payload) as resp:
            body = await resp.text()
            if resp.status != 200:
                raise RuntimeError(f"Ollama embed 실패({resp.status}): {body}")
            data = await resp.json()

    embeddings = data.get("embeddings")
    if not embeddings:
        raise RuntimeError(f"응답에 embeddings가 없습니다: {data}")

    arr = np.array(embeddings, dtype=np.float32)
    if arr.ndim != 2:
        raise RuntimeError(f"임베딩 shape 비정상: {arr.shape}")
    if not np.isfinite(arr).all():
        raise RuntimeError("임베딩 결과에 NaN/Inf 포함")
    if arr.shape[0] != len(sample_inputs):
        raise RuntimeError(f"입력/출력 개수 불일치: in={len(sample_inputs)} out={arr.shape[0]}")

    print("✅ sample_doc 기반 Ollama 임베딩 정상 응답")
    print(f"- model: {OLLAMA_EMBED_MODEL}")
    print(f"- base_url: {OLLAMA_BASE_URL}")
    print(f"- file: {file_path}")
    print(f"- paragraphs tested: {len(sample_inputs)}")
    print(f"- embedding shape: {arr.shape}")
    print(f"- dtype: {arr.dtype}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="", help="검증할 텍스트 파일 경로")
    parser.add_argument("--max-samples", type=int, default=50, help="최대 문단 샘플 수")
    args = parser.parse_args()

    if args.file:
        asyncio.run(test_embed_with_file(Path(args.file), max_samples=args.max_samples))
    else:
        asyncio.run(test_embed())
