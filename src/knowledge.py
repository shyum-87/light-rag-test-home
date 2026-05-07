"""src/knowledge.py — LightRAG 래핑: 도메인 지식 인덱싱 및 검색."""
import os
import logging
import importlib
import numpy as np
from pathlib import Path

from lightrag import LightRAG, QueryParam
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import EmbeddingFunc, Tokenizer, setup_logger

logger = logging.getLogger(__name__)


class OfflineCharTokenizer:
    """폐쇄망용 문자 기반 토크나이저."""
    def encode(self, content: str) -> list[int]:
        return [ord(ch) for ch in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(t) for t in tokens)


# ONNX 임베딩 싱글턴
_ort_session = None
_ort_tokenizer = None


def _get_ort_model(model_path: str):
    global _ort_session, _ort_tokenizer
    if _ort_session is None:
        import onnxruntime as ort
        from transformers import AutoTokenizer

        onnx_file = Path(model_path) / "model.onnx"
        if not onnx_file.exists():
            raise RuntimeError(f"ONNX 모델 없음: {onnx_file}")

        logger.info(f"ONNX 모델 로딩: {model_path}")
        _ort_session = ort.InferenceSession(str(onnx_file))
        _ort_tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), local_files_only=True
        )
    return _ort_session, _ort_tokenizer


async def _onnx_embed(texts: list[str], model_path: str) -> np.ndarray:
    session, tokenizer = _get_ort_model(model_path)
    input_names = [i.name for i in session.get_inputs()]
    all_embeddings = []
    for i in range(0, len(texts), 32):
        batch = texts[i:i + 32]
        inputs = tokenizer(batch, return_tensors="np", padding=True,
                           truncation=True, max_length=8192)
        feeds = {k: v for k, v in inputs.items() if k in input_names}
        outputs = session.run(None, feeds)
        all_embeddings.append(outputs[0][:, 0, :])
    embeddings = np.vstack(all_embeddings).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return embeddings / norms


class KnowledgeBase:
    """LightRAG 기반 도메인 지식 검색."""

    def __init__(self, working_dir: str, embed_model_path: str,
                 llm_model_func, llm_model_name: str):
        self._working_dir = working_dir
        self._embed_model_path = embed_model_path
        self._llm_model_func = llm_model_func
        self._llm_model_name = llm_model_name
        self._rag = None

    async def initialize(self):
        """LightRAG 인스턴스를 초기화한다."""
        Path(self._working_dir).mkdir(parents=True, exist_ok=True)

        embed_path = self._embed_model_path

        self._rag = LightRAG(
            working_dir=self._working_dir,
            tokenizer=Tokenizer(
                model_name="offline-char-tokenizer",
                tokenizer=OfflineCharTokenizer(),
            ),
            tiktoken_model_name="text-embedding-3-small",
            chunk_token_size=800,
            chunk_overlap_token_size=80,
            llm_model_func=self._llm_model_func,
            llm_model_name=self._llm_model_name,
            llm_model_kwargs={},
            embedding_func=EmbeddingFunc(
                embedding_dim=1024,
                max_token_size=8192,
                func=lambda texts: _onnx_embed(texts, embed_path),
            ),
        )
        await self._rag.initialize_storages()
        await initialize_pipeline_status()

    async def index_documents(self, docs_dir: str):
        """docs_dir 내의 모든 .txt 파일을 인덱싱한다."""
        docs_path = Path(docs_dir)
        txt_files = sorted(docs_path.rglob("*.txt"))
        if not txt_files:
            logger.warning(f"인덱싱할 .txt 파일 없음: {docs_dir}")
            return

        for f in txt_files:
            logger.info(f"인덱싱: {f}")
            text = f.read_text(encoding="utf-8")
            await self._rag.ainsert(text)

        logger.info(f"{len(txt_files)}개 문서 인덱싱 완료")

    async def query(self, question: str, mode: str = "hybrid") -> str:
        """도메인 지식을 검색하여 답변을 반환한다."""
        return await self._rag.aquery(question, param=QueryParam(mode=mode))

    async def finalize(self):
        if self._rag:
            await self._rag.finalize_storages()
