# LightRAG 테스트 샘플

HKUDS의 [LightRAG](https://github.com/HKUDS/LightRAG)를 바로 돌려볼 수 있는 최소한의 한국어 샘플입니다.

## 구성 파일

| 파일 | 용도 |
|------|------|
| `sample_doc.txt` | 테스트용 한국어 문서 (양자 컴퓨팅 주제) |
| `lightrag_demo.py` | **OpenAI API** 기반 데모 (권장) |
| `lightrag_ollama_demo.py` | **Ollama 로컬 LLM** 기반 데모 (API 키 불필요) |
| `requirements.txt` | 필요한 파이썬 패키지 |

---

## 방법 A — OpenAI로 돌리기 (가장 빠름)

```bash
# 1) 가상환경
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2) 설치
pip install -r requirements.txt

# 3) API 키
export OPENAI_API_KEY="sk-..."   # Windows PowerShell: $env:OPENAI_API_KEY="sk-..."

# 4) 실행
python lightrag_demo.py
```

### 예상 비용
- 문서 1KB당 대략 수백~수천 토큰이 엔티티 추출에 쓰입니다.
- 샘플 문서(약 2KB)는 gpt-4o-mini 기준 **$0.01 ~ $0.03** 수준.
- 쿼리 5회는 추가로 몇 센트 정도.

### 실행 시간
- 인덱싱(엔티티 추출): **30초 ~ 2분**
- 쿼리 하나당: **5 ~ 15초**

---

## 방법 B — Ollama 로컬 LLM으로 돌리기 (무료)

```bash
# 1) Ollama 설치: https://ollama.com
# 2) 모델 다운로드 (엔티티 추출은 7B+ 권장)
ollama pull qwen2.5:7b
ollama pull bge-m3:latest

# 3) Ollama 서버가 켜져있는지 확인
ollama list

# 4) 실행
pip install -r requirements.txt
python lightrag_ollama_demo.py
```

> ⚠️ 로컬 LLM은 훨씬 느리고, 7B 이하 모델은 엔티티 추출 품질이 떨어져 그래프가 빈약해질 수 있습니다. qwen2.5:14b 이상, 또는 그 이상을 권장합니다.

---

## 돌리고 나면 보이는 것

### 1. 지식 그래프 저장소
`./rag_storage/` 디렉터리가 생기고 그 안에 이런 파일들이 쌓입니다:
```
rag_storage/
├─ graph_chunk_entity_relation.graphml   ← 지식 그래프 (Gephi/Cytoscape로 열람 가능)
├─ kv_store_full_docs.json
├─ kv_store_text_chunks.json
├─ vdb_chunks.json                       ← 청크 벡터
├─ vdb_entities.json                     ← 엔티티 벡터
└─ vdb_relationships.json                ← 관계 벡터
```
`.graphml` 파일을 [Gephi](https://gephi.org) 같은 그래프 뷰어로 열면 추출된 엔티티·관계를 시각적으로 볼 수 있습니다.

### 2. 같은 질문에 대한 5가지 모드 답변 비교

샘플 질문: *"IBM과 구글은 양자 컴퓨팅 분야에서 각각 어떤 접근을 하고 있으며, 그들의 주요 성과는 무엇인가?"*

일반적으로 모드별로 이런 경향이 관찰됩니다:
- **naive**: 원문을 짧게 인용하는 수준. 관계 추론이 약함.
- **local**: IBM, 구글, Sycamore, Condor 같은 **구체적 엔티티** 정보 위주.
- **global**: 초전도 큐비트 방식이라는 **공통 테마**를 중심으로 비교.
- **hybrid**: 위 둘을 섞어 가장 균형 잡힌 답.
- **mix**: 벡터 검색까지 섞어 맥락이 가장 풍부. 일반적으로 **최고 품질**이지만 토큰도 제일 많이 먹음.

---

## 다음으로 해볼 만한 것

1. **다른 문서로 교체** — `sample_doc.txt`를 본인의 문서(사내 정책, 논문, 책 등)로 갈아끼우기만 하면 됩니다.
2. **WebUI로 시각화** — `pip install "lightrag-hku[api]"` 후 `lightrag-server`를 돌리면 브라우저에서 그래프를 돌려볼 수 있습니다.
3. **스토리지 백엔드 교체** — 프로덕션은 Neo4j + PostgreSQL 조합을 많이 씁니다. `.env`로 설정 가능.
4. **스트리밍 응답** — `QueryParam(stream=True)`으로 토큰 단위 스트리밍 가능.

---

## 자주 나는 오류

| 증상 | 원인 / 해결 |
|------|------------|
| `ModuleNotFoundError: lightrag` | 가상환경 활성화를 안 했거나 `pip install -r requirements.txt` 누락 |
| 임베딩 차원 불일치 오류 | 이전 실행의 `rag_storage/`를 안 지움. 스크립트가 매번 지우지만, 수동 변경 시 주의 |
| Ollama에서 JSON 파싱 실패 반복 | 모델이 너무 작음. 7B → 14B로 올리거나 OpenAI로 전환 |
| `OPENAI_API_KEY` 관련 에러 | 환경변수 재설정 후 새 터미널에서 실행 |

---

## 참고
- 공식 리포: https://github.com/HKUDS/LightRAG
- 논문 (EMNLP 2025 Findings): https://aclanthology.org/2025.findings-emnlp.568/
- arXiv: https://arxiv.org/abs/2410.05779
