# LightRAG 테스트 프로젝트

이 디렉터리는 [HKUDS/LightRAG](https://github.com/HKUDS/LightRAG)를 로컬에서
테스트해보기 위한 최소 샘플입니다. Claude Code로 실행/수정하려는 목적입니다.

## 빠른 실행 (Windows + Claude Code)

```powershell
# 1) 가상환경
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # PowerShell
# cmd: .venv\Scripts\activate.bat

# 2) 의존성
pip install -r requirements.txt

# 3) API 키 설정 (.env.example을 복사해서 .env 만들고 키 채우기)
copy .env.example .env
# 그런 다음 .env 파일을 열어 OPENAI_API_KEY 값을 채우기

# 4) 실행
python lightrag_demo.py
```

> macOS/Linux라면 `source .venv/bin/activate`, `cp .env.example .env` 사용.

## 파일 구조

```
light-rag-test/
├─ CLAUDE.md               ← 지금 이 파일 (Claude Code 가이드)
├─ README.md               ← 사람이 읽는 일반 가이드
├─ requirements.txt
├─ .env.example            ← API 키 템플릿
├─ sample_doc.txt          ← 테스트용 한국어 문서 (양자 컴퓨팅 주제)
├─ lightrag_demo.py        ← OpenAI 버전 메인 데모
└─ lightrag_ollama_demo.py ← Ollama 로컬 LLM 버전 (선택)
```

실행 후 생성되는 디렉터리 (gitignore 대상):
- `rag_storage/` — OpenAI 버전 인덱스/그래프 저장
- `rag_storage_ollama/` — Ollama 버전 인덱스/그래프 저장
- `.venv/` — 가상환경

## 이 프로젝트의 목적

LightRAG의 **다섯 가지 쿼리 모드**가 같은 질문에 어떻게 다르게 답하는지 비교하는 것이
핵심입니다. 모드별 특징:

| 모드 | 강점 | 주로 쓸 상황 |
|------|------|--------------|
| `naive` | 단순 벡터 검색 (베이스라인) | 비교 기준 |
| `local` | 특정 엔티티 중심 사실 | "X는 누구/무엇?" |
| `global` | 테마/커뮤니티 요약 | "전반적으로…", "비교해줘" |
| `hybrid` | local + global 결합 | 일반적 용도 |
| `mix` | KG + vector 모두 | 최고 품질 (토큰 많이 씀) |

## Claude Code에서 해볼 만한 작업

1. **`sample_doc.txt`를 다른 도메인 문서로 교체** — 가장 먼저 해볼 것
2. **`lightrag_demo.py`의 질문 리스트 수정** — `main()` 함수 안에 있음
3. **모델 교체** — `gpt_4o_mini_complete` → `gpt_4o_complete`로 바꾸면 품질↑ 비용↑
4. **WebUI 띄워보기** — `pip install "lightrag-hku[api]"` → `lightrag-server`
5. **스트리밍 답변** — `QueryParam(stream=True, mode="hybrid")`로 바꾸고 async iterator로 받기

## 주의사항 (Claude Code가 작업할 때)

- 인덱싱은 LLM을 많이 호출하므로 **실행 시 비용 발생**. 샘플 문서는 보통 $0.01~0.03 수준.
- `rag_storage/` 폴더를 수동으로 지우지 말고 스크립트에 맡길 것. 임베딩 모델/차원
  바뀌면 기존 벡터 DB와 충돌남.
- `.env` 파일은 **절대 커밋하지 말 것** (`.gitignore` 포함됨).
- 인덱싱은 한 번 돌고 나면 `rag_storage/`에 결과가 남습니다. 매 실행마다 전체
  재인덱싱하는 건 `lightrag_demo.py` 코드가 폴더를 지우고 시작하기 때문 —
  실험 중이면 주석 처리하세요.

## 참고 링크

- 공식 리포: https://github.com/HKUDS/LightRAG
- EMNLP 2025 논문: https://aclanthology.org/2025.findings-emnlp.568/
- 공식 examples: https://github.com/HKUDS/LightRAG/tree/main/examples
