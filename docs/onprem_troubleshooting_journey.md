# On-Prem LightRAG 개선 이력 및 남은 이슈 정리

이 문서는 `lightrag_onprem_demo.py`를 폐쇄망 환경에서 실행하기 위해 진행한 수정 이력, 발생 에러, 개선 과정, 그리고 현재 남아있는 리스크를 정리한 문서입니다.

---

## 1) 초기 상태와 목표

- 목표:
  - 폐쇄망에서도 문서 인덱싱/그래프 생성/질의응답이 가능해야 함
  - 외부 네트워크 의존도를 최소화해야 함
  - 사내 LLM Gateway 헤더 규격을 준수해야 함
- 초기 상태:
  - LLM/Reranker는 사내 API 호출 구조
  - 임베딩/토크나이저는 폐쇄망 제약을 충분히 고려하지 못한 상태

---

## 2) 주요 에러와 원인 분석, 개선 조치

### A. `CERTIFICATE_VERIFY_FAILED` (`openaipublic.blob.core.windows.net`)

- 증상:
  - `tiktoken`이 인코딩 파일(`o200k_base`, `cl100k_base`)을 원격 다운로드 시도
  - 폐쇄망/사내 SSL 환경에서 인증서 검증 실패
- 원인:
  - 런타임 시점에 tokenizer 리소스 원격 fetch 발생
- 조치:
  - 고정 tokenizer 설정 추가 (`ONPREM_TIKTOKEN_MODEL`)
  - 오프라인 토크나이저 옵션 추가 (`ONPREM_USE_OFFLINE_CHAR_TOKENIZER`)
  - 이후 xlm-roberta 로컬 토크나이저 옵션까지 확장 (`ONPREM_TOKENIZER_TYPE=xlmr`)

---

### B. 임베딩 API 미구축 / 온프렘 임베딩 경로 불일치

- 증상:
  - 기존 임베딩 API 가정 코드로 사내 환경에서 동작 불가
- 조치:
  - 임베딩 경로를 로컬 Ollama로 전환
  - `ONPREM_OLLAMA_BASE_URL`, `ONPREM_OLLAMA_EMBED_MODEL`, `ONPREM_EMBED_DIM` 도입
  - `/api/tags` 사전 점검 추가 (`ensure_ollama_embed_model`)

---

### C. `list` 타입 임베딩 반환으로 인한 `.size` 오류

- 증상:
  - `AttributeError: 'list' object has no attribute 'size'`
- 원인:
  - LightRAG 기대 타입(`numpy.ndarray`)과 불일치
- 조치:
  - 임베딩 반환을 `np.ndarray(dtype=np.float32)`로 통일

---

### D. `미등록 Send-System-Name` (LLM 400)

- 증상:
  - 인덱싱/질의 단계에서 LLM 호출이 400으로 실패
- 원인:
  - 사내 Gateway 필수 헤더 값 미등록/오입력
- 조치:
  - `ONPREM_SEND_SYSTEM_NAME`, `ONPREM_USER_ID` 필수 검증 추가
  - 실행 전 LLM preflight 추가 (`ensure_llm_gateway_ready`)
  - LLM 호출을 사내 샘플 규격에 맞게 `AsyncOpenAI` wrapper로 정렬

---

### E. `Gleaning stopped ... Input tokens exceeded limit`

- 증상:
  - 추출 단계 토큰 초과 경고
- 원인:
  - 오프라인 char tokenizer 사용 시 토큰 추정이 보수적으로 커짐
  - 추출 프롬프트/문맥까지 합쳐져 입력 토큰이 증가
- 조치:
  - `ONPREM_CHUNK_TOKEN_SIZE`, `ONPREM_CHUNK_OVERLAP_TOKEN_SIZE`,
    `ONPREM_MAX_EXTRACT_INPUT_TOKENS` 환경변수화
  - 문서/환경에 맞춰 운영 중 튜닝 가능하게 변경

---

### F. Ollama 임베딩 500 (`unsupported value: NaN`)

- 증상:
  - 특정 입력에서 임베딩 API가 NaN 포함 응답 직렬화 실패
  - entity upsert 재시도 후 실패
- 조치:
  - 배치 실패 시 단건 fallback
  - 단건도 실패하면 zero-vector 대체
  - 대체 입력을 run별 로그 파일에 기록:
    - `bad_embed_inputs_YYYYMMDD_HHMMSS.log`
    - 동일 문자열 중복 count summary 포함

---

## 3) 검증 보조 스크립트 추가

- 파일: `tokenizer_ollama_embed_test.py`
- 목적:
  - 로컬 Ollama 모델 존재 및 `/api/embed` 응답 검증
  - shape/dtype/NaN 여부 확인
  - `--file sample_doc.txt --max-samples N`으로 실제 문서 기반 점검 가능

---

## 4) 현재 코드 기준 운영 플로우 요약

1. `.env.onprem` 로드
2. 실행 전 점검
   - Ollama 모델 존재 (`/api/tags`)
   - LLM Gateway 헤더/인증 최소 호출 점검
3. LightRAG 초기화
   - tokenizer 타입 선택 (`char` / `xlmr`)
   - local Ollama embedding 함수 연결
4. 인덱싱
   - NaN 임베딩 시 fallback/로그
   - `vdb_chunks.json` 생성/내용 검증
5. 질의 모드별 실행

---

## 5) 아직 남아있는 문제/리스크

1. **Zero-vector 대체의 정확도 저하 리스크**
   - 시스템 가용성은 높이지만 검색/랭킹 품질 저하 가능
2. **환경 편차**
   - 집/사내에서 Ollama 버전, 모델 digest, 하드웨어 차이로 재현성 차이 가능
3. **문자열 특이 케이스**
   - 특정 엔티티명(특수문자/기호/코드형 문자열)에서 NaN 재발 가능
4. **폐쇄망 패키지 동기화**
   - wheels와 실제 실행 인터프리터 불일치 시 런타임 문제 발생 가능

---

## 6) 권장 후속 작업

1. `bad_embed_inputs_*.log` 주기 수집 및 패턴 전처리 룰화
2. Ollama 버전/모델 digest 집/사내 정렬
3. 필요 시 임베딩 모델 대체 실험 (A/B)
4. 폐쇄망 설치 표준 절차 문서화
   - venv 생성
   - wheel 설치
   - tokenizer 로컬 파일 배치
   - 사전점검 스크립트 실행 순서

---

## 7) 참고 파일

- `lightrag_onprem_demo.py`
- `.env.onprem.example`
- `tokenizer_ollama_embed_test.py`

