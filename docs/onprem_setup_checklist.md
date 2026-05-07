# 폐쇄망 Bin 불량 분석 시스템 — 사내 셋업 체크리스트

## 사전 준비 (인터넷 PC에서)

- [ ] git clone 또는 zip 다운로드
- [ ] ONNX 임베딩 모델 변환 및 복사
  ```bash
  pip install optimum[onnxruntime]
  optimum-cli export onnx --model BAAI/bge-m3 ./models/bge-m3-onnx
  ```
- [ ] `models/bge-m3-onnx/` 폴더를 USB로 폐쇄망 PC에 복사

---

## Step 1: 환경 설치 (폐쇄망 PC)

```powershell
cd light-rag-test

# 가상환경 생성 (Python 3.10)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 오프라인 설치
pip install --no-index --find-links=wheels_cp310/ -r requirements_onprem.txt
```

- [ ] 가상환경 생성 완료
- [ ] pip install 오류 없이 완료
- [ ] `models/bge-m3-onnx/` 폴더에 model.onnx, tokenizer.json 등 파일 있는지 확인

---

## Step 2: .env.onprem 설정

`.env.onprem` 파일을 열어 아래 값들을 채우세요 (기존 LLM/Reranker 설정은 이미 있음):

```ini
# --- MSSQL (분석 시스템용) --- 아래 4줄 추가
MSSQL_HOST=10.x.x.x           # 실제 MSSQL 서버 IP
MSSQL_DATABASE=PACKAGE_DB      # 실제 데이터베이스명
MSSQL_USER=분석용_계정          # DB 사용자명
MSSQL_PASSWORD=비밀번호         # DB 비밀번호
```

- [ ] MSSQL 접속 정보 4개 입력 완료
- [ ] 기존 LLM 관련 설정 (ONPREM_LLM_CREDENTIAL_KEY 등) 확인

---

## Step 3: 테이블 스키마 작성 (가장 중요!)

`docs/domain/table_schemas/package_tables.txt` 파일을 열어 실제 테이블 정보를 작성하세요.

### 최소 필요 테이블 (Bin 불량 분석 기준)

#### 1) Bin 테스트 결과 테이블
Step 1 (설비별 불량률 개요), Step 2 (시계열 추이)에서 사용.
```
### 테이블: [실제_테이블명]
용도: Bin별 최종 테스트 결과

| 컬럼명 | 타입 | 설명 |
|--------|------|------|
| ???    | VARCHAR | Lot 식별자 |
| ???    | VARCHAR | 설비 식별자 |
| ???    | VARCHAR | 챔버/위치 식별자 |
| ???    | INT     | Bin 코드 (1=양품, 2~=불량유형) |
| ???    | DATETIME| 테스트 일시 |
| ???    | VARCHAR | 공정 코드 |
```

#### 2) Map 데이터 테이블
Step 3 (Wafer Map / Frame Map 패턴)에서 사용.
```
### 테이블: [실제_테이블명]
용도: 좌표별 Pass/Fail 결과

| 컬럼명 | 타입 | 설명 |
|--------|------|------|
| ???    | VARCHAR | Lot 식별자 |
| ???    | INT     | X좌표 |
| ???    | INT     | Y좌표 |
| ???    | INT     | Bin 코드 |
| ???    | VARCHAR | 설비 식별자 |
```

#### 3) 정밀 검증용 테이블
Step 4 (정밀 Bin 불량률 재계산)에서 사용.
```
### 테이블: [실제_테이블명]
용도: 교차 검증용 상세 데이터

| 컬럼명 | 타입 | 설명 |
| ...
```

#### 4) 테이블 간 조인 관계
```
- [테이블1] ↔ [테이블2]: ???_ID로 조인
- [테이블2] ↔ [테이블3]: ???_ID + ???_ID로 조인
```

- [ ] Bin 테스트 결과 테이블 작성 완료
- [ ] Map 데이터 테이블 작성 완료
- [ ] 정밀 검증용 테이블 작성 완료
- [ ] 조인 관계 작성 완료

---

## Step 4: config.yaml 테이블 화이트리스트 등록

`config.yaml`을 열어 Step 3에서 작성한 테이블명을 등록하세요:

```yaml
safety:
  # ...
  table_whitelist: [테이블명1, 테이블명2, 테이블명3]  # ← 여기에 실제 테이블명 입력
```

- [ ] table_whitelist에 허용 테이블명 등록 완료

---

## Step 5: 연결 테스트

```powershell
# DB 연결 테스트
python -c "
from dotenv import load_dotenv; load_dotenv('.env.onprem')
from src.config import load_config
from src.db_client import DatabaseClient
cfg = load_config('config.yaml')
db = DatabaseClient(cfg.database.host, cfg.database.port, cfg.database.database, cfg.database.username, cfg.database.password)
print('DB:', 'OK' if db.test_connection() else 'FAIL')
"
```

```powershell
# 단위 테스트 전체 실행
python -m pytest tests/ -v
```

- [ ] DB 연결 테스트 OK
- [ ] 단위 테스트 14개 PASS

---

## Step 6: 도메인 지식 인덱싱 + 첫 질문 테스트

```powershell
python analyze.py
```

처음 실행 시 `docs/domain/` 내 txt 파일들을 LightRAG에 인덱싱합니다 (몇 분 소요).
인덱싱 완료 후 아래 예시 질문으로 테스트:

```
질문: Die Attach 공정에서 Bin3 불량이 급증했는데 원인 설비 찾아줘
```

### 예상 흐름
1. 쿼리 분석 → `{process: "Die Attach", bin_code: 3, issue: "급증"}`
2. 도메인 지식 검색 → Bin 불량 분석 플로우 + 테이블 스키마
3. 분석 계획 수립 → Step 1~4 표시
4. 진행 여부 확인 → `y` 입력
5. Step별로 SQL 표시 → 승인(`y`) 후 실행 → 결과 + 해석
6. 최종 리포트 출력

- [ ] 인덱싱 완료
- [ ] 첫 질문에 분석 계획이 생성됨
- [ ] SQL이 올바른 테이블/컬럼을 참조함
- [ ] SQL 실행 결과가 정상적으로 나옴
- [ ] 최종 리포트에 혐의 설비가 표시됨

---

## 트러블슈팅

| 증상 | 원인 / 해결 |
|------|------------|
| `No module named 'xxx'` | 가상환경 활성화 안 됨 → `.venv\Scripts\Activate.ps1` |
| `DB 연결 실패` | .env.onprem의 MSSQL_* 값 확인, 방화벽/포트 확인 |
| `model.onnx가 없습니다` | models/bge-m3-onnx/ 폴더에 모델 복사 안 됨 |
| `허용되지 않는 테이블` | config.yaml의 table_whitelist에 테이블 등록 안 됨 |
| `LLM 게이트웨이 사전 점검 실패` | ONPREM_LLM_CREDENTIAL_KEY / SEND_SYSTEM_NAME 확인 |
| SQL이 잘못된 컬럼 참조 | package_tables.txt의 스키마 정보가 부정확 → 수정 후 rag_storage_analysis/ 삭제하고 재실행 |
| 인덱싱이 너무 오래 걸림 | 사내 LLM 응답 속도에 의존. 정상적으로 5~10분 소요 |
| `JSON 파싱 실패` | 사내 LLM이 JSON 형식을 잘 못 지키는 경우. 로그 확인 후 프롬프트 조정 필요 |

---

## 검증 완료 후 다음 단계

- [ ] 분석 모델 추가: `docs/domain/analysis_flows/`에 새 txt 파일 추가 → `rag_storage_analysis/` 삭제 후 재실행
- [ ] 적용 공정 추가: `docs/domain/equipment_knowledge/`에 공정별 문서 추가
- [ ] Phase 2 전환: config.yaml에서 `execution.mode: auto`로 변경
- [ ] 웹 UI 구축: Streamlit/Gradio로 analyze.py 로직 래핑
