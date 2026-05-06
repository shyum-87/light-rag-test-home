# Bin 불량 분석 시스템 설계 (LightRAG + LLM Agent)

## 개요

반도체 패키지 공정에서 Bin 불량률 기반 원인 설비 추적을 자동화하는 시스템.
LightRAG로 도메인 지식을 검색하고, LLM Agent가 단계별로 SQL 생성/실행/해석을 수행한다.

## 요구사항

- **환경:** 폐쇄망, Python 3.10, MSSQL, 사내 LLM API, ONNX 임베딩
- **첫 POC:** 공정 1개, 분석 모델 1개 (Bin 불량률 기반 원인 설비 추적)
- **자동화 단계:** Phase 1 사용자 승인 후 실행 → Phase 2 완전 자동
- **인터페이스:** CLI로 테스트 → 최종 웹 UI (Streamlit/Gradio)
- **보안:** SELECT-only 코드 제어, 테이블 화이트리스트

## 아키텍처

```
사용자 쿼리
    |
    v
[Query Analyzer] --- 쿼리에서 공정/Bin/키워드 추출
    |
    v
[LightRAG] --- 도메인 지식 검색 (분석 플로우, 테이블 스키마, 판단 기준)
    |
    v
[Plan Generator] --- 분석 계획 수립 (단계별 할 일 리스트)
    |
    v
[Step Executor] --- 단계별 실행 루프
    |   SQL 생성 (LLM)
    |   SQL 검증 (SELECT-only, 화이트리스트)
    |   사용자 승인 (Phase 1) 또는 자동 실행 (Phase 2)
    |   MSSQL 실행
    |   통계 검정 (scipy)
    |   결과 해석 (LLM)
    |   다음 단계 판단
    |
    v
[Report Generator] --- 혐의 설비 + 예상 원인 + 근거 리포트
```

### 구성요소

| 구성요소 | 역할 | 의존성 |
|---------|------|--------|
| Query Analyzer | 자연어에서 공정/Bin/설비 등 파라미터 추출 | LLM |
| LightRAG | 도메인 지식 검색 (분석 절차, 테이블 스키마, 판단 기준) | 기존 셋업 |
| Plan Generator | 검색된 지식 바탕으로 단계별 분석 계획 생성 | LLM + LightRAG 결과 |
| Step Executor | SQL 생성 -> 검증 -> 실행 -> 해석 루프 | LLM + MSSQL |
| Report Generator | 전체 결과를 종합하여 혐의 설비/원인 리포트 | LLM |

## 분석 플로우 (Bin 불량 분석 모델)

### Step 1: 설비별 Bin 불량률 개요

- 목적: 전체 설비 중 특이 설비 선별
- 행동: SQL로 설비+챔버별 Bin 불량률 집계
- 특이 판단: 평균 + 3sigma 초과
- 출력: 특이 설비 리스트

### Step 2: 특이 설비 시계열 확인

- 목적: 불량 추이 확인 (갑작스런 변화 vs 점진적 악화)
- 입력: Step 1의 특이 설비 ID
- 행동: 일별/주별 Bin 불량률 추이 SQL
- 출력: 혐의 설비 + 시점 범위 확정

### Step 3: Map 패턴 확인

- 목적: 불량의 공간적 분포 패턴 파악
- 입력: 혐의 설비 + 시점 범위
- 행동: Wafer Map / Frame Map 데이터 SQL
- 패턴 유형: Edge / Center / Random / Cluster
- 출력: 패턴 유형 + 혐의 범위 축소

### Step 4: 정밀 Bin 불량률 재계산

- 목적: 별도 테이블로 교차 검증
- 행동: 정밀 불량률 SQL + 통계 검정 (카이제곱, p-value < 0.05)
- 출력: 확정된 혐의 설비, 불량률, p-value

### Step 5: 최종 리포트

- 혐의 설비 (설비ID + 챔버ID)
- 불량률 수치 (전체 평균 대비 N sigma)
- 통계 유의성 (p-value)
- Map 패턴 기반 예상 원인
- 권장 조치

## LLM vs 코드 역할 분담

| 작업 | LLM | Python 코드 |
|------|-----|------------|
| SQL 생성 | O | X |
| SQL 안전 검증 | X | O |
| SQL 실행 | X | O (pymssql) |
| 결과 해석 | O | X |
| 통계 검정 | X | O (scipy.stats) |
| Map 패턴 판단 | O (좌표 집계 결과 기반) | 보조 (좌표 분포 계산) |
| 다음 Step 결정 | O | X |

핵심 원칙: 숫자 계산은 코드가, 판단/해석은 LLM이 한다.

## SQL 안전장치

### 검증 순서

1. 허용 키워드: SELECT, WITH만 허용. INSERT/UPDATE/DELETE/DROP/ALTER/TRUNCATE/EXEC 거부.
2. 테이블 화이트리스트: 사전 등록된 테이블만 조회 가능.
3. 결과 제한: TOP/LIMIT 없으면 자동으로 TOP 10000 추가.

### 실행 모드

- Phase 1 (approval): SQL을 사용자에게 표시, 승인 후 실행.
- Phase 2 (auto): 검증 통과 시 자동 실행, 로그 기록.

## 프로젝트 파일 구조

```
light-rag-test/
  analyze.py                          # CLI 진입점
  config.yaml                         # DB/안전장치/실행모드 설정
  src/
    __init__.py
    query_analyzer.py                  # 쿼리 파라미터 추출
    knowledge.py                       # LightRAG 래핑
    plan_generator.py                  # 분석 계획 수립
    step_executor.py                   # SQL 생성/검증/실행/해석 루프
    sql_safety.py                      # SQL 안전 검증
    db_client.py                       # MSSQL 연결/실행
    stats.py                           # 통계 검정
    report.py                          # 리포트 생성
  docs/domain/
    analysis_flows/
      bin_defect_analysis.txt          # Bin 불량 분석 플로우
    table_schemas/
      package_tables.txt               # 테이블 스키마 (별도 제공)
    judgment_criteria/
      statistical_thresholds.txt       # 통계 판단 기준
    equipment_knowledge/
      da_process.txt                   # 공정별 설비 지식
  rag_storage_analysis/                # LightRAG 인덱스 (gitignore)
  logs/                                # SQL 실행 로그 (gitignore)
```

## 도메인 지식 문서 작성 가이드

LightRAG에 인덱싱할 txt 파일 형식. 문서 품질이 분석 품질을 결정한다.

### 분석 플로우 문서 (analysis_flows/*.txt)

각 분석 모델마다 1개 파일. 포함할 내용:
- 적용 조건 (언제 이 분석을 쓰는가)
- Step별 목적, 조회 테이블, 조인 조건, 집계 방식, 판단 기준
- 최종 리포트 포맷

### 테이블 스키마 문서 (table_schemas/*.txt)

테이블마다 포함할 내용:
- 테이블명, 용도
- 컬럼명, 타입, 설명
- 테이블 간 조인 관계 (어떤 키로 연결)

### 통계 판단 기준 문서 (judgment_criteria/*.txt)

- sigma 기준 (몇 sigma 이상이면 특이)
- p-value 기준 (0.05 미만이면 유의)
- 불량률 임계값 등

### 설비 지식 문서 (equipment_knowledge/*.txt)

- 공정별 설비 종류, 챔버 구조
- Map 패턴별 의미와 원인 후보
- 과거 유사 사례

## 설정 파일 (config.yaml)

```yaml
database:
  driver: mssql
  host: ${MSSQL_HOST}
  port: 1433
  database: ${MSSQL_DATABASE}
  username: ${MSSQL_USER}
  password: ${MSSQL_PASSWORD}

safety:
  allowed_statements: [SELECT, WITH]
  blocked_keywords: [INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, EXEC]
  max_rows: 10000
  table_whitelist_file: ./docs/domain/table_schemas/package_tables.txt

execution:
  mode: approval
  log_all_sql: true
  log_path: ./logs/sql_execution.log

lightrag:
  working_dir: ./rag_storage_analysis
  domain_docs_dir: ./docs/domain
  embed_model_path: ./models/bge-m3-onnx
```

## 확장 계획

첫 POC 검증 후:
- 분석 모델 추가: docs/domain/analysis_flows/에 txt 파일 추가 + 재인덱싱
- 적용 공정 추가: docs/domain/equipment_knowledge/에 공정별 문서 추가
- Phase 2 전환: config.yaml에서 execution.mode를 auto로 변경
- 웹 UI: Streamlit/Gradio로 analyze.py의 로직을 웹으로 래핑
