# 8일 POC 전략 & 보충 학습 가이드

## 현실 점검

### 가지고 있는 것
- 반도체 수율 분석 도메인 지식 (핵심 자산)
- LightRAG 기본 셋업 완료 (인터넷 PC)
- 사내 LLM/Reranker API 연동 코드 준비됨
- 폐쇄망 배포 패키지 준비됨

### 리스크
- 사내 LLM의 엔티티 추출 품질이 GPT-4o 대비 떨어질 수 있음
- 폐쇄망 환경에서 예상 못한 에러 발생 가능
- LightRAG 프레임워크 내부를 잘 모르는 상태에서 문제 해결이 느릴 수 있음

### 8일에 현실적으로 가능한 것
- 분석 모델 1개를 **제대로** POC하는 것
- 2개는 첫 번째가 순조로울 경우에만
- "동작하는 데모"를 보여주는 것이 목표, 완성품이 아님

---

## 일정 계획

### Day 1-2: 폐쇄망 환경 셋업 + 사내 LLM 검증

이것이 **가장 중요한 관문**입니다. 여기서 막히면 일정 전체가 밀립니다.

```
[Day 1 오전] 폐쇄망 환경 구축
├── zip 풀고 pip 오프라인 설치
├── .env.onprem 설정
└── lightrag_onprem_demo.py 실행 시도

[Day 1 오후] 에러 대응
├── 안 되면: 에러 메시지 기록 → 집에서 수정 → 다음날 재시도
└── 되면: Day 2 작업으로 바로 넘어감

[Day 2] 사내 LLM 품질 검증 ← 매우 중요
├── sample_doc.txt로 인덱싱 실행
├── 생성된 지식 그래프(.graphml) 열어서 확인
│   → 엔티티가 제대로 추출됐는가?
│   → 관계가 말이 되는가?
│   → 누락된 것은 없는가?
├── 5가지 모드 쿼리 실행 → 답변 품질 확인
└── GPT-4o 결과와 비교 (집에서 돌린 것 vs 사내 LLM)
```

**Day 2의 판단 기준:**

| 사내 LLM 결과 | 다음 행동 |
|-------------|----------|
| 엔티티/관계 추출이 괜찮음 | → Day 3으로 진행 |
| 추출은 되는데 품질이 낮음 | → 문서 형식을 더 단순하게 수정 |
| 추출이 거의 안 됨 | → 사내 LLM 모델 변경 검토 또는 프롬프트 튜닝 |

### Day 3-4: 실제 분석 모델 1개 문서 작성

**POC 대상 선택 기준:** 가장 잘 아는 분석 업무 1개를 고릅니다.

예시: "수율 특이점 발생 시 원인 설비 추적"

```
[Day 3] 도메인 문서 작성 ← 이 프로젝트에서 가장 중요한 작업
│
│  LightRAG의 성능 = 입력 문서의 품질
│  코딩이 아니라 "글쓰기"가 핵심입니다
│
├── 문서 1: 분석 프로세스 전체 흐름
│   "수율 특이점이 감지되면 → Step 1 → Step 2 → ... → 결론"
│
├── 문서 2: 사용하는 테이블/컬럼 상세
│   "LOT_HISTORY 테이블: LOT_ID(PK), EQP_ID, STEP_ID, YIELD, ..."
│   "EQP_SENSOR_DATA 테이블: ..."
│   "조인 조건: LOT_HISTORY.LOT_ID = EQP_SENSOR_DATA.LOT_ID"
│
├── 문서 3: 통계 기법 설명
│   "Welch t-test: 두 그룹의 평균 차이 검정, p-value < 0.05이면 유의미"
│   "Commonality Analysis: 불량 LOT의 공통 설비/챔버를 찾는 방법"
│
└── 문서 4: 실제 판단 사례
    "2024년 3월 ETCH 3호기 사례: Temperature 센서 이상 → Chamber Clean으로 해결"

[Day 4] 인덱싱 + 그래프 검증 + 문서 수정
├── 작성한 문서를 LightRAG에 투입
├── 생성된 그래프 확인
│   → 기대한 엔티티가 다 나왔는가?
│   → 테이블 간 관계가 올바른가?
│   → 분석 프로세스의 순서가 반영됐는가?
├── 부족한 부분 → 문서 수정 → 재인덱싱
└── 이 과정을 2-3회 반복 (문서 ↔ 그래프 핑퐁)
```

### Day 5-6: 쿼리 테스트 + 데모 시나리오 구성

```
[Day 5] 다양한 질문으로 테스트
├── 사실 확인 질문: "LOT_HISTORY 테이블의 컬럼은?"
├── 프로세스 질문: "수율 특이점 분석 절차를 단계별로 알려줘"
├── 관계 추론 질문: "ETCH 설비에서 수율이 떨어지면 어떤 센서를 봐야 해?"
├── 의사결정 질문: "Temperature 이상이 감지되면 어떤 정비를 해야 해?"
└── 각 질문별로 5가지 모드 결과 비교 → 어떤 모드가 가장 좋은지 판단

[Day 6] 데모 시나리오 정리
├── "이런 질문을 하면 이런 답이 나옵니다" 시나리오 3-5개
├── 각 시나리오별 스크린샷 또는 결과 저장
├── naive vs hybrid vs mix 비교 → "지식 그래프가 있으니 이렇게 다릅니다" 증명
└── (시간이 남으면) 두 번째 분석 모델 문서 작성 시작
```

### Day 7: 발표/보고 자료 준비

```
├── POC 결과 정리
│   ├── 무엇을 했는가 (입력 문서 → 지식 그래프 → 쿼리 답변)
│   ├── 결과가 어떠한가 (모드별 비교, 답변 품질)
│   ├── 한계는 무엇인가 (솔직하게)
│   └── 다음 단계는 무엇인가 (정비 이력 연결, 장기 비전)
│
└── 장기 로드맵 한 장
    분석 온톨로지 → 정비 이력 → AI 판단 → 휴머노이드 액션
```

### Day 8: 버퍼 + 두 번째 모델

```
├── Day 1-7에서 밀린 작업 처리
├── 예상 못한 이슈 대응
├── (여유 있으면) 두 번째 분석 모델 POC
└── 최종 코드/문서 정리 + git push
```

---

## 문서 작성 가이드 (가장 중요)

LightRAG에서 코딩보다 중요한 것이 **입력 문서의 품질**입니다.
문서를 어떻게 쓰느냐가 지식 그래프의 품질을 결정합니다.

### 좋은 문서 작성 원칙

#### 1. 엔티티를 명확하게 쓴다

```
나쁜 예: "데이터를 조회해서 분석합니다"
좋은 예: "LOT_HISTORY 테이블에서 YIELD 컬럼을 조회하여 Welch t-test를 수행합니다"
```

LightRAG가 추출할 수 있는 것: LOT_HISTORY, YIELD, Welch t-test
"데이터", "분석"에서는 의미있는 엔티티를 추출할 수 없습니다.

#### 2. 관계를 명시적으로 쓴다

```
나쁜 예: "여러 테이블을 연결해서 봅니다"
좋은 예: "LOT_HISTORY와 EQP_SENSOR_DATA를 LOT_ID 컬럼으로 조인합니다"
```

#### 3. 프로세스는 순서를 매긴다

```
나쁜 예: "데이터를 보고 원인을 찾고 조치합니다"
좋은 예:
  Step 1: LOT_HISTORY에서 최근 7일간 YIELD < (평균 - 3σ)인 LOT을 추출한다.
  Step 2: 해당 LOT의 EQP_ID로 EQP_SENSOR_DATA를 조인한다.
  Step 3: 정상 LOT과 불량 LOT의 센서값을 Welch t-test로 비교한다.
  Step 4: p-value < 0.05인 센서를 원인 후보로 선정한다.
  Step 5: 원인 후보 센서의 시계열 트렌드를 확인한다.
```

#### 4. 하나의 문서에 하나의 주제

```
나쁜 예: 한 파일에 모든 분석 방법 + 테이블 구조 + 설비 정보를 다 넣음
좋은 예:
  analysis_process.txt    → 분석 프로세스 흐름
  table_schema.txt        → 테이블/컬럼 상세
  statistical_methods.txt → 통계 기법 설명
  equipment_info.txt      → 설비/챔버 정보
  case_studies.txt        → 실제 분석 사례
```

#### 5. 약어는 풀어쓴다

```
나쁜 예: "CA로 EQP 매칭 확인"
좋은 예: "Commonality Analysis(CA)를 통해 설비(Equipment, EQP) 매칭을 확인한다"
```

LLM이 약어를 모를 수 있고, 같은 약어가 다른 의미로 쓰이면 혼란이 생깁니다.

### 문서 템플릿

```markdown
# [분석 모델명]

## 개요
[이 분석이 무엇이고 언제 사용하는지 2-3줄]

## 트리거 조건
[이 분석을 시작하는 조건]
예: "일간 수율 모니터링에서 특정 공정의 수율이 3시그마 이상 하락했을 때"

## 사용 테이블
| 테이블명 | 주요 컬럼 | 역할 |
|---------|----------|------|
| LOT_HISTORY | LOT_ID, EQP_ID, STEP_ID, YIELD | LOT별 수율 및 경유 설비 |
| EQP_SENSOR_DATA | LOT_ID, SENSOR_ID, VALUE, TIME | 설비 센서 측정값 |

## 테이블 조인 관계
- LOT_HISTORY.LOT_ID = EQP_SENSOR_DATA.LOT_ID
- LOT_HISTORY.STEP_SEQ = EQP_SENSOR_DATA.STEP_SEQ

## 분석 절차
Step 1: [구체적 행동]
Step 2: [구체적 행동]
...

## 사용하는 통계 기법
- Welch t-test: [언제, 왜 사용하는지]
- Commonality Analysis: [언제, 왜 사용하는지]

## 판단 기준
- p-value < 0.05이면 해당 센서를 원인 후보로 선정
- Commonality ratio > 80%이면 해당 설비를 주요 원인으로 판단

## 과거 사례
[실제 분석했던 사례 1-2개, 구체적으로]
```

---

## 보충 학습 가이드

8일 안에 모든 것을 공부할 수 없습니다.
POC에 **직접 필요한 것만** 우선순위를 매겼습니다.

### 반드시 알아야 하는 것 (Day 1-2에 익히기)

#### 1. LightRAG의 .graphml 파일 열어보기

인덱싱 후 생성되는 `graph_chunk_entity_relation.graphml`을 열어봐야
지식 그래프가 제대로 만들어졌는지 판단할 수 있습니다.

```
방법 1: Gephi (무료, 추천)
  - https://gephi.org 에서 다운로드
  - 파일 열기 → .graphml 선택
  - 노드 = 엔티티, 엣지 = 관계
  - 노드 크기 = 연결이 많을수록 큼

방법 2: Python으로 간단히 확인
  import networkx as nx
  G = nx.read_graphml("rag_storage/graph_chunk_entity_relation.graphml")
  print(f"엔티티 수: {G.number_of_nodes()}")
  print(f"관계 수: {G.number_of_edges()}")
  for node in list(G.nodes)[:20]:
      print(f"  - {node}")

방법 3: VS Code 확장
  - "GraphML Viewer" 확장 설치
```

#### 2. LightRAG에 여러 문서 넣는 방법

현재 코드는 `sample_doc.txt` 하나만 넣고 있지만,
실제로는 여러 문서를 넣어야 합니다.

```python
# 방법 1: 한 파일씩
await rag.ainsert(open("analysis_process.txt").read())
await rag.ainsert(open("table_schema.txt").read())
await rag.ainsert(open("case_studies.txt").read())

# 방법 2: 폴더 안의 모든 .txt 파일
from pathlib import Path
for doc_path in Path("./docs_input").glob("*.txt"):
    text = doc_path.read_text(encoding="utf-8")
    await rag.ainsert(text)
```

#### 3. 인덱싱 결과 보존하기

매번 `shutil.rmtree(WORKING_DIR)`로 지우고 있는데,
문서를 수정할 때마다 전체 재인덱싱하면 시간과 비용이 낭비됩니다.

```python
# lightrag_onprem_demo.py에서 이 부분을 주석 처리:
# if WORKING_DIR.exists():
#     shutil.rmtree(WORKING_DIR)

# 그러면 기존 인덱스를 유지한 채 새 문서만 추가됩니다.
# 단, 문서 내용을 수정했으면 지우고 다시 해야 합니다.
```

### 알면 좋지만 급하지 않은 것

#### 4. Python async/await 기초

LightRAG는 비동기(async) 함수를 사용합니다.
지금은 "이 패턴이 이런 뜻이구나" 정도만 이해하면 됩니다.

```python
# async = "이 함수는 비동기입니다" 선언
# await = "이 작업이 끝날 때까지 기다립니다"

async def main():
    rag = LightRAG(...)           # 일반 함수 → 그냥 호출
    await rag.ainsert(text)       # 비동기 함수 → await 붙여서 호출
    result = await rag.aquery(q)  # 비동기 함수 → await 붙여서 호출

asyncio.run(main())  # 비동기 함수를 실행하는 시작점
```

규칙: `a`로 시작하는 함수(ainsert, aquery)는 `await`를 붙인다.

#### 5. 환경변수와 .env 파일

```python
# .env 파일:
ONPREM_LLM_CREDENTIAL_KEY=my_secret_key

# Python에서 읽기:
import os
key = os.getenv("ONPREM_LLM_CREDENTIAL_KEY")  # → "my_secret_key"

# dotenv가 .env 파일을 환경변수로 로드해줌:
from dotenv import load_dotenv
load_dotenv(".env.onprem")  # .env.onprem 파일의 내용을 환경변수로 설정
```

### POC 이후에 공부할 것 (지금은 안 해도 됨)

| 주제 | 왜 필요한지 | 언제 |
|------|-----------|------|
| NetworkX | 지식 그래프를 코드로 분석/수정 | 그래프 품질 개선 시 |
| LangChain Agent | LightRAG + SQL 실행 + 통계 엔진 결합 | 하이브리드 시스템 구축 시 |
| Neo4j | 대규모 지식 그래프 저장/쿼리 | 프로덕션 전환 시 |
| Prompt Engineering | LLM 엔티티 추출 품질 개선 | 사내 LLM 품질이 낮을 때 |
| Streamlit | 간단한 웹 데모 UI | 데모/발표 시 |

---

## 위기 대응 시나리오

### "폐쇄망에서 pip install이 안 됩니다"

```
1차: wheels 폴더 경로 확인, Python 버전 확인
2차: Docker .tar 방식으로 전환
3차: 집에서 에러 메시지 보고 수정 → 다음날 재시도
```

### "사내 LLM이 엔티티를 제대로 못 뽑습니다"

```
1차: 문서를 더 단순하고 명시적으로 수정
     (긴 문장 → 짧은 문장, 표 형식 활용)
2차: 사내 LLM 모델 변경 가능한지 확인
     (gpt-oss-120b → 더 큰 모델?)
3차: 인덱싱은 집(인터넷 PC)에서 GPT-4o로 하고,
     쿼리만 사내 LLM으로 하는 "하이브리드" 전략
     → rag_storage 폴더를 통째로 폐쇄망에 복사
```

### "인덱싱에 너무 오래 걸립니다"

```
1차: 문서 크기를 줄인다 (핵심만 남김)
2차: 한 번 인덱싱하고 rag_storage를 보존한다
     (매번 지우는 코드 주석 처리)
```

### "8일 안에 2개 모델이 안 될 것 같습니다"

```
1개 모델을 확실하게 하는 것이 2개를 대충 하는 것보다 낫습니다.

POC의 목표는 "LightRAG + 사내 LLM으로 도메인 지식을
구조화하고 활용할 수 있다"는 것을 증명하는 것입니다.
1개 모델로도 충분히 증명할 수 있습니다.
```

---

## 체크리스트

### Day 1-2
- [ ] 폐쇄망 pip 설치 성공
- [ ] lightrag_onprem_demo.py 실행 성공
- [ ] 사내 LLM으로 sample_doc.txt 인덱싱 성공
- [ ] .graphml 파일 열어서 그래프 확인
- [ ] 5가지 모드 쿼리 결과 확인

### Day 3-4
- [ ] POC 대상 분석 모델 1개 선정
- [ ] 도메인 문서 3개 이상 작성 (프로세스, 테이블, 통계기법)
- [ ] 문서 투입 후 그래프 확인
- [ ] 문서 수정 → 재인덱싱 1-2회 반복

### Day 5-6
- [ ] 테스트 질문 10개 이상 실행
- [ ] 모드별 답변 품질 비교
- [ ] 데모 시나리오 3-5개 정리

### Day 7
- [ ] 결과 보고서/발표 자료 작성

### Day 8
- [ ] 버퍼/마무리
- [ ] 코드/문서 최종 정리 + git push
