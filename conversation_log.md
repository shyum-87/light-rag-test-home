# LightRAG 테스트 대화 로그

> 날짜: 2026-04-21
> 목적: LightRAG를 활용한 반도체 FAB 수율 분석 온톨로지 구축 테스트

---

## 1. 초기 셋업

### 환경 구성
- Python 가상환경(`.venv`) 생성
- `requirements.txt` 의존성 설치 (lightrag-hku 1.4.15, openai, ollama 등)
- OpenRouter API 연동 설정 (`.env`에 API 키 + base URL 설정)
- `lightrag_demo.py`를 OpenRouter 호환으로 수정 (모델명: `openai/gpt-4o-mini`)

### 첫 번째 실행 (양자 컴퓨팅 샘플)
기본 제공된 양자 컴퓨팅 한국어 문서로 5가지 쿼리 모드(naive, local, global, hybrid, mix) 테스트 성공.

---

## 2. 반도체 수율 분석 샘플 데이터 구축

### 목표
엔지니어의 데이터 분석 업무를 LightRAG로 온톨로지화하여, "수율 특이점 분석해줘" 같은 쿼리에 대해 테이블 정보, 분석 프로세스, 통계 기법 등을 정확히 반환하는 시스템 구축.

### 작성한 샘플 데이터 (`sample_doc.txt`) 구성
1. **수율 분석 업무 개요** - FAB 엔지니어의 역할과 목표
2. **분석 프로세스 6단계**
   - Step 1: 수율 특이점 탐지 (SPC, WAT, Bin 분포)
   - Step 2: 불량 Lot 그룹핑 (Bad/Good Lot 분류)
   - Step 3: 공정 이력 추적 (설비/챔버 매핑)
   - Step 4: 통계적 검증 (Chi-Square, t-Test, Kruskal-Wallis)
   - Step 5: 웨이퍼 맵 분석 (Edge, Center, Scratch 등 패턴)
   - Step 6: 원인 설비 확정 및 PM 조치
3. **핵심 데이터 테이블 8개** (컬럼 정보 포함)
   - YIELD_SUMMARY, LOT_HISTORY, EQUIPMENT_MASTER, WAT_DATA
   - WAFER_MAP, DEFECT_DATA, EQP_SENSOR_DATA, PM_HISTORY
4. **분석 로직 상세**
   - 설비 기여도 분석, Commonality Analysis, Multi-Step Correlation, 시계열 분석
5. **주요 공정 단계** - Photo, Etch, Deposition, CMP, Ion Implantation, Diffusion, Cleaning
6. **실전 시나리오 3개**
   - ETCH 챔버 Edge Ring 마모
   - Photo-Etch 상호작용
   - PM 주기 관련 수율 열화

---

## 3. 테스트 쿼리 및 결과

### 질문 1: 수율 특이점 원인 설비 추적 프로세스 (5가지 모드 비교)
> "수율 특이점이 발생했을 때 원인 설비를 찾아내는 분석 프로세스와 사용하는 테이블, 통계 기법을 단계별로 설명해줘."

**결과**: 5가지 모드 모두 6단계 프로세스를 정확히 설명. 각 단계별 사용 테이블(YIELD_SUMMARY, LOT_HISTORY 등)과 통계 기법(Chi-Square, t-Test 등)을 연결하여 응답.

### 질문 2: ETCH 설비 챔버 원인 분석 (hybrid)
> "ETCH 설비의 특정 챔버에서 수율이 떨어질 때, 어떤 테이블을 조인해서 어떤 통계 검정으로 원인을 확정하는지 구체적으로 알려줘."

**결과**: LOT_HISTORY와 YIELD_SUMMARY를 LOT_ID로 조인, Chi-Square/t-Test로 검정하는 구체적 방법 응답.

### 질문 3: 테이블 스키마 및 조인 관계 (local)
> "LOT_HISTORY 테이블과 EQP_SENSOR_DATA 테이블의 컬럼 정보와 조인 관계를 알려줘."

**결과**: 두 테이블의 전체 컬럼 정보(타입, 설명 포함)와 EQUIPMENT_ID + LOT_ID + WAFER_ID 조인키를 정확히 반환.

### 질문 4: 웨이퍼 맵 패턴 + Commonality 분석 (mix)
> "웨이퍼 맵에서 Edge 불량 패턴이 나타났을 때, 어떤 공정과 설비를 의심해야 하며 Commonality Analysis는 어떻게 수행하나?"

**결과**: 코팅/식각/CMP를 의심 공정으로 지목, Commonality Rate 계산식(Bad/Good Lot Commonality, Delta > 30% 기준) 정확 응답.

---

## 4. 저장 구조 분석

### rag_storage/ 디렉토리 구조

#### Vector RAG (벡터 검색) - `naive` 모드에서 사용
| 파일 | 역할 | 내용 |
|------|------|------|
| `vdb_chunks.json` | 텍스트 청크 벡터 DB | 원문 6개 청크의 1536차원 임베딩 |
| `kv_store_text_chunks.json` | 청크 원문 저장소 | 청크 6개의 실제 텍스트 |
| `kv_store_full_docs.json` | 원본 문서 저장소 | 인덱싱된 전체 문서 원문 |

#### Graph RAG (지식 그래프) - `local`, `global`, `hybrid`, `mix` 모드에서 사용
| 파일 | 역할 | 내용 |
|------|------|------|
| `graph_chunk_entity_relation.graphml` | 지식 그래프 본체 | 161개 엔티티(Node), 74개 관계(Edge) |
| `vdb_entities.json` | 엔티티 벡터 DB | 161개 엔티티의 1536차원 임베딩 |
| `vdb_relationships.json` | 관계 벡터 DB | 74개 관계의 1536차원 임베딩 |
| `kv_store_full_entities.json` | 엔티티 메타데이터 | 엔티티 원문 설명 |
| `kv_store_full_relations.json` | 관계 메타데이터 | 관계 원문 설명 |

#### 추출된 지식 그래프 예시
```
[엔티티 유형별]
  person:       Yield Analysis Engineer
  organization: Semiconductor FAB
  concept:      Excursion, Root Cause Equipment
  data:         YIELD_SUMMARY, LOT_HISTORY, EQUIPMENT_MASTER, WAFER_MAP, WAT_DATA, ...
  method:       Chi-Square Test, t-Test, Kruskal-Wallis, SPC, ...

[관계 예시]
  Yield Analysis Engineer --사용--> YIELD_SUMMARY
  Yield Analysis Engineer --수행--> Chi-Square Test
  YIELD_SUMMARY --LOT_ID로 조인--> LOT_HISTORY
  YIELD_SUMMARY --LOT_ID로 조인--> WAT_DATA
```

#### 쿼리 모드별 데이터 소스
| 모드 | Vector DB | Knowledge Graph |
|------|-----------|----------------|
| `naive` | vdb_chunks (텍스트 유사도 검색) | 사용 안 함 |
| `local` | vdb_entities (엔티티 검색) | 해당 엔티티의 이웃 관계 탐색 |
| `global` | 사용 안 함 | 그래프 커뮤니티 요약 |
| `hybrid` | vdb_entities | local + global 결합 |
| `mix` | vdb_chunks + vdb_entities | KG + Vector 모두 사용 |

---

## 5. 모드별 특성 요약

| 모드 | 강점 | 적합한 질문 유형 |
|------|------|-----------------|
| `naive` | 단순 벡터 검색 (베이스라인) | 비교 기준 |
| `local` | 특정 엔티티 중심 사실 조회 | "LOT_HISTORY 테이블 컬럼 알려줘" |
| `global` | 전체 프로세스/테마 요약 | "수율 분석 전체 흐름 설명해줘" |
| `hybrid` | local + global 결합 | 일반적 용도 (가장 실용적) |
| `mix` | KG + Vector 모두 활용 | 최고 품질 (토큰 많이 사용) |

---

## 6. 기술 스택

- **LightRAG**: v1.4.15 (lightrag-hku)
- **LLM**: OpenAI GPT-4o-mini (OpenRouter 경유)
- **Embedding**: OpenAI text-embedding-3-small (1536차원)
- **Storage**: 로컬 파일 기반 (JSON + GraphML)
- **Python**: 3.14
