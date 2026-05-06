# LightRAG 테스트 샘플

HKUDS의 [LightRAG](https://github.com/HKUDS/LightRAG)를 바로 돌려볼 수 있는 최소한의 한국어 샘플입니다.

## 구성 파일

| 파일 | 용도 |
|------|------|
| `sample_doc.txt` | 테스트용 한국어 문서 (반도체 수율 분석 온톨로지) |
| `lightrag_demo.py` | **OpenAI API** 기반 데모 (인터넷 환경) |
| `lightrag_onprem_demo.py` | **사내 폐쇄망** 전용 데모 (커스텀 헤더 인증 + ONNX 임베딩) |
| `lightrag_ollama_demo.py` | **Ollama 로컬 LLM** 기반 데모 (API 키 불필요) |
| `requirements.txt` | 느슨한 의존성 (인터넷 환경용) |
| `requirements_onprem.txt` | 폐쇄망 환경 버전 고정 (Python 3.10 + ONNX Runtime) |
| `requirements.lock` | 기존 버전 고정 (참고용) |
| `models/bge-m3-onnx/` | ONNX 변환된 bge-m3 모델 위치 (별도 변환/복사 필요) |
| `Dockerfile` | Docker 이미지 빌드용 |

---

# 인터넷 환경 (방법 A, B)

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
- 샘플 문서(약 2KB)는 gpt-4o-mini 기준 **$0.01 ~ $0.03** 수준.
- 쿼리 5회는 추가로 몇 센트 정도.

### 실행 시간
- 인덱싱(엔티티 추출): **30초 ~ 2분**
- 쿼리 하나당: **5 ~ 15초**

---

## 방법 B — Ollama 로컬 LLM으로 돌리기 (무료)

```bash
# 1) Ollama 설치: https://ollama.com
# 2) 모델 다운로드
ollama pull qwen2.5:7b
ollama pull bge-m3:latest

# 3) 실행
pip install -r requirements.txt
python lightrag_ollama_demo.py
```

> ⚠️ 로컬 LLM은 훨씬 느리고, 7B 이하 모델은 엔티티 추출 품질이 떨어질 수 있습니다. qwen2.5:14b 이상을 권장합니다.

---

# 폐쇄망 환경 (방법 C, D)

인터넷이 안 되는 사내 폐쇄망에서 실행하는 방법입니다.
**두 가지 방법** 중 상황에 맞는 것을 선택하세요.

| | 방법 C (zip 방식) | 방법 D (Docker 방식) |
|---|---|---|
| **난이도** | 쉬움 | 중간 (Docker/WSL 첫 설치 필요) |
| **필요한 것** | Python이 사내 PC에 설치되어 있을 것 | WSL2 + Docker 설치 |
| **장점** | 단순하고 빠름 | Python 버전 걱정 없음, 환경 완전 격리 |
| **단점** | Python 버전이 맞아야 함 | Docker 이미지 파일이 큼 (~500MB) |

---

## 방법 C — zip 파일로 옮겨서 pip 오프라인 설치

### 개념

인터넷이 되는 PC에서 모든 패키지(.whl 파일)를 미리 다운로드한 뒤,
git zip에 포함시켜 폐쇄망 PC로 옮깁니다.
폐쇄망에서는 `pip install --no-index`로 로컬 파일에서만 설치합니다.

### 이미 준비된 것

이 리포에는 Python 3.10 + Windows x64 용 wheels가 포함되어 있습니다:

| 폴더 | Python 버전 | 플랫폼 | 포함 내용 |
|------|------------|--------|-----------|
| `wheels_cp310/` | 3.10 | Windows x64 | LightRAG + ONNX Runtime (72MB) |

> ONNX Runtime으로 bge-m3를 직접 실행하여 Ollama 서빙 경로의 NaN 문제를 해결합니다. torch 불필요.

### Step 1: 인터넷 PC에서 zip 만들기

```powershell
# git 리포를 zip으로 압축 (GitHub에서 "Download ZIP"도 동일)
git archive --format=zip --output=lightrag-onprem.zip HEAD
```

또는 GitHub 웹에서 **Code > Download ZIP** 을 클릭합니다.

### Step 2: 폐쇄망 PC에서 설치

zip 파일을 USB 등으로 폐쇄망 PC에 복사한 뒤:

```powershell
# 1) zip 압축 해제
# 2) 해당 폴더로 이동
cd light-rag-test

# 3) 가상환경 생성 (Python 3.10)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 4) 오프라인 설치
pip install --no-index --find-links=wheels_cp310/ -r requirements_onprem.txt

# 5) bge-m3 ONNX 임베딩 모델 배치
#    인터넷 PC에서 변환한 models/bge-m3-onnx/ 폴더를 복사
#    (변환: pip install optimum[onnxruntime] && optimum-cli export onnx --model BAAI/bge-m3 ./models/bge-m3-onnx)

# 6) 환경 설정
copy .env.onprem.example .env.onprem
# .env.onprem 파일을 열어 아래 값들을 채우기:
#   ONPREM_LLM_CREDENTIAL_KEY=DS API HUB에서 발급받은 credential key
#   ONPREM_USER_ID=본인 KNOX ID
#   ONPREM_RERANK_CREDENTIAL_KEY=Reranker용 credential key

# 7) 실행
python lightrag_onprem_demo.py
```

### 만약 Python 버전이 다르다면?

인터넷이 되는 PC에서 해당 버전용 wheels를 다시 다운로드해야 합니다:

```powershell
# 예: Python 3.11용
pip download -r requirements_onprem.txt -d wheels_cp311/ --platform win_amd64 --python-version 3.11 --only-binary=:all: --no-deps
```

### GPU를 사용하고 싶다면?

ONNX Runtime GPU 버전을 사용할 수 있습니다:

```powershell
# 인터넷 PC에서 GPU 버전 다운로드
pip download onnxruntime-gpu --python-version 3.10 --platform win_amd64 --only-binary=:all: --no-deps -d wheels_cp310/
# 기존 CPU 버전 삭제
del wheels_cp310\onnxruntime-*-cp310-cp310-win_amd64.whl
```

### 트러블슈팅

| 증상 | 원인 / 해결 |
|------|------------|
| `No matching distribution found` | Python 버전 불일치. `python --version`으로 3.10 확인 |
| `Could not find a version that satisfies` | wheels 폴더 경로가 틀림. `--find-links=wheels_cp310/` 확인 |
| `ModuleNotFoundError` | 가상환경 활성화를 안 했거나 설치가 안 된 패키지 있음 |
| `model.onnx가 없습니다` | models/bge-m3-onnx/ 폴더에 ONNX 모델이 없음. `optimum-cli export onnx` 필요 |

---

## 방법 D — Docker 이미지로 옮기기 (WSL2 사용)

Docker는 앱과 모든 의존성을 하나의 **이미지 파일**로 묶는 기술입니다.
Python 버전, OS 차이를 걱정할 필요 없이 어디서든 동일하게 실행됩니다.

### 전체 흐름 요약

```
[인터넷 PC]                         [폐쇄망 PC]
   |                                     |
   |- Docker 설치                        |- WSL2 설치
   |- docker build (이미지 빌드)          |- Docker 설치 (오프라인)
   |- docker save (파일로 저장)           |- docker load (파일에서 불러오기)
   |- .tar 파일을 USB로 복사 -----------> |- docker run (실행)
```

### Step 1: 인터넷 PC에 Docker Desktop 설치

1. https://www.docker.com/products/docker-desktop/ 에서 **Docker Desktop for Windows** 다운로드
2. 설치 실행 — 설치 중 "Use WSL 2 instead of Hyper-V" 옵션 체크
3. 설치 후 PC 재시작
4. Docker Desktop 실행하고 로그인 (무료 계정)
5. 터미널에서 확인:
   ```powershell
   docker --version
   # Docker version 28.x.x 같은 출력이 나오면 성공
   ```

### Step 2: Docker 이미지 빌드 (인터넷 PC에서)

이 프로젝트 폴더에서:

```powershell
# 이미지 빌드 (최초 1회, 몇 분 소요)
docker build -t lightrag-onprem .

# 빌드 잘 됐는지 확인
docker images lightrag-onprem
# REPOSITORY        TAG       IMAGE ID       SIZE
# lightrag-onprem   latest    xxxxxxxxxxxx   약 500MB
```

> `docker build` 명령은 `Dockerfile`을 읽어서 Python + 모든 패키지가 설치된 리눅스 환경을 만듭니다. Docker 사용 시에는 Linux용 wheels를 별도로 준비해야 합니다 (현재 `wheels_cp310/`은 Windows 전용).

### Step 3: 이미지를 파일로 저장

```powershell
# .tar 파일로 저장 (약 500MB)
docker save -o lightrag-onprem.tar lightrag-onprem

# 파일 확인
dir lightrag-onprem.tar
```

이 `lightrag-onprem.tar` 파일을 USB에 복사합니다.

### Step 4: 폐쇄망 PC에 WSL2 설치

폐쇄망 PC가 Windows 10(2004+) 또는 Windows 11이면 WSL2를 설치할 수 있습니다.

#### 4-1. 오프라인 WSL2 설치

인터넷 PC에서 필요한 파일을 미리 다운로드합니다:

1. **WSL2 커널 업데이트 패키지** 다운로드:
   - https://aka.ms/wsl2kernel 에서 `wsl_update_x64.msi` 다운로드

2. **Ubuntu WSL 배포판** 다운로드:
   - https://aka.ms/wslubuntu2204 에서 `.appx` 파일 다운로드
   - 또는 Microsoft Store에서 "Ubuntu 22.04" 검색 후 다운로드

3. 위 파일들을 USB에 복사

폐쇄망 PC에서:

```powershell
# 1) Windows 기능 켜기 (관리자 PowerShell)
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart

# 2) PC 재시작
Restart-Computer

# 3) 재시작 후 — WSL2 커널 업데이트 설치
# wsl_update_x64.msi를 더블클릭하여 설치

# 4) WSL2를 기본 버전으로 설정
wsl --set-default-version 2

# 5) Ubuntu 설치 (다운받은 .appx 파일)
Add-AppxPackage .\Ubuntu_2204.appx

# 6) Ubuntu 실행하여 초기 설정 (사용자명/비밀번호 설정)
ubuntu
```

#### 4-2. 오프라인 Docker 설치 (WSL2 안에서)

Docker Desktop 대신 WSL2 Ubuntu 안에서 직접 Docker를 설치합니다.

인터넷 PC에서 Docker 바이너리를 다운로드합니다:

```bash
# 인터넷 PC의 WSL/Linux 또는 브라우저에서 다운로드
# https://download.docker.com/linux/static/stable/x86_64/
# 최신 docker-xx.xx.x.tgz 파일 다운로드
```

또는 **더 쉬운 방법**: 인터넷 PC에서 Docker Desktop 설치 파일을 다운로드합니다:

1. https://www.docker.com/products/docker-desktop/ 에서 설치파일 다운로드
2. 설치파일을 USB로 복사
3. 폐쇄망 PC에서 Docker Desktop 설치 (오프라인 설치 가능)

```powershell
# 폐쇄망 PC에서 Docker Desktop 설치 후 확인
docker --version
```

### Step 5: 이미지 불러오기 & 실행 (폐쇄망 PC에서)

```powershell
# 1) .tar 파일에서 이미지 불러오기
docker load -i lightrag-onprem.tar
# Loaded image: lightrag-onprem:latest

# 2) 불러온 이미지 확인
docker images lightrag-onprem

# 3) 실행 — 환경변수를 직접 전달
docker run --rm ^
  -e ONPREM_LLM_CREDENTIAL_KEY=여기에_credential_key ^
  -e ONPREM_USER_ID=여기에_KNOX_ID ^
  -e ONPREM_RERANK_CREDENTIAL_KEY=여기에_reranker_key ^
  lightrag-onprem
```

또는 `.env.onprem` 파일을 만들어서 전달:

```powershell
# .env.onprem 파일을 먼저 작성한 뒤:
docker run --rm --env-file .env.onprem lightrag-onprem
```

### Step 6: 결과 확인 & 파일 꺼내기

Docker 컨테이너 안에서 생성된 `rag_storage_onprem/` 등을 호스트 PC로 꺼내려면:

```powershell
# 볼륨 마운트로 실행 (결과가 현재 폴더에 저장됨)
docker run --rm ^
  --env-file .env.onprem ^
  -v "%cd%\output:/app/rag_storage_onprem" ^
  lightrag-onprem
```

### Docker 용어 정리 (처음 쓰는 분을 위해)

| 용어 | 설명 |
|------|------|
| **이미지 (Image)** | 앱 + 모든 의존성이 묶인 패키지. 설치 CD 같은 것 |
| **컨테이너 (Container)** | 이미지를 실행한 상태. 프로그램이 돌아가는 격리된 공간 |
| **Dockerfile** | 이미지를 만드는 레시피 파일 |
| **docker build** | Dockerfile을 읽어 이미지를 만드는 명령 |
| **docker save** | 이미지를 .tar 파일로 내보내기 |
| **docker load** | .tar 파일에서 이미지를 불러오기 |
| **docker run** | 이미지를 실행하여 컨테이너를 만드는 명령 |
| **WSL2** | Windows 안에서 Linux를 돌리는 기능. Docker가 이 위에서 동작 |

---

# 사내 API 설정 가이드

`lightrag_onprem_demo.py`는 사내 커스텀 헤더 인증 방식의 LLM API와 Reranker API를 사용합니다.

### 설정 파일

`.env.onprem.example`을 `.env.onprem`으로 복사한 뒤 아래 값을 채웁니다:

```ini
# LLM API
ONPREM_LLM_CREDENTIAL_KEY=DS API HUB에서 발급받은 credential key
ONPREM_USER_ID=본인 KNOX ID

# Reranker API (선택 — 비워두면 reranker 없이 실행)
ONPREM_RERANK_CREDENTIAL_KEY=Reranker용 credential key
```

### 사내 API 엔드포인트 변경

기본값은 stg(스테이징) 환경입니다. prod로 바꾸려면 `.env.onprem`에서:

```ini
ONPREM_LLM_BASE_URL=http://apigw.shyum.net:8000/gpt-oss/1/gpt-oss-120b/v1
ONPREM_RERANK_BASE_URL=http://apigw.shyum.net:8000/reranker/1/v2/rerank
```

---

# 실행 결과

## 지식 그래프 저장소

`rag_storage/` (또는 `rag_storage_onprem/`) 디렉터리에 아래 파일들이 생성됩니다:

```
rag_storage/
├─ graph_chunk_entity_relation.graphml   ← 지식 그래프 (Gephi/Cytoscape로 열람 가능)
├─ kv_store_full_docs.json
├─ kv_store_text_chunks.json
├─ vdb_chunks.json                       ← 청크 벡터
├─ vdb_entities.json                     ← 엔티티 벡터
└─ vdb_relationships.json                ← 관계 벡터
```

`.graphml` 파일을 [Gephi](https://gephi.org) 같은 그래프 뷰어로 열면 추출된 엔티티/관계를 시각적으로 볼 수 있습니다.

## 5가지 쿼리 모드 비교

| 모드 | 강점 | 주로 쓸 상황 |
|------|------|--------------|
| `naive` | 단순 벡터 검색 (베이스라인) | 비교 기준 |
| `local` | 특정 엔티티 중심 사실 | "X는 누구/무엇?" |
| `global` | 테마/커뮤니티 요약 | "전반적으로...", "비교해줘" |
| `hybrid` | local + global 결합 | 일반적 용도 |
| `mix` | KG + vector 모두 | 최고 품질 (토큰 많이 씀) |

---

# 다음으로 해볼 만한 것

1. **다른 문서로 교체** — `sample_doc.txt`를 본인의 문서(사내 정책, 논문, 책 등)로 교체
2. **WebUI로 시각화** — `pip install "lightrag-hku[api]"` 후 `lightrag-server` 실행
3. **스토리지 백엔드 교체** — 프로덕션은 Neo4j + PostgreSQL 조합을 많이 씁니다
4. **스트리밍 응답** — `QueryParam(stream=True)`으로 토큰 단위 스트리밍 가능

---

# 자주 나는 오류

| 증상 | 원인 / 해결 |
|------|------------|
| `ModuleNotFoundError: lightrag` | 가상환경 활성화를 안 했거나 `pip install` 누락 |
| `No matching distribution found` | Python 버전 불일치. `wheels_cp310/`은 Python 3.10 전용 |
| 임베딩 차원 불일치 오류 | `rag_storage/` 삭제 후 재실행 |
| Ollama에서 NaN/zero-vector | bge-m3 Ollama 서빙 경로 문제. FlagEmbedding 버전(`lightrag_onprem_demo.py`) 사용 |
| `OPENAI_API_KEY` 관련 에러 | 환경변수 확인 |
| Docker에서 네트워크 오류 | 사내 API 주소가 Docker 컨테이너에서 접근 가능한지 확인. `--network host` 옵션 추가 |
| ONNX 모델 경로 에러 | `models/bge-m3-onnx/`에 모델 파일 배치 확인 |

---

# 참고
- 공식 리포: https://github.com/HKUDS/LightRAG
- 논문 (EMNLP 2025 Findings): https://aclanthology.org/2025.findings-emnlp.568/
- arXiv: https://arxiv.org/abs/2410.05779
