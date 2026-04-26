FROM python:3.12-slim

WORKDIR /app

# 시스템 패키지 (빌드 도구 등)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# wheels 먼저 복사해서 Docker 레이어 캐시 활용
COPY wheels_cp312/ /tmp/wheels/
COPY requirements.lock .

# 오프라인 설치 (인터넷 불필요)
RUN pip install --no-cache-dir --no-index --find-links=/tmp/wheels/ -r requirements.lock && \
    rm -rf /tmp/wheels/

# 소스 코드 복사
COPY sample_doc.txt .
COPY lightrag_onprem_demo.py .
COPY .env.onprem* ./

# 실행
CMD ["python", "lightrag_onprem_demo.py"]
