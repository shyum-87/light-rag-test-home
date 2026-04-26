FROM python:3.12-slim

WORKDIR /app

# 시스템 패키지 (빌드 도구 등)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# requirements 먼저 복사해서 Docker 레이어 캐시 활용
# (소스가 바뀌어도 패키지 재설치 불필요)
COPY requirements.lock .

# 인터넷에서 패키지 설치 (빌드는 인터넷 되는 PC에서 수행)
# 빌드 완료된 이미지(.tar)를 폐쇄망으로 옮기므로 이후 인터넷 불필요
RUN pip install --no-cache-dir -r requirements.lock

# 소스 코드 복사
COPY sample_doc.txt .
COPY lightrag_onprem_demo.py .
COPY .env.onprem* ./

# 실행
CMD ["python", "lightrag_onprem_demo.py"]
