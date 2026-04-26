#!/usr/bin/env bash
# =============================================================
# export_packages.sh
# 인터넷이 되는 PC에서 실행하여 모든 패키지를 wheels/ 폴더에 다운로드
# =============================================================
# 사용법:
#   bash scripts/export_packages.sh
#
# 결과물:
#   wheels/           <- 모든 .whl 파일
#   requirements.lock <- 정확한 버전이 고정된 requirements
# =============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "=== 1) 현재 환경의 정확한 패키지 버전 고정 ==="
pip freeze > requirements.lock
echo "   -> requirements.lock 생성 완료 ($(wc -l < requirements.lock) 패키지)"

echo ""
echo "=== 2) wheels/ 폴더에 모든 패키지 다운로드 ==="
mkdir -p wheels

# --platform, --python-version 은 대상 환경에 맞게 수정
# 아래는 Windows x86_64 + Python 3.11 기준
pip download \
    -r requirements.lock \
    -d wheels/ \
    --platform win_amd64 \
    --python-version 3.11 \
    --only-binary=:all: \
    || {
        echo ""
        echo "!! 일부 패키지가 pure-python이 아니어서 --only-binary 실패."
        echo "!! source 포함하여 재다운로드합니다."
        pip download -r requirements.lock -d wheels/
    }

echo ""
echo "=== 완료 ==="
echo "wheels/ 폴더와 requirements.lock 을 USB 등으로 폐쇄망 PC에 복사하세요."
echo ""
echo "폐쇄망에서 설치 명령:"
echo "  python -m venv .venv"
echo "  .venv\\Scripts\\activate"
echo "  pip install --no-index --find-links=wheels/ -r requirements.lock"
