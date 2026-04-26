# =============================================================
# export_packages.ps1
# 인터넷이 되는 PC(Windows)에서 실행하여 모든 패키지를 wheels/ 폴더에 다운로드
# =============================================================
# 사용법 (PowerShell):
#   .\scripts\export_packages.ps1
# =============================================================

$ErrorActionPreference = "Stop"

$ProjectDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $ProjectDir

Write-Host "=== 1) 현재 환경의 정확한 패키지 버전 고정 ===" -ForegroundColor Cyan
pip freeze | Out-File -Encoding utf8 requirements.lock
$count = (Get-Content requirements.lock | Measure-Object -Line).Lines
Write-Host "   -> requirements.lock 생성 완료 ($count 패키지)"

Write-Host ""
Write-Host "=== 2) wheels/ 폴더에 모든 패키지 다운로드 ===" -ForegroundColor Cyan
New-Item -ItemType Directory -Force -Path wheels | Out-Null

# 현재 환경과 동일한 조건으로 다운로드 (같은 OS/Python이면 가장 확실)
pip download -r requirements.lock -d wheels/

Write-Host ""
Write-Host "=== 완료 ===" -ForegroundColor Green
Write-Host ""
Write-Host "다음 항목을 USB 등으로 폐쇄망 PC에 복사하세요:" -ForegroundColor Yellow
Write-Host "  1. wheels/ 폴더 전체"
Write-Host "  2. requirements.lock 파일"
Write-Host "  3. 프로젝트 소스 코드"
Write-Host ""
Write-Host "폐쇄망에서 설치 명령:" -ForegroundColor Yellow
Write-Host "  python -m venv .venv"
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "  pip install --no-index --find-links=wheels/ -r requirements.lock"
