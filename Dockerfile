# 더 최신/작은 베이스
FROM python:3.11-slim

# 실행 편의 & 캐시 최소화
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# 시스템 최소 의존성
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
 && rm -rf /var/lib/apt/lists/*

# 먼저 requirements만 복사 -> 레이어 캐시 극대화
COPY requirements.txt .

# 1) torch는 CPU 전용 휠로 별도 설치 (CUDA 미포함)
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    torch==2.3.1

# 2) 나머지 패키지 설치 (이미 torch는 설치됨)
#    (requirements.txt에서 torch는 제거되어 있어야 함)
RUN pip install --no-cache-dir -r requirements.txt

# 앱 소스 복사 (dockerignore가 불필요물 차단)
COPY . .

EXPOSE 8000

# Railway가 PORT env 넣어줌. 없으면 8000.
CMD ["bash", "-lc", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
