# 1) 베이스 이미지 (슬림)
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TRANSFORMERS_CACHE=/cache/hf \
    HF_HOME=/cache/hf

# 2) (옵션) 시스템 의존성 – 필요 시만 사용
# tokenizers 등은 보통 wheel 제공되어 문제 없음.
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     git curl ca-certificates \
#  && rm -rf /var/lib/apt/lists/*

# 3) 워킹 디렉토리
WORKDIR /app

# 4) 의존성 먼저 설치(빌드 캐시 극대화)
COPY requirements.txt .
RUN pip install -r requirements.txt

# 5) 애플리케이션 복사
COPY . .

# 6) 포트 (Railway는 PORT 환경변수로 내려줌)
EXPOSE 8000

# 7) 시작 커맨드: PORT 없으면 8000 기본
CMD ["bash", "-lc", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
