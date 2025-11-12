FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf-cache \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas-dev libgomp1 \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.3.1
RUN pip install --no-cache-dir -r requirements.txt


# Hugging Face 캐시 경로
ENV HF_HOME=/opt/hf-cache

# 캐시 폴더 보장
RUN mkdir -p /opt/hf-cache && chmod -R 777 /opt/hf-cache

# 로컬 캐시 주입 (있으면 복사되어 cold start ↓)
COPY .hf-cache/ /opt/hf-cache/

# ✅ 앱 파일을 명시적으로만 복사 (중복/대용량 방지)
COPY app.py ./app.py
COPY config.yaml ./config.yaml
COPY wing_ai ./wing_ai
# data가 런타임에 필요하면 다음 라인 주석 해제
# COPY data ./data

EXPOSE 8000
ENV TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
