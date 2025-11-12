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
    libopenblas-dev libgomp1 git curl \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.3.1
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir sentence-transformers==2.7.0 transformers==4.45.2 huggingface_hub

RUN mkdir -p /opt/hf-cache && chmod -R 777 /opt/hf-cache

# 🔥 두 모델 모두 미리 다운로드 (embedding + sentiment)
RUN python - <<'PY'
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
print("✅ Embedding model cached")

AutoTokenizer.from_pretrained("snunlp/KR-FinBERT-SC")
AutoModelForSequenceClassification.from_pretrained("snunlp/KR-FinBERT-SC")
print("✅ Sentiment model cached")
PY

COPY app.py ./app.py
COPY config.yaml ./config.yaml
COPY wing_ai ./wing_ai
EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
