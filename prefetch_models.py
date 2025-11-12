# prefetch_models.py  (로컬에서 1회 실행)
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

print("📥 Downloading embedding...")
SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
print("✅ Embedding cached")

print("📥 Downloading FinBERT...")
AutoTokenizer.from_pretrained("snunlp/KR-FinBert-SC")
AutoModelForSequenceClassification.from_pretrained("snunlp/KR-FinBert-SC")
print("✅ FinBERT cached")
