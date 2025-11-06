import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class ModelManager:
    def __init__(self, config):
        self.cfg = config
        self.device = torch.device("cpu")
        self.embedding_model = None
        self.sentiment_tokenizer = None
        self.sentiment_model = None
        self.sentiment_device = None  # ✅ 감성모델 실제 디바이스 기록

    def set_device(self, device: torch.device):
        self.device = device

    def load_embedding_model(self):
        name = self.cfg["models"]["embedding"]
        print(f"Loading embedding model: {name}")
        model = SentenceTransformer(name, device=str(self.device))
        self.embedding_model = model
        return self.embedding_model

    def load_sentiment_model(self):
        name = self.cfg["models"]["sentiment"]
        print(f"Loading sentiment model: {name}")
        tok = AutoTokenizer.from_pretrained(name)
        mdl = AutoModelForSequenceClassification.from_pretrained(name, use_safetensors=True)

        # ✅ CPU에서만 양자화 적용, 그 외(MPS/CUDA)는 양자화 건너뛰고 해당 디바이스로 올림
        if self.device.type == "cpu":
            # (Apple Silicon 등 ARM CPU) 양자화 엔진 지정
            try:
                torch.backends.quantized.engine = "qnnpack"
            except Exception:
                pass
            mdl = torch.quantization.quantize_dynamic(mdl, {nn.Linear}, dtype=torch.qint8)
            mdl.to("cpu")
            self.sentiment_device = torch.device("cpu")
        else:
            # MPS / CUDA일 땐 양자화 X (커널 없음) → 원본 모델을 가속 디바이스로
            mdl.to(self.device)
            self.sentiment_device = self.device

        mdl.eval()
        self.sentiment_tokenizer = tok
        self.sentiment_model = mdl
        return self.sentiment_tokenizer, self.sentiment_model
