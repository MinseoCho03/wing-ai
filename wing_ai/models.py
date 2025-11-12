# wing_ai/models.py
import os
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DEFAULT_EMBED_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_SENTI_ID = "snunlp/KR-FinBERT-SC"


def _hf_cache_dir() -> Optional[str]:
    """HF_HOME 환경변수 반환 (로컬 fallback 포함)"""
    cache = os.getenv("HF_HOME") or os.getenv("HUGGINGFACE_HUB_CACHE")
    
    # 로컬 개발 환경 fallback
    if not cache:
        # 프로젝트 루트의 .hf-cache 디렉토리 찾기
        current_dir = Path(__file__).resolve().parent.parent
        local_cache = current_dir / ".hf-cache"
        if local_cache.exists():
            print(f"⚠️  HF_HOME not set, using local cache: {local_cache}")
            return str(local_cache)
    
    return cache


def _resolve_local_snapshot(repo_id: str) -> str:
    """
    오프라인 캐시에서 모델 경로 직접 탐색 (대소문자 무시)
    - HF Hub 캐시 구조: hub/models--{org}--{name}/snapshots/{hash}/
    """
    cache_dir = _hf_cache_dir()
    if not cache_dir:
        raise RuntimeError("HF_HOME is not set; cannot resolve local snapshot path.")
    
    # repo_id를 캐시 디렉토리명으로 변환
    cache_name = "models--" + repo_id.replace("/", "--")
    model_cache = Path(cache_dir) / "hub" / cache_name
    
    # 대소문자 불일치 대응: 정확히 일치하지 않으면 case-insensitive 검색
    if not model_cache.exists():
        hub_dir = Path(cache_dir) / "hub"
        if hub_dir.exists():
            # 모든 캐시 디렉토리를 대소문자 무시하고 검색
            for candidate in hub_dir.iterdir():
                if candidate.name.lower() == cache_name.lower():
                    model_cache = candidate
                    print(f"⚠️  Case mismatch: '{repo_id}' found as '{candidate.name}'")
                    break
    
    if not model_cache.exists():
        raise RuntimeError(
            f"Model cache not found: {model_cache}\n"
            f"Expected repo '{repo_id}' in {cache_dir}/hub/"
        )
    
    # snapshots 디렉토리에서 가장 최근 스냅샷 찾기
    snapshots_dir = model_cache / "snapshots"
    if not snapshots_dir.exists():
        raise RuntimeError(f"No snapshots directory in {model_cache}")
    
    snapshots = list(snapshots_dir.iterdir())
    if not snapshots:
        raise RuntimeError(f"No snapshot found in {snapshots_dir}")
    
    # config.json이 있는 스냅샷 찾기 (우선순위)
    valid_snapshot = None
    for snapshot in snapshots:
        if (snapshot / "config.json").exists():
            valid_snapshot = snapshot
            break
    
    # config.json이 없으면 첫 번째 스냅샷 사용
    if not valid_snapshot:
        print(f"⚠️  No config.json found, using first snapshot: {snapshots[0].name}")
        valid_snapshot = snapshots[0]
    
    print(f"✅ Resolved '{repo_id}' -> {valid_snapshot}")
    return str(valid_snapshot)


class ModelManager:
    def __init__(self, config):
        self.cfg = config or {}
        self.device = torch.device("cpu")
        self.embedding_model = None
        self.sentiment_tokenizer = None
        self.sentiment_model = None
        self.sentiment_device = None

    def set_device(self, device: torch.device):
        self.device = device

    # -------------------------------
    # 임베딩 모델
    # -------------------------------
    def load_embedding_model(self):
        name = (self.cfg.get("models") or {}).get("embedding", DEFAULT_EMBED_ID)
        print(f"Loading embedding model: {name}")
        
        local_dir = _resolve_local_snapshot(name)
        model = SentenceTransformer(local_dir, device=str(self.device))
        self.embedding_model = model
        return self.embedding_model

    # -------------------------------
    # 감성 모델
    # -------------------------------
    def load_sentiment_model(self):
        name = (self.cfg.get("models") or {}).get("sentiment", DEFAULT_SENTI_ID)
        print(f"Loading sentiment model: {name}")
        
        local_dir = _resolve_local_snapshot(name)
        
        tok = AutoTokenizer.from_pretrained(local_dir, local_files_only=True)
        mdl = AutoModelForSequenceClassification.from_pretrained(
            local_dir,
            local_files_only=True,
        )
        
        if self.device.type == "cpu":
            # CPU 양자화
            try:
                torch.backends.quantized.engine = "qnnpack"
            except Exception:
                pass
            mdl = torch.quantization.quantize_dynamic(
                mdl, {nn.Linear}, dtype=torch.qint8
            )
            mdl.to("cpu")
            self.sentiment_device = torch.device("cpu")
        else:
            mdl.to(self.device)
            self.sentiment_device = self.device
        
        mdl.eval()
        self.sentiment_tokenizer = tok
        self.sentiment_model = mdl
        return self.sentiment_tokenizer, self.sentiment_model