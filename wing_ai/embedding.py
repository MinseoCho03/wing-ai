import os
import hashlib
import numpy as np
from typing import Dict

def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode('utf-8')).hexdigest()

class EmbeddingProcessor:
    """기사 임베딩 처리 클래스 (간단 캐시 지원)"""
    def __init__(self, embedding_model, cfg):
        self.embedding_model = embedding_model
        self.cfg = cfg
        self.cache_dir = cfg.get("cache", {}).get("embedding_dir")
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

    def _cache_path(self, text: str) -> str:
        k = _sha1(text)
        return os.path.join(self.cache_dir, f"{k}.npy")

    def _load_cache(self, text: str):
        if not self.cache_dir:
            return None
        p = self._cache_path(text)
        if os.path.exists(p):
            try:
                return np.load(p)
            except Exception:
                return None
        return None

    def _save_cache(self, text: str, vec: np.ndarray):
        if not self.cache_dir:
            return
        p = self._cache_path(text)
        try:
            np.save(p, vec)
        except Exception:
            pass

    def embed_articles(self, processed_data: Dict) -> Dict:
        embeddings_by_keyword = {}
        bs = int(self.cfg["embedding"]["batch_size"])
        show = bool(self.cfg["embedding"]["show_progress_bar"])

        for keyword, articles in processed_data.items():
            texts = [article['full_text'] for article in articles]

            # 캐시 확인
            cached_vecs, to_compute_idx, to_compute_texts = [], [], []
            for idx, t in enumerate(texts):
                v = self._load_cache(t)
                if v is None:
                    to_compute_idx.append(idx)
                    to_compute_texts.append(t)
                else:
                    cached_vecs.append((idx, v))

            new_vecs = []
            if to_compute_texts:
                new_vecs = self.embedding_model.encode(
                    to_compute_texts,
                    batch_size=bs,
                    show_progress_bar=show,
                    convert_to_numpy=True
                )
                for i, vec in enumerate(new_vecs):
                    self._save_cache(to_compute_texts[i], vec)

            # 원래 순서대로 재조립
            emb = [None] * len(texts)
            for i, v in cached_vecs:
                emb[i] = v
            for j, idx in enumerate(to_compute_idx):
                emb[idx] = new_vecs[j]
            embeddings_by_keyword[keyword] = np.vstack(emb) if emb else np.zeros((0,))

        return embeddings_by_keyword

    def calculate_keyword_vector(self, embeddings: np.ndarray) -> np.ndarray:
        return np.mean(embeddings, axis=0)
