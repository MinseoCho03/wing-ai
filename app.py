# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict
from typing import Any, Dict, List, Optional
import time
import torch
import numpy as np
import os

from wing_ai.pipeline import WINGAIPipeline

# ---------------------------------------------------------
# 환경 변수 세팅 (Dockerfile과 일치)
# ---------------------------------------------------------
os.environ.setdefault("HF_HOME", "/opt/hf-cache")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

app = FastAPI(title="WING AI API", version="0.1.4")

# ---------------------------------------------------------
# Globals
# ---------------------------------------------------------
pipeline: Optional[WINGAIPipeline] = None
_ready: bool = False

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def _build_articles_by_keyword(results_list: List[Dict]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for block in results_list:
        kw = block.get("query")
        if not kw:
            continue
        items = block.get("items", []) or []
        bucket = out.setdefault(kw, [])
        for it in items:
            bucket.append({
                "title": it.get("title"),
                "description": it.get("description"),
                "link": it.get("link"),
                "originallink": it.get("originallink"),
                "pubDate": it.get("pubDate"),
            })
    return out


def _to_py(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: _to_py(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_py(x) for x in obj]
    if isinstance(obj, tuple):
        return tuple(_to_py(x) for x in obj)
    return obj


# ---------------------------------------------------------
# Schemas
# ---------------------------------------------------------
class CrawlingItem(BaseModel):
    model_config = ConfigDict(extra='ignore')
    link: Optional[str] = None
    title: Optional[str] = None
    pubDate: Optional[str] = None
    originallink: Optional[str] = None
    description: Optional[str] = None


class CrawlingResultBlock(BaseModel):
    model_config = ConfigDict(extra='ignore')
    query: str
    need: Optional[int] = None
    collectedCount: Optional[int] = None
    totalEstimated: Optional[int] = None
    items: List[CrawlingItem] = Field(default_factory=list)
    done: Optional[bool] = None
    nextStartHint: Optional[str] = None


class CrawlingPayload(BaseModel):
    model_config = ConfigDict(extra='ignore')
    mainKeyword: Optional[str] = None
    subKeywords: Optional[List[str]] = None
    queryCount: Optional[int] = None
    results: List[CrawlingResultBlock]


class GraphNode(BaseModel):
    model_config = ConfigDict(extra='allow')
    id: str
    importance: float


class GraphEdge(BaseModel):
    model_config = ConfigDict(extra='allow')
    source: str
    target: str
    weight: Optional[float] = None
    cooccurrence: Optional[float] = None
    similarity: Optional[float] = None
    articles: Optional[List[Dict[str, Any]]] = None
    sentiment_score: Optional[float] = None
    sentiment_label: Optional[str] = None
    sentiment_subject: Optional[str] = None
    sentiment_derivation: Optional[str] = None
    hops_to_main: Optional[int] = None


class GraphMetadata(BaseModel):
    model_config = ConfigDict(extra='allow')
    total_nodes: int
    total_edges: int
    processing_time: Dict[str, float]


class GraphResponse(BaseModel):
    model_config = ConfigDict(extra='allow')
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    metadata: GraphMetadata


# ---------------------------------------------------------
# Lifecycles
# ---------------------------------------------------------
@app.on_event("startup")
def load_pipeline():
    global pipeline, _ready
    boot_t0 = time.time()
    try:
        pipeline = WINGAIPipeline(config_path="config.yaml")

        # 모델 로드/웜업 (이미 캐시가 bake-in 되어있어 매우 빠름)
        pipeline._ensure_embedding_model_loaded()
        pipeline._ensure_sentiment_model_loaded()

        with torch.inference_mode():
            _ = pipeline.embedding_model.encode(["warmup"], convert_to_tensor=False)
            sa = pipeline.sentiment_analyzer
            inputs = sa.tokenizer("warmup", return_tensors="pt", truncation=True, max_length=32)
            _ = sa.model(**inputs)

        _ready = True
        print(f"[startup] ✅ Warm-up complete in {(time.time()-boot_t0):.2f}s")
    except Exception as e:
        _ready = False
        print(f"[startup] ⚠️ Warm-up failed: {e}")


# ---------------------------------------------------------
# Health / Status
# ---------------------------------------------------------
@app.get("/")
def root():
    return {"service": "wing-ai", "version": app.version, "status": "ok" if _ready else "warming"}


@app.get("/health")
def health():
    return {"status": "ok" if _ready else "warming"}


# ---------------------------------------------------------
# Main Endpoint
# ---------------------------------------------------------
@app.post("/process", response_model=GraphResponse)
def process_news(payload: CrawlingPayload, mode: str = "investment"):
    if pipeline is None or not _ready:
        raise HTTPException(status_code=503, detail="Pipeline not ready")

    total_t0 = time.time()
    prep_t0 = time.time()
    results_list = [r.model_dump() for r in payload.results]
    articles_by_kw = _build_articles_by_keyword(results_list)
    prep_ms = (time.time() - prep_t0) * 1000.0

    pipe_t0 = time.time()
    result: Dict[str, Any] = pipeline.process(
        articles_by_kw,
        mode=mode,
        main_keyword=payload.mainKeyword
    )
    pipe_ms = (time.time() - pipe_t0) * 1000.0

    resp_t0 = time.time()
    nodes_list: List[GraphNode] = [
        GraphNode(id=str(kw), importance=float(imp))
        for kw, imp in result.get("nodes", {}).items()
    ]
    edges_list: List[GraphEdge] = []
    for (src, tgt), data in result.get("edges", {}).items():
        edge_payload: Dict[str, Any] = {"source": src, "target": tgt}
        edge_payload.update(_to_py(data))
        articles = edge_payload.get("articles")
        if isinstance(articles, list):
            for art in articles:
                if isinstance(art, dict):
                    art.pop("sentiment", None)
                    art.pop("sentiment_score", None)
        edges_list.append(GraphEdge(**edge_payload))

    resp_ms = (time.time() - resp_t0) * 1000.0
    total_ms = (time.time() - total_t0) * 1000.0

    meta = GraphMetadata(
        total_nodes=len(nodes_list),
        total_edges=len(edges_list),
        processing_time={
            "total_ms": round(total_ms, 2),
            "preparation_ms": round(prep_ms, 2),
            "pipeline_ms": round(pipe_ms, 2),
            "response_ms": round(resp_ms, 2),
        }
    )

    return GraphResponse(nodes=nodes_list, edges=edges_list, metadata=meta)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=False)
