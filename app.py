# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional

from data.demo_data import _build_articles_by_keyword
from wing_ai.pipeline import WINGAIPipeline

app = FastAPI(title="WING AI API", version="0.1.1")

pipeline: Optional[WINGAIPipeline] = None

@app.on_event("startup")
def load_pipeline():
    global pipeline
    pipeline = WINGAIPipeline(config_path="config.yaml")

# ---------- Input Schemas (네 JSON과 1:1 매칭) ----------
class CrawlingItem(BaseModel):
    link: Optional[str] = None
    title: Optional[str] = None
    pubDate: Optional[str] = None
    originallink: Optional[str] = None
    description: Optional[str] = None

class CrawlingResultBlock(BaseModel):
    query: str
    need: Optional[int] = None
    collectedCount: Optional[int] = None
    totalEstimated: Optional[int] = None
    items: List[CrawlingItem] = Field(default_factory=list)  # <- 중요!
    done: Optional[bool] = None
    nextStartHint: Optional[str] = None

class CrawlingPayload(BaseModel):
    mainKeyword: Optional[str] = None        # camelCase 그대로
    subKeywords: Optional[List[str]] = None
    queryCount: Optional[int] = None
    results: List[CrawlingResultBlock]        # 네 JSON이 요구하는 필수 필드

# ---------- Output Schemas (docs에서 보기 좋게) ----------
class GraphNode(BaseModel):
    id: str
    importance: float

class GraphEdge(BaseModel):
    source: str
    target: str
    cooccurrence: Optional[float] = None
    similarity: Optional[float] = None
    weight: Optional[float] = None
    articles: Optional[List[Dict[str, Any]]] = None

class GraphResponse(BaseModel):
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    metadata: Dict[str, int]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/process", response_model=GraphResponse)
def process_news(payload: CrawlingPayload, mode: str = "investment"):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready")

    # results[].items[]를 _build_articles_by_keyword가 먹는 형태로 전달
    results_list = [r.model_dump() for r in payload.results]
    articles_by_kw = _build_articles_by_keyword(results_list)

    result = pipeline.process(
        articles_by_kw,
        mode=mode,
        main_keyword=payload.mainKeyword,   # camelCase 그대로 사용
    )

    edges_list = []
    for (source, target), data in result["edges"].items():
        edge_payload = {"source": source, "target": target}
        edge_payload.update(data)
        edges_list.append(GraphEdge(**edge_payload))

    nodes_list = [GraphNode(id=kw, importance=float(imp))
                  for kw, imp in result["nodes"].items()]

    return GraphResponse(
        nodes=nodes_list,
        edges=edges_list,
        metadata={"total_nodes": len(nodes_list), "total_edges": len(edges_list)},
    )
