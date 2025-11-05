# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, List, Optional

from data.demo_data import _build_articles_by_keyword
from wing_ai.pipeline import WINGAIPipeline

app = FastAPI(title="WING AI API", version="0.1.0")

pipeline: Optional[WINGAIPipeline] = None   # 1) 처음엔 빈 상태


@app.on_event("startup")
def load_pipeline():
    global pipeline
    # 2) 서버가 켜질 때 한 번만 로드
    pipeline = WINGAIPipeline(config_path="config.yaml")


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
    items: List[CrawlingItem] = []
    done: Optional[bool] = None
    nextStartHint: Optional[str] = None


class CrawlingPayload(BaseModel):
    mainKeyword: Optional[str] = None
    subKeywords: Optional[List[str]] = None
    queryCount: Optional[int] = None
    results: List[CrawlingResultBlock]

'''
@app.get("/health")
def health():
    return {"status": "ok"}
'''
'''
#2. health 체크 API
GET http://127.0.0.1:8000/health

정상 응답:
{"status": "ok"}
'''


@app.post("/process")
def process_news(payload: CrawlingPayload, mode: str = "investment"):
    if pipeline is None:
        # 아직 로딩 안 됐을 때
        raise HTTPException(status_code=503, detail="Pipeline not ready")

    articles_by_kw = _build_articles_by_keyword(
        [r.model_dump() for r in payload.results]
    )

    result = pipeline.process(
        articles_by_kw,
        mode=mode,
        main_keyword=payload.mainKeyword,
    )

    edges_list = []
    for (source, target), data in result["edges"].items():
        edges_list.append(
            {
                "source": source,
                "target": target,
                **data,
            }
        )

    nodes_list = [
        {"id": kw, "importance": float(imp)}
        for kw, imp in result["nodes"].items()
    ]

    return {
        "nodes": nodes_list,
        "edges": edges_list,
        "metadata": {
            "total_nodes": len(nodes_list),
            "total_edges": len(edges_list),
        },
    }
