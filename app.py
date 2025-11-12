# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict
from typing import Any, Dict, List, Optional
import time
import torch
import numpy as np
import os
import re

from wing_ai.pipeline import WINGAIPipeline

# ---------------------------------------------------------
# 환경 변수 세팅 (Dockerfile과 일치)
# ---------------------------------------------------------
os.environ.setdefault("HF_HOME", "/opt/hf-cache")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

app = FastAPI(title="WING AI API", version="0.1.6")

# ---------------------------------------------------------
# Globals
# ---------------------------------------------------------
pipeline: Optional[WINGAIPipeline] = None
_ready: bool = False

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def _normalize_text(s: Optional[str]) -> str:
    if not s:
        return ""
    # 필요시 추가 정규화(대소문자, 공백 등)
    return s.strip()

def _denamespace_kw(raw_q: str, main_keyword: Optional[str]) -> str:
    """
    '엔비디아 이재용'처럼 메인키워드가 query 접두로 붙은 케이스를 정규화하여
    서브키워드만 남기거나, 메인키워드 자체면 그대로 반환.
    """
    q = _normalize_text(raw_q)
    if not main_keyword:
        return q

    mk = _normalize_text(main_keyword)
    if q == mk:
        return q

    # 접두 제거 패턴들: "메인키워드 " / "메인키워드-" / "메인키워드_" / "메인키워드:"
    patterns = [
        rf"^{re.escape(mk)}\s+",
        rf"^{re.escape(mk)}[-_:/|]\s*",
    ]
    for pat in patterns:
        q2 = re.sub(pat, "", q)
        if q2 != q:
            return q2.strip()

    return q

def _article_contains_main(art: Dict[str, Any], main_keyword: str) -> bool:
    if not main_keyword:
        return False
    t = (_normalize_text(art.get("title")) + " " + _normalize_text(art.get("description"))).strip()
    if not t:
        return False
    # 한국어는 띄어쓰기 기반 포함으로도 충분히 보수적. 필요 시 형태소/정규식 경계 강화 가능.
    return main_keyword in t

def _build_articles_by_keyword(
    results_list: List[Dict],
    main_keyword: Optional[str],
) -> Dict[str, List[Dict[str, Any]]]:
    """
    크롤링 결과 블록을 {정규화된_키워드: [기사들]} 형태로 변환.
    - query에서 메인키워드 접두를 제거(네임스페이스 제거)
    - 메인키워드 노드는 기사 유무와 무관하게 반드시 생성
    - 🔥 하이드레이션: 서브 버킷 기사 중 제목/요약에 메인키워드가 포함된 기사를 메인 버킷에도 함께 담아 co-occurrence 보장
    """
    out: Dict[str, List[Dict[str, Any]]] = {}

    # 1) 기본 빌드 (+ 네임스페이스 제거)
    for block in results_list:
        raw_q = block.get("query")
        if not raw_q:
            continue
        kw = _denamespace_kw(raw_q, main_keyword)
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

    # 2) 메인 노드 보장
    if main_keyword:
        out.setdefault(main_keyword, [])

        # 3) 🔥 메인 버킷 하이드레이션
        main_bucket = out[main_keyword]
        seen = {a.get("link") for a in main_bucket if isinstance(a, dict)}
        for kw, bucket in out.items():
            if kw == main_keyword:
                continue
            for art in bucket:
                if not isinstance(art, dict):
                    continue
                lk = art.get("link")
                if lk in seen:
                    continue
                if _article_contains_main(art, main_keyword):
                    main_bucket.append(art)
                    seen.add(lk)

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

    # 1) 입력 정리
    prep_t0 = time.time()
    results_list = [r.model_dump() for r in payload.results]
    # ⚠️ 메인키워드 전달하여 네임스페이스 제거 + 하이드레이션 수행
    articles_by_kw = _build_articles_by_keyword(results_list, main_keyword=payload.mainKeyword)
    prep_ms = (time.time() - prep_t0) * 1000.0

    # 2) 파이프라인 실행
    pipe_t0 = time.time()
    result: Dict[str, Any] = pipeline.process(
        articles_by_kw,
        mode=mode,
        main_keyword=payload.mainKeyword
    )
    pipe_ms = (time.time() - pipe_t0) * 1000.0

    # 3) 응답 직렬화
    resp_t0 = time.time()
    nodes_list: List[GraphNode] = [
        GraphNode(id=str(kw), importance=float(imp))
        for kw, imp in result.get("nodes", {}).items()
    ]
    edges_list: List[GraphEdge] = []
    for (src, tgt), data in result.get("edges", {}).items():
        edge_payload: Dict[str, Any] = {"source": src, "target": tgt}
        edge_payload.update(_to_py(data))
        # 불필요/중복 감성 필드 정리
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
