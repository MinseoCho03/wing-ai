# crawling_parser.py
from typing import Dict, List, Any
import json
from pathlib import Path

def load_crawling_json(path: str | Path = "data/crawling.json") -> Dict[str, Any]:
    """crawling.json 파일을 읽어 dict로 반환"""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def to_pipeline_format(payload: Dict[str, Any]) -> Dict[str, List[Dict]]:
    """백엔드 crawling.json 포맷 -> {keyword: [articles...]} 형태로 변환"""
    by_kw: Dict[str, List[Dict]] = {}
    seen: Dict[str, set] = {}

    main_kw = (payload.get("mainKeyword") or "").strip()
    results = payload.get("results") or []

    for block in results:
        query = (block.get("query") or "").strip()
        if not query:
            continue
        keywords = [kw.strip() for kw in query.split() if kw.strip()]
        items: List[Dict] = block.get("items") or []

        for item in items:
            article = {
                "link": item.get("link"),
                "title": item.get("title"),
                "pubDate": item.get("pubDate"),
                "description": item.get("description"),
            }
            for kw in keywords:
                if kw not in by_kw:
                    by_kw[kw] = []
                    seen[kw] = set()
                link = article["link"]
                if link and link not in seen[kw]:
                    by_kw[kw].append(article)
                    seen[kw].add(link)

    if main_kw and main_kw not in by_kw:
        by_kw[main_kw] = []

    return by_kw
