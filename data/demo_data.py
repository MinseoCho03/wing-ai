# data/demo_data.py
from typing import Dict, List
import json
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent 
CRAWLING_JSON = DATA_PATH / "crawling.json"


def load_crawling_json(path: Path) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # {mainKeyword, subKeywords, queryCount, results: [...] } 에서 results 리스트만 사용
    return data["results"] 


def _build_articles_by_keyword(feeds: List[Dict]) -> Dict[str, List[Dict]]:
    by_kw: Dict[str, List[Dict]] = {}
    seen_per_kw: Dict[str, set] = {}

    for feed in feeds:
        query = (feed.get("query") or "").strip()
        if not query:
            continue

        # "테슬라 BYD" → ["테슬라", "BYD"]
        keywords = [kw.strip() for kw in query.split() if kw.strip()]
        items: List[Dict] = feed.get("items") or []

        for it in items:
            article = {
                "link": it.get("link"),
                "title": it.get("title"),
                "pubDate": it.get("pubDate"),
                "description": it.get("description"),
            }
            for kw in keywords:
                if kw not in by_kw:
                    by_kw[kw] = []
                    seen_per_kw[kw] = set()

                link = article["link"]
                if link and link not in seen_per_kw[kw]:
                    by_kw[kw].append(article)
                    seen_per_kw[kw].add(link)

    return by_kw


# ─────────────────────────────────────────
# 여기서 JSON을 읽어서 DEMO_DATA를 만듦
# ─────────────────────────────────────────
_feeds = load_crawling_json(CRAWLING_JSON)
DEMO_DATA: Dict[str, List[Dict]] = _build_articles_by_keyword(_feeds)

__all__ = ["DEMO_DATA"]
