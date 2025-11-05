import re
from typing import Dict, List
from datetime import datetime
import numpy as np

class ArticlePreprocessor:
    """기사 데이터 전처리 클래스"""
    def __init__(self, media_trust_scores: Dict[str, float], topk: int | None = None):
        self.media_trust_scores = media_trust_scores
        self.topk = topk

    def clean_html_tags(self, text: str) -> str:
        clean = re.compile('<.*?>')
        return re.sub(clean, '', text)

    def extract_media_from_link(self, link: str) -> str:
        media_mapping = {
            "yna.co.kr": "연합뉴스",
            "hankyung.com": "한국경제",
            "mk.co.kr": "매일경제",
            "edaily.co.kr": "이데일리",
            "fnnews.com": "파이낸셜뉴스"
        }
        for domain, media in media_mapping.items():
            if domain in link:
                return media
        return "default"

    def calculate_recency_score(self, pub_date: str, current_date: datetime = None) -> float:
        """최신성 점수 계산 (0~1)"""
        if current_date is None:
            current_date = datetime.now()
        try:
            article_date = datetime.strptime(pub_date, "%a, %d %b %Y %H:%M:%S %z")
            article_date = article_date.replace(tzinfo=None)
        except Exception:
            return 0.5
        days_diff = (current_date - article_date).days
        recency_score = np.exp(-days_diff / 7.0)
        return float(max(0.0, min(1.0, recency_score)))

    def calculate_article_trust(self, article: Dict) -> float:
        """신뢰도 = 0.6 * 언론사_신뢰도 + 0.4 * 최신성"""
        media = self.extract_media_from_link(article['link'])
        media_trust = self.media_trust_scores.get(media, self.media_trust_scores['default'])
        recency = self.calculate_recency_score(article['pubDate'])
        return float(0.6 * media_trust + 0.4 * recency)

    def preprocess_articles(self, articles_by_keyword: Dict[str, List[Dict]]) -> Dict:
        processed_data = {}
        for keyword, articles in articles_by_keyword.items():
            processed_articles = []
            for article in articles:
                clean_title = self.clean_html_tags(article['title'])
                clean_desc = self.clean_html_tags(article['description'])
                trust_score = self.calculate_article_trust(article)
                processed_article = {
                    'link': article['link'],
                    'title': clean_title,
                    'description': clean_desc,
                    'pubDate': article['pubDate'],
                    'trust_score': trust_score,
                    'full_text': f"{clean_title} {clean_desc}"
                }
                processed_articles.append(processed_article)

            # 신뢰도 기준 정렬 후 topk 적용
            processed_articles.sort(key=lambda x: x['trust_score'], reverse=True)
            if self.topk:
                processed_articles = processed_articles[: self.topk]

            processed_data[keyword] = processed_articles
        return processed_data
