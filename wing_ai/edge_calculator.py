import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict

class EdgeWeightCalculator:
    """엣지 가중치 계산 클래스"""
    def __init__(self, embedding_processor, snippet_len: int = 80):
        self.embedding_processor = embedding_processor
        # 기사 본문을 몇 글자로 잘라 넣을지 설정 (50~100자 사이 추천)
        self.snippet_len = snippet_len

    def calculate_cooccurrence(self, processed_data: Dict) -> Dict[Tuple[str, str], int]:
        cooccurrence = defaultdict(int)
        article_keywords = defaultdict(set)

        # processed_data 구조 예:
        # {
        #   "테슬라": [
        #       {"link": ..., "title": ..., "pubDate": ..., "trust_score": ..., "description": "..."},
        #       ...
        #   ],
        #   ...
        # }
        for keyword, articles in processed_data.items():
            for article in articles:
                article_keywords[article['link']].add(keyword)

        for _link, kws in article_keywords.items():
            kws_list = list(kws)
            for i in range(len(kws_list)):
                for j in range(i + 1, len(kws_list)):
                    kw1, kw2 = sorted([kws_list[i], kws_list[j]])
                    cooccurrence[(kw1, kw2)] += 1
        return dict(cooccurrence)

    def _make_snippet(self, article: Dict) -> str:
        """
        기사 원본 dict에서 description이나 content가 있으면 앞부분만 잘라서 반환.
        없으면 "" 반환.
        """
        # 크롤러가 어디에 본문을 넣는지에 따라 순서 바꿔도 됨
        text = article.get("description") or article.get("content") or ""
        if not text:
            return ""
        # 너무 긴 경우 자르고 ... 붙이기
        if len(text) > self.snippet_len:
            return text[:self.snippet_len].rstrip() + "..."
        return text

    def get_related_articles(self, kw1: str, kw2: str, processed_data: Dict) -> List[Dict]:
        articles1 = {a['link']: a for a in processed_data.get(kw1, [])}
        articles2 = {a['link']: a for a in processed_data.get(kw2, [])}
        common_links = set(articles1) & set(articles2)

        related_articles = []
        for link in common_links:
            a = articles1[link]  # 공통 링크의 기사 하나(동일)
            related_articles.append({
                'link': a['link'],
                'title': a.get('title', ''),
                'pubDate': a.get('pubDate', ''),
                'trust_score': a.get('trust_score', 0.0),
                # ← 여기 추가
                'description': self._make_snippet(a)
            })

        # 신뢰도 높은 기사부터
        related_articles.sort(key=lambda x: x['trust_score'], reverse=True)
        return related_articles

    def calculate_weights(
        self,
        processed_data: Dict,
        embeddings_by_keyword: Dict,
        alpha: float = 0.5,
        beta: float = 0.5
    ) -> Dict[Tuple[str, str], Dict]:
        keywords = list(processed_data.keys())

        # 키워드마다 대표 벡터 만들기
        keyword_vectors = {
            kw: self.embedding_processor.calculate_keyword_vector(embeddings_by_keyword[kw])
            for kw in keywords if len(embeddings_by_keyword.get(kw, [])) > 0
        }

        # 1) 동시출현 계산
        cooccurrence = self.calculate_cooccurrence(processed_data)
        if cooccurrence:
            max_cooccur = max(cooccurrence.values())
            normalized_cooccur = {
                k: (v / max_cooccur if max_cooccur > 0 else 0.0)
                for k, v in cooccurrence.items()
            }
        else:
            normalized_cooccur = {}

        edge_weights = {}
        for i in range(len(keywords)):
            for j in range(i + 1, len(keywords)):
                kw1, kw2 = sorted([keywords[i], keywords[j]])

                # (1) 동시출현 점수
                cooccur_score = normalized_cooccur.get((kw1, kw2), 0.0)

                # (2) 의미적 유사도
                vec1 = keyword_vectors.get(kw1)
                vec2 = keyword_vectors.get(kw2)
                if vec1 is None or vec2 is None:
                    similarity = 0.0
                else:
                    denom = (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                    similarity = float(np.dot(vec1, vec2) / denom) if denom > 0 else 0.0
                # [-1, 1] -> [0, 1]
                similarity = (similarity + 1) / 2

                # (3) 최종 엣지 가중치
                weight = float(alpha * cooccur_score + beta * similarity)

                # (4) 이 두 키워드에 공통으로 등장한 기사들 + 짧은 본문
                related_articles = self.get_related_articles(kw1, kw2, processed_data)

                edge_weights[(kw1, kw2)] = {
                    'weight': weight,
                    'cooccurrence': float(cooccur_score),
                    'similarity': float(similarity),
                    'articles': related_articles
                }

        return edge_weights
