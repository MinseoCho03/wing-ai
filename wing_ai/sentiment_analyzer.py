# sentiment_analysis.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import re
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch


class SentimentAnalyzer:
    """
    투자 모드 감정 분석기 (메인 키워드 기준)
    - direct: 메인 ↔ 서브 엣지 → title + (메인/서브 포함 문장) 기반 + 금융 휴리스틱 블렌딩
    - propagated: 메인과 직접 연결되지 않은 엣지 → base(연속 점수, 중립정규화) × hop 감쇠
    """

    def __init__(self, sentiment_tokenizer, sentiment_model,
                 label_threshold: float = 0.25,  # 라벨링 임계값(스케일 보정 적용 후)
                 propagate_alpha: float = 0.7):
        self.tokenizer = sentiment_tokenizer
        self.model = sentiment_model
        self.label_threshold = float(label_threshold)
        self.propagate_alpha = float(propagate_alpha)

    # ---------------------------------------------------------------------
    # 기본 모델 호출 (512토큰 초과 시 truncation)
    #  - sentiment_score: 기존 호환용 정수(-1/0/+1)
    #  - sentiment_cont : 연속 점수(-1..+1) = p(pos) - p(neg)
    #  - sentiment_cont_adj : 중립정규화 연속 점수 = (p(pos)-p(neg))/(1-p(neu)+eps)
    # ---------------------------------------------------------------------
    def _model_predict(self, text: str) -> Dict:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )

        # 사람이 읽을 수 있는 모델 입력 텍스트(특수토큰 포함해 디코딩)
        truncated_text = self.tokenizer.decode(
            inputs["input_ids"][0], skip_special_tokens=False
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits  # [B, 3] = [neg, neu, pos]
            probs = torch.softmax(logits, dim=1)[0]
            pred = int(torch.argmax(probs).item())
            conf = float(probs[pred].item())

        labels = ['negative', 'neutral', 'positive']
        p_neg, p_neu, p_pos = float(probs[0]), float(probs[1]), float(probs[2])

        # 기본 연속 점수
        cont = p_pos - p_neg  # -1..+1

        # 중립정규화(스케일 보정): 중립이 클수록 양/음의 상대 대립을 확대
        denom = max(1e-6, 1.0 - p_neu)
        cont_adj = cont / denom
        cont_adj = max(-1.0, min(1.0, cont_adj))  # 안전 클리핑

        return {
            'label': labels[pred],
            'score': conf,
            'sentiment_score': int(pred) - 1,   # -1, 0, +1  (호환)
            'sentiment_cont': cont,             # 원 연속 점수
            'sentiment_cont_adj': cont_adj,     # 보정 연속 점수
            'probs': {'neg': p_neg, 'neu': p_neu, 'pos': p_pos},
            'model_input_text': truncated_text
        }

    # ---------------------------------------------------------------------
    # 문장 분리 (간단 정규식)
    # ---------------------------------------------------------------------
    _SPLIT_RE = re.compile(r'(?<=[\.!?])\s+|\n+')

    def _split_sentences(self, text: str) -> List[str]:
        sents = self._SPLIT_RE.split(text or "")
        return [s.strip() for s in sents if s and s.strip()]

    # ---------------------------------------------------------------------
    # 금융 감정 패턴 (휴리스틱)
    # ---------------------------------------------------------------------
    _POS_PAT = re.compile(r"(급등|상승|강세|반등|급반등|사상 최고|신고가|호재)")
    _NEG_PAT = re.compile(r"(급락|하락|약세|부진|폭락|추락|악재)")
    _NUM_PAT = re.compile(r"([+-]?\d+(?:\.\d+)?\s*%|[+-]?\d+(?:\.\d+)?\s*p)")

    _TOKEN_SPLIT = re.compile(r"\s+")
    _CONTRAST_PAT = re.compile(r"(반면|대조|하지만|대비|vs|VS|그러나)")

    # ---------------------------------------------------------------------
    # 금융 휴리스틱 규칙
    # ---------------------------------------------------------------------
    def _find_spans(self, tokens: List[str], needle: str) -> List[int]:
        return [i for i, t in enumerate(tokens) if needle in t]

    def _window_indices(self, center_idx: int, n_tokens: int, radius: int = 6) -> Tuple[int, int]:
        s = max(0, center_idx - radius)
        e = min(n_tokens, center_idx + radius + 1)
        return s, e

    def _count_polarity_in_window(self, tokens: List[str], start: int, end: int) -> int:
        window_text = " ".join(tokens[start:end])
        pos = bool(self._POS_PAT.search(window_text))
        neg = bool(self._NEG_PAT.search(window_text))
        if pos and not neg:
            return +1
        if neg and not pos:
            return -1
        return 0

    def _num_strength_near_entity(self, tokens: List[str], start: int, end: int) -> float:
        return 1.2 if self._NUM_PAT.search(" ".join(tokens[start:end])) else 1.0

    def _finance_heuristic_polarity_entity_aware(
        self, sent: str, main_kw: str, other_kw: Optional[str]
    ) -> Tuple[int, float]:
        tokens = self._TOKEN_SPLIT.split(sent.strip())
        if not tokens:
            return 0, 1.0

        contrasted = bool(self._CONTRAST_PAT.search(sent))
        main_idxs = self._find_spans(tokens, main_kw)
        if not main_idxs:
            return 0, 1.0

        rule_score, strength = 0, 1.0
        for midx in main_idxs:
            s, e = self._window_indices(midx, len(tokens))
            local_pol = self._count_polarity_in_window(tokens, s, e)
            if local_pol != 0:
                rule_score = local_pol
                strength = max(strength, self._num_strength_near_entity(tokens, s, e))

        if contrasted and other_kw:
            other_idxs = self._find_spans(tokens, other_kw)
            for oidx in other_idxs:
                s2, e2 = self._window_indices(oidx, len(tokens))
                other_pol = self._count_polarity_in_window(tokens, s2, e2)
                if other_pol != 0 and rule_score != 0 and (other_pol * rule_score) < 0:
                    strength = max(strength, 1.25)
        return rule_score, strength

    def _blend_rule_with_model(self, rule_score: int, model_score_cont_adj: float) -> float:
        """
        rule_score: -1/0/+1 (휴리스틱)
        model_score_cont_adj: -1..+1, 중립정규화 연속 점수
        """
        if rule_score == 0:
            return float(model_score_cont_adj)
        return 0.6 * float(rule_score) + 0.4 * float(model_score_cont_adj)

    # ---------------------------------------------------------------------
    # 메인/서브 문장 추출
    # ---------------------------------------------------------------------
    def _extract_targeted_sentences(
        self, title: str, desc: str, main_kw: str, other_kw: Optional[str]
    ) -> Tuple[List[str], List[str], List[str]]:
        sents = self._split_sentences(desc or "")
        main_only = [s for s in sents if main_kw in s]
        both = [s for s in main_only if other_kw and other_kw in s] if other_kw else []
        main_only = [s for s in main_only if s not in both]
        title_sents = [title] if (title and main_kw in title) else []
        return title_sents, both, main_only

    # ---------------------------------------------------------------------
    # 기사 1건을 메인 관점으로 분석 (sentiment_description 포함)
    #  - 중립정규화 연속 점수 사용 + 휴리스틱 블렌딩
    # ---------------------------------------------------------------------
    def analyze_article_for_main(self, article: Dict, main_kw: str, other_kw: Optional[str] = None) -> Dict:
        title = article.get("title", "") or ""
        desc = article.get("description", "") or ""
        title_sents, both, main_only = self._extract_targeted_sentences(title, desc, main_kw, other_kw)

        scores, weights = [], []
        used_input_texts: List[str] = []   # 기사 단위 sentiment_description용

        def infer(s: str, base_w: float):
            model_out = self._model_predict(s)
            model_base = float(model_out.get("sentiment_cont_adj",
                                             model_out.get("sentiment_cont",
                                                           model_out["sentiment_score"])))
            rule, s_strength = self._finance_heuristic_polarity_entity_aware(s, main_kw, other_kw)
            blended = self._blend_rule_with_model(rule, model_base)
            scores.append(blended)
            weights.append(base_w * s_strength)
            used_input_texts.append(model_out.get("model_input_text", s))

        for s in title_sents:
            infer(s, 0.8)
        for s in both:
            infer(s, 1.0)
        for s in main_only:
            infer(s, 0.7)

        if not scores:
            # 메인 텍스트가 거의 없을 때의 폴백
            fallback = title if (title and main_kw in title) else (title or desc[:200])
            pred = self._model_predict(fallback)
            cont_adj = float(pred.get("sentiment_cont_adj",
                                      pred.get("sentiment_cont",
                                               pred["sentiment_score"])))
            return {
                "label": "positive" if cont_adj > self.label_threshold
                         else ("negative" if cont_adj < -self.label_threshold else "neutral"),
                "score": 1.0,
                "sentiment_score": cont_adj,
                "sentiment_description": pred.get("model_input_text", fallback),
            }

        wsum = float(np.sum(weights)) if np.sum(weights) > 0 else 1.0
        avg = float(np.dot(scores, weights) / wsum)
        label = "positive" if avg > self.label_threshold else ("negative" if avg < -self.label_threshold else "neutral")

        return {
            "label": label,
            "score": 1.0,
            "sentiment_score": avg,  # 보정 연속 점수 평균
            "sentiment_description": "\n".join(used_input_texts) if used_input_texts else None,
        }

    # ---------------------------------------------------------------------
    # 엣지 일반 컨텍스트 (propagated/base용)
    # ---------------------------------------------------------------------
    def _extract_generic_context_for_edge(self, title: str, desc: str, kw_a: str, kw_b: str, max_chars: int = 1800) -> str:
        sents = self._split_sentences(desc or "")
        both = [s for s in sents if kw_a in s and kw_b in s]
        only_a = [s for s in sents if kw_a in s and kw_b not in s]
        only_b = [s for s in sents if kw_b in s and kw_a not in s]
        bag = []
        if title:
            bag.append(title)
        bag.extend(both)
        bag.extend(only_a)
        bag.extend(only_b)
        out, total = [], 0
        for s in bag:
            if total + len(s) > max_chars:
                break
            out.append(s)
            total += len(s)
        return " ".join(out) if out else (title or "")

    # ---------------------------------------------------------------------
    # 보조: 두 키워드 동시 문장(both)에서 약한 규칙 신호 추출(서브↔서브 base 강화용)
    # ---------------------------------------------------------------------
    def _weak_rule_from_both(self, desc: str, kw_a: str, kw_b: str) -> int:
        sents = self._split_sentences(desc or "")
        both = [s for s in sents if kw_a in s and kw_b in s]
        sig = 0
        for s in both:
            tokens = self._TOKEN_SPLIT.split(s.strip())
            pol = self._count_polarity_in_window(tokens, 0, len(tokens))
            sig += pol
        if sig > 0:
            return +1
        if sig < 0:
            return -1
        return 0

    # ---------------------------------------------------------------------
    # 엣지 감정 계산 (메인 관점: direct / propagated)
    # ---------------------------------------------------------------------
    def calculate_edge_sentiment_main_subject(
        self,
        edge_weights: Dict[Tuple[str, str], Dict],
        processed_data: Dict[str, List[Dict]],
        main_keyword: str,
        propagate_alpha: Optional[float] = None
    ) -> Dict[Tuple[str, str], Dict]:

        alpha = float(self.propagate_alpha if propagate_alpha is None else propagate_alpha)

        # --- (A) 링크 인덱스: 모든 키워드 버킷을 통틀어 O(1)로 기사 찾기 ---
        link_index: Dict[str, Dict] = {}
        for kw, recs in processed_data.items():
            for rec in recs:
                link = rec.get("link")
                if link and link not in link_index:
                    link_index[link] = rec

        # 그래프 구성
        graph: Dict[str, List[str]] = {}
        for (a, b) in edge_weights.keys():
            graph.setdefault(a, []).append(b)
            graph.setdefault(b, []).append(a)

        def weighted_mean(scores: List[float], weights: List[float]) -> float:
            if not scores:
                return 0.0
            s = float(np.dot(scores, weights))
            w = float(np.sum(weights)) if np.sum(weights) > 0 else 1.0
            return s / w

        out: Dict[Tuple[str, str], Dict] = {}

        # base 점수와 enriched articles 캐시
        base_cache: Dict[Tuple[str, str], Dict] = {}

        # --- (B) 기본 감정 계산(모든 엣지): 보정 연속 점수(cont_adj) + 약한 규칙 보강 ---
        for (a, b), info in edge_weights.items():
            articles = info.get("articles", [])
            scores, weights = [], []
            enriched_articles = []

            for art in articles:
                full = link_index.get(art.get("link", ""))
                if full:
                    title = full.get("title", "")
                    desc = full.get("description", "")
                    ctx = self._extract_generic_context_for_edge(title, desc, a, b)
                    s = self._model_predict(ctx)

                    # 보정 연속 점수
                    cont_adj = float(s.get("sentiment_cont_adj",
                                           s.get("sentiment_cont",
                                                 s["sentiment_score"])))

                    # 두 키워드 동시 문장에 금융 키워드가 있으면 약한 규칙 신호(±0.2) 가산
                    weak_rule = self._weak_rule_from_both(desc, a, b)
                    if weak_rule != 0:
                        cont_adj = 0.8 * cont_adj + 0.2 * float(weak_rule)

                    scores.append(cont_adj)
                    weights.append(float(full.get("trust_score", 0.7)))

                    enriched = dict(art)
                    enriched["sentiment_description"] = s.get("model_input_text", ctx)
                    enriched_articles.append(enriched)
                else:
                    enriched_articles.append(art)

            base = weighted_mean(scores, weights)
            base_cache[(a, b)] = {
                "base_score": base,
                "enriched_articles": enriched_articles
            }

        # --- (C) 메인과 직접 연결된 엣지: 기사별 메인 관점 분석 (블렌딩 포함) ---
        for (a, b), info in edge_weights.items():
            if main_keyword in (a, b):
                other = b if a == main_keyword else a
                articles = info.get("articles", [])
                scores, weights = [], []
                enriched_articles = []

                for art in articles:
                    full = link_index.get(art.get("link", ""))
                    if full:
                        s = self.analyze_article_for_main(full, main_kw=main_keyword, other_kw=other)
                        scores.append(float(s["sentiment_score"]))  # 이미 보정 연속 점수 기반
                        weights.append(float(full.get("trust_score", 0.7)))

                        enriched = dict(art)
                        enriched["sentiment_description"] = s.get("sentiment_description")
                        enriched_articles.append(enriched)
                    else:
                        enriched_articles.append(art)

                direct_score = weighted_mean(scores, weights)
                label = "positive" if direct_score > self.label_threshold else \
                        ("negative" if direct_score < -self.label_threshold else "neutral")
                out[(a, b)] = {
                    **info,
                    "articles": enriched_articles,
                    "sentiment_score": float(direct_score),
                    "sentiment_label": label,
                    "sentiment_subject": main_keyword,
                    "sentiment_derivation": "direct"
                }

        # --- (D) propagated 엣지: base(cont_adj) × hop 감쇠 + 증거(기사별 sentiment_description) 포함 ---
        def _shortest_hop(g, start, goal):
            if start == goal:
                return 0
            if start not in g or goal not in g:
                return None
            from collections import deque
            dq = deque([(start, 0)])
            seen = {start}
            while dq:
                node, d = dq.popleft()
                for nxt in g.get(node, []):
                    if nxt == goal:
                        return d + 1
                    if nxt not in seen:
                        seen.add(nxt)
                        dq.append((nxt, d + 1))
            return None

        for (a, b), info in edge_weights.items():
            if (a, b) in out:  # 이미 direct로 처리됨
                continue

            da = _shortest_hop(graph, a, main_keyword)
            db = _shortest_hop(graph, b, main_keyword)
            hops = [d for d in [da, db] if d is not None]

            base_info = base_cache.get((a, b), {"base_score": 0.0, "enriched_articles": info.get("articles", [])})
            base = float(base_info["base_score"])
            enriched_articles = base_info["enriched_articles"]

            if hops:
                h = min(hops)
                # 감쇠 적용
                propagated = (alpha ** h) * base
                label = "positive" if propagated > self.label_threshold else \
                        ("negative" if propagated < -self.label_threshold else "neutral")
                out[(a, b)] = {
                    **info,
                    "articles": enriched_articles,  # propagated에도 기사별 sentiment_description 노출
                    "sentiment_score": float(propagated),
                    "sentiment_label": label,
                    "sentiment_subject": main_keyword,
                    "sentiment_derivation": "propagated",
                    "hops_to_main": int(h)
                }
            else:
                out[(a, b)] = {
                    **info,
                    "articles": enriched_articles,
                    "sentiment_score": 0.0,
                    "sentiment_label": "neutral",
                    "sentiment_subject": main_keyword,
                    "sentiment_derivation": "propagated",
                    "hops_to_main": None,
                    "note": "no_path_to_main"
                }

        return out


__all__ = ["SentimentAnalyzer"]
