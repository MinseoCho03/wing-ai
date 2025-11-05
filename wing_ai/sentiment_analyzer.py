# wing_ai/sentiment_analyzer.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import re
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch


class SentimentAnalyzer:
    """
    투자 모드 감정 분석기 (메인 키워드 기준)
    - direct: 메인 ↔ 서브 엣지 → title+메인 포함 문장 기반
    - propagated: 메인과 직접 연결되지 않은 엣지 → hop 감쇠
    """

    def __init__(self, sentiment_tokenizer, sentiment_model):
        self.tokenizer = sentiment_tokenizer
        self.model = sentiment_model

    # ---------------------------------------------------------------------
    # 기본 모델 호출 (512토큰 초과 시 truncation)
    # ---------------------------------------------------------------------
    def _model_predict(self, text: str) -> Dict:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )

        # 사람이 읽을 수 있는 모델 입력 텍스트
        truncated_text = self.tokenizer.decode(
            inputs["input_ids"][0], skip_special_tokens=False
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            conf = probs[0][pred].item()

        labels = ['negative', 'neutral', 'positive']

        return {
            'label': labels[pred],
            'score': float(conf),
            'sentiment_score': int(pred) - 1,  # -1, 0, +1
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
    def _find_spans(self, tokens: list[str], needle: str) -> list[int]:
        return [i for i, t in enumerate(tokens) if needle in t]

    def _window_indices(self, center_idx: int, n_tokens: int, radius: int = 6) -> tuple[int, int]:
        s = max(0, center_idx - radius)
        e = min(n_tokens, center_idx + radius + 1)
        return s, e

    def _count_polarity_in_window(self, tokens, start, end) -> int:
        window_text = " ".join(tokens[start:end])
        pos = bool(self._POS_PAT.search(window_text))
        neg = bool(self._NEG_PAT.search(window_text))
        if pos and not neg:
            return +1
        if neg and not pos:
            return -1
        return 0

    def _num_strength_near_entity(self, tokens, start, end) -> float:
        return 1.2 if self._NUM_PAT.search(" ".join(tokens[start:end])) else 1.0

    def _finance_heuristic_polarity_entity_aware(
        self, sent: str, main_kw: str, other_kw: str | None
    ) -> tuple[int, float]:
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

    def _blend_rule_with_model(self, rule_score: int, model_score: float) -> float:
        if rule_score == 0:
            return model_score
        return 0.6 * float(rule_score) + 0.4 * float(model_score)

    # ---------------------------------------------------------------------
    # 메인/서브 문장 추출
    # ---------------------------------------------------------------------
    def _extract_targeted_sentences(
        self, title: str, desc: str, main_kw: str, other_kw: Optional[str]
    ) -> tuple[list[str], list[str], list[str]]:
        sents = self._split_sentences(desc or "")
        main_only = [s for s in sents if main_kw in s]
        both = [s for s in main_only if other_kw and other_kw in s] if other_kw else []
        main_only = [s for s in main_only if s not in both]
        title_sents = [title] if (title and main_kw in title) else []
        return title_sents, both, main_only

    # ---------------------------------------------------------------------
    # 기사 1건을 메인 관점으로 분석 (sentiment_description만 출력)
    # ---------------------------------------------------------------------
    def analyze_article_for_main(self, article: Dict, main_kw: str, other_kw: Optional[str] = None) -> Dict:
        title = article.get("title", "") or ""
        desc = article.get("description", "") or ""
        title_sents, both, main_only = self._extract_targeted_sentences(title, desc, main_kw, other_kw)

        scores, weights = [], []
        used_input_texts: List[str] = []   # 기사 단위 sentiment_description용

        def infer(s: str, base_w: float):
            model_out = self._model_predict(s)
            rule, s_strength = self._finance_heuristic_polarity_entity_aware(s, main_kw, other_kw)
            blended = self._blend_rule_with_model(rule, float(model_out["sentiment_score"]))
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
            fallback = title if (title and main_kw in title) else (title or desc[:200])
            pred = self._model_predict(fallback)
            return {
                "label": pred["label"],
                "score": 1.0,
                "sentiment_score": float(pred["sentiment_score"]),
                "sentiment_description": pred.get("model_input_text", fallback),
            }

        wsum = float(np.sum(weights)) if np.sum(weights) > 0 else 1.0
        avg = float(np.dot(scores, weights) / wsum)
        label = "positive" if avg > 0.3 else ("negative" if avg < -0.3 else "neutral")

        return {
            "label": label,
            "score": 1.0,
            "sentiment_score": avg,
            "sentiment_description": "\n".join(used_input_texts) if used_input_texts else None,
        }

    # ---------------------------------------------------------------------
    # 엣지 일반 컨텍스트 (propagated용)
    # ---------------------------------------------------------------------
    def _extract_generic_context_for_edge(self, title, desc, kw_a, kw_b, max_chars=1800):
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
    # 엣지 감정 계산
    # ---------------------------------------------------------------------
    def calculate_edge_sentiment_main_subject(
        self,
        edge_weights: Dict[Tuple[str, str], Dict],
        processed_data: Dict[str, List[Dict]],
        main_keyword: str,
        propagate_alpha: float = 0.7
    ) -> Dict[Tuple[str, str], Dict]:

        graph: Dict[str, List[str]] = {}
        for (a, b) in edge_weights.keys():
            graph.setdefault(a, []).append(b)
            graph.setdefault(b, []).append(a)

        def weighted_mean(scores, weights):
            if not scores:
                return 0.0
            s = float(np.dot(scores, weights))
            w = float(np.sum(weights)) if np.sum(weights) > 0 else 1.0
            return s / w

        out: Dict[Tuple[str, str], Dict] = {}
        base_sentiments = {}

        # 1) 기본 감정 계산 (propagation용)
        for (a, b), info in edge_weights.items():
            articles = info.get("articles", [])
            scores, weights = [], []
            for art in articles:
                full = None
                for kw in (a, b):
                    for rec in processed_data.get(kw, []):
                        if rec["link"] == art["link"]:
                            full = rec
                            break
                    if full:
                        break
                if full:
                    s = self._model_predict(
                        self._extract_generic_context_for_edge(full.get("title",""), full.get("description",""), a, b)
                    )
                    scores.append(float(s["sentiment_score"]))
                    weights.append(float(full.get("trust_score", 0.7)))
            base = weighted_mean(scores, weights)
            base_sentiments[(a, b)] = base

        # 2) 메인과 직접 연결된 엣지
        for (a, b), info in edge_weights.items():
            if main_keyword in (a, b):
                other = b if a == main_keyword else a
                articles = info.get("articles", [])
                scores, weights = [], []
                enriched_articles = []

                for art in articles:
                    full = None
                    for kw in (a, b):
                        for rec in processed_data.get(kw, []):
                            if rec["link"] == art["link"]:
                                full = rec
                                break
                        if full:
                            break
                    if full:
                        s = self.analyze_article_for_main(full, main_kw=main_keyword, other_kw=other)
                        scores.append(float(s["sentiment_score"]))
                        weights.append(float(full.get("trust_score", 0.7)))
                        enriched_art = dict(art)
                        enriched_art["sentiment_description"] = s.get("sentiment_description")
                        enriched_articles.append(enriched_art)
                    else:
                        enriched_articles.append(art)

                direct_score = weighted_mean(scores, weights)
                label = "positive" if direct_score > 0.3 else ("negative" if direct_score < -0.3 else "neutral")
                out[(a, b)] = {
                    **info,
                    "articles": enriched_articles,
                    "sentiment_score": float(direct_score),
                    "sentiment_label": label,
                    "sentiment_subject": main_keyword,
                    "sentiment_derivation": "direct"
                }

        # 3) propagated edges
        def _shortest_hop(graph, start, goal):
            if start == goal:
                return 0
            if start not in graph or goal not in graph:
                return None
            from collections import deque
            dq = deque([(start, 0)])
            seen = {start}
            while dq:
                node, d = dq.popleft()
                for nxt in graph.get(node, []):
                    if nxt == goal:
                        return d + 1
                    if nxt not in seen:
                        seen.add(nxt)
                        dq.append((nxt, d + 1))
            return None

        for (a, b), info in edge_weights.items():
            if (a, b) in out:
                continue
            da = _shortest_hop(graph, a, main_keyword)
            db = _shortest_hop(graph, b, main_keyword)
            hops = [d for d in [da, db] if d is not None]

            if hops:
                h = min(hops)
                base = base_sentiments.get((a, b), 0.0)
                propagated = (propagate_alpha ** h) * base
                label = "positive" if propagated > 0.3 else ("negative" if propagated < -0.3 else "neutral")
                out[(a, b)] = {
                    **info,
                    "sentiment_score": float(propagated),
                    "sentiment_label": label,
                    "sentiment_subject": main_keyword,
                    "sentiment_derivation": "propagated",
                    "hops_to_main": int(h)
                }
            else:
                out[(a, b)] = {
                    **info,
                    "sentiment_score": 0.0,
                    "sentiment_label": "neutral",
                    "sentiment_subject": main_keyword,
                    "sentiment_derivation": "propagated",
                    "hops_to_main": None,
                    "note": "no_path_to_main"
                }
        return out
