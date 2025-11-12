# -*- coding: utf-8 -*-
"""
핵심 변경사항
- S4(novelty) 완전 제거. 중요도는 S1(trust) + S2(weighted degree) + S3(weighted PageRank)만 사용.
- 기본 가중치: trust 0.4, degree 0.3, pagerank 0.3.
- embeddings_by_keyword 인자는 하위호환을 위해 남기되, 내부에서 사용하지 않음(Deprecated 표기).
- 나머지 로직(메인 소프트 캡, 재정규화, IDF 옵션)은 동일.
"""
from __future__ import annotations
from typing import Dict, Tuple, Optional
import math
import numpy as np


class NodeImportanceCalculator:
    """
    다신호 결합형 중요도 계산기 (novelty 제거):
      - S1: trust(=신뢰도 합) 로그 스케일 (+선택적 약한 IDF)
      - S2: weighted degree (엣지 weight 합)
      - S3: weighted PageRank (전역 영향력)
    + 메인 노드 소프트 캡 + 재정규화
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = (config or {}).get("importance", {})
        # 기본 가중치(심플화)
        self.w_trust = float(cfg.get("w_trust", 0.40))
        self.w_degree = float(cfg.get("w_degree", 0.30))
        self.w_pagerank = float(cfg.get("w_pagerank", 0.30))

        # IDF 옵션(자주 등장하는 범용 키워드 과대 억제용)
        self.use_idf = bool(cfg.get("use_idf", False))
        self.idf_strength = float(cfg.get("idf_strength", 0.3))  # 0~1, 낮게 유지 권장

        # 메인 소프트 캡 / 재정규화
        self.main_cap = float(cfg.get("main_cap", 0.45))
        self.renormalize_after_cap = bool(cfg.get("renormalize_after_cap", True))

    # ---------- utils ----------
    def _safe_log(self, x: float) -> float:
        # 합이 0이어도 안전하도록 log(1+x)
        return math.log(max(x, 0.0) + 1.0)

    def _normalize_dict(self, d: Dict[str, float]) -> Dict[str, float]:
        if not d:
            return {}
        mx = max(d.values())
        if mx <= 0:
            return {k: 0.0 for k in d}
        return {k: v / mx for k, v in d.items()}

    # ---------- signals ----------
    def _signal_trust(self, processed_data: Dict[str, list]) -> Dict[str, float]:
        """키워드별 trust_score 합을 로그 스케일로 압축. 선택적으로 약한 IDF 혼합."""
        trust_sum: Dict[str, float] = {}
        df: Dict[str, float] = {}
        all_links = set()
        for kw, arts in processed_data.items():
            s, links = 0.0, set()
            for a in arts:
                s += float(a.get("trust_score", 0.7))
                if a.get("link"):
                    links.add(a["link"])
                    all_links.add(a["link"])
            trust_sum[kw] = s
            df[kw] = float(len(links))

        if not self.use_idf:
            return {kw: self._safe_log(trust_sum.get(kw, 0.0)) for kw in processed_data.keys()}

        # 약한 IDF 결합
        N_docs = float(len(all_links)) if all_links else sum(df.values())
        out: Dict[str, float] = {}
        for kw in processed_data.keys():
            tf = self._safe_log(trust_sum.get(kw, 0.0))
            idf = math.log((N_docs + 1.0) / (df.get(kw, 0.0) + 1.0)) if N_docs > 0 else 0.0
            out[kw] = tf * (1.0 - self.idf_strength) + tf * max(idf, 0.0) * self.idf_strength
        return out

    def _signal_degree(self, edge_weights: Optional[Dict[Tuple[str, str], Dict]]) -> Dict[str, float]:
        """가중 차수: 각 노드로 유입되는 엣지 weight의 합."""
        deg: Dict[str, float] = {}
        if not edge_weights:
            return deg
        for (a, b), info in edge_weights.items():
            w = float(info.get("weight", 0.0))
            deg[a] = deg.get(a, 0.0) + w
            deg[b] = deg.get(b, 0.0) + w
        return deg

    def _signal_pagerank(
        self,
        edge_weights: Optional[Dict[Tuple[str, str], Dict]],
        damping: float = 0.85,
        max_iter: int = 50,
        tol: float = 1e-6,
    ) -> Dict[str, float]:
        """가중 PageRank: 무방향 그래프를 양방향 전이로 가정하여 열 정규화 전이행렬 구성."""
        if not edge_weights:
            return {}
        # 노드 인덱싱
        nodes = set()
        for (a, b) in edge_weights.keys():
            nodes.add(a); nodes.add(b)
        nodes = sorted(nodes)
        idx = {n: i for i, n in enumerate(nodes)}
        n = len(nodes)

        # 가중 인접행렬 W (무방향 → 양방향)
        W = np.zeros((n, n), dtype=float)
        for (a, b), info in edge_weights.items():
            w = float(info.get("weight", 0.0))
            ia, ib = idx[a], idx[b]
            W[ia, ib] += w
            W[ib, ia] += w

        # 열 정규화(출력합=1) → 전이확률행렬 M
        col_sums = W.sum(axis=0)
        M = np.zeros_like(W)
        for j in range(n):
            if col_sums[j] > 0:
                M[:, j] = W[:, j] / col_sums[j]
            else:
                M[:, j] = 1.0 / n  # dangling 처리

        pr = np.ones(n) / n
        teleport = np.ones(n) / n

        for _ in range(max_iter):
            new_pr = damping * (M @ pr) + (1 - damping) * teleport
            if np.linalg.norm(new_pr - pr, 1) < tol:
                pr = new_pr
                break
            pr = new_pr

        return {nodes[i]: float(pr[i]) for i in range(n)}

    # ---------- main ----------
    '''
    def calculate_importance(
        self,
        processed_data: Dict[str, list],
        edge_weights: Optional[Dict[Tuple[str, str], Dict]] = None,
        embeddings_by_keyword: Optional[Dict[str, np.ndarray]] = None,  # Deprecated: 미사용(호환 위해 유지)
        main_keyword: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        중요도 = w_trust*S1 + w_degree*S2 + w_pagerank*S3 (모두 0~1 정규화 후 가중합)
        - novelty 신호는 제거됨
        - embeddings_by_keyword 인자는 무시됨(향후 제거 예정)
        """
        if not processed_data:
            return {}

        # S1: trust
        sig_trust = self._normalize_dict(self._signal_trust(processed_data))
        # S2: degree
        sig_deg = self._normalize_dict(self._signal_degree(edge_weights))
        # S3: pagerank
        sig_pr = self._normalize_dict(self._signal_pagerank(edge_weights))

        # 결합 (없는 키는 0으로 취급)
        keys = set(processed_data.keys()) | set(sig_deg.keys()) | set(sig_pr.keys())
        raw: Dict[str, float] = {}
        for kw in keys:
            s = (
                self.w_trust    * sig_trust.get(kw, 0.0) +
                self.w_degree   * sig_deg.get(kw, 0.0) +
                self.w_pagerank * sig_pr.get(kw, 0.0)
            )
            raw[kw] = s

        # 메인 소프트 캡 (상한 적용으로 과비대 방지)
        if main_keyword and main_keyword in raw and self.main_cap is not None:
            raw[main_keyword] = min(raw[main_keyword], self.main_cap)

        # 최종 정규화
        final = self._normalize_dict(raw) if self.renormalize_after_cap else raw
        return final
'''

    def calculate_importance(
        self,
        processed_data,
        edge_weights=None,
        embeddings_by_keyword=None,
        main_keyword=None,
    ):
        if not processed_data:
            return {}

        # --- 신호별 계산 ---
        sig_trust_raw = self._signal_trust(processed_data)
        sig_trust = self._normalize_dict(sig_trust_raw)

        sig_deg_raw = self._signal_degree(edge_weights)
        sig_deg = self._normalize_dict(sig_deg_raw)

        sig_pr_raw = self._signal_pagerank(edge_weights)
        sig_pr = self._normalize_dict(sig_pr_raw)

        # ✅ 정규화 전후 비교 출력
        print("\n=== [S1: TRUST] raw vs normalized ===")
        for k in sig_trust_raw:
            print(f"{k:10s} raw={sig_trust_raw[k]:.4f}  norm={sig_trust.get(k,0):.4f}")

        print("\n=== [S2: DEGREE] raw vs normalized ===")
        for k in sig_deg_raw:
            print(f"{k:10s} raw={sig_deg_raw[k]:.4f}  norm={sig_deg.get(k,0):.4f}")

        print("\n=== [S3: PAGERANK] raw vs normalized ===")
        for k in sig_pr_raw:
            print(f"{k:10s} raw={sig_pr_raw[k]:.4f}  norm={sig_pr.get(k,0):.4f}")

        # --- 결합 ---
        keys = set(processed_data.keys()) | set(sig_deg.keys()) | set(sig_pr.keys())
        raw = {}
        for kw in keys:
            s = (
                self.w_trust    * sig_trust.get(kw, 0.0) +
                self.w_degree   * sig_deg.get(kw, 0.0) +
                self.w_pagerank * sig_pr.get(kw, 0.0)
            )
            raw[kw] = s

        # 메인 상한 적용
        if main_keyword and main_keyword in raw and self.main_cap is not None:
            raw[main_keyword] = min(raw[main_keyword], self.main_cap)

        # 최종 정규화
        final = self._normalize_dict(raw) if self.renormalize_after_cap else raw

        print("\n=== [FINAL IMPORTANCE raw vs normalized] ===")
        for k in raw:
            print(f"{k:10s} raw={raw[k]:.4f}  norm={final.get(k,0):.4f}")

        return final