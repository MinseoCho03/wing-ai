# wing_ai/pipeline.py
from typing import Dict, Optional
from .models import ModelManager
from .preprocessor import ArticlePreprocessor
from .node_calculator import NodeImportanceCalculator
from .embedding import EmbeddingProcessor
from .edge_calculator import EdgeWeightCalculator
from .sentiment_analyzer import SentimentAnalyzer
from .config_utils import load_config, resolve_device, set_global_seed

class WINGAIPipeline:
    """WING 프로젝트의 메인 AI 파이프라인 (메인 키워드 기준 투자 감성 지원)"""

    def __init__(self, config_path: Optional[str] = "config.yaml"):
        self.config = load_config(config_path)
        self.device = resolve_device(self.config["runtime"].get("device", "auto"))
        set_global_seed(self.config["runtime"].get("seed", 42))

        self.model_manager = ModelManager(self.config)
        self.preprocessor = ArticlePreprocessor(self.config.get("media_trust", {}))
        self.node_calculator = NodeImportanceCalculator()

        self.embedding_model = None
        self.embedding_processor = None
        self.edge_calculator = None
        self.sentiment_analyzer = None


    def _ensure_embedding_model_loaded(self):
        if self.embedding_model is None:
            self.embedding_model = self.model_manager.load_embedding_model()
            self.embedding_processor = EmbeddingProcessor(self.embedding_model, self.config)  # ← cfg 전달
            self.edge_calculator = EdgeWeightCalculator(self.embedding_processor)


    def _ensure_sentiment_model_loaded(self):
        if self.sentiment_analyzer is None:
            tokenizer, model = self.model_manager.load_sentiment_model()
            self.sentiment_analyzer = SentimentAnalyzer(tokenizer, model)

    '''
    def process(
        self,
        articles_by_keyword: Dict,
        mode: str = 'normal',
        alpha: float = 0.5,
        beta: float = 0.5,
        main_keyword: Optional[str] = None,
        propagate_alpha: float = 0.7
    ) -> Dict:
        """
        전체 파이프라인 실행
        - mode='investment'일 때, main_keyword가 있으면 메인 기준 감정 계산(직접 + 전파)
        - main_keyword가 없으면 기존 방식(엣지 기사 기반 평균 감정)으로 동작
        """
        print("Step 1: Preprocessing articles...")
        processed_data = self.preprocessor.preprocess_articles(articles_by_keyword)

        print("Step 2: Calculating node importance...")
        node_importance = self.node_calculator.calculate_importance(processed_data)

        print("Step 3: Loading embedding model and embedding articles...")
        self._ensure_embedding_model_loaded()
        embeddings_by_keyword = self.embedding_processor.embed_articles(processed_data)

        print("Step 4: Calculating edge weights...")
        edge_weights = self.edge_calculator.calculate_weights(
            processed_data,
            embeddings_by_keyword,
            alpha=alpha,
            beta=beta
        )

        if mode == 'investment':
            print("Step 5: Analyzing sentiment (Investment Mode)...")
            self._ensure_sentiment_model_loaded()
            if main_keyword:
                edge_weights = self.sentiment_analyzer.calculate_edge_sentiment_main_subject(
                    edge_weights=edge_weights,
                    processed_data=processed_data,
                    main_keyword=main_keyword,
                    propagate_alpha=propagate_alpha
                )
            else:
                # 이전 방식(메인 미지정): 기사 full_text 1샷 평균
                # 기존 SentimentAnalyzer.calculate_edge_sentiment(...)가 있었다면 호출
                # 여기서는 후방 호환 위해 간단 중립 처리
                # 필요 시 기존 함수를 남겨두고 호출해도 OK
                for k, v in edge_weights.items():
                    v['sentiment_score'] = 0.0
                    v['sentiment_label'] = 'neutral'
                    v['sentiment_subject'] = None
                    v['sentiment_derivation'] = 'none'

        print("Processing complete!")

        return {
            'nodes': node_importance,
            'edges': edge_weights,
            'processed_articles': processed_data
        }
'''


    def process(
        self,
        articles_by_keyword: Dict,
        mode: str = 'normal',
        alpha: float = 0.5,
        beta: float = 0.5,
        main_keyword: Optional[str] = None,
        propagate_alpha: float = 0.7
    ) -> Dict:
        """
        전체 파이프라인 실행
        - mode='investment'일 때, main_keyword가 있으면 메인 기준 감정 계산(직접 + 전파)
        - main_keyword가 없으면 기존 방식(엣지 기사 기반 평균 감정)으로 동작
        """
        print("Step 1: Preprocessing articles...")
        processed_data = self.preprocessor.preprocess_articles(articles_by_keyword)

        print("Step 2: Loading embedding model and embedding articles...")
        self._ensure_embedding_model_loaded()
        embeddings_by_keyword = self.embedding_processor.embed_articles(processed_data)

        print("Step 3: Calculating edge weights...")
        edge_weights = self.edge_calculator.calculate_weights(
            processed_data,
            embeddings_by_keyword,
            alpha=alpha,
            beta=beta
        )

        # ✅ 이제 엣지 기반 신호(S2, S3)가 생겼으니 여기서 중요도 계산
        print("Step 4: Calculating node importance...")
        node_importance = self.node_calculator.calculate_importance(
            processed_data=processed_data,
            edge_weights=edge_weights,
            main_keyword=main_keyword,      # 필요 시 main_cap 적용
        )

        if mode == 'investment':
            print("Step 5: Analyzing sentiment (Investment Mode)...")
            self._ensure_sentiment_model_loaded()
            if main_keyword:
                edge_weights = self.sentiment_analyzer.calculate_edge_sentiment_main_subject(
                    edge_weights=edge_weights,
                    processed_data=processed_data,
                    main_keyword=main_keyword,
                    propagate_alpha=propagate_alpha
                )
            else:
                # 메인 미지정 시 간단 중립 처리(후방 호환)
                for k, v in edge_weights.items():
                    v['sentiment_score'] = 0.0
                    v['sentiment_label'] = 'neutral'
                    v['sentiment_subject'] = None
                    v['sentiment_derivation'] = 'none'

        print("Processing complete!")

        return {
            'nodes': node_importance,
            'edges': edge_weights,
            'processed_articles': processed_data
        }
