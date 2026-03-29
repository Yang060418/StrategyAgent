from typing import List
from strategy_ideation_engine.data.research.arxiv_client import ArxivClient
from strategy_ideation_engine.data.research.search_client import SearchClient
from strategy_ideation_engine.schemas.research import LiteratureContext, ResearchInsight
from strategy_ideation_engine.config import logger

class ResearchOrchestrator:
    """Orchestrates Phase 3 literature review."""
    
    def __init__(self):
        self.arxiv = ArxivClient()
        self.search = SearchClient()

    def get_literature_review(self, query: str, tickers: List[str]) -> LiteratureContext:
        logger.info(f"Orchestrating academic literature review for: {tickers}")
        all_insights = []
        
        # 1. Search ArXiv (Academic - Targets quantitative papers)
        # We craft a query focusing on anomalies and backtests for the specific sector/ticker
        academic_query = f"abs:({tickers[0]} OR trading strategy) AND (backtest OR 'alpha factor' OR 'market anomaly')"
        academic_insights = self.arxiv.search(query=academic_query, max_results=3)
        all_insights.extend(academic_insights)
        
        # 2. Search News/Tavily (Real-time - Targets structural catalysts)
        news_query = f"structural catalysts and quantitative risks for {', '.join(tickers)} 2026"
        news_insights = self.search.search_news(query=news_query, max_results=3)
        all_insights.extend(news_insights)
        
        return LiteratureContext(
            query_topic=query,
            insights=all_insights
        )
