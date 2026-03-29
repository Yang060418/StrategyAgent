from typing import List, Optional
from tavily import TavilyClient as Tavily
from strategy_ideation_engine.config import settings, logger
from strategy_ideation_engine.schemas.research import ResearchInsight, SourceCredibility

class SearchClient:
    """Uses Tavily for real-time financial news and search."""
    
    def __init__(self):
        self.client = Tavily(api_key=settings.TAVILY_API_KEY) if settings.TAVILY_API_KEY else None

    def search_news(self, query: str, max_results: int = 5) -> List[ResearchInsight]:
        if not self.client:
            logger.warning("Tavily API key missing. Skipping search.")
            return []
            
        logger.info(f"Searching Tavily for news: {query}")
        insights = []
        
        try:
            response = self.client.search(
                query=query,
                search_depth="advanced",
                max_results=max_results
            )
            
            for result in response.get('results', []):
                insights.append(ResearchInsight(
                    title=result.get('title', 'No Title'),
                    source_url=result.get('url'),
                    credibility=SourceCredibility.FINANCIAL_NEWS,
                    key_findings=[result.get('content', '')[:1000]] # Slice to keep context manageable
                ))
        except Exception as e:
            logger.error(f"Error searching Tavily: {e}")
            
        return insights
