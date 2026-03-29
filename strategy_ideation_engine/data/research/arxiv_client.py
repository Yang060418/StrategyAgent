from typing import List
import arxiv
import time
import random
from strategy_ideation_engine.config import logger
from strategy_ideation_engine.schemas.research import ResearchInsight, SourceCredibility

class ArxivClient:
    """Queries ArXiv for academic trading precedents with rate-limit awareness."""
    
    _last_request_time = 0

    def search(self, query: str, max_results: int = 5) -> List[ResearchInsight]:
        # Enforce 3-second rule between global requests
        elapsed = time.time() - ArxivClient._last_request_time
        if elapsed < 3.0:
            wait_time = 3.0 - elapsed + random.uniform(0.1, 0.5)
            logger.info(f"ArXiv Rate Limit: Waiting {wait_time:.2f}s...")
            time.sleep(wait_time)

        logger.info(f"Searching ArXiv for: {query}")
        insights = []
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                ArxivClient._last_request_time = time.time()
                search = arxiv.Search(
                    query=query,
                    max_results=max_results,
                    sort_by=arxiv.SortCriterion.Relevance
                )
                
                # arxiv library fetches results lazily during iteration
                for result in search.results():
                    insights.append(ResearchInsight(
                        title=result.title,
                        source_url=result.pdf_url,
                        credibility=SourceCredibility.ACADEMIC_PAPER,
                        authors_or_publisher=", ".join([a.name for a in result.authors]),
                        published_date=result.published,
                        key_findings=[result.summary[:500] + "..."]
                    )
                )
                return insights # Success!

            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    backoff = (attempt + 1) * 5 + random.uniform(0, 2)
                    logger.warning(f"ArXiv 429 Error. Retrying in {backoff:.2f}s... (Attempt {attempt+1}/{max_retries})")
                    time.sleep(backoff)
                else:
                    logger.error(f"Error searching ArXiv: {e}")
                    break
            
        return insights
