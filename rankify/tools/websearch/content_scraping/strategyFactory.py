import os
from typing import Optional
from crawl4ai.extraction_strategy import (
    LLMExtractionStrategy,
    NoExtractionStrategy,
    JsonCssExtractionStrategy,
    JsonXPathExtractionStrategy,
    CosineStrategy,
    create_llm_config
)
from rankify.tools.websearch.content_scraping.config import (
    DEFAULT_PROVIDER,
    DEFAULT_PROVIDER_API_KEY
)
llm_config = create_llm_config(
    provider=DEFAULT_PROVIDER,
    api_token=os.environ.get(DEFAULT_PROVIDER_API_KEY),
)

class StrategyFactory:
    """Factory-pattern-based class to create extraction strategy"""

    @staticmethod
    def create_llm_strategy(
            input_format: str = 'markdown',
            instruction: str = "Extract relevant content from the provided text, only return the text, no markdown formatting, remove all footnotes, citations, and other metadata and only keep the main content",
            verbose=True,
    ) -> LLMExtractionStrategy:
        return LLMExtractionStrategy(
            input_format=input_format,
            llm_config=create_llm_config(
                provider=DEFAULT_PROVIDER,
                api_token=os.environ.get(DEFAULT_PROVIDER_API_KEY),
                ),
            verbose=verbose,
            instruction=instruction
        )

    @staticmethod
    def create_no_extraction_strategy() -> NoExtractionStrategy:
        return NoExtractionStrategy()

    @staticmethod
    def create_cosine_strategy(semantic_filter: Optional[str] = None,
                               word_count_threshold: int = 10,
                               max_dist: float = 0.2,
                               sim_threshold: float = 0.3,
                               debug: bool = False) -> CosineStrategy:
        return CosineStrategy(
            semantic_filter=semantic_filter,
            word_count_threshold=word_count_threshold,
            sim_threshold=sim_threshold,
            max_dist=max_dist,
            verbose=debug
        )
    @staticmethod
    def create_css_strategy() -> JsonCssExtractionStrategy:
        schema = {
            "baseSelector": ".product",
            "fields": [
                {"name": "title", "selector": "h1.product-title", "type": "text"},
                {"name": "price", "selector": ".price", "type": "text"},
                {"name": "description", "selector": ".description", "type": "text"},
            ],
        }
        return JsonCssExtractionStrategy(schema=schema)
    @staticmethod
    def create_xpath_strategy() -> JsonXPathExtractionStrategy:
        schema = {
            "baseSelector": "//div[@class='product']",
            "fields": [
                {"name": "title", "selector": ".//h1[@class='product-title']/text()", "type": "text"},
                {"name": "price", "selector": ".//span[@class='price']/text()", "type": "text"},
                {"name": "description", "selector": ".//div[@class='description']/text()", "type": "text"},
            ],
        }
        return JsonXPathExtractionStrategy(schema=schema)
