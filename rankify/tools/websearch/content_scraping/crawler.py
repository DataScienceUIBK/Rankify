"""
    Web scraper implementation using Crawl4AI
"""
import asyncio
from typing import Dict, List, Optional

from loguru import logger
from crawl4ai import BrowserConfig , CrawlerRunConfig , CacheMode , AsyncWebCrawler
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.content_filter_strategy import PruningContentFilter

from rankify.tools.websearch.content_scraping.scrapedResult import ScrapedResult, print_extracted_result
from rankify.tools.websearch.content_scraping.strategyFactory import StrategyFactory
from rankify.tools.websearch.content_scraping.config import ExtractionConfig


class WebScraper:
    #'markdown_llm', 'html_llm','fit_markdown_llm','css', 'xpath','cosine'
    def __init__(self,browser_config:Optional[BrowserConfig]=None,strategies : List[str]= [   'no_extraction', ],llm_instruction:str="Extract relevant content from the provided text, only return the text, no markdown formatting, remove all footnotes, citations, and other metadata and only keep the main content",
                 query:Optional[str]=None,filter_content:bool=True):
        self.browser_config = browser_config
        self.strategies = strategies
        self.llm_instruction = llm_instruction
        self.query = query
        self.filter_content = filter_content
        self.strategy_factory = StrategyFactory()
        valid_strategies = {  'no_extraction', } #'markdown_llm', 'html_llm', 'fit_markdown_llm','css', 'xpath','cosine'
        invalid_strategies = set(strategies) - valid_strategies
        if invalid_strategies:
            raise ValueError(f"Invalid strategies: {invalid_strategies}")
        self.strategy_map = {
            'markdown_llm': lambda : self.strategy_factory.create_llm_strategy(),
            'fit_markdown_llm': lambda: self.strategy_factory.create_llm_strategy('fit_markdown', self.llm_instruction),
            'html_llm': lambda: self.strategy_factory.create_llm_strategy('html', self.llm_instruction),
            'cosine': lambda:self.strategy_factory.create_cosine_strategy(),
            'no_extraction': lambda:self.strategy_factory.create_no_extraction_strategy(),
            'css': self.strategy_factory.create_css_strategy,
            'xpath': self.strategy_factory.create_xpath_strategy,
        }

    def _create_crawler_config_old(self):
        filter_content = PruningContentFilter(user_query=self.query) if self.query else PruningContentFilter()
        return CrawlerRunConfig(cache_mode=CacheMode.BYPASS,markdown_generator=DefaultMarkdownGenerator(content_filter=filter_content))
    def _create_crawler_config(self):
        prune_filter = PruningContentFilter(
            # Lower → more content retained, higher → more content pruned
            threshold=0.45,           
            # "fixed" or "dynamic"
            threshold_type="dynamic",  
            # Ignore nodes with <5 words
            min_word_threshold=5      
        )
        # Step 2: Insert it into a Markdown Generator
        md_generator = DefaultMarkdownGenerator(content_filter=prune_filter)

        # Step 3: Pass it to CrawlerRunConfig
        config = CrawlerRunConfig(
            markdown_generator=md_generator
        )
        return config

    async def extract(self, extraction_config: ExtractionConfig, url: str) -> ScrapedResult:
            """ Method to perform extraction using a strategy(e.g., markdown strategy)"""
            logger.info("Extracting...")
            try:
                config = self._create_crawler_config()
                config.extraction_strategy = extraction_config.strategy
                #config=self.browser_config
                async with AsyncWebCrawler() as crawler:
                    if isinstance(url, list):
                        results = await crawler.arun_many(urls=url, config=config)
                    else:
                        results = await crawler.arun(url=url, config=config)
                    # print(results.success)
                    # print(results.markdown.fit_markdown)
                    # print("*" * 80)
                    # print(results.markdown.raw_markdown)
                    # ddddd
                content = None
                fit_markdown = None
                if results.success:
                    #print(extraction_config.name)
                    if extraction_config.name in ['no_extraction', 'cosine']:
                        if hasattr(results, 'markdown'):
                            content = results.markdown.raw_markdown
                        elif hasattr(results, 'raw_html'):
                            content = results.raw_html
                        elif hasattr(results, 'extracted_content') and results.extracted_content:
                            if isinstance(results.extracted_content, list):
                                content = '\n'.join(item.get('content', '') for item in results.extracted_content)
                            else:
                                content = results.extracted_content
                        if hasattr(results.markdown, 'fit_markdown'):
                            fit_markdown = results.markdown.fit_markdown
                        if self.filter_content and content:
                            from rankify.tools.websearch.content_scraping.utils import filter_quality_content
                            content = filter_quality_content(content)

                        #asdasdasdasd
                    else:
                        content = results.extracted_content
                        from rankify.tools.websearch.content_scraping.utils import filter_quality_content
                        content = filter_quality_content(content)
                #print(f"Content extracted: {content[:100]}...")  # Debug print
                #print(f"Fit markdown: {fit_markdown[:100]}...")  # Debug print
                extracted_result = ScrapedResult(
                    name=extraction_config.name,
                    success=results.success,
                    content=content,
                    fit_markdown = fit_markdown,
                    error=getattr(results, 'error', None)
                )
                if results.success:
                    extracted_result.raw_markdown_len = len(results.markdown.raw_markdown)
                    extracted_result.citations_markdown_len = len(results.markdown.markdown_with_citations)

                return extracted_result

            except Exception as e:
                return ScrapedResult(
                    name=extraction_config.name,
                    success=False,
                    error=str(e)
                )

    async def scrape(self, url: str) -> dict[str, ScrapedResult]:
        """ Scrape URL using the chosen strategy"""
        logger.info(f"Scraping {url}")
        results = {}
        # Handle Wikipedia URLs
        # if 'wikipedia.org/wiki/' in url:
        #     from rankify.tools.websearch.content_scraping.utils import get_wikipedia_content
        #     try:
        #         content = get_wikipedia_content(url)
        #         results['no_extraction'] = ScrapedResult(name='no_extraction', success=True, content=content)
               
        #     except Exception as e:
        #         raise ValueError(f"Debug: Wikipedia extraction failed {str(e)}")
        
        for strategy_name in self.strategies:
            config = ExtractionConfig(
                name=strategy_name, strategy=self.strategy_map[strategy_name]()
            )
            result = await self.extract(config, url=url)
            results[strategy_name] = result

        return results

    async def scrape_many(self, urls: List[str], pro_mode:Optional[bool]=True) -> Dict[str, Dict[str, ScrapedResult]]:
        """ Extracted multiple URLs concurrently using chosen strategies """
       
        tasks = [self.scrape(url) for url in urls]
        results_list = await asyncio.gather(*tasks)

        results = {}
        for url, result in zip(urls, results_list):
            results[url] = result
            #print(f"Scraped {url} with results: {result}")
        return results

async def main():
    urls = [
        "https://en.wikipedia.org/wiki/Apple_Inc.",
        "https://python.org",
        "https://github.com"
    ]
    scraper = WebScraper(strategies=['no_extraction'])
    multi_results = await scraper.scrape_many(urls)

    # Print multiple URL results
    for url, url_results in multi_results.items():
        print(f"\nResults for {url}:")
        for result in url_results.values():
            print_extracted_result(result)


if __name__ == "__main__":
    asyncio.run(main())