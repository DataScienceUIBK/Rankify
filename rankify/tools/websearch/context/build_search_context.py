
import os
import json
import subprocess

from pyserini.search.lucene import LuceneSearcher

from rankify.tools.websearch.content_scraping.crawler import WebScraper
from rankify.tools.websearch.content_scraping.scrapedResult import print_extracted_result
from typing import List, TypeVar
import tempfile
T = TypeVar('T')


async def build_search_context(sources: T) -> T:
    if sources is None:
        return []
    stra= ['no_extraction'] #, 'cosine'
    # Filter only Wikipedia sources
    filtered_sources = [
        (i, source)
        for i, source in enumerate(sources.data['organic'])
        #if 'wikipedia.org' in source.get('link', '')
    ]

    urls = [source[1]['link'] for source in filtered_sources] 
    print(f"Filtered Wikipedia URLs: {urls}")

    scraper = WebScraper(strategies=stra) #strategies=['no_extraction']
    multi_results = await scraper.scrape_many(urls)

    sources_with_html: List[T] = []
    for _, source in filtered_sources:
        url = source['link']
        if url in multi_results:
            source['html'] = multi_results[url]['no_extraction'].content
            source['fit_markdown'] = multi_results[url]['no_extraction'].fit_markdown
            #print(multi_results[url]['no_extraction'] , source['fit_markdown'])
            sources_with_html.append(source)

    return sources_with_html


