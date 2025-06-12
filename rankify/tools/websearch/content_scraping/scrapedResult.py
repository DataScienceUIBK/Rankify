

from dataclasses import dataclass
from typing import Optional


@dataclass
class ScrapedResult:
    def __init__(self, name:str,success:bool, content:Optional[str]=None,error:Optional[str]=None , fit_markdown:Optional[str]=None):
        print(f"results:{content}")
        self.name = name
        self.success = success
        self.content = content
        self.error = error
        self.raw_markdown_len= 0
        self.citations_markdown_len = 0
        self.fit_markdown =fit_markdown


def print_extracted_result(result: ScrapedResult):
    """ Method to print out the extracted results"""
    if result.success:
        print(f"\n=={result.name} Results ===")
        print(f"Extracted content: {result.content}")
        print(f"Raw markdown length: {result.raw_markdown_len}")
        print(f"Citations markdown length: {result.citations_markdown_len}")
    else:
        print(f"Error in {result.name}: {result.error}") 
