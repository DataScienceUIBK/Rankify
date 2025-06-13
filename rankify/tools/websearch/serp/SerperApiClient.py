import requests
from rankify.tools.websearch.models.SerpResults import SerpResult
from rankify.tools.websearch.serp.SearchAPIClient import SearchAPIClient
from typing import Optional, Dict, Any, List

from rankify.tools.websearch.serp.config import SerpConfig
from rankify.tools.websearch.serp.errors import SerperAPIException


class SerperApiClient(SearchAPIClient):

    def __init__(self, api_key:Optional[str]=None,config:Optional[SerpConfig]=None):
        if api_key:
            self.config = SerpConfig(api_key=api_key)
        elif config:
            self.config = config
        else:
            self.config = SerpConfig.load_env_vars()
        self.header = {
            'Accept': 'application/json',
            'X-API-Key': self.config.api_key,
        }

    def search_web(self,query:str,num_results:int = 10 , search_location:Optional[str]=None) -> SerpResult[
        Dict[str, Any]]:
        
        
        if not query.strip():
            return SerperAPIException(error=f"query is required")
        try:
            payload = {
                'q': query,
                'numResults': num_results,
                'gl': search_location or self.config.default_location
            }
            response = requests.post(
                url=self.config.api_url,
                headers=self.header,
                json=payload,
                timeout=self.config.timeout,
            )
            response.raise_for_status()

            data = response.json()

            results = {
                'organic': self.extract_fields(data.get('organic',[]),['title','link','snippet','date']),
                'topStories': self.extract_fields(data.get('topStories',[]),['title','imageUrl']),
                'images':self.extract_fields(data.get('images',[])[:6],['title','imageUrl']),
                'answerBox':data.get('answerBox'),
                'peopleAlsoAsk':data.get('peopleAlsoAsk'),
                'relatedSearches':data.get('relatedSearches')
            }

            return SerpResult(data=results)

        except requests.RequestException as e:
            return SerpResult(error=f"Serper API request failed {str(e)}")
        except Exception as e:
            return SerpResult(error=f" Unexcpected error {str(e)}")

    @staticmethod
    def extract_fields(items:List[Dict[str, Any]], field:List[str]) -> List[Dict[str, Any]]:
        return [{key:item.get(key,"") for key in field if key in item} for item in items]

