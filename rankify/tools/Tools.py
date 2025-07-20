import asyncio
from typing import Optional

import nest_asyncio

from rankify.tools.websearch.context.build_search_context import build_search_context
from rankify.tools.websearch.serp.SearchAPIClient import SearchAPIClient
from rankify.tools.websearch.serp.SerperApiClient import SerperApiClient
# noinspection PyTypeHints


class Tool:
    """
        A base class for the functions used by the agent. Subclass this and implement the `forward` method as well as the
        following class attributes:

        - **description** (`str`) -- A short description of what your tool does, the inputs it expects and the output(s) it
          will return. For instance, 'This is a tool that downloads a file from a `url`. It takes the `url` as input, and
          returns the text contained in the file'.
        - **name** (`str`) -- A performative name that will be used for your tool in the prompt to the agent. For instance
          `"text-classifier"` or `"image_generator"`.
        - **inputs** (`Dict[str, Dict[str, Union[str, type, bool]]]`) -- The dict of modalities expected for the inputs.
          It has one `type' key and a `description`key.
          This is used by `launch_gradio_demo` or to make a nice space from your tool, and also can be used in the generated
          description for your tool.
        - **output_type** (`type`) -- The type of the tool output. This is used by `launch_gradio_demo`
          or to make a nice space from your tool, and also can be used in the generated description for your tool.
"""
    name: str
    description: str
    inputs: dict[str, dict[str, str | type | bool]]
    output_type: str

    def __init__(self):
        self.is_initialized = False

    def validate_arguments(self):
        required_attributes = {
            "description": str,
            "name": str,
            "inputs": dict[str, dict[str, str | type | bool]],
            "output_type": str
        }
        for attr, expected_type in required_attributes.items():
            attr_value = getattr(self, attr, None)
            if attr_value is None:
                raise ValueError(f"Attribute {attr} is required for tool {self.name}")
            if not isinstance(attr_value, expected_type):# noqa
                raise ValueError(f"Attribute {attr} for tool {self.name} must be of type {expected_type}")

    def forward(self, query: str):
        return NotImplementedError("Write this method in your subclass of `Tool`.")

    def setup(self):
        """
        Overwrite this method here for any operation that is expensive and needs to be executed before you start using
        your tool. Such as loading a big model.
        """
        self.is_initialized = True


class WebSearchTool(Tool):

    name = "web_search"
    description = "Performs web search using user query and returns the top n results."
    inputs = {
        'query': {
            'type': 'string',
            'description': 'The user search query'
        }
    }
    output_type = "string"

    def __init__(self, model: Optional[str]=None, search_provider: str = 'serper',
                 search_provider_api_key: Optional[str] = None):
        super().__init__()
        self.search_client = None
        self.model = model
        self.search_provider = search_provider
        self.search_provider_api_key = search_provider_api_key
#        self.validate_arguments()

    @staticmethod
    def create_search_api(search_provider: str = 'SERPER', api_key: Optional[str] = None) -> SearchAPIClient:
        """
        Instantiate a search API client to fetch SERP results from a SERP provider e.g., SERPER.dev.
        Args:
            search_provider (str): The name of the search provider.
            Api_key (str): The API key for the search provider. Can be stored and loaded from env vars.
        Returns:
            SearchAPIClient: An instance of the search API client. E.g., SerpAPIClient.
        """
        if search_provider.lower() == 'serper':
            #print(api_key)
            return SerperApiClient(api_key=api_key)
        else:
            raise ValueError(f'Invalid search provider{search_provider}')

    def forward(self, query: str,num_result:int=10):
        sources = self.search_client.search_web(query=query,num_results=num_result)
        #print(sources, len(sources.data['organic']), num_result)

        
        #asdadsasdad
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                nest_asyncio.apply()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(build_search_context(sources))



    def setup(self):
        """
        Performs time-consuming setup operations here e.g., model loading.
        """
        self.search_client = self.create_search_api(api_key=self.search_provider_api_key)
        self.is_initialized = True
