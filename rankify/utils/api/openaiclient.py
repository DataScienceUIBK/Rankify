
from openai import OpenAI
import openai
import time

class OpenaiClient:
    def __init__(self, keys=None, base_url=None, start_id=None, proxy=None, api_version=None):

        if isinstance(keys, str):
            keys = [keys]
        if keys is None:
            raise "Please provide OpenAI Key."

        self.key = keys
        self.base_url = base_url
        self.key_id = start_id or 0
        self.key_id = self.key_id % len(self.key)
        self.api_key = self.key[self.key_id % len(self.key)]
        print(self.base_url)

        # Use AzureOpenAI when base_url points to an Azure endpoint
        if base_url and ".openai.azure.com" in base_url:
            try:
                from openai import AzureOpenAI
                import os
                # Extract api_version from env or passed param
                _api_version = (api_version
                                or os.getenv("RANKIFY_AZURE_API_VERSION", "2025-01-01-preview"))
                # Extract endpoint (everything before /openai/...)
                _endpoint = base_url.split("/openai/")[0] if "/openai/" in base_url else base_url
                self.client = AzureOpenAI(
                    azure_endpoint=_endpoint,
                    api_key=self.api_key,
                    api_version=_api_version,
                )
                # Store deployment name from the base_url path for chat calls
                self._azure_deployment = base_url.split("/deployments/")[-1].split("/")[0] if "/deployments/" in base_url else None
            except ImportError:
                self.client = OpenAI(api_key=self.api_key, base_url=base_url)
                self._azure_deployment = None
        else:
            self.client = OpenAI(api_key=self.api_key, base_url=base_url)
            self._azure_deployment = None


    def chat(self, *args, return_text=False, reduce_length=False, **kwargs):
        while True:
            try:
                completion = self.client.chat.completions.create(*args, **kwargs, timeout=30)
                break
            except Exception as e:
                print(str(e))
                if "This model's maximum context length is" in str(e):
                    print('reduce_length')
                    return 'ERROR::reduce_length'
                time.sleep(0.1)
        if return_text:
            completion = completion.choices[0].message.content
        return completion

    def text(self, *args, return_text=False, reduce_length=False, **kwargs):
        while True:
            try:
                completion = self.client.completions.create(
                    *args, **kwargs
                )
                break
            except Exception as e:
                print(e)
                if "This model's maximum context length is" in str(e):
                    print('reduce_length')
                    return 'ERROR::reduce_length'
                time.sleep(0.1)
        if return_text:
            completion = completion.choices[0].text
        return completion
