import os
import subprocess
import tempfile
import json
from typing import List, Dict, Any

from loguru import logger
from tqdm import tqdm

# DPR evaluation
from pyserini.eval.evaluate_dpr_retrieval import has_answers, SimpleTokenizer
# Lucene search
from pyserini.search.lucene import LuceneSearcher

# Dataset & tools
from rankify.dataset.dataset import Document, Context
from rankify.tools.Tools import WebSearchTool

# HTML extraction
import trafilatura

# LLM API imports
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from openai import AzureOpenAI
    AZURE_OPENAI_AVAILABLE = True
except ImportError:
    AZURE_OPENAI_AVAILABLE = False

try:
    from together import Together
    TOGETHER_AVAILABLE = True
except ImportError:
    TOGETHER_AVAILABLE = False

# LangChain splitter
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

class LLMHTMLCleaner:
    def __init__(self, llm_provider: str, api_key: str, model: str = None):
        self.llm_provider = llm_provider.lower()
        self.api_key = api_key

        self.model = model or ("gpt-3.5-turbo" if self.llm_provider == "openai" else "gpt-4o-4")

        if self.llm_provider == "openai" and OPENAI_AVAILABLE:
            openai.api_key = self.api_key
        elif self.llm_provider == "azure_openai" and AZURE_OPENAI_AVAILABLE:
            self.azure_client = AzureOpenAI(
                azure_endpoint="https://openaireceiptwestus.openai.azure.com/",
                api_key=self.api_key,
                api_version="2024-05-01-preview"
            )
        elif self.llm_provider == "together" and TOGETHER_AVAILABLE:
            self.together_client = Together(api_key=self.api_key)

        self.prompt = """You are an HTML content extractor. Your job is to clean HTML content and extract ONLY the main readable text.

CRITICAL RULES:
1. DO NOT generate, create, or write any new content
2. DO NOT summarize, paraphrase, or rephrase existing text
3. DO NOT add your own words, opinions, or explanations
4. ONLY extract and clean the existing text content

WHAT TO REMOVE:
- All HTML tags (<div>, <p>, <span>, etc.)
- Navigation menus and links
- Advertisements and promotional content
- Header and footer elements
- Social media buttons and widgets
- JavaScript and CSS code
- Comments and metadata
- Cookie notices and privacy banners
- \"Read more\", \"Share this\", \"Subscribe\" buttons
- Website navigation (\"Home\", \"About\", \"Contact\")
- Breadcrumbs and site structure elements

WHAT TO KEEP:
- Main article/page content
- Headings and subheadings
- Paragraphs of actual content
- Lists that contain meaningful information
- Table data (if relevant to main content)

OUTPUT FORMAT:
- Return clean text with natural paragraph breaks
- Keep section headings
- Maintain logical text flow
- Use double line breaks between sections
- Do NOT add any prefixes like \"Here is the cleaned content:\"

Now clean this HTML content:"""

    def clean(self, html: str) -> str:
        prompt = f"{self.prompt}\n\nHTML Content:\n{html}"
        try:
            if self.llm_provider == "openai":
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.prompt},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=16384,
                    temperature=0.1,
                    top_p=0.9
                )
                return response.choices[0].message.content.strip()
            elif self.llm_provider == "azure_openai":
                response = self.azure_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.prompt},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=16384,
                    temperature=0.1,
                    top_p=0.9
                )
                return response.choices[0].message.content.strip()
            elif self.llm_provider == "together":
                response = self.together_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.prompt},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=16384,
                    temperature=0.1,
                    top_p=0.9
                )
                return response.choices[0].message.content.strip()
            else:
                raise ValueError("Unsupported LLM provider.")
        except Exception as e:
            print(f"LLM cleaning failed: {e}")
            return ""

class HTMLContentExtractor:
    def __init__(
        self,
        method: str = 'trafilatura',
        llm_provider: str = None,
        llm_api_key: str = None,
        llm_model: str = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        self.method = method
        self.llm_cleaner = None
        if method == 'llm_filter' and llm_provider and llm_api_key:
            self.llm_cleaner = LLMHTMLCleaner(llm_provider, llm_api_key, llm_model)

        if LANGCHAIN_AVAILABLE:
            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len
            )
        else:
            self.splitter = None

    def extract_passages(
        self,
        html_content: str,
        title: str = "",
        source_url: str = "",
        max_passages: int = 5
    ) -> List[Dict[str, Any]]:
        if not html_content or not html_content.strip():
            return []

        if self.method == 'llm_filter' and self.llm_cleaner:
            text = self.llm_cleaner.clean(html_content)
        else:
            text = trafilatura.extract(
                html_content,
                include_comments=False,
                include_tables=False,
                include_links=False,
                favor_precision=True,
                favor_recall=False
            )

        if not text or len(text.strip()) < 50:
            return []

        if self.splitter:
            chunks = self.splitter.split_text(text)
        else:
            chunks = [p for p in text.split("\n\n") if len(p.split()) >= 50]

        passages: List[Dict[str, Any]] = []
        for i, chunk in enumerate(chunks[:max_passages]):
            passages.append({
                'id': f"passage_{i}",
                'title': title or f'Passage {i+1}',
                'contents': chunk.strip(),
                'source_url': source_url,
                'metadata': {
                    'source': self.method,
                    'word_count': len(chunk.split())
                }
            })
        return passages
# ... (Previous classes remain unchanged)

class OnlineRetriever:
    def __init__(
        self,
        n_docs: int = 10,
        api_key: str = None,
        extraction_method: str = 'trafilatura',
        llm_provider: str = None,
        llm_api_key: str = None,
        llm_model: str = None,
        chunk_size: int = 150,
        chunk_overlap: int = 50
    ) -> None:
        self.n_docs = n_docs
        self.searcher = WebSearchTool(search_provider_api_key=api_key)
        self.tokenizer = SimpleTokenizer()
        self.content_extractor = HTMLContentExtractor(
            method=extraction_method,
            llm_provider=llm_provider,
            llm_api_key=llm_api_key,
            llm_model=llm_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def _search_web(self, query: str) -> List[Any]:
        if not self.searcher.is_initialized:
            self.searcher.setup()
        return self.searcher.forward(query, num_result=self.n_docs)

    def retrieve(self, documents: List[Document]) -> List[Document]:
        for idx, doc in enumerate(tqdm(documents, desc="Fetching docs...")):
            question = doc.question.question
            logger.info(f"Fetching contexts for Q{idx}: {question}")

            sources = self._search_web(question)
            processed = []
            for s_idx, source in enumerate(sources):
                html = source.get('html', '')
                with open("log.txt", "w",  encoding="utf-8") as f:
                    f.writelines(html)
                if html:
                    passages = self.content_extractor.extract_passages(
                        html_content=html,
                        title=source.get('title',''),
                        source_url=source.get('link',''),
                        max_passages=self.n_docs
                    )
                    for p_idx, p in enumerate(passages):
                        processed.append({'id': f's{ s_idx }_p{ p_idx }', 'contents': p['contents']})
                else:
                    snippet = source.get('snippet','')
                    if len(snippet.split()) > 20:
                        processed.append({'id': f's{ s_idx }_snip', 'contents': snippet})

            # Indexing
            with tempfile.TemporaryDirectory() as tmpdir:
                json.dump(processed, open(os.path.join(tmpdir,'docs.json'),'w'))
                subprocess.run([
                    'python','-m','pyserini.index.lucene',
                    '-collection','JsonCollection','-generator','DefaultLuceneDocumentGenerator',
                    '-input',tmpdir,'-index',tmpdir,
                    '-storePositions','-storeDocvectors','-storeRaw'
                ], check=True)

                searcher = LuceneSearcher(tmpdir)
                hits = searcher.search(question, k=self.n_docs)
                contexts: List[Context] = []
                for h_idx, hit in enumerate(hits):
                    raw = searcher.doc(hit.docid).raw()
                    data = json.loads(raw)
                    text = data.get('contents','')
                    title = text.split("\n")[0]
                    try:
                        cid = int(hit.docid)
                    except ValueError:
                        cid = h_idx
                    contexts.append(Context(
                        id=cid,
                        title=title,
                        text=text,
                        score=hit.score,
                        has_answer=has_answers(text, doc.answers.answers, self.tokenizer)
                    ))
                doc.contexts = contexts
        return documents
