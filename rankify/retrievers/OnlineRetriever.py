import asyncio
import os
import subprocess
from typing import List, TypeVar, Dict, Any
import tempfile
from loguru import logger
from pyserini.eval.evaluate_dpr_retrieval import has_answers, SimpleTokenizer
from tqdm import tqdm

from rankify.dataset.dataset import Document, Question, Context, Answer
from rankify.tools.Tools import WebSearchTool
import json
from pyserini.search.lucene import LuceneSearcher

import re
import html2text
from bs4 import BeautifulSoup
from typing import List
T = TypeVar('T')

import trafilatura
from bs4 import BeautifulSoup
import re
from typing import List, Dict, Any

# LangChain imports
try:
    from langchain.text_splitter import (
        RecursiveCharacterTextSplitter,
        MarkdownHeaderTextSplitter,
        CharacterTextSplitter
    )
    from langchain.schema import Document as LangChainDocument
    LANGCHAIN_AVAILABLE = True
except ImportError:
    print("LangChain not available. Please install: pip install langchain")
    LANGCHAIN_AVAILABLE = False

# LLM API imports
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    from openai import AzureOpenAI
    AZURE_OPENAI_AVAILABLE = True
except ImportError:
    AZURE_OPENAI_AVAILABLE = False

class LLMHTMLCleaner:
    """Clean HTML content using LLM APIs (OpenAI GPT or Together AI)"""
    
    def __init__(self, llm_provider: str = "openai", api_key: str = None, model: str = None):
        """
        Initialize LLM HTML cleaner
        
        Args:
            llm_provider: "openai", "together", or "azure_openai"
            api_key: API key for the LLM provider
            model: Model name (e.g., "gpt-3.5-turbo", "gpt-4o-4")
        """
        self.llm_provider = llm_provider.lower()
        self.api_key = api_key
        
        # Set default models
        if model is None:
            if self.llm_provider == "openai":
                self.model = "gpt-3.5-turbo"
            elif self.llm_provider == "azure_openai":
                self.model = "gpt-4o-4"
            else:
                self.model = "gpt-3.5-turbo"
        else:
            self.model = model
        
        # Initialize API client
        if self.llm_provider == "openai" and OPENAI_AVAILABLE:
            openai.api_key = self.api_key
        elif self.llm_provider == "azure_openai" and AZURE_OPENAI_AVAILABLE:
            self.azure_client = AzureOpenAI(
                azure_endpoint="https://openaireceiptwestus.openai.azure.com/",
                api_key=self.api_key,
                api_version="2024-05-01-preview",
            )

        if LANGCHAIN_AVAILABLE:
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=200,  # ~150 words
                chunk_overlap=50,  # Some overlap to maintain context
                length_function=len,
            )
        else:
            self.text_splitter = None
            print("LangChain not available. Using fallback splitting.")    
        
        self.cleaning_prompt = self._create_cleaning_prompt()
    
    def _create_cleaning_prompt(self) -> str:
        """Create a detailed prompt for HTML content cleaning"""
        return """You are an HTML content extractor. Your job is to clean HTML content and extract ONLY the main readable text.

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
- "Read more", "Share this", "Subscribe" buttons
- Website navigation ("Home", "About", "Contact")
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
- Do NOT add any prefixes like "Here is the cleaned content:"

EXAMPLE:
Input: <div><h1>Article Title</h1><nav>Home | About</nav><p>This is the main content.</p><footer>Copyright 2023</footer></div>
Output: Article Title

This is the main content.

Now clean this HTML content:"""

    def clean_html_with_llm(self, html_content: str, title: str = "") -> str:
        """Clean HTML content using LLM API"""
        
        if not html_content or len(html_content.strip()) < 100:
            return ""
        
        # Truncate very long content to avoid API limits
        # max_chars = 8000  # Adjust based on your API limits
        # if len(html_content) > max_chars:
        #     html_content = html_content[:max_chars] + "..."
        
        # Prepare the prompt
        full_prompt = f"{self.cleaning_prompt}\n\nHTML Content:\n{html_content}"
        
        try:
            if self.llm_provider == "openai":
                return self._clean_with_openai(full_prompt)
            elif self.llm_provider == "azure_openai":
                return self._clean_with_azure_openai(full_prompt)
            else:
                print(f"Unknown LLM provider: {self.llm_provider}")
                return ""
        except Exception as e:
            print(f"LLM cleaning failed: {e}")
            return ""
    
    def _clean_with_openai(self, prompt: str) -> str:
        """Clean content using OpenAI API"""
        if not OPENAI_AVAILABLE:
            print("OpenAI not available. Please install: pip install openai")
            return ""
        
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert HTML content extractor. Extract only the main readable content, removing all HTML markup and navigation elements. Do not generate new content."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=16384,  # Adjust based on needs
                temperature=0.1,  # Low temperature for consistent extraction
                top_p=0.9
            )
            
            cleaned_content = response.choices[0].message.content.strip()
            return cleaned_content
            
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return ""
    def _clean_with_azure_openai(self, prompt: str) -> str:
        """Clean content using Azure OpenAI API"""
        if not AZURE_OPENAI_AVAILABLE:
            print("Azure OpenAI not available. Please install: pip install openai")
            return ""
        
        try:
            response = self.azure_client.chat.completions.create(
                model=self.model,  # This will be "gpt-4o-4" 
                messages=[
                    {"role": "system", "content": "You are an expert HTML content extractor. Extract only the main readable content, removing all HTML markup and navigation elements. Do not generate new content."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=16384,
                temperature=0.1,
                top_p=0.9
            )
            
            cleaned_content = response.choices[0].message.content.strip()
            print(cleaned_content)
            #sadasd
            return cleaned_content
            
        except Exception as e:
            print(f"Azure OpenAI API error: {e}")
            return ""
    def _clean_with_together(self, prompt: str) -> str:
        """Clean content using Together AI API"""
        if not TOGETHER_AVAILABLE:
            print("Together AI client not available. Please install: pip install together")
            return ""
        
        try:
            client = Together(api_key=self.api_key)
            
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert HTML content extractor. Extract only the main readable content, removing all HTML markup and navigation elements. Do not generate new content."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100000,
                temperature=0.1,
                top_p=0.9
            )
            
            cleaned_content = response.choices[0].message.content.strip()
            return cleaned_content
            
        except Exception as e:
            print(f"Together AI API error: {e}")
            return ""
    
    def extract_passages_with_llm(self, html_content: str, title: str = "", source_url: str = "", max_passages: int = 5) -> List[Dict[str, Any]]:
        """Extract clean passages using LLM and LangChain splitting"""
        
        print(f"Using LLM ({self.llm_provider}) to clean HTML content")
        
        # Clean content with LLM
        cleaned_text = self.clean_html_with_llm(html_content, title)
        
        if not cleaned_text or len(cleaned_text.strip()) < 100:
            print("LLM cleaning produced insufficient content")
            return []
        
        print(f"Cleaned text length: {len(cleaned_text)} characters")
        
        # Split cleaned content into passages using LangChain
        passages = []
        
        if LANGCHAIN_AVAILABLE and self.text_splitter:
            print("Using LangChain RecursiveCharacterTextSplitter")
            
            # Split text using LangChain
            text_chunks = self.text_splitter.split_text(cleaned_text)
            
            for passage_id, chunk in enumerate(text_chunks):
                chunk = chunk.strip()
                word_count = len(chunk.split())
                
                # Only keep substantial passages
                if word_count >= 10:
                    passages.append({
                        'id': f"llm_passage_{passage_id}",
                        'contents': chunk,
                        'title': title,
                        'source_url': source_url,
                        'metadata': {
                            'source': 'llm_cleaned',
                            'llm_provider': self.llm_provider,
                            'word_count': word_count,
                            'splitting_method': 'langchain_recursive'
                        }
                    })
            
            print(f"LangChain splitting created {len(passages)} passages")
        
        else:
            print("LangChain not available, using fallback splitting")
            # Fallback to original splitting method
            sections = cleaned_text.split('\n\n')
            current_passage = []
            current_word_count = 0
            passage_id = 0
            
            for section in sections:
                section = section.strip()
                if not section:
                    continue
                
                section_words = len(section.split())
                
                # If adding this section would make passage too long, save current and start new
                if current_word_count + section_words > 150 and current_passage:
                    passage_text = '\n\n'.join(current_passage)
                    if len(passage_text.split()) >= 20:  # Only keep substantial passages
                        passages.append({
                            'id': f"llm_passage_{passage_id}",
                            'contents': passage_text,
                            'title': title,
                            'source_url': source_url,
                            'metadata': {
                                'source': 'llm_cleaned',
                                'llm_provider': self.llm_provider,
                                'word_count': len(passage_text.split()),
                                'splitting_method': 'fallback_manual'
                            }
                        })
                        passage_id += 1
                    
                    current_passage = [section]
                    current_word_count = section_words
                else:
                    current_passage.append(section)
                    current_word_count += section_words
            
            # Don't forget the last passage
            if current_passage:
                passage_text = '\n\n'.join(current_passage)
                if len(passage_text.split()) >= 20:
                    passages.append({
                        'id': f"llm_passage_{passage_id}",
                        'contents': passage_text,
                        'title': title,
                        'source_url': source_url,
                        'metadata': {
                            'source': 'llm_cleaned',
                            'llm_provider': self.llm_provider,
                            'word_count': len(passage_text.split()),
                            'splitting_method': 'fallback_manual'
                        }
                    })
            
            print(f"Fallback splitting created {len(passages)} passages")
        
        return passages

class WikipediaProcessor:
    """Enhanced Wikipedia content processor using LangChain"""
    
    def __init__(self, chunk_size: int = 128, chunk_overlap: int = 20):
        self.chunk_size = chunk_size * 5  # Approximate words to characters
        self.chunk_overlap = chunk_overlap
        
        if LANGCHAIN_AVAILABLE:
            # Initialize LangChain splitters
            self.recursive_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                #separators=["\n\n", "\n", "## ", "# ", ". ", " ", ""]
            )
            
            self.markdown_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=[
                    ("#", "Header 1"),
                    ("##", "Header 2"), 
                    ("###", "Header 3"),
                ]
            )
        else:
            # Fallback to simple splitting
            self.recursive_splitter = None
            self.markdown_splitter = None
    
    def clean_wikipedia_content(self, text: str) -> str:
        """Clean Wikipedia-specific markup and navigation elements"""
        
        # Remove navigation elements
        text = re.sub(r'\[Jump to content\].*?(?=\n# )', '', text, flags=re.DOTALL)
        text = re.sub(r'Main menu.*?(?=\n# )', '', text, flags=re.DOTALL)
        
        # Clean Wikipedia links but preserve text
        text = re.sub(r'\[([^\]]+)\]\([^)]+\s+"[^"]*"\)', r'\1', text)
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        
        # Remove edit links - multiple variations
        text = re.sub(r'\[\[edit\]\]', '', text)
        text = re.sub(r'\[edit\]', '', text)  # Simple edit markers
        text = re.sub(r'\(edit\)', '', text)
        text = re.sub(r'\(https://en\.wikipedia\.org[^)]+\)', '', text)
        
        # Clean table formatting
        text = re.sub(r'\|([^|\n]+)\|', r'\1', text)
        text = re.sub(r'^---+$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*\|\s*', '', text, flags=re.MULTILINE)
        
        # Remove language links section
        text = re.sub(r'\n\d+ languages.*?(?=\n#)', '', text, flags=re.DOTALL)
        
        # Remove references and citations
        text = re.sub(r'\[\d+\]', '', text)
        text = re.sub(r'\[citation needed\]', '', text)
        text = re.sub(r'\[\^[^\]]+\]', '', text)
        
        # Remove footer/navigation content
        text = re.sub(r'Retrieved from.*$', '', text, flags=re.DOTALL)
        text = re.sub(r'Categories:.*$', '', text, flags=re.DOTALL)
        text = re.sub(r'Hidden categories:.*$', '', text, flags=re.DOTALL)
        
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        text = re.sub(r'^\s+', '', text, flags=re.MULTILINE)
        
        return text.strip()
    
    def extract_sections(self, text: str) -> List[Dict]:
        """Extract and process Wikipedia sections using markdown splitter"""
        
        # Clean the content first
        cleaned_text = self.clean_wikipedia_content(text)
        
        if LANGCHAIN_AVAILABLE and self.markdown_splitter:
            try:
                # Split by markdown headers first
                markdown_docs = self.markdown_splitter.split_text(cleaned_text)
                return [{'content': doc.page_content, 'metadata': doc.metadata} for doc in markdown_docs]
            except Exception as e:
                print(f"Markdown splitting failed: {e}")
        
        # Fallback: manual section splitting with better regex
        sections = []
        current_section = []
        current_header = "Introduction"
        
        lines = cleaned_text.split('\n')
        for line in lines:
            line = line.strip()
            
            # Detect headers - look for # patterns or Wikipedia section patterns
            if (line.startswith('##') or line.startswith('#') or 
                (line and line.isupper() and len(line.split()) <= 4) or
                re.match(r'^[A-Z][a-z\s]+$', line) and len(line) < 50):
                
                # Save previous section
                if current_section:
                    content = '\n'.join(current_section).strip()
                    if len(content) > 100:  # Only keep substantial sections
                        sections.append({
                            'content': content,
                            'metadata': {'section': current_header}
                        })
                
                # Start new section
                current_header = line.strip('#').strip()
                if not current_header:
                    current_header = ""
                current_section = []
            else:
                if line:  # Skip empty lines
                    current_section.append(line)
        
        # Don't forget the last section
        if current_section:
            content = '\n'.join(current_section).strip()
            if len(content) > 100:
                sections.append({
                    'content': content,
                    'metadata': {'section': current_header}
                })
        
        # If no sections found, treat entire content as one section
        if not sections and cleaned_text:
            sections.append({
                'content': cleaned_text,
                'metadata': {'section': 'Main Content'}
            })
        
        print(f"Detected sections: {[s['metadata']['section'] for s in sections]}")
        return sections
    
    def chunk_documents(self, sections: List[Dict]) -> List[Dict]:
        """Further chunk sections into smaller, word-limited pieces"""
        
        final_chunks = []
        
        for section in sections:
            content = section['content']
            metadata = section['metadata']
            
            if LANGCHAIN_AVAILABLE and self.recursive_splitter:
                # Use LangChain splitter
                chunks = self.recursive_splitter.split_text(content)
            else:
                # Fallback: simple paragraph splitting
                chunks = self._simple_chunk_split(content)
            
            for i, chunk in enumerate(chunks):
                # Filter out very short chunks
                word_count = len(chunk.split())
                if word_count < 10:
                    continue
                    
                # Skip if mostly navigation/metadata
                if self._is_navigation_chunk(chunk):
                    continue
                
                # Create passage metadata
                chunk_metadata = {
                    'section': metadata.get('section', ''),
                    'subsection': metadata.get('Header 3', ''),
                    'chunk_index': i,
                    'word_count': word_count
                }
                
                final_chunks.append({
                    'contents': chunk.strip(),
                    'metadata': chunk_metadata
                })
        
        return final_chunks
    
    def _simple_chunk_split(self, text: str, max_words: int = 128) -> List[str]:
        """Simple fallback chunking when LangChain is not available"""
        
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = []
        current_word_count = 0
        
        for paragraph in paragraphs:
            para_words = len(paragraph.split())
            
            if current_word_count + para_words <= max_words:
                current_chunk.append(paragraph)
                current_word_count += para_words
            else:
                # Save current chunk
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                # Start new chunk
                current_chunk = [paragraph]
                current_word_count = para_words
        
        # Don't forget the last chunk
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        return chunks
    
    def _is_navigation_chunk(self, chunk: str) -> bool:
        """Check if chunk is mostly navigation/metadata"""
        nav_indicators = [
            'Retrieved from', 'Categories:', 'Hidden categories:',
            'Privacy policy', 'About Wikipedia', 'Contact Wikipedia',
            'Toggle the table', 'Edit links', 'External links',
            'References', 'Citations', 'Bibliography', 'Contents',
            'Languages', 'Search', 'Main menu', 'Navigation',
            '[edit]', '(edit)', 'Jump to content', 'move to sidebar',
            'Personal tools', 'Pages for logged out', 'Contribute',
            'Create account', 'Log in', 'Donate', 'Help', 'Learn to edit',
            'Community portal', 'Recent changes', 'Upload file',
            'Special pages', 'What links here', 'Related changes',
            'Permanent link', 'Page information', 'Cite this page'
        ]
        
        chunk_lower = chunk.lower()
        
        # Check for navigation indicators
        nav_count = sum(1 for indicator in nav_indicators if indicator.lower() in chunk_lower)
        if nav_count >= 2:  # Multiple navigation indicators
            return True
        
        # Check if chunk is mostly links/references
        if chunk.count('[') > len(chunk.split()) * 0.3:
            return True
            
        # Check if chunk is very repetitive (common in navigation)
        words = chunk.split()
        if len(words) > 5 and len(set(words)) < len(words) * 0.4:
            return True
        
        return any(indicator.lower() in chunk_lower for indicator in nav_indicators)
    
    def process_wikipedia_content(self, content: str, title: str = "", source_url: str = "") -> List[Dict]:
        """Main processing pipeline for Wikipedia content"""
        
        if not content:
            return []
        
        print(f"Processing Wikipedia content: {title}")
        
        # Extract sections
        sections = self.extract_sections(content)
        print(f"Extracted {len(sections)} sections")
        
        # Further chunk into smaller pieces
        final_chunks = self.chunk_documents(sections)
        print(f"Created {len(final_chunks)} final chunks")
        
        # Add document metadata
        for i, chunk in enumerate(final_chunks):
            chunk['id'] = f"wiki_{i}"
            chunk['title'] = title or 'Wikipedia Article'
            chunk['source_url'] = source_url
        
        return final_chunks

class HTMLContentExtractor:
    """Extract clean passages from HTML content using multiple strategies"""
    
    def __init__(self, method='hybrid', llm_config: Dict = None):
        """
        Initialize HTML content extractor
        
        Args:
            method: 'hybrid', 'trafilatura', 'beautifulsoup', 'wikipedia_filter', or 'llm_filter'
            llm_config: Dict with LLM configuration for llm_filter method
                       {'provider': 'openai'/'together', 'api_key': 'xxx', 'model': 'gpt-3.5-turbo'}
        """
        self.method = method
        
        # Initialize Wikipedia processor
        self.wikipedia_processor = WikipediaProcessor(chunk_size=128, chunk_overlap=20)
        
        # Initialize LLM cleaner if needed
        self.llm_cleaner = None
        if method == 'llm_filter' and llm_config:
            self.llm_cleaner = LLMHTMLCleaner(
                llm_provider=llm_config.get('provider', 'openai'),
                api_key=llm_config.get('api_key'),
                model=llm_config.get('model')
            )
            if not self.llm_cleaner.api_key:
                print("Warning: No API key provided for LLM filter. Falling back to hybrid method.")
                self.method = 'hybrid'
                self.llm_cleaner = None
    
    def extract_with_trafilatura(self, html_content: str) -> List[str]:
        """Extract content using trafilatura with better error handling"""
        try:
            if not html_content or not html_content.strip():
                return []
            
            html_content = html_content.strip()
            if not html_content.startswith('<'):
                if len(html_content) > 100:
                    return [html_content]
                return []
            
            text = trafilatura.extract(
                html_content, 
                include_comments=False, 
                include_tables=True,
                include_links=False,
                favor_precision=True,
                favor_recall=False
            )
            
            if not text or len(text.strip()) < 50:
                return []
            
            paragraphs = []
            for p in text.split('\n\n'):
                p = p.strip()
                if len(p) > 80 and len(p.split()) > 10:
                    paragraphs.append(p)
            
            return paragraphs
        except Exception as e:
            print(f"Trafilatura extraction failed: {e}")
            return []
    
    def extract_with_beautifulsoup(self, html_content: str) -> List[str]:
        """Extract content using BeautifulSoup with smart detection"""
        try:
            if not html_content or not html_content.strip():
                return []
            
            if not html_content.strip().startswith('<'):
                if len(html_content) > 100:
                    return [html_content.strip()]
                return []
            
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove unwanted elements
            unwanted_selectors = [
                'script', 'style', 'nav', 'header', 'footer', 'aside', 
                'menu', 'form', 'button', 'input', 'select', 'textarea',
                '.navigation', '.nav', '.menu', '.sidebar', '.footer', 
                '.header', '.ad', '.ads', '.advertisement', '.social',
                '#navigation', '#nav', '#menu', '#sidebar', '#footer',
                '#header', '#ad', '#ads', '[class*="nav"]', '[class*="menu"]',
                '[class*="sidebar"]', '[class*="footer"]', '[class*="header"]',
                '[class*="ad"]', '[id*="nav"]', '[id*="menu"]',
                '[id*="sidebar"]', '[id*="footer"]', '[id*="header"]',
                '[id*="ad"]', '.mw-editsection', '.reference', '.citation',
                '.infobox', '.navbox', '.metadata'
            ]
            
            for selector in unwanted_selectors:
                for element in soup.select(selector):
                    element.decompose()
            
            # Find main content areas with priority order
            content_selectors = [
                'article', 'main', '[role="main"]', '.mw-parser-output',
                '#mw-content-text', '.content', '.post-content', '.entry-content',
                '.article-content', '#content', '.main-content',
                '.post-body', '.entry-body', '.article-body',
                '.wiki-content', '.page-content'
            ]
            
            main_content = None
            for selector in content_selectors:
                elements = soup.select(selector)
                if elements:
                    main_content = elements[0]
                    break
            
            if not main_content:
                main_content = soup.find('body') or soup
            
            text_elements = main_content.find_all([
                'p', 'div', 'section', 'article', 'li', 'td', 'th'
            ])
            
            paragraphs = []
            seen_texts = set()
            
            for element in text_elements:
                if element.find_parent(['script', 'style']):
                    continue
                
                text = element.get_text(separator=' ', strip=True)
                
                # Clean text
                text = re.sub(r'\s+', ' ', text)
                text = re.sub(r'\[.*?\]', '', text)
                text = re.sub(r'\(edit\)', '', text)
                text = text.strip()
                
                if (len(text) > 80 and 
                    len(text.split()) > 10 and
                    not self._is_unwanted_text(text) and
                    text not in seen_texts):
                    
                    paragraphs.append(text)
                    seen_texts.add(text)
            
            if not paragraphs:
                all_text = main_content.get_text(separator='\n', strip=True)
                lines = all_text.split('\n')
                
                current_paragraph = []
                for line in lines:
                    line = line.strip()
                    if len(line) > 5:
                        current_paragraph.append(line)
                    elif current_paragraph:
                        paragraph = ' '.join(current_paragraph)
                        if len(paragraph) > 10 and not self._is_unwanted_text(paragraph):
                            paragraphs.append(paragraph)
                        current_paragraph = []
                
                if current_paragraph:
                    paragraph = ' '.join(current_paragraph)
                    if len(paragraph) > 10 and not self._is_unwanted_text(paragraph):
                        paragraphs.append(paragraph)
            
            return paragraphs #[:20]
            
        except Exception as e:
            print(f"BeautifulSoup extraction failed: {e}")
            return []
    
    def _is_unwanted_text(self, text: str) -> bool:
        """Check if text should be filtered out"""
        text_lower = text.lower()
        
        unwanted_patterns = [
            'click here', 'read more', 'share this', 'follow us',
            'subscribe', 'advertisement', 'sponsored', 'cookie',
            'privacy policy', 'terms of service', 'all rights reserved',
            'copyright', '© 20', 'search', 'menu', 'navigation',
            'jump to content', 'main menu', 'toggle', 'edit links',
            'categories:', 'hidden categories:', 'retrieved from',
            'this page was last edited', 'privacy policy', 'about wikipedia',
            'disclaimers', 'contact wikipedia', 'donate', 'create account',
            'log in', 'talk', 'contributions', 'special pages'
        ]
        
        for pattern in unwanted_patterns:
            if pattern in text_lower:
                return True
        
        if len(re.sub(r'[^a-zA-Z\s]', '', text)) < len(text) * 0.5:
            return True
        
        words = text.split()
        if len(set(words)) < len(words) * 0.3 and len(words) > 5:
            return True
        
        return False
    
    def extract_passages(self, html_content: str, title: str = "", source_url: str = "", max_passages: int = 5) -> List[Dict[str, Any]]:
        """Extract clean passages from HTML content with enhanced processing options"""
        
        if not html_content or not html_content.strip():
            print("Empty HTML content provided")
            return []
        
        print(f"Attempting to extract content using method: {self.method}")
        
        # Method 1: LLM Filter
        if self.method == 'llm_filter' and self.llm_cleaner:
            return self.llm_cleaner.extract_passages_with_llm(html_content, title, source_url, max_passages)
        
        # Method 2: Wikipedia Filter (existing logic)
        if self.method == 'wikipedia_filter':
            is_wikipedia = 'wikipedia.org' in source_url.lower() or 'wikipedia' in title.lower()
            if is_wikipedia:
                print("Detected Wikipedia content, using specialized processor")
                chunks = self.wikipedia_processor.process_wikipedia_content(
                    html_content, title, source_url
                )
                
                # Filter and sort chunks by quality
                quality_chunks = []
                for chunk in chunks:
                    if (len(chunk['contents'].split()) >= 10 and 
                        not self.wikipedia_processor._is_navigation_chunk(chunk['contents']) and
                        '[edit]' not in chunk['contents']):
                        quality_chunks.append(chunk)
                
                quality_chunks.sort(key=lambda x: x['metadata']['word_count'], reverse=True)
                
                passages = []
                for i, chunk in enumerate(quality_chunks[:max_passages]):
                    section_name = chunk['metadata'].get('section', '')
                    passages.append({
                        'id': f"wiki_passage_{i}",
                        'contents': chunk['contents'],
                        'title': f"{section_name} - {title}",
                        'source_url': source_url,
                        'metadata': chunk['metadata']
                    })
                
                print(f"Wikipedia processor created {len(passages)} quality passages from {len(chunks)} total chunks")
                return passages
        
        # Method 3: Traditional methods (hybrid, trafilatura, beautifulsoup)
        paragraphs = []
        
        if self.method == 'trafilatura':
            paragraphs = self.extract_with_trafilatura(html_content)
            if not paragraphs:
                print("Trafilatura failed, falling back to BeautifulSoup")
                paragraphs = self.extract_with_beautifulsoup(html_content)
        elif self.method == 'beautifulsoup':
            paragraphs = self.extract_with_beautifulsoup(html_content)
        else:  # hybrid or fallback
            paragraphs = self.extract_with_trafilatura(html_content)
            if not paragraphs:
                print("Trafilatura failed, trying BeautifulSoup")
                paragraphs = self.extract_with_beautifulsoup(html_content)
        
        print(f"Extracted {len(paragraphs)} paragraphs before filtering")
        
        if not paragraphs:
            print("All methods failed, trying simple text extraction")
            soup = BeautifulSoup(html_content, 'html.parser')
            text = soup.get_text()
            if text and len(text.strip()) > 100:
                simple_paragraphs = re.split(r'\n\n|\. (?=[A-Z])', text)
                for p in simple_paragraphs:
                    p = p.strip()
                    if len(p) > 100 and not self._is_unwanted_text(p):
                        paragraphs.append(p)
        
        # Convert to passage format
        passages = []
        for i, paragraph in enumerate(paragraphs):
            passages.append({
                'id': f"passage_{i}",
                'contents': paragraph,
                'title': title or f'Passage {i+1}',
                'source_url': source_url,
                'metadata': {
                    'source': 'web',
                    'extraction_method': self.method,
                    'word_count': len(paragraph.split())
                }
            })
        
        print(f"Final extracted passages: {len(passages)}")
        return passages

# Modified OnlineRetriever class
class OnlineRetriever:
    def __init__(self, model: str = 'online_retriever', n_docs: int = 10, 
                 llm_model: str = None, batch_size: int = 10, api_key=None, 
                 extraction_method='hybrid', llm_config: Dict = None) -> None:
        """
        Initialize OnlineRetriever with flexible content extraction methods
        
        Args:
            extraction_method: 'hybrid', 'trafilatura', 'beautifulsoup', 'wikipedia_filter', or 'llm_filter'
            llm_config: Dict for LLM configuration when using 'llm_filter'
                       {'provider': 'openai'/'together', 'api_key': 'xxx', 'model': 'gpt-3.5-turbo'}
        """
        self.n_docs = n_docs
        print(api_key)
        self.searcher = WebSearchTool(search_provider_api_key=api_key)
        self.model = model
        self.llm_model = llm_model
        self.sources = []
        self.relevant_docs: List[Document] = []
        self.batch_size = batch_size
        self.tokenizer = SimpleTokenizer()
        
        # Initialize content extractor with LLM support
        self.content_extractor = HTMLContentExtractor(method=extraction_method, llm_config=llm_config)

    def _search_web(self, query: str) -> List[Any]:
        if not self.searcher.is_initialized:
            self.searcher.setup()
        return self.searcher.forward(query, num_result=self.n_docs)

    def retrieve(self, documents: List[Document]) -> List[Document]:
        question_texts = [doc.question.question for doc in documents]

        for i, question in enumerate(tqdm(question_texts, desc="Fetching documents...")):
            document = documents[i]
            logger.info(f"🌐 Retrieving contexts for q{i}:{question}...")
            
            # Get search results
            sources = self._search_web(question)
            print(f"Retrieved {len(sources)} sources")
            
            # Extract clean passages from HTML content with enhanced processing
            processed_contexts = []
            for idx, source in enumerate(sources):
                if 'html' in source and source['html']:
                    # Extract passages from HTML (now with LLM or Wikipedia support)
                    passages = self.content_extractor.extract_passages(
                        html_content=source['html'],
                        title=source.get('title', 'Web Result'),
                        source_url=source.get('link', ''),
                        max_passages=1000  # Get multiple passages per source
                    )
                    
                    for passage_idx, passage in enumerate(passages):
                        processed_contexts.append({
                            'id': f"source_{idx}_passage_{passage_idx}",
                            'title': passage['title'],
                            'contents': passage['contents'],
                            'source_url': passage['source_url'],
                            'metadata': passage.get('metadata', {})
                        })
                else:
                    # Fallback to snippet if no HTML
                    snippet = source.get('snippet', '')
                    if len(snippet) > 50:
                        processed_contexts.append({
                            'id': f"snippet_{idx}",
                            'title': source.get('title', ''),
                            'contents': snippet,
                            'source_url': source.get('link', ''),
                            'metadata': {'source': 'snippet'}
                        })
            
            print(f"Extracted {len(processed_contexts)} clean passages")
            
            # Show sample of what was extracted
            # if processed_contexts:
            #     print(f"\nSample extracted passages:")
            #     for i, ctx in enumerate(processed_contexts):
            #         print(f"Passage {i+1}:")
            #         print(f"  Title: {ctx['title']}")
            #         print(f"  Words: {len(ctx['contents'].split())}")
            #         print(f"  Preview: {ctx['contents'][:150]}...")
            #         if 'metadata' in ctx:
            #             if 'section' in ctx['metadata']:
            #                 print(f"  Section: {ctx['metadata']['section']}")
            #             if 'llm_provider' in ctx['metadata']:
            #                 print(f"  Cleaned by: {ctx['metadata']['llm_provider']}")
            #         print()
            
            # Continue with existing indexing and search logic
            with tempfile.TemporaryDirectory() as tmpdirname:
                contexts = [
                    {
                        'id': ctx['id'],
                        'contents': ctx['contents']
                    }
                    for ctx in processed_contexts
                ]

                print(len(contexts))
                
                json.dump(contexts, open(tmpdirname + '/context.json', 'w'))
                subprocess.run([
                    "python", "-m", "pyserini.index.lucene",
                    "-collection", "JsonCollection",
                    "-generator", "DefaultLuceneDocumentGenerator",
                    "-input", tmpdirname,
                    "-index", tmpdirname,
                    "-storePositions", "-storeDocvectors", "-storeRaw"
                ], check=True)

                lucene_searcher = LuceneSearcher(tmpdirname)
                hits = lucene_searcher.search(question, k=self.n_docs)
                print(self.n_docs, len(hits))
                contexts: List[Context] = []
                for idx, hit in enumerate(hits):
                    try:
                        lucene_doc = lucene_searcher.doc(hit.docid)
                        raw_content = json.loads(lucene_doc.raw())
                        text = raw_content.get("contents", "")
                        title = raw_content.get("contents", "").split("\n")[0]
                        
                        # Handle ID conversion - use index if Context expects integer
                        try:
                            context_id = int(hit.docid)
                        except ValueError:
                            # If hit.docid is string, use index as ID
                            context_id = idx
                        
                        context = Context(
                            id=context_id,
                            title=title,
                            text=text,
                            score=hit.score,
                            has_answer=has_answers(text, document.answers.answers, self.tokenizer)
                        )
                        contexts.append(context)
                    except Exception as e:
                        print(f"Error processing document ID {hit.docid}: {e}")
                        
                document.contexts = contexts

        return documents