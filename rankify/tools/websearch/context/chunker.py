from typing import List, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re
class Chunker:
    """A modular text chunking class that splits text into smaller, overlapping segments.
    
    This class provides a flexible way to break down large texts into smaller chunks
    while maintaining context through configurable overlap. It uses RecursiveCharacterTextSplitter
    from langchain under the hood.
    
    Attributes:
        chunk_size (int): The target size for each text chunk.
        chunk_overlap (int): The number of characters to overlap between chunks.
        separators (List[str]): List of separators to use for splitting, in order of preference.
        length_function (callable): Function to measure text length (default: len).
    """

    def __init__(
        self,
        chunk_size: int = 1200,
        chunk_overlap: int = 100,
        min_chunk_size: int = 800, 
        separators: Optional[List[str]] = None,
        length_function: callable = len
    ):
        """Initialize the Chunker with specified parameters.
        
        Args:
            chunk_size (int, optional): Target size for each chunk. Defaults to 250.
            chunk_overlap (int, optional): Number of characters to overlap. Defaults to 50.
            separators (List[str], optional): Custom separators for splitting.
                Defaults to ["\n\n", "\n", " "].
            length_function (callable, optional): Function to measure text length.
                Defaults to len.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n",]
        self.length_function = length_function
        self.min_chunk_size = min_chunk_size
        
        self.splitter = RecursiveCharacterTextSplitter(
            separators=self.separators,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=self.length_function
        )
    
    def split_text(self, text: str) -> List[str]:
        """Split text and combine small chunks."""
        chunks = self.splitter.split_text(text)
        
        # Combine chunks that are too small
        combined_chunks = []
        current_chunk = ""
        
        for chunk in chunks:
            chunk = self.remove_links(chunk)
            if len(current_chunk) + len(chunk) < self.min_chunk_size:
                current_chunk += (" " if current_chunk else "") + chunk
            else:
                if current_chunk:
                    combined_chunks.append(current_chunk)
                current_chunk = chunk
        
        if current_chunk:
            combined_chunks.append(current_chunk)
            
        return combined_chunks
    def remove_links(self, text: str) -> str:
        """Remove various types of links from text."""
        patterns = [
            r'https?://[^\s<>"{}|\\^`\[\]]+',  # HTTP/HTTPS URLs
            r'www\.[^\s<>"{}|\\^`\[\]]+',      # www. links
            r'ftp://[^\s<>"{}|\\^`\[\]]+',     # FTP links
            r'\[[^\]]*\]\([^)]*\)',            # Markdown links [text](url)
            r'<a[^>]*>.*?</a>',                # HTML links
        ]
        
        for pattern in patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Clean up extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    def split_texts(self, texts: List[str]) -> List[List[str]]:
        """Split multiple texts into chunks.
        
        Args:
            texts (List[str]): A list of input texts to be split into chunks.
            
        Returns:
            List[List[str]]: A list of lists, where each inner list contains
                the chunks for one input text.
        """
        return [self.split_text(text) for text in texts]