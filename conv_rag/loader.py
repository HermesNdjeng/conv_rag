from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import os

class LoaderConfig(BaseModel):
    """Configuration for document loader"""
    source_dir: str = Field(default="data/raw", description="Directory containing source documents")
    chunk_size: int = Field(default=1000, description="Size of text chunks for processing")
    chunk_overlap: int = Field(default=200, description="Overlap between chunks for context preservation")

class LoadResult(BaseModel):
    """Result of document loading operation"""
    original_count: int = Field(description="Number of original documents loaded")
    chunk_count: int = Field(description="Number of chunks created")
    chunks: List[Document] = Field(description="List of document chunks")
    
    class Config:
        arbitrary_types_allowed = True

class DocumentLoader:
    def __init__(self, config: Optional[LoaderConfig] = None):
        """
        Initialize the document loader for PDF files.
        
        Args:
            config: Loader configuration
        """
        self.config = config or LoaderConfig()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            length_function=len,
        )
        
    def load_documents(self, file_path: Optional[str] = None) -> LoadResult:
        """
        Load documents from a specific PDF file or all PDFs in the source directory.
        
        Args:
            file_path: Optional specific file path to load
            
        Returns:
            LoadResult containing document chunks and metadata
        """
        documents = []
        
        if file_path and os.path.exists(file_path):
            if file_path.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
        else:
            # Load all PDFs in the source directory
            for filename in os.listdir(self.config.source_dir):
                if filename.endswith('.pdf'):
                    file_path = os.path.join(self.config.source_dir, filename)
                    loader = PyPDFLoader(file_path)
                    documents.extend(loader.load())
        
        # Split documents into chunks
        chunked_documents = self.text_splitter.split_documents(documents)
        
        return LoadResult(
            original_count=len(documents),
            chunk_count=len(chunked_documents),
            chunks=chunked_documents
        )