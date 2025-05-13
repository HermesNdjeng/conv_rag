from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import os
import tempfile
from ocr_processor import OCRProcessor, OCRConfig
from utils.logging_utils import setup_logger

# Set up loader-specific logger
logger = setup_logger("loader")

class LoaderConfig(BaseModel):
    """Configuration for document loader"""
    source_dir: str = Field(default="data/raw", description="Directory containing source documents")
    chunk_size: int = Field(default=1000, description="Size of text chunks for processing")
    chunk_overlap: int = Field(default=200, description="Overlap between chunks for context preservation")
    ocr_enabled: bool = Field(default=True, description="Whether to use OCR for scanned documents")
    ocr_output_dir: Optional[str] = Field(default="data/ocr", description="Directory to save OCR'd text files")

class LoadResult(BaseModel):
    """Result of document loading operation"""
    original_count: int = Field(description="Number of original documents loaded")
    chunk_count: int = Field(description="Number of chunks created")
    chunks: List[Document] = Field(description="List of document chunks")
    ocr_processed: int = Field(default=0, description="Number of documents processed with OCR")
    
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
        logger.info(f"Initializing document loader with chunk size={self.config.chunk_size}, overlap={self.config.chunk_overlap}")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            length_function=len,
        )
        
        # Initialize OCR processor if enabled
        self.ocr_processor = None
        if self.config.ocr_enabled:
            ocr_config = OCRConfig(
                languages=["fr", "en"],  # French and English
                output_dir=self.config.ocr_output_dir,
                gpu=True  # Enable GPU if available
            )
            self.ocr_processor = OCRProcessor(config=ocr_config)
            # Create OCR output directory
            if self.config.ocr_output_dir:
                os.makedirs(self.config.ocr_output_dir, exist_ok=True)
    
    def load_documents(self, file_path: Optional[str] = None) -> LoadResult:
        """
        Load documents from a specific PDF file or all PDFs in the source directory.
        
        Args:
            file_path: Optional specific file path to load
            
        Returns:
            LoadResult containing document chunks and metadata
        """
        documents = []
        ocr_count = 0
        
        if file_path and os.path.exists(file_path):
            if file_path.endswith('.pdf'):
                documents.extend(self._process_pdf(file_path, ocr_count))
        else:
            # Load all PDFs in the source directory
            logger.info(f"Searching for PDFs in directory: {self.config.source_dir}")
            try:
                for filename in os.listdir(self.config.source_dir):
                    if filename.endswith('.pdf'):
                        file_path = os.path.join(self.config.source_dir, filename)
                        docs, ocr_processed = self._process_pdf(file_path)
                        documents.extend(docs)
                        ocr_count += ocr_processed
            except FileNotFoundError:
                logger.error(f"Directory not found: {self.config.source_dir}")
            except Exception as e:
                logger.error(f"Error loading documents: {str(e)}")
        
        # Split documents into chunks
        chunked_documents = self.text_splitter.split_documents(documents)
        logger.info(f"Split {len(documents)} documents into {len(chunked_documents)} chunks")
        
        return LoadResult(
            original_count=len(documents),
            chunk_count=len(chunked_documents),
            chunks=chunked_documents,
            ocr_processed=ocr_count
        )
    
    def _process_pdf(self, file_path: str) -> tuple[List[Document], int]:
        """
        Process a PDF file, using OCR if it's a scanned document.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Tuple of (list of documents, whether OCR was used)
        """
        logger.info(f"Processing PDF: {file_path}")
        
        # Check if OCR is needed and enabled
        is_scanned = False
        if self.config.ocr_enabled and self.ocr_processor:
            is_scanned = self.ocr_processor.is_scanned_pdf(file_path)
        
        if is_scanned and self.ocr_processor:
            logger.info(f"Detected scanned PDF, using OCR: {file_path}")
            
            # Process with OCR
            extracted_text = self.ocr_processor.process_pdf(file_path)
            
            # Save to temporary file and load with TextLoader
            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as tmp:
                tmp.write(extracted_text)
                tmp_path = tmp.name
            
            try:
                loader = TextLoader(tmp_path, encoding="utf-8")
                docs = loader.load()
                
                # Update metadata to include original source
                for doc in docs:
                    doc.metadata["source"] = file_path
                    doc.metadata["ocr_processed"] = True
                
                os.unlink(tmp_path)  # Delete temporary file
                return docs, 1
            except Exception as e:
                logger.error(f"Error loading OCR-processed text: {str(e)}")
                os.unlink(tmp_path)  # Clean up even on error
                return [], 0
        else:
            # Use standard PDF loader for machine-readable PDFs
            logger.info(f"Loading machine-readable PDF: {file_path}")
            try:
                loader = PyPDFLoader(file_path)
                return loader.load(), 0
            except Exception as e:
                logger.error(f"Error loading PDF: {str(e)}")
                return [], 0