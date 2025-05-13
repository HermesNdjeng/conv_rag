from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import easyocr
import os
import pdf2image
from PIL import Image
import tempfile
from utils.logging_utils import setup_logger

# Set up module-specific logger
logger = setup_logger("ocr_processor")

class OCRConfig(BaseModel):
    """Configuration for the OCR processor"""
    languages: List[str] = Field(
        default=["fr", "en"],
        description="Languages to detect in documents (fr for French, en for English)"
    )
    gpu: bool = Field(
        default=False,
        description="Whether to use GPU for OCR processing"
    )
    batch_size: int = Field(
        default=4,
        description="Batch size for OCR processing"
    )
    output_dir: Optional[str] = Field(
        default=None,
        description="Directory to save OCR'd text files"
    )

# Modifiez la classe OCRProcessor pour utiliser le GPU Mac

class OCRProcessor:
    """Processes scanned documents with OCR to extract text"""
    
    def __init__(self, config: Optional[OCRConfig] = None):
        """
        Initialize the OCR processor.
        
        Args:
            config: OCR configuration
        """
        self.config = config or OCRConfig()
        
        # Détection automatique du GPU Mac
        self.use_gpu = self._check_mac_gpu() if self.config.gpu else False
        logger.info(f"Initializing OCR processor with languages: {self.config.languages}, GPU: {self.use_gpu}")
        
        self.reader = None  # Lazy initialization to avoid loading models unnecessarily
        
        # Create output directory if specified
        if self.config.output_dir:
            os.makedirs(self.config.output_dir, exist_ok=True)
    
    def _check_mac_gpu(self):
        """Vérifie si le GPU Mac (Metal) est disponible pour PyTorch"""
        try:
            import torch
            # Vérifier si MPS est disponible (Mac avec Apple Silicon)
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                logger.info("GPU Apple Silicon (MPS) détecté et activé")
                return True
            # Vérifier CUDA comme fallback
            elif torch.cuda.is_available():
                logger.info("GPU CUDA détecté et activé")
                return True
            else:
                logger.info("Aucun GPU détecté, utilisation du CPU")
                return False
        except Exception as e:
            logger.warning(f"Erreur lors de la vérification du GPU: {e}")
            return False
    
    def _init_reader(self):
        """Initialize OCR reader if not already initialized"""
        if self.reader is None:
            logger.info("Loading OCR models, this may take a moment...")
            try:
                # Initialiser EasyOCR avec GPU si disponible
                self.reader = easyocr.Reader(
                    self.config.languages,
                    gpu=self.use_gpu,
                    model_storage_directory="data/ocr_models"
                )
                logger.info("OCR models loaded successfully")
            except Exception as e:
                logger.error(f"Error loading OCR models: {e}")
                # Fallback to CPU if GPU fails
                logger.info("Falling back to CPU mode")
                self.reader = easyocr.Reader(
                    self.config.languages,
                    gpu=False,
                    model_storage_directory="data/ocr_models"
                )
    
    def process_pdf(self, pdf_path: str) -> str:
        """Process a PDF file with OCR to extract text."""
        logger.info(f"Processing PDF with OCR: {pdf_path}")
        
        # Initialize OCR reader
        self._init_reader()
        
        # Use PyMuPDF to convert PDF to images
        import fitz  # PyMuPDF
        
        # Create temporary directory for images
        with tempfile.TemporaryDirectory() as temp_dir:
            doc = fitz.open(pdf_path)
            all_text = []
            
            # Process each page
            for i, page in enumerate(doc):
                logger.info(f"Processing page {i+1}/{len(doc)}")
                
                # Convert page to image
                pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
                img_path = os.path.join(temp_dir, f"page_{i+1}.png")
                pix.save(img_path)
                
                # Process with OCR
                result = self.reader.readtext(img_path)
                page_text = "\n".join([text for _, text, _ in result])
                all_text.append(f"=== Page {i+1} ===\n{page_text}")
            
            # Combine and save results
            full_text = "\n\n".join(all_text)
            
            if self.config.output_dir:
                output_path = os.path.join(
                    self.config.output_dir,
                    os.path.basename(pdf_path).replace(".pdf", ".txt")
                )
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(full_text)
                logger.info(f"Saved OCR text to {output_path}")
            
            return full_text

    def is_scanned_pdf(self, pdf_path: str, sample_pages: int = 3) -> bool:
        """
        Check if a PDF file is a scanned document by checking for extractable text.
        
        Args:
            pdf_path: Path to the PDF file
            sample_pages: Number of pages to sample
            
        Returns:
            True if the PDF is likely scanned, False otherwise
        """
        import PyPDF2
        
        logger.info(f"Checking if PDF is scanned: {pdf_path}")
        
        # Open the PDF
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            num_pages = len(reader.pages)
            
            # Check a sample of pages
            pages_to_check = min(sample_pages, num_pages)
            total_text = 0
            
            for i in range(pages_to_check):
                page = reader.pages[i]
                text = page.extract_text() or ""
                total_text += len(text.strip())
            
            # If there's very little text, it's likely a scanned document
            avg_text_per_page = total_text / pages_to_check
            logger.info(f"Average text per page: {avg_text_per_page} characters")
            
            return avg_text_per_page < 100  # Threshold for determining if it's scanned