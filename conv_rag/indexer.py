from typing import List, Optional, Literal, Union
from pydantic import BaseModel, Field
from langchain_community.vectorstores import FAISS, Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
import os

class IndexerConfig(BaseModel):
    """Configuration for document indexer"""
    embedding_model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Name of the HuggingFace embedding model to use"
    )
    index_path: str = Field(
        default="data/indexes",
        description="Directory to store vector indexes"
    )

class IndexResult(BaseModel):
    """Result of indexing operation"""
    index_type: Literal["faiss", "chroma"] = Field(description="Type of index created")
    document_count: int = Field(description="Number of documents indexed")
    index_path: str = Field(description="Path where the index is stored")
    
    class Config:
        arbitrary_types_allowed = True

class DocumentIndexer:
    def __init__(self, config: Optional[IndexerConfig] = None):
        """
        Initialize the document indexer with a specified embedding model.
        
        Args:
            config: Indexer configuration
        """
        self.config = config or IndexerConfig()
        self.embeddings = HuggingFaceEmbeddings(model_name=self.config.embedding_model_name)
        
        # Create index directory if it doesn't exist
        os.makedirs(self.config.index_path, exist_ok=True)
        
    def create_faiss_index(self, documents: List[Document], index_name: str = "um_nyobe_index") -> IndexResult:
        """
        Create a FAISS vector store from the provided documents.
        
        Args:
            documents: List of document chunks to index
            index_name: Name for the index
            
        Returns:
            IndexResult with metadata about the created index
        """
        vector_store = FAISS.from_documents(documents, self.embeddings)
        
        # Save the index
        save_path = os.path.join(self.config.index_path, index_name)
        vector_store.save_local(save_path)
        
        return IndexResult(
            index_type="faiss",
            document_count=len(documents),
            index_path=save_path
        )
    
    def create_chroma_index(self, documents: List[Document], index_name: str = "um_nyobe_chroma") -> IndexResult:
        """Create a Chroma vector store from the provided documents."""
        save_path = os.path.join(self.config.index_path, index_name)
        
        # Delete existing Chroma directory if it exists
        if os.path.exists(save_path):
            import shutil
            shutil.rmtree(save_path)
        
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=save_path
        )
        vector_store.persist()
        
        return IndexResult(
            index_type="chroma",
            document_count=len(documents),
            index_path=save_path
        )
    
    def load_faiss_index(self, index_name: str = "um_nyobe_index") -> Optional[FAISS]:
        """Load a previously saved FAISS index"""
        load_path = os.path.join(self.config.index_path, index_name)
        if os.path.exists(load_path):
            return FAISS.load_local(load_path, self.embeddings, allow_dangerous_deserialization=True)
        else:
            return None
    
    def load_chroma_index(self, index_name: str = "um_nyobe_chroma") -> Optional[Chroma]:
        """Load a previously saved Chroma index"""
        load_path = os.path.join(self.config.index_path, index_name)
        if os.path.exists(load_path):
            return Chroma(persist_directory=load_path, embedding_function=self.embeddings)
        else:
            return None