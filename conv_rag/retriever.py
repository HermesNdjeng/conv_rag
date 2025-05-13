from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field
from langchain.schema import Document
from langchain_community.vectorstores import FAISS, Chroma
from indexer import DocumentIndexer, IndexerConfig
from utils.logging_utils import setup_logger

# Set up module-specific logger
logger = setup_logger("retriever")

class RetrievalConfig(BaseModel):
    """Configuration for the retrieval process"""
    index_type: Literal["faiss", "chroma"] = Field(
        default="faiss", 
        description="Type of vector index to use for retrieval"
    )
    index_name: str = Field(
        default=None,
        description="Name of the index to retrieve from"
    )
    top_k: int = Field(
        default=5, 
        description="Number of documents to retrieve for each query"
    )
    score_threshold: Optional[float] = Field(
        default=0.3, 
        description="Minimum similarity score threshold (0-1)"
    )
    
    def model_post_init(self, __context: Any) -> None:
        if self.index_name is None:
            self.index_name = "um_nyobe_index" if self.index_type == "faiss" else "um_nyobe_chroma"

class RetrievalResult(BaseModel):
    """Results from a retrieval operation"""
    query: str = Field(description="Original query that was processed")
    documents: List[Document] = Field(description="Retrieved documents sorted by relevance")
    scores: Optional[List[float]] = Field(description="Similarity scores for each document")
    
    class Config:
        arbitrary_types_allowed = True

class DocumentRetriever:
    """Retrieves relevant documents from a vector store based on user queries"""
    
    def __init__(self, config: Optional[RetrievalConfig] = None, indexer: Optional[DocumentIndexer] = None):
        """
        Initialize the document retriever.
        
        Args:
            config: Retrieval configuration
            indexer: Optional pre-configured indexer instance
        """
        self.config = config or RetrievalConfig()
        self.indexer = indexer or DocumentIndexer()
        logger.info(f"Initializing retriever with {self.config.index_type} index: {self.config.index_name}")
        self.vector_store = self._load_vector_store()
    
    def _load_vector_store(self):
        """Load the appropriate vector store based on configuration"""
        if self.config.index_type == "faiss":
            store = self.indexer.load_faiss_index(self.config.index_name)
            if store:
                logger.info(f"Loaded FAISS index: {self.config.index_name}")
                return store
        else:  # chroma
            store = self.indexer.load_chroma_index(self.config.index_name)
            if store:
                logger.info(f"Loaded Chroma index: {self.config.index_name}")
                return store
        
        logger.error(f"Failed to load {self.config.index_type} index: {self.config.index_name}")
        raise ValueError(f"Could not load {self.config.index_type} index. Make sure it exists.")
    
    def retrieve(self, query: str) -> RetrievalResult:
        """
        Retrieve relevant documents based on the user query.
        
        Args:
            query: User's question or search query
            
        Returns:
            RetrievalResult containing relevant documents and metadata
        """
        logger.info(f"Processing query: {query}")
        
        if self.config.index_type == "faiss":
            docs_with_scores = self.vector_store.similarity_search_with_score(
                query, k=self.config.top_k
            )
            # FAISS returns (doc, score) tuples
            documents = [doc for doc, score in docs_with_scores]
            scores = [1.0 - score for doc, score in docs_with_scores]  # Convert distance to similarity
            
        else:  # chroma
            docs_with_scores = self.vector_store.similarity_search_with_score(
                query, k=self.config.top_k
            )
            # Chroma returns (doc, score) tuples
            documents = [doc for doc, score in docs_with_scores]
            scores = [score for doc, score in docs_with_scores]
        
        # Filter by threshold if specified
        if self.config.score_threshold is not None:
            filtered_results = [(doc, score) for doc, score in zip(documents, scores) 
                               if score >= self.config.score_threshold]
            
            if filtered_results:
                documents = [doc for doc, _ in filtered_results]
                scores = [score for _, score in filtered_results]
            else:
                logger.warning(f"No documents met the similarity threshold of {self.config.score_threshold}")
        
        logger.info(f"Retrieved {len(documents)} relevant documents")
        return RetrievalResult(
            query=query,
            documents=documents,
            scores=scores
        )

    def retrieve_for_rag(self, query: str) -> List[Document]:
        """
        Simplified retrieval method that returns documents for RAG.
        
        Args:
            query: User's question or search query
            
        Returns:
            List of relevant documents
        """
        result = self.retrieve(query)
        return result.documents