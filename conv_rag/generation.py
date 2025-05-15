from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, SecretStr
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
from langchain.schema.runnable import RunnableMap
from langchain.callbacks import get_openai_callback
from retriever import DocumentRetriever
from utils.logging_utils import setup_logger
import os

# Set up module-specific logger
logger = setup_logger("generator")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

class GeneratorConfig(BaseModel):
    """Configuration for the LLM generator"""
    model_name: str = Field(
        default="gpt-3.5-turbo", 
        description="Name of the LLM model to use"
    )
    temperature: float = Field(
        default=0.7, 
        description="Temperature for generation (0.0-1.0)"
    )
    max_tokens: int = Field(
        default=1024,
        description="Maximum number of tokens in the response"
    )
    api_key: Optional[SecretStr] = Field(
        default=None,
        description="OpenAI API key (will use environment variable if not provided)"
    )
    streaming: bool = Field(
        default=False,
        description="Whether to stream the LLM response"
    )
    track_token_usage: bool = Field(
        default=True,
        description="Whether to track token usage"
    )

class GenerationResult(BaseModel):
    """Results from a generation operation"""
    query: str = Field(description="Original query that was processed")
    answer: str = Field(description="Generated answer from the LLM")
    source_documents: Optional[List[Document]] = Field(
        default=None,
        description="Source documents used for generation"
    )
    token_usage: Optional[Dict[str, Union[int, float]]] = Field(
        default=None,
        description="Token usage statistics, with integers for token counts and float for cost"
    )
    
    class Config:
        arbitrary_types_allowed = True

class RAGGenerator:
    """Generates answers using a retrieval-augmented LLM"""
    
    def __init__(
        self, 
        config: Optional[GeneratorConfig] = None, 
        retriever: Optional[DocumentRetriever] = None
    ):
        """
        Initialize the RAG generator.
        
        Args:
            config: Generator configuration
            retriever: Document retriever instance
        """
        self.config = config or GeneratorConfig()
        self.retriever = retriever
        
        # Get API key from config or environment
        api_key = None
        if self.config.api_key:
            api_key = self.config.api_key.get_secret_value()
        else:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.error("No OpenAI API key provided. Set OPENAI_API_KEY environment variable or provide in config.")
                raise ValueError("OpenAI API key is required")
        
        # Initialize LLM
        logger.info(f"Initializing LLM with model: {self.config.model_name}")
        self.llm = ChatOpenAI(
            model_name=self.config.model_name,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            openai_api_key=api_key,
            streaming=self.config.streaming,
        )
        
        # Initialize QA chain if retriever is provided
        if self.retriever:
            self._setup_qa_chain()
    

    def _setup_qa_chain(self):
        """Set up the QA chain with the current retriever and conversation memory"""
        
        # Conserver l'approche ChatPromptTemplate
        chat_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""Vous êtes un assistant IA spécialisé dans l'histoire du Cameroun, 
            particulièrement sur Ruben Um Nyobe et la période de lutte pour l'indépendance.
            Répondez aux questions de manière précise et factuelle en vous basant uniquement 
            sur les informations contenues dans les documents fournis.
            Si vous ne connaissez pas la réponse, dites-le clairement sans inventer d'information."""),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{query}"),
            MessagesPlaceholder(variable_name="context"),
        ])
        
        # Importer ce qui est nécessaire pour créer des messages
        from langchain_core.messages import HumanMessage, AIMessage
        
        # Créer une fonction pour convertir les documents en messages
        def docs_to_messages(docs):
            messages = []
            for i, doc in enumerate(docs):
                source = doc.metadata.get('source', 'Unknown source')
                page = doc.metadata.get('page', 'Unknown page')
                # Créer un message pour chaque document
                messages.append(HumanMessage(
                    content=f"Document {i+1} (Source: {source}, Page: {page}):\n{doc.page_content}"
                ))
            return messages
        
        # Modifier le RunnableMap pour fournir une liste de messages
        rag_chain = RunnableMap({
            "context": lambda x: docs_to_messages(self.retriever.get_relevant_documents(x["query"])),
            "chat_history": lambda x: x["chat_history"],
            "query": lambda x: x["query"]
        }) | chat_prompt | self.llm

        self.qa_chain = rag_chain
        logger.info("QA chain initialized successfully with conversation history integration")
    
    def set_retriever(self, retriever: DocumentRetriever):
        """Set or update the retriever instance"""
        self.retriever = retriever
        self._setup_qa_chain()
    
    def generate(self, query: str, conversation_manager=None) -> GenerationResult:
        """Generate an answer to the user query using RAG."""
        logger.info(f"Generating answer for query: {query}")
        
        # Get chat history if a conversation manager is provided
        chat_history = []
        if conversation_manager:
            chat_history = conversation_manager.get_chat_history()
            logger.info(f"Using {len(chat_history)} messages from conversation history")
        
        try:
            # Récupérer les documents d'abord pour pouvoir les logger
            source_docs = self.retriever.get_relevant_documents(query)
            
            # Log des chunks récupérés
            total_tokens = sum(len(doc.page_content.split()) * 1.3 for doc in source_docs)
            logger.info(f"===== LLM Context: {len(source_docs)} chunks (approx. {int(total_tokens)} tokens) =====")
            for i, doc in enumerate(source_docs):
                source = doc.metadata.get('source', 'Unknown source')
                page = doc.metadata.get('page', 'Unknown page')
                words = len(doc.page_content.split())
                logger.info(f"CONTEXT CHUNK {i+1}: Source={source}, Page={page}, Words={words}")
            
            # Utiliser .invoke() avec l'API runnable de LangChain
            result = self.qa_chain.invoke({
                "query": query,
                "chat_history": chat_history
            })
            
            # Avec l'API Runnable, le résultat est directement la réponse du LLM
            answer = result.content if hasattr(result, 'content') else str(result)
            
            # Calculate token usage if tracking is enabled
            token_usage = None
            if self.config.track_token_usage:
                token_usage = self._calculate_token_usage(query, answer, source_docs)
            
            return GenerationResult(
                query=query,
                answer=answer,
                source_documents=source_docs,
                token_usage=token_usage
            )
            
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())  # Afficher la stack trace complète
            return GenerationResult(
                query=query,
                answer=f"Une erreur s'est produite lors de la génération de la réponse: {str(e)}",
                source_documents=[],
                token_usage=None
            )

    # Ajoutez cette méthode à votre classe RAGGenerator
    def _calculate_token_usage(self, query: str, answer: str, source_docs: List[Document]) -> Dict[str, Union[int, float]]:
        """
        Calculate token usage for the query, answer, and source documents.
        
        Args:
            query: User's question
            answer: Generated answer
            source_docs: Source documents used for generation
            
        Returns:
            Dictionary with token usage statistics
        """
        try:
            # Use OpenAI's callback to track token usage
            with get_openai_callback() as cb:
                # Re-run the query to get token counts
                self.llm.predict(text=f"Question: {query}\n\nAnswer: {answer}")
                
                # Log token usage details
                logger.info(f"===== Token Usage =====")
                logger.info(f"Prompt tokens: {cb.prompt_tokens}")
                logger.info(f"Completion tokens: {cb.completion_tokens}")
                logger.info(f"Total tokens: {cb.total_tokens}")
                logger.info(f"Cost: ${cb.total_cost:.6f}")
                
                return {
                    "prompt_tokens": cb.prompt_tokens,
                    "completion_tokens": cb.completion_tokens,
                    "total_tokens": cb.total_tokens,
                    "cost": cb.total_cost
                }
        except Exception as e:
            logger.error(f"Error calculating token usage: {str(e)}")
            return {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cost": 0.0,
                "error": str(e)
            }