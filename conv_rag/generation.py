from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, SecretStr
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.callbacks import get_openai_callback
from retriever import DocumentRetriever
from utils.logging_utils import setup_logger
import os

# Set up module-specific logger
logger = setup_logger("generator")

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
        """Set up the QA chain with the current retriever"""
        # Define system and human message templates
        system_template = """
        Tu es un assistant expert sur Ruben Um Nyobe et l'histoire du Cameroun.
        
        Utilise les informations du contexte ci-dessous pour répondre à la question de l'utilisateur.
        Si tu ne connais pas la réponse à partir du contexte, dis simplement que tu ne sais pas, 
        n'invente pas d'information.
        
        Le contexte est constitué d'extraits d'un livre sur Um Nyobe et les maquis camerounais.
        Réponds toujours en français, et avec respect pour l'histoire camerounaise.
        
        Contexte:
        {context}
        """
        
        human_template = "{question}"
        
        # Create chat prompt
        system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
        chat_prompt = ChatPromptTemplate.from_messages(
            [system_message_prompt, human_message_prompt]
        )
        
        # Create QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",  # Stuff all documents into the context
            retriever=self.retriever.vector_store.as_retriever(
                search_kwargs={"k": self.retriever.config.top_k}
            ),
            return_source_documents=True,
            chain_type_kwargs={"prompt": chat_prompt}
        )
        
        logger.info("QA chain initialized successfully")
    
    def set_retriever(self, retriever: DocumentRetriever):
        """Set or update the retriever instance"""
        self.retriever = retriever
        self._setup_qa_chain()
    
    def generate(self, query: str) -> GenerationResult:
        """
        Generate an answer based on the query and retrieved documents.
        
        Args:
            query: User's question
            
        Returns:
            GenerationResult containing the answer and metadata
        """
        if not self.retriever:
            logger.error("Retriever not set. Call set_retriever() before generating answers.")
            raise ValueError("Retriever not initialized")
        
        logger.info(f"Generating answer for query: {query}")
        
        token_usage = None
        if self.config.track_token_usage:
            with get_openai_callback() as cb:
                result = self.qa_chain({"query": query})
                token_usage = {
                    "prompt_tokens": cb.prompt_tokens,
                    "completion_tokens": cb.completion_tokens,
                    "total_tokens": cb.total_tokens,
                    "cost": cb.total_cost
                }
                logger.info(f"Token usage: {token_usage['total_tokens']} tokens, ${token_usage['cost']:.5f}")
        else:
            result = self.qa_chain({"query": query})
        
        answer = result.get("result", "No answer generated")
        source_docs = result.get("source_documents", [])
        
        logger.info(f"Generated answer of length {len(answer)}")
        
        return GenerationResult(
            query=query,
            answer=answer,
            source_documents=source_docs,
            token_usage=token_usage
        )