from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage, SystemMessage, BaseMessage
from utils.logging_utils import setup_logger

# Set up module-specific logger
logger = setup_logger("conversation")

class Message(BaseModel):
    """Model for a single conversation message"""
    role: str = Field(description="Role of the message sender (user, assistant, system)")
    content: str = Field(description="Content of the message")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata about the message")

class ConversationHistory(BaseModel):
    """Model for storing conversation history"""
    messages: List[Message] = Field(default_factory=list, description="List of conversation messages")
    
    def add_user_message(self, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Add a user message to the conversation"""
        self.messages.append(Message(role="user", content=content, metadata=metadata))
    
    def add_assistant_message(self, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Add an assistant message to the conversation"""
        self.messages.append(Message(role="assistant", content=content, metadata=metadata))
    
    def add_system_message(self, content: str):
        """Add a system message to the conversation"""
        self.messages.append(Message(role="system", content=content))
    
    def get_messages(self, limit: Optional[int] = None) -> List[Message]:
        """Get the most recent messages up to the limit"""
        if limit is None:
            return self.messages
        return self.messages[-limit:]
    
    def clear(self):
        """Clear all messages from history"""
        self.messages = []

class ConversationManager:
    """Manages conversation state and history for the RAG system"""
    
    def __init__(self, system_message: str = None):
        """
        Initialize the conversation manager.
        
        Args:
            system_message: Optional system message to start the conversation
        """
        self.history = ConversationHistory()
        self.memory = ConversationBufferMemory(return_messages=True)
        
        # Add initial system message if provided
        if system_message:
            self.history.add_system_message(system_message)
            self.memory.chat_memory.add_message(SystemMessage(content=system_message))
            logger.info("Initialized conversation with system message")
    
    def add_user_message(self, message: str) -> None:
        """
        Add a user message to the conversation.
        
        Args:
            message: User's message content
        """
        self.history.add_user_message(message)
        self.memory.chat_memory.add_message(HumanMessage(content=message))
        logger.debug(f"Added user message: {message[:50]}{'...' if len(message) > 50 else ''}")
    
    def add_assistant_message(self, message: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add an assistant message to the conversation.
        
        Args:
            message: Assistant's message content
            metadata: Optional metadata about the message (e.g., source docs, token usage)
        """
        self.history.add_assistant_message(message, metadata)
        self.memory.chat_memory.add_message(AIMessage(content=message))
        logger.debug(f"Added assistant message: {message[:50]}{'...' if len(message) > 50 else ''}")
    
    def get_chat_history(self) -> List[BaseMessage]:
        """Get the chat history in a format compatible with LangChain"""
        return self.memory.chat_memory.messages
    
    def get_formatted_history(self, include_system: bool = False) -> str:
        """
        Get the conversation history as a formatted string.
        
        Args:
            include_system: Whether to include system messages
            
        Returns:
            Formatted conversation history
        """
        formatted = []
        for msg in self.history.messages:
            if msg.role == "system" and not include_system:
                continue
            formatted.append(f"{msg.role.capitalize()}: {msg.content}")
        
        return "\n".join(formatted)
    
    def clear_history(self) -> None:
        """Clear the conversation history"""
        self.history.clear()
        self.memory.clear()
        logger.info("Cleared conversation history")