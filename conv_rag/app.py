import streamlit as st
import os
from indexer import DocumentIndexer, IndexerConfig
from retriever import DocumentRetriever, RetrievalConfig
from generation import RAGGenerator, GeneratorConfig
from conversation import ConversationManager
from langchain_openai import ChatOpenAI
from utils.logging_utils import setup_logger

# Set up logger
logger = setup_logger("streamlit_app")

# Page configuration
st.set_page_config(
    page_title="Assistant Um Nyobe",
    page_icon="📚",
    layout="wide",
)

# Initialize session state for conversation history
if "conversation_manager" not in st.session_state:
    system_message = """Tu es un assistant expert sur Ruben Um Nyobe et l'histoire du Cameroun.
    Tu utilises les informations d'un livre sur Um Nyobe et les maquis camerounais pour répondre aux questions.
    Réponds toujours en français, et avec respect pour l'histoire camerounaise.
    Si tu ne connais pas la réponse à partir des informations disponibles, dis simplement que tu ne sais pas.
    """
    st.session_state.conversation_manager = ConversationManager(system_message=system_message)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize RAG components
@st.cache_resource
def initialize_rag_components():
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY") and not st.session_state.get("openai_api_key"):
        return None, None, None
    
    api_key = os.getenv("OPENAI_API_KEY") or st.session_state.get("openai_api_key")
    
    # Initialize components
    indexer_config = IndexerConfig(
        embedding_model_name="dangvantuan/sentence-camembert-base",
        index_path="data/indexes"
    )
    indexer = DocumentIndexer(config=indexer_config)
    
    retrieval_config = RetrievalConfig(
        index_type="faiss",
        index_name="um_nyobe_index",
        top_k=4,
        score_threshold=0.5
    )
    retriever = DocumentRetriever(config=retrieval_config, indexer=indexer)
    
    generator_config = GeneratorConfig(
        model_name="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=1024,
        api_key=api_key if not os.getenv("OPENAI_API_KEY") else None,
        track_token_usage=True
    )
    generator = RAGGenerator(config=generator_config, retriever=retriever)
    
    logger.info("RAG components initialized successfully")
    return indexer, retriever, generator

def display_source_extract(source, content_preview):
    st.write(f"**Source:** {source.get('title', 'Document')} (Page: {source.get('page', 'N/A')})")
    st.markdown("**Extrait:**")
    st.markdown(f"<div style='background-color:#e6f2ff;padding:10px;border-radius:5px;font-size:0.9em;border:1px solid #b3d9ff;color:#003366;'>{content_preview}</div>", unsafe_allow_html=True)
    st.markdown("---")

# Main app layout
st.title("📚 Assistant Um Nyobe")
st.subheader("Conversation sur l'histoire d'Um Nyobe et des maquis camerounais")

# Sidebar for API key and settings
with st.sidebar:
    st.header("Configuration")
    
    # API Key input
    if not os.getenv("OPENAI_API_KEY"):
        api_key = st.text_input("OpenAI API Key", type="password")
        if api_key:
            st.session_state.openai_api_key = api_key
    
    # Model selection
    model_option = st.selectbox(
        "Modèle LLM",
        options=["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
        index=0
    )
    
    # Temperature setting
    temperature = st.slider(
        "Température",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Contrôle la créativité des réponses. Valeurs basses = réponses plus déterministes."
    )
    
    # Clear conversation button
    if st.button("Effacer la conversation"):
        st.session_state.conversation_manager.clear_history()
        st.session_state.messages = []
        st.experimental_rerun()

# Initialize RAG components
indexer, retriever, generator = initialize_rag_components()

# Check if the components are initialized
if indexer is None or retriever is None or generator is None:
    st.warning("Veuillez configurer votre clé API OpenAI dans le panneau latéral pour commencer.")
    st.stop()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message.get("sources"):
            with st.expander("Sources"):
                for i, source in enumerate(message["sources"]):
                    st.write(f"**Source {i+1}:** {source.get('title', 'Document')} (Page: {source.get('page', 'N/A')})")
                    # Afficher l'aperçu du contenu s'il existe
                    if source.get('content_preview'):
                        st.markdown("**Extrait:**")
                        st.markdown(f"<div style='background-color:#e6f2ff;padding:10px;border-radius:5px;font-size:0.9em;border:1px solid #b3d9ff;color:#003366;'>{source['content_preview']}</div>", unsafe_allow_html=True)
                        st.markdown("---")

# User input
user_query = st.chat_input("Posez votre question sur Ruben Um Nyobe et l'histoire du Cameroun...")

if user_query:
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.session_state.conversation_manager.add_user_message(user_query)
    
    # Display user message
    with st.chat_message("user"):
        st.write(user_query)
    
    # Display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Réflexion en cours..."):
            # Update generator settings if changed
            if generator.config.model_name != model_option or generator.config.temperature != temperature:
                generator.config.model_name = model_option
                generator.config.temperature = temperature
                generator.llm = ChatOpenAI(
                    model_name=generator.config.model_name,
                    temperature=generator.config.temperature,
                    max_tokens=generator.config.max_tokens,
                    openai_api_key=os.getenv("OPENAI_API_KEY") or st.session_state.get("openai_api_key"),
                    streaming=generator.config.streaming,
                )
                generator._setup_qa_chain()
            
            # Generate response
            result = generator.generate(user_query)
            
            # Display response
            st.write(result.answer)
            
            # Display sources if available
            sources = []
            if result.source_documents:
                with st.expander("Sources"):
                    for i, doc in enumerate(result.source_documents[:3]):  # Show top 3 sources
                        source = doc.metadata.get('source', 'Document inconnu')
                        page = doc.metadata.get('page', 'N/A')
                        content_preview = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
                        
                        st.write(f"**Source {i+1}:** {source} (Page: {page})")
                        st.markdown("**Extrait:**")
                        st.markdown(f"<div style='background-color:#e6f2ff;padding:10px;border-radius:5px;font-size:0.9em;border:1px solid #b3d9ff;color:#003366;'>{content_preview}</div>", unsafe_allow_html=True)
                        st.markdown("---")
                        
                        # Add to sources list for history
                        sources.append({
                            "title": source,
                            "page": page,
                            "content_preview": content_preview
                        })
            
            # Display token usage
            if result.token_usage:
                with st.expander("Statistiques"):
                    st.write(f"**Tokens utilisés:** {result.token_usage['total_tokens']}")
                    st.write(f"**Coût estimé:** ${result.token_usage['cost']:.5f}")
    
    # Add assistant message to chat history
    st.session_state.messages.append({
        "role": "assistant", 
        "content": result.answer,
        "sources": sources if sources else None
    })
    st.session_state.conversation_manager.add_assistant_message(
        result.answer, 
        metadata={"sources": sources, "token_usage": result.token_usage}
    )

# Display instructions for first-time users
if not st.session_state.messages:
    st.info("""
    👋 Bienvenue à l'Assistant Um Nyobe!
    
    Cet assistant utilise l'intelligence artificielle pour répondre à vos questions sur Ruben Um Nyobe et l'histoire des maquis camerounais.
    
    Posez une question ci-dessous pour commencer la conversation.
    """)