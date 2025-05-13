import os
from loader import DocumentLoader, LoaderConfig
from indexer import DocumentIndexer, IndexerConfig
from retriever import DocumentRetriever, RetrievalConfig
from utils.logging_utils import setup_logger
from generation import RAGGenerator, GeneratorConfig

# Set up module-specific logger
logger = setup_logger("main")

# Mettre à jour la fonction index_documents pour ajouter l'information OCR

def index_documents():
    """Handle the document indexing process"""
    # Initialize the document loader with Pydantic config
    loader_config = LoaderConfig(
        source_dir="data/raw",
        chunk_size=1000,
        chunk_overlap=200,
        ocr_enabled=True,
        ocr_output_dir="data/ocr"
    )
    loader = DocumentLoader(config=loader_config)
    
    # Initialize the indexer with a French-optimized embedding model
    indexer_config = IndexerConfig(
        # Using a model specifically optimized for French text
        embedding_model_name="dangvantuan/sentence-camembert-base",
        index_path="data/indexes"
    )
    indexer = DocumentIndexer(config=indexer_config)
    
    # Load and chunk documents
    load_result = loader.load_documents()
    logger.info(f"Loaded {load_result.original_count} documents and split into {load_result.chunk_count} chunks")
    if load_result.ocr_processed > 0:
        logger.info(f"Processed {load_result.ocr_processed} scanned documents with OCR")
    
    if load_result.chunks:
        # Create both FAISS and Chroma indexes for comparison
        faiss_result = indexer.create_faiss_index(load_result.chunks)
        chroma_result = indexer.create_chroma_index(load_result.chunks)
        
        logger.info(f"Created FAISS index with {faiss_result.document_count} documents at {faiss_result.index_path}")
        logger.info(f"Created Chroma index with {chroma_result.document_count} documents at {chroma_result.index_path}")
        
        return indexer, True
    else:
        logger.warning("No documents found to index.")
        return indexer, False

def test_retrieval(indexer=None):
    """Test the retrieval functionality with sample queries"""
    # Initialize the indexer if not provided
    if indexer is None:
        indexer_config = IndexerConfig(
            embedding_model_name="dangvantuan/sentence-camembert-base",
            index_path="data/indexes"
        )
        indexer = DocumentIndexer(config=indexer_config)
    
    # Initialize the retriever
    retrieval_config = RetrievalConfig(
        index_type="faiss",
        index_name="um_nyobe_index",
        top_k=3,
        score_threshold=0.5
    )
    retriever = DocumentRetriever(config=retrieval_config, indexer=indexer)
    
    # Sample test queries
    test_queries = [
        "Qui était Ruben Um Nyobe?",
        "Parlez-moi des maquis au Cameroun",
        "Quelles étaient les revendications de l'UPC?",
        "Comment la France a réagi au mouvement indépendantiste?",
    ]
    
    # Test retrieval for each query
    for query in test_queries:
        logger.info(f"\n--- Testing query: {query} ---")
        result = retriever.retrieve(query)
        
        logger.info(f"Found {len(result.documents)} relevant documents")
        
        # Display the top results with their scores
        for i, (doc, score) in enumerate(zip(result.documents, result.scores)):
            logger.info(f"\nResult {i+1} (Score: {score:.4f}):")
            
            # Display source information
            source = doc.metadata.get('source', 'Unknown source')
            page = doc.metadata.get('page', 'Unknown page')
            logger.info(f"Source: {source}, Page: {page}")
            
            # Display a preview of the content
            content_preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            logger.info(f"Content: {content_preview}")

def test_generation(indexer=None):
    """Test the generation functionality with sample queries"""
    # Initialize the indexer if not provided
    if indexer is None:
        indexer_config = IndexerConfig(
            embedding_model_name="dangvantuan/sentence-camembert-base",
            index_path="data/indexes"
        )
        indexer = DocumentIndexer(config=indexer_config)
    
    # Initialize the retriever
    retrieval_config = RetrievalConfig(
        index_type="faiss",
        index_name="um_nyobe_index",
        top_k=4,
        score_threshold=0.5
    )
    retriever = DocumentRetriever(config=retrieval_config, indexer=indexer)
    
    # Initialize the generator
    generator_config = GeneratorConfig(
        model_name="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=512,
        track_token_usage=True
    )
    generator = RAGGenerator(config=generator_config, retriever=retriever)
    
    # Sample test queries
    test_queries = [
        "Qui était Ruben Um Nyobe?",
        "Parlez-moi des maquis au Cameroun",
    ]
    
    # Test generation for each query
    for query in test_queries:
        logger.info(f"\n--- Testing generation for: {query} ---")
        
        # Generate answer
        result = generator.generate(query)
        
        # Display the answer
        logger.info(f"\nAnswer: {result.answer}")
        
        # Display token usage
        if result.token_usage:
            logger.info(f"Token usage: {result.token_usage['total_tokens']} tokens, ${result.token_usage['cost']:.5f}")

def main():
    """Main execution function"""
    logger.info("Starting the RAG pipeline process")
    
    # Prompt user for what to run
    print("\nWhat would you like to do?")
    print("1. Index documents")
    print("2. Test retrieval")
    print("3. Test generation (RAG)")
    print("4. Do all steps")
    choice = input("Enter your choice (1-4): ")
    
    indexer = None
    success = True
    
    if choice in ['1', '4']:
        logger.info("Starting document indexing process")
        indexer, success = index_documents()
    
    if (choice in ['2', '4']) and success:
        logger.info("Starting retrieval testing process")
        test_retrieval(indexer)
    
    if (choice in ['3', '4']) and success:
        logger.info("Starting generation testing process")
        test_generation(indexer)
    
    logger.info("Process completed")

if __name__ == "__main__":
    main()