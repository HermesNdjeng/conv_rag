import os
import streamlit
import subprocess
import sys
from utils.logging_utils import setup_logger

# Set up logger
logger = setup_logger("run_app")

def main():
    # Check if streamlit is installed
    try:
        import streamlit
        logger.info("Streamlit is already installed.")
    except ImportError:
        logger.info("Installing streamlit...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])
        logger.info("Streamlit installed successfully.")
    
    # Ensure necessary directories exist
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/indexes", exist_ok=True)
    
    # Check if FAISS index exists
    if not os.path.exists("data/indexes/um_nyobe_index"):
        logger.warning("FAISS index not found. Please run indexing first with: poetry run python conv_rag/main_indexer.py")
        print("FAISS index not found. You need to index your documents first.")
        print("Run: poetry run python conv_rag/main_indexer.py")
        print("Select option 1 to index documents, then try running the app again.")
        return
    
    # Start the Streamlit app
    logger.info("Starting Streamlit app...")
    app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")
    subprocess.run(["streamlit", "run", app_path])

if __name__ == "__main__":
    main()