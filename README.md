# Conv-RAG: Conversational Retrieval-Augmented Generation

Conv-RAG is a conversational agent built with a Retrieval-Augmented Generation (RAG) architecture. It combines the power of large language models (LLMs) with document retrieval to provide accurate and context-aware answers. This project is designed to answer questions about Ruben Um Nyobe, the UPC, and the history of the Cameroonian maquis.

## Features

- **Document Retrieval**: Retrieve relevant documents from a knowledge base.
- **LLM Integration**: Generate answers using OpenAI's GPT models.
- **OCR Support**: Process scanned documents using OCR for indexing.
- **Token Usage Tracking**: Monitor token usage and associated costs.
- **Streamlit Interface**: Interactive web-based interface for user interaction.

## Installation

### Prerequisites

- Python 3.9 (>=3.9, < 3.9.7 || >3.9.7,<=3.12)
- Poetry (for dependency management)

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/conv-rag.git
   cd conv-rag
   ```

2. Install dependencies:
   ```bash
   poetry install
   ```

3. Set up environment variables:
   - Create a `.env` file in the project root.
   - Add your OpenAI API key:
     ```
     OPENAI_API_KEY=your_openai_api_key
     ```

4. Run the application:
   ```bash
   poetry run streamlit run conv_rag/app.py
   ```

## Usage

### Indexing Documents

1. Place your documents in the `data/raw` directory.
2. Run the indexing pipeline:
   ```bash
   poetry run python conv_rag/main_indexer.py
   ```

### Starting the Assistant

Launch the Streamlit app:
```bash
poetry run streamlit run conv_rag/app.py
```

Interact with the assistant by asking questions in the provided interface.

### Testing

- Test retrieval:
  ```bash
  poetry run python conv_rag/main_indexer.py
  ```
  Select option `3` when prompted.
- Test generation:
  ```bash
  poetry run python conv_rag/main_indexer.py
  ```
  Select option `4` when prompted.

## Project Structure

```
conv_rag/
├── conv_rag/
│   ├── app.py               # Streamlit application
│   ├── generation.py        # RAG generation logic
│   ├── loader.py            # Document loading and OCR processing
│   ├── main_indexer.py      # Main indexing and testing pipeline
│   ├── utils/               # Utility functions (e.g., logging)
├── data/                    # Directory for raw and processed data
├── pyproject.toml           # Project dependencies and configuration
└── README.md                # Project documentation
```

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a feature branch:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add feature-name"
   ```
4. Push to your fork:
   ```bash
   git push origin feature-name
   ```
5. Open a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

- [LangChain](https://github.com/hwchase17/langchain) for retrieval and LLM integration.
- [Streamlit](https://streamlit.io/) for the interactive interface.
- OpenAI for GPT models.

