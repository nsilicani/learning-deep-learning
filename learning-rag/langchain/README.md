# Learning Langchain
This repo contains some Langchain examples.

## Build RAG with Huggingface and Chroma
This notebook demonstrates how to build a Retrieval-Augmented Generation (RAG) pipeline using Langchain, Huggingface models, and Chroma. The pipeline is designed to process and query a financial document (Bank of America FORM 10-Q) and generate answers based on the retrieved context.

### Overview

The notebook is structured as follows:

1. **Load Data and Parameters**: Load the PDF document and define key parameters for processing.
2. **Chunking**: Split the document into manageable chunks using the `RecursiveCharacterTextSplitter`.
3. **Custom Embeddings**: Implement a custom embedding strategy using CLS pooling from a Huggingface model.
4. **Indexing**: Index the document chunks into a Chroma vector store.
5. **Load Generative Model**: Load a generative language model from Huggingface to generate responses.
6. **Set up LLM Chain**: Create a Langchain pipeline that uses the indexed data to generate responses to queries.
7. **Generation**: Execute a sample query to demonstrate the RAG pipeline in action.

## Dependencies

The following libraries are required to run this notebook:

- `langchain`, `langchain_community`, `langchain-huggingface`, `langchain-text-splitters`, `langchain-chroma`
- `pypdf`, `tqdm`, `accelerate`, `bitsandbytes`, `python-dotenv`
- `transformers` (for Huggingface models)

To install the necessary packages, run:

```bash
!pip install --upgrade langchain langchain_community langchain-huggingface langchain-text-splitters langchain-chroma pypdf tqdm accelerate bitsandbytes python-dotenv
```

### Steps
1. Load Data and Parameters
    - Load the PDF document and set chunking parameters.
    - Set up Huggingface models for encoding and generation.
2. Chunking
    - Use `RecursiveCharacterTextSplitter` to chunk the document into sections for processing.
3. Custom Embeddings
    - Implement custom embeddings using `CLS pooling` from a Huggingface model.
4. Indexing
    - Index the document chunks into Chroma for efficient similarity search.
5. Load Generative Model
    - Load a generative model with quantization options to handle large models efficiently.
6. Set up LLM Chain
    - Create a Langchain pipeline to process and respond to queries using the indexed data.
7. Generation
    - Run a sample query to generate a response based on the document context.

### Notes
- Ensure you have the necessary [access tokens](https://huggingface.co/docs/hub/security-tokens) set up if using models that require authentication. If you use Google Colab, you can set your Huggingface access token via [secret](https://x.com/GoogleColab/status/1719798406195867814). In particular, you need export the env var in `.env` file:
```bash
USE_COLAB_SECRET=true
```
And set you secret as `HF_TOKEN`. \
Alternatively, set the token directly in `.env`:
```bash
USE_COLAB_SECRET=false
HF_TOKEN=<HF_TOKEN>
```
- Adjust parameters like `CHUNK_SIZE`, `TOP_K`, and `generation_kwargs` based on your specific use case and computational resources.
