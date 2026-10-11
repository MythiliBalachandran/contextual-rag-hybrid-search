# Contextual RAG System with Hybrid Search and Reranking

A local Retrieval-Augmented Generation (RAG) pipeline that combines semantic vector search, BM25 keyword search, and cross-encoder reranking to retrieve relevant document passages and generate grounded answers.

## Overview

Retrieval quality is a major challenge in RAG systems. Semantic search can miss exact terminology, while keyword-based search may miss passages with similar meaning. This project combines both retrieval approaches and reranks their candidate passages before generating an answer.

## Components

- **LLM:** Llama 3.2 3B, served locally using Ollama
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`
- **Vector database:** Chroma with cosine similarity
- **Keyword retrieval:** BM25 using `rank_bm25`
- **Hybrid retrieval:** LangChain `EnsembleRetriever`
- **Reranker:** `BAAI/bge-reranker-v2-m3`
- **Framework:** LangChain
- **Document processing:** PyMuPDF and Recursive Character Text Splitter

## Pipeline

1. **Document loading:** Load the Wikidata JSONL dataset and research-paper PDFs.
2. **Chunking:** Split PDF content into chunks while retaining metadata such as source and page number.
3. **Embedding and indexing:** Generate local sentence embeddings and store document vectors in Chroma.
4. **Semantic retrieval:** Retrieve passages using vector similarity.
5. **Keyword retrieval:** Retrieve passages using BM25.
6. **Hybrid search:** Combine the two retrievers using a weighted ensemble.
7. **Reranking:** Score query–document pairs with the cross-encoder and select the top five passages.
8. **Answer generation:** Pass the retrieved context and question to the local LLM, with instructions to answer from the context and admit when the answer is unavailable.

## Evaluation

The project includes five manually selected test questions covering machine learning, Vision Transformers, and Transformer attention, plus an out-of-scope question about the 1998 FIFA World Cup.

- Saved generated answers in `rag_evaluation_results.json`.
- Saved retrieved-document comparisons across vector search, BM25, hybrid search, and hybrid search with reranking in `rag_retrieval_comparison.json`.
- Manually inspected the answers and retrieval outputs to understand where retrieval methods succeed or return irrelevant passages.

These are small, manually inspected tests—not a statistically representative benchmark. No general retrieval-accuracy claim is made.

## Run Locally

### Prerequisites

- Python 3.12
- Ollama
- The `llama3.2:3b` model
- The Python dependencies required by the notebook

### Setup

1. Clone this repository and open its folder.
2. Create and activate a Python virtual environment.
3. Install the dependencies required by the notebook.
4. Start Ollama and download the model:

   ```bash
   ollama pull llama3.2:3b
   ```

5. Open `Contextual_RAG_System.ipynb` in Jupyter or VS Code.
6. Select the project environment as the notebook kernel and run the cells in order.

The notebook uses local inference and local embeddings; an OpenAI API key is not required for this pipeline.

## Project Structure

- `Contextual_RAG_System.ipynb` — end-to-end RAG pipeline
- `rag_docs/` — source documents used by the notebook
- `images/` — project diagram
- `rag_evaluation_results.json` — saved answer examples
- `rag_retrieval_comparison.json` — saved retrieval comparisons

## Limitations

- Evaluation uses a small set of manually selected questions.
- Reranking adds computational cost and latency.
- The local LLM may still generate incomplete or incorrect answers.
- Retrieval of irrelevant passages is possible, even when the final answer correctly abstains.

## Future Improvements

- Create a larger, manually labelled retrieval evaluation set.
- Measure retrieval metrics such as Precision@k, Recall@k, and MRR using passage-level relevance judgements.
- Compare answer quality and latency across retrieval configurations.
- Add automated tests for retrieval and answer grounding.