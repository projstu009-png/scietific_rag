
# Scientific RAG System

**A retrieval-augmented generation (RAG) platform for scientific documents, built for high correctness, configurability, and advanced research use. Optimized for physics and its subdomains, supporting both theoretical and experimental scientific literature.**

***

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Configuration](#configuration)
- [Usage](#usage)
    - [Ingesting Papers](#ingesting-papers)
    - [Querying the System](#querying-the-system)
    - [Citation Graph Retrieval](#citation-graph-retrieval)
    - [Benchmarking and Evaluation](#benchmarking-and-evaluation)
- [Customization & Extensions](#customization--extensions)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [References](#references)

***

## Overview

This platform is a production-oriented scientific QA and assistant system leveraging Retrieval-Augmented Generation (RAG) with agentic workflows and advanced retrieval orchestration. It is designed for high-stakes scientific contexts, supporting:

- Open-source, local model deployments (embedding & LLM)
- Physics-specific document handling (equations, citations, figures)
- Multi-strategy retrieval (dense, sparse, hybrid, citation-graph)
- Citation verification and equation normalization

***

## Features

- **Domain-tuned Embeddings:** Supports scientific embeddings (e.g., nomic-embed-text-v1.5)
- **Vector and BM25 Hybrid Database:** Uses Qdrant for retrieval, with tunable quantization
- **Agentic Orchestration (LangGraph):** Multi-step, reflective agent with document verification
- **Citation Graph Retrieval:** Traverse citation relationships to extend context beyond direct matches
- **Equation Normalization:** Semantic LaTeX handling with SymPy, bigram and Greek preservation
- **Configurable & Extensible:** Every major component and strategy is controlled by config files
- **Scientific Benchmarking:** Integrates RAGAS, domain-specific benchmarks, and custom scripts
- **Production-Ready:** Logging, error-handling, safe defaults, and best practices for research integrity

***

## System Architecture

```mermaid
graph TD
    subgraph Ingestion
        A[PDF/ArXiv Paper] --> B[PDF Parser]
        B --> C[Chunker]
        C --> D[Embedding]
        C --> E[Metadata + Citations + Equations]
        D --> F[Qdrant Vector Store]
        E --> G[Citation Graph Builder]
    end
    subgraph Retrieval/QA
        H[User Query] --> I[Embedder]
        I --> J[Dense Retrieval]
        H --> K[Sparse Retrieval (BM25, bigram, Greek)]
        H --> L[Graph Traversal (optional)]
        J --> M[Fusion & Reflection]
        K --> M
        L --> M
        M --> N[LLM Generator + Agent Workflow]
        N --> O[Final Answer + Citations]
    end
```

***

## Installation

### Prerequisites

- Python 3.9+ (recommended: 3.11+)
- Docker (for Qdrant)
- [Ollama](https://ollama.com/) (for local LLM and embedding API)
- Node.js (optional, for frontend components)
- Scientific PDFs for ingestion

### Setup

1. **Clone the Repository**
    ```bash
    git clone https://github.com/yourorg/scientific-rag
    cd scientific-rag
    ```

2. **Install Python Dependencies**
    ```bash
    python3 -m venv env
    source env/bin/activate
    pip install -r requirements.txt
    ```

3. **Launch Qdrant**
    ```bash
    docker run -p 6333:6333 -p 6334:6334 -v "$(pwd)/qdrant_storage:/qdrant/storage:z" qdrant/qdrant
    ```

4. **Install & Start Ollama**
    - [Install Ollama](https://ollama.com/download)
    - Pull required models (e.g., nomic-embed-text, Qwen):
        ```bash
        ollama pull nomic-embed-text:v1.5
        ollama pull qwen
        ```

***

## Quickstart

**Ingest a scientific paper:**

```bash
python3 scripts/ingest_papers.py --papers ~/path/to/papers/my_paper.pdf
```

**Ask a question:**

```bash
python3 scripts/query_system.py --question "What is Xe-135?"
```

***

## Configuration

All major options are exposed in `config/config.yaml`. Key sections:

- **Embeddings:** Model, dimensions, task type
- **LLM:** Provider, endpoint, parameters
- **Vectorstore:** Qdrant host, port, quantization settings
- **Processing:** Chunking, separator logic, LaTeX extraction
- **Retrieval:** Hybrid weights, reranking, multi-query, citation graph
- **Scientific:** Equation/citation handling, section weighting
- **Advanced retrieval:** Query expansions, section-awareness, citation graph parameters

_Example:_
```yaml
embeddings:
  provider: "ollama"
  model: "nomic-embed-text:v1.5"
  ...
vectorstore:
  provider: "qdrant"
  mode: "server"
  quantization:
    enabled: false
...
scientific:
  advanced_retrieval:
    citation_graph:
      enabled: true
      max_depth: 2
```

***

## Usage

### Ingesting Papers

- Supports individual PDFs or a directory of papers.
- Extracts text, equations, tables, citations, and splits into chunks for indexing.
  
```bash
python3 scripts/ingest_papers.py --papers ~/Downloads/20050613.pdf
```

### Querying the System

- Pass your query via CLI; agentic workflow orchestrates retrieval and LLM response.
- Returns answer with inline citations and sources.

```bash
python3 scripts/query_system.py --question "Explain the role of Xe-135 in reactors."
```

### Citation Graph Retrieval

- With citation graph enabled, system traverses citation links up to configured depth, retrieving context from scientifically related papers.
- Useful for multi-hop or literature survey questions.

### Benchmarking and Evaluation

- Evaluate retrieval and QA performance using Ragas and custom scripts.
- Supports physics-specific datasets (e.g., PHYSICS).
  
```bash
python3 scripts/run_benchmark.py --config config/config.yaml --dataset tests/test_data/eval_dataset.json
```

***

## Customization & Extensions

- **Add new retrieval strategies:** Implement in `src/retrieval/` and reference in config.
- **Plug different LLMs or embedders:** Swap out the provider/model and ensure API compatibility.
- **Expand scientific tokenization:** Edit `ScientificBM25` to preserve more multiword, Greek, or domain-sensitive tokens.
- **Agent logic:** Edit `src/agents/rag_agent.py` to adjust reflection, reranking, quality thresholds, or introduce new agent nodes.
- **Cite graph depth, section weights, and other heuristics** are all configurable.
- **Front-end**: (Optional) You can add an API or webapp layer on top for researchers.

***

## Best Practices

- For best results, ingest as many relevant papers as possible and ensure text extraction quality.
- Use domain-specific embeddings/models for higher accuracy.
- Adjust chunk and retrieval sizes for your hardware and use case.
- Regularly benchmark with domain datasets and update the citation graph.
- Review cited context for every LLM answer in high-stakes or publication settings.

***

## Troubleshooting

- **Qdrant connection issues:** Verify Docker and network settings, ensure ports aren't blocked.
- **Ollama model errors:** Ensure models are pulled and the Ollama service is running.
- **“ValueError: empty separator” or similar:** Review chunking config (see README FAQ).
- **Agent infinite loop:** Check `max_iterations` and `max_no_improve` in config.

***

## Contributing

Contributions are welcome! Please:
- File issues for bugs or scientific feature requests.
- PRs should include appropriate tests and docstrings.
- New scientific domains/extensions: add new tokenization rules/templates and cite your data/test sets.

***

## License

This project is licensed under the MIT License.

***

## References

- [LangGraph](https://github.com/langchain-ai/langgraph) - Agent orchestration
- [Qdrant](https://qdrant.tech/) - Vector database
- [Ollama](https://ollama.com/) - Local models
- [SymPy](https://www.sympy.org/) - Equation normalization
- [RAGAS](https://github.com/explodinggradients/ragas) - RAG evaluation

***

