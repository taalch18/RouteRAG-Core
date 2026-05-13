# Adaptive RAG Query Routing

An intelligent Retrieval-Augmented Generation (RAG) system that dynamically selects the optimal retrieval strategy for a query before answer generation.

Instead of treating every query the same way, the system analyzes the query’s characteristics and routes it through the most effective retrieval path:

- Direct LLM generation
- Dense vector retrieval
- Sparse BM25 retrieval
- Web search retrieval

The project focuses on improving:
- Retrieval quality
- Cost efficiency
- Latency optimization
- Faithfulness in RAG systems

# Motivation

Most production RAG systems assume that every query should use the same retrieval strategy.

This creates major inefficiencies:

| Query Type | Problem With Uniform Retrieval |
|---|---|
| "What is gradient descent?" | Dense retrieval is unnecessary |
| "SIGKILL-5 on pod xyz-123" | Dense retrieval fails exact matching |
| "Latest OpenAI pricing today" | Internal vector store becomes stale |

This project introduces a lightweight routing layer that predicts the best retrieval modality based on query characteristics.


# Core Research Question

Can a lightweight query classifier intelligently route queries between multiple retrieval modalities while reducing cost and preserving answer quality?


# System Architecture

```text
User Query
    │
    ▼
Query Router
(Logistic Regression / BERT)
    │
    ├──► LLM Direct
    ├──► Dense Retrieval
    ├──► Sparse BM25
    └──► Web Search
            │
            ▼
      Context Fusion
            │
            ▼
      GPT-4o Generator
            │
            ▼
   Evaluation + Cost Tracking
```


# Retrieval Paths

## 1. LLM Direct
Used when the model can answer directly from parametric memory.

Example:
- "What is attention mechanism?"

Benefits:
- Lowest latency
- Zero retrieval cost

## 2. Dense Vector Retrieval

Uses:
- Sentence Transformers
- Pinecone
- Cosine similarity

Best for:
- Semantic understanding
- Conceptual similarity

Example:
- "How does attention differ from RNNs?"

## 3. Sparse BM25 Retrieval

Uses:
- rank-bm25

Best for:
- Exact token matching
- Error logs
- IDs
- Dates

Example:
- "SIGKILL-5 on pod abc-123"

## 4. Web Search Retrieval

Uses:
- Serper API

Best for:
- Real-time external information

Example:
- "Latest Claude pricing today"

# Routing Signals

The router predicts the retrieval strategy using three signal groups:

## Parametric Answerability
Can the LLM answer without retrieval?

## Token Specificity
Does the query require exact token matching?

## Information Source Type
Where does the answer exist?
- Internal corpus
- Public internet
- Parametric memory

# Tech Stack

| Component | Technology |
|---|---|
| Orchestration | LangGraph |
| LLM | GPT-4o |
| Dense Retrieval | Pinecone |
| Embeddings | all-MiniLM-L6-v2 |
| Sparse Retrieval | rank-bm25 |
| Web Search | Serper API |
| NLP | spaCy |
| ML Router | scikit-learn |
| Evaluation | RAGAS |
| Validation | Pydantic |

# Evaluation Metrics

The system is benchmarked using:

- Faithfulness
- Context Precision
- Answer Relevancy
- Cost per query
- Latency

Baselines:
- Always Dense
- Always Sparse
- Always Web
- Random Router

# Repository Structure

```text
adaptive-rag-routing/
│
├── data/
├── notebooks/
├── src/
│   ├── router/
│   ├── retrieval/
│   ├── generation/
│   ├── evaluation/
│   ├── pipelines/
│   └── utils/
│
├── experiments/
├── tests/
├── requirements.txt
├── README.md
└── .env
```

# Setup

## Clone Repository

```bash
git clone https://github.com/yourusername/adaptive-rag-routing.git
cd adaptive-rag-routing
```

## Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate
```

Windows:

```bash
venv\Scripts\activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

# Environment Variables

Create a `.env` file:

```env
OPENAI_API_KEY=your_key
PINECONE_API_KEY=your_key
SERPER_API_KEY=your_key
```

# Future Work

- Hybrid dense+sparse retrieval
- Reinforcement-learning routers
- Multi-hop retrieval
- Adaptive chunking
- Agentic retrieval planners
- Confidence-aware fallback routing

# Research Positioning

This project explores retrieval modality routing for cost-aware and quality-aware RAG systems.

Related areas:
- Adaptive RAG
- Self-RAG
- Retrieval optimization
- Agentic AI systems
- Query planning systems

# License

MIT License


# Author

Taal Chawla

Machine Learning Engineering • Retrieval Systems • Agentic AI
