<div align="center">

# 🔀 Adaptive RAG Query Router

### A production-grade framework that routes queries to the optimal retrieval strategy — saving cost without sacrificing quality

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-1C1C1C?style=flat&logo=langchain&logoColor=white)](https://github.com/langchain-ai/langgraph)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat)](LICENSE)
[![Status: Active](https://img.shields.io/badge/Status-Active-brightgreen?style=flat)]()
[![Eval: RAGAS](https://img.shields.io/badge/Eval-RAGAS-blueviolet?style=flat)](https://docs.ragas.io)

**[Live Demo (coming soon)]** · **[Architecture](#architecture)** · **[Benchmarks](#benchmarks)** · **[Quickstart](#quickstart)**


*Every RAG system today sends every query through the same retrieval strategy. That's expensive and often wrong. This project fixes it.*

</div>

## The Problem

Two queries hit the same RAG pipeline:

```
"What is the attention mechanism in transformers?"
"What caused SIGKILL-5 on pod sre-agent-abc-123 at 03:41 UTC?"
```

A standard RAG system routes **both** through dense vector retrieval. This is wrong twice over:

- The first query is **parametrically answerable** — the LLM already knows this from training. Dense retrieval adds latency and cost with zero quality gain.
- The second needs **exact token matching** — the pod ID `abc-123` is a lookup key, not a semantic concept. Dense retrieval finds plausible-sounding but wrong documents.

**This router fixes that.** It classifies the incoming query, selects the right retrieval strategy, and tracks exactly what that decision saved.


## How It Works

The router extracts three signals from every query:

| Signal | What It Detects | Routes To |
|---|---|---|
| **Parametric Answerability** | Can the LLM answer this from training weights alone? | LLM Direct (no retrieval) |
| **Token Specificity** | Does it need exact match — error codes, IDs, dates? | Sparse BM25 |
| **Information Source Type** | Is the answer external / real-time / post-cutoff? | Web Search |
| *(default)* | Semantic / conceptual query | Dense Vector |

A lightweight logistic regression classifier (trained on 10 handcrafted features, <5ms inference on CPU) makes the routing decision. Every run is evaluated with RAGAS faithfulness and a formal cost model.

## Architecture

```
                   ┌──────────────────────────────────┐
  User Query ────► │          Query Router             │
                   │   LogisticRegression + spaCy      │
                   │                                   │
                   │  · Parametric Answerability       │
                   │  · Token Specificity              │
                   │  · Information Source Type        │
                   └──────┬──────┬──────┬──────┬───────┘
                          │      │      │      │
                   ┌──────┘      │      │      └──────┐
                   ▼             ▼      ▼             ▼
            ┌──────────┐  ┌──────────┐ ┌──────────┐ ┌──────────┐
            │  Direct  │  │  Dense   │ │  Sparse  │ │   Web    │
            │ (no ret) │  │ Pinecone │ │  BM25    │ │  Search  │
            │          │  │ MiniLM   │ │ rank_bm25│ │  Serper  │
            └────┬─────┘  └────┬─────┘ └────┬─────┘ └────┬─────┘
                 │             │             │             │
                 │        ┌────┴─────────────┘             │
                 │        │     RRF Fusion (k=60)          │
                 │        └──────────┐                     │
                 └───────────────────┼─────────────────────┘
                                     ▼
                            ┌────────────────┐
                            │  LLM Generator │
                            │    GPT-4o      │
                            └───────┬────────┘
                                    │
                       ┌────────────┴───────────┐
                       ▼                        ▼
                  RAGAS Eval              Cost Tracker
             Faithfulness · Precision   USD · tokens · ms
```

## Stack

| Layer | Technology | Why |
|---|---|---|
| Orchestration | **LangGraph** | Stateful routing graph with typed conditional edges |
| Dense Retrieval | **Pinecone** + `all-MiniLM-L6-v2` | Managed vector store, local embeddings (zero API cost) |
| Sparse Retrieval | **rank-bm25** (BM25Okapi) | Pure Python, zero infra, exact token match |
| Web Search | **Serper API** | Generous free tier, Google Search results |
| Router | **scikit-learn** LR + **spaCy** | Interpretable, sub-5ms inference, CPU-only |
| Evaluation | **RAGAS** | Faithfulness + Context Precision without human labels |
| Data Validation | **Pydantic v2** | Typed schemas at every system boundary |
| Demo | **Streamlit** | Live cost and quality tracking per query |
| Generation | **OpenAI GPT-4o** | Final answer conditioned on retrieved context |

## Project Structure

```
adaptive-rag-router/
├── router/
│   ├── classifier.py        # Query classifier (LR + BERT variants)
│   ├── features.py          # Feature extraction — 10 routing signal features
│   └── routing_signals.py   # Parametric / specificity / source detectors
├── retrieval/
│   ├── dense.py             # Pinecone dense retrieval
│   ├── sparse.py            # BM25Okapi retrieval
│   ├── web_search.py        # Serper API wrapper
│   └── llm_direct.py        # Direct generation, no retrieval
├── pipeline/
│   ├── graph.py             # LangGraph state machine
│   └── schemas.py           # Pydantic I/O models
├── eval/
│   ├── ragas_runner.py      # RAGAS evaluation harness
│   ├── cost_tracker.py      # Token + API cost logging per query
│   └── benchmark.py         # Full benchmark runner across strategies
├── demo/
│   └── app.py               # Streamlit demo with live cost tracking
├── data/
│   ├── label_queries.py     # Oracle routing labeler
│   └── datasets/            # NQ, HotpotQA, TriviaQA, synthetic SRE
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_classifier_training.ipynb
│   └── 03_results_analysis.ipynb
├── docker-compose.yml
├── .env.example
├── requirements.txt
└── README.md
```

## Quickstart

### Prerequisites

- Python 3.10+
- Pinecone account (free tier)
- OpenAI API key
- Serper API key (free tier: 2,500 queries/month)

### Installation

```bash
git clone https://github.com/taalchawla/adaptive-rag-router.git
cd adaptive-rag-router

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Environment Setup

```bash
cp .env.example .env
# Fill in:
# OPENAI_API_KEY=sk-...
# PINECONE_API_KEY=...
# PINECONE_INDEX=adaptive-rag
# SERPER_API_KEY=...
```

### Run a Query

```python
from pipeline.graph import build_graph

graph = build_graph()

result = graph.invoke({
    "query": "What caused SIGKILL-5 on pod sre-agent-abc-123 at 03:41 UTC?",
    "route": None,
    "retrieved_chunks": [],
    "answer": "",
    "cost_usd": 0.0,
    "faithfulness": 0.0
})

print(f"Route:      {result['route']}")        # sparse
print(f"Answer:     {result['answer']}")
print(f"Cost (USD): ${result['cost_usd']:.4f}")
print(f"Faith:      {result['faithfulness']:.2f}")
```

### Run the Demo

```bash
streamlit run demo/app.py
```

### Run Full Benchmark

```bash
python eval/benchmark.py \
  --datasets nq hotpotqa trivia synthetic_sre \
  --strategies all \
  --output results/benchmark.csv
```

## Benchmarks

> Results being populated as benchmark suite completes.

| Strategy | Faithfulness↑ | Ctx Precision↑ | Cost / 1K queries↓ | Latency p50↓ |
|---|---|---|---|---|
| Always-Dense (baseline) | — | — | — | — |
| Always-BM25 (baseline) | — | — | — | — |
| Always-Web (baseline) | — | — | — | — |
| Random Router | — | — | — | — |
| **LR Router (ours)** | — | — | — | — |
| **BERT Router (ours)** | — | — | — | — |


## Cost Model

```
Total Cost = C_retrieval + C_generation

C_retrieval:
  Dense  → local MiniLM ($0) + Pinecone (~$0 free tier)
  BM25   → local computation ≈ $0
  Web    → $0.001 / query (Serper)
  Direct → $0

C_generation:
  input_tokens  × $0.0000025  (GPT-4o)
  output_tokens × $0.0000100  (GPT-4o)
```

## Author

**Taal Chawla** — ECE (AI/ML), MAIT GGSIPU Delhi

[![LinkedIn](https://img.shields.io/badge/LinkedIn-taalchawla18-0A66C2?style=flat&logo=linkedin)](https://linkedin.com/in/taalchawla18)


<div align="center">
<sub>Built as a production AI engineering project. Star to follow progress.</sub>
</div>
