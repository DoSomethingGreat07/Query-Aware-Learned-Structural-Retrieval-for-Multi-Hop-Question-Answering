# GraphRAG: Multi-Hop Retrieval-Augmented Generation on HotpotQA

A graph-based multi-hop retrieval pipeline for complex question answering. Built on HotpotQA, this system goes beyond standard dense retrieval by constructing a hybrid document graph and using graph traversal, PCST, and GNN-based reranking to find **bridge documents** that standard RAG systems miss.

---

## The Problem

Multi-hop questions require connecting facts across multiple documents:

> *"Where was the director of the film Titanic born?"*

Standard dense retrieval finds documents similar to the query — but misses the **bridge document** that connects the two required facts. This system is designed to find both.

```
Standard RAG:   Query → [Titanic] → ❌ misses birthplace
GraphRAG:       Query → [Titanic] → [James Cameron] → [Kapuskasing] ✅
```

---

## Architecture

```
Query
  ↓
Dense Retrieval (all-MiniLM-L6-v2)    ← Hop-1: top-20 seed chunks via FAISS
  ↓
Hybrid Graph Expansion                 ← Hop-2: expand via bridge edges
  ↓
Re-ranking (PCST / GNN)               ← score and select best subgraph
  ↓
LLM Answer Generation (GPT-4o-mini)   ← answer from top-5 retrieved chunks
```

---

## Graph Construction

The hybrid document graph connects 139k chunks via four edge types:

| Edge Type | Signal | Weight |
|---|---|---|
| **Hyperlink** | Title mention in text (Wikipedia proxy) | 1.00 |
| **Adjacent chunk** | Sequential chunks from same article | 0.90 |
| **Same title** | All chunks from same article | 0.95 |
| **Keyword overlap** | Shared named entities across articles | 0.55–0.85 |

At 10k samples: **139,504 nodes**, **1,060,371 edges**, **979,249 hyperlink edges**.

Semantic KNN edges are automatically disabled above 10k nodes to avoid O(N²) cost.

---

## Retrieval Methods

### 1. Graph Two-Hop Traversal
Expands from hop-1 seeds via bridge edges. Hop-1 and hop-2 get **separate quotas** — they never compete for the same slots.

```
Hop-1: top-5 from dense retrieval
Hop-2: top-5 from bridge neighbors, scored by:
       (1 - w) * query_sim + w * bridge_score
```

### 2. PCST (Prize-Collecting Steiner Tree)
Finds the minimum-cost connected subgraph that maximizes node prizes. Seeds are locked anchors — PCST only controls expansion beyond them.

### 3. GNN Reranking (GraphSAGE)
Trained with contrastive loss on bridge-connected pairs. GNN embeddings are used **only for hop-2** — hop-1 uses base embeddings to preserve recall.

```
hop-1 triplets:  9,210   (at 10k samples)
hop-2 triplets:  1,350   (23x more than 200-sample run)
```

---

## Project Structure

```
src/
├── loading.py                 # HotpotQA dataset loading
├── chunking.py                # Document chunking
├── embeddings.py              # Chunk embedding generation
├── hybrid_graph_builder.py    # Graph construction (all edge types)
├── graph_retrieval.py         # Two-hop graph traversal
├── pcst.py                    # PCST-based retrieval
├── gnn_train.py               # GraphSAGE training
├── retrieval_eval.py          # Baseline dense retrieval eval
├── graph_retrieval_eval.py    # Graph two-hop eval
├── pcst_eval.py               # PCST eval
├── gnn_rerank_eval.py         # GNN reranking eval
└── llm_eval.py                # End-to-end LLM accuracy eval
```

---

## Setup

```bash
# Clone repo
git clone https://github.com/DoSomethingGreat07/graphrag-hotpotqa
cd graphrag-hotpotqa

# Create virtual environment
python3 -m venv graphrag_env
source graphrag_env/bin/activate

# Install dependencies
pip install torch torch-geometric sentence-transformers
pip install datasets networkx openai python-dotenv tqdm
```

Create a `.env` file:
```
OPENAI_API_KEY=your_key_here
```

---

## Run Order

```bash
# 1. Build graph (one-time, ~15 min at 10k samples)
python src/hybrid_graph_builder.py

# 2. Baseline evaluation
python src/retrieval_eval.py

# 3. Graph two-hop evaluation
python src/graph_retrieval_eval.py

# 4. PCST evaluation
python src/pcst_eval.py

# 5. GNN training + evaluation
python src/gnn_rerank_eval.py

# 6. End-to-end LLM accuracy
python src/llm_eval.py
```

---

## Key Design Decisions

**Hyperlink edges as primary bridge signal**
HotpotQA is Wikipedia-based. When chunk A's text contains the title of article B, a hyperlink edge is added. This creates 979k bridge connections at 10k scale.

**Separate hop-1 / hop-2 quotas**
Hop-2 candidates previously lost to hop-1 in score ranking. Giving each hop its own `final_k` slots ensures bridge documents always have a chance to appear.

**GNN only for hop-2**
Using GNN embeddings to rerank hop-1 seeds dropped `hop1_contributed` from 0.94 → 0.76. Base embeddings are kept for hop-1; GNN is used only for hop-2 bridge scoring.

**seed_k=20 at 10k scale**
At 200 samples, `seed_k=10` covered 0.19% of the pool. At 10k samples with 139k chunks, the same `seed_k=10` covers only 0.007%. Increasing to 20 compensates for the larger pool.

---

## Dataset

[HotpotQA](https://hotpotqa.github.io/) — distractor setting, train split.

```
Total questions:   ~90,447
Used:              10,000
Bridge questions:  ~81%
Comparison:        ~19%
```

---

## Tech Stack

| Component | Library |
|---|---|
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Graph | NetworkX |
| GNN | PyTorch Geometric (GraphSAGE) |
| LLM | OpenAI GPT-4o-mini |
| Dataset | HuggingFace datasets |

---

## Author

**Nikhil Juluri** — MS Computer Science, University of Illinois Chicago  
[GitHub](https://github.com/DoSomethingGreat07) · [LinkedIn](https://linkedin.com/in/nikhiljuluri)
