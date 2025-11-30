# AI Tutor: Intelligent RAG System

An adaptive tutoring system that recommends personalized practice questions for competitive exams (JEE, NEET) using Retrieval-Augmented Generation (RAG). The system understands student context, learns from interaction patterns, and dynamically calibrates question difficulty.

## 🚀 Features

-   **Intelligent Retrieval**: 3-stage pipeline (Vector Search -> Multi-Signal Ranking -> LLM Refinement).
-   **Adaptive Difficulty**: Vygotsky-inspired calibration to keep students in their "Zone of Proximal Development".
-   **Context Awareness**: Analyzes chat history to detect learning gaps and performance signals (struggling, bored, ready for challenge).
-   **Low Latency**: Optimized for <500ms response times using local embeddings and Groq LPU inference.
-   **Production Ready**: Async architecture, modular design, and comprehensive error handling.

## 🛠️ Tech Stack

-   **Language**: Python 3.9+
-   **Vector DB**: Qdrant (Local or Cloud)
-   **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (Local)
-   **LLM**: Groq (`llama-3.1-8b-instant`) for signal detection and re-ranking.
-   **API**: FastAPI
-   **Orchestration**: Custom Python pipeline

## 📂 Project Structure

```
.
├── data/
│   └── data.json            # Question corpus (Biology, Physics, Chemistry)
├── src/
│   ├── api.py               # FastAPI application
│   ├── config.py            # Configuration settings
│   ├── ingestion.py         # Data loading & embedding pipeline
│   ├── orchestrator.py      # Main system coordinator
│   ├── retrieval.py         # Core RAG logic (3 stages)
│   └── logger.py            # Logging utility
├── demo.py                  # End-to-end demo script
├── requirements.txt         # Dependencies
└── README.md                # Project documentation
```

## ⚡ Quick Start

### 1. Prerequisites

-   Python 3.9+
-   [Qdrant](https://qdrant.tech/) (Running locally on port 6333)
    ```bash
    docker run -p 6333:6333 qdrant/qdrant
    ```

### 2. Installation

1.  Clone the repository:
    ```bash
    git clone <repository-url>
    cd Ai_tutor
    ```

2.  Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4.  Set up environment variables:
    Create a `.env` file in the root directory:
    ```env
    QDRANT_URL=http://localhost:6333
    QDRANT_API_KEY=
    GROQ_API_KEY=your_groq_api_key_here
    ```

### 3. Data Ingestion

Load the question corpus into Qdrant. This script generates embeddings locally and indexes metadata.

```bash
python -m src.ingestion
```

### 4. Running the Server

Start the FastAPI server:

```bash
uvicorn src.api:app --reload
```
The API will be available at `http://localhost:8000`.

### 5. Running the Demo

Run the end-to-end demo script to see the system in action. This script simulates 5 different student scenarios (Beginner, Advanced, Chemistry Concept, etc.) and saves the output to `outputs/output.json`.

```bash
python demo.py
```

## 🔧 Configuration

You can customize the system behavior in `src/config.py` or via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `QDRANT_URL` | `http://localhost:6333` | URL of Qdrant instance |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Local embedding model name |
| `ENABLE_LLM_RANKING` | `False` | Enable Stage 3 LLM re-ranking (adds latency) |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Model used for signal detection |
| `MAX_TOTAL_LATENCY_MS` | `500` | Latency budget target |

## 📊 Latency & Performance

The system is designed to be fast. Typical latency breakdown:

-   **Signal Detection**: ~150ms (Async)
-   **Retrieval (Stage 1)**: ~30ms
-   **Ranking (Stage 2)**: <5ms
-   **Total End-to-End**: **~200-300ms** (without LLM re-ranking)

---

# System Design & Architecture

## 1. System Overview

The AI Tutor is an intelligent Retrieval-Augmented Generation (RAG) system designed to recommend personalized practice questions to students preparing for competitive exams (JEE, NEET, etc.). The system orchestrates a 4-stage pipeline that understands student context, retrieves relevant questions with intelligent caching, and ranks them based on pedagogical principles (Vygotsky's Zone of Proximal Development).

### High-Level Architecture

![Architecture Diagram](architecture_diagram.png)

The diagram above illustrates the complete request-response flow through four distinct stages:

#### Request Flow
1. **Student/Client** sends a request containing their profile (grade, expertise level, weak topics) and chat history to the **FastAPI/Orchestrator**
2. The orchestrator initializes the **Retrieval Pipeline** which coordinates the four stages in parallel/sequence
3. Results flow back as a **JSON Response** containing 3 personalized question recommendations

#### Pipeline Stages

**STAGE 0: Context Understanding (Blue)**
- **Input**: Chat history from student conversations
- **Component**: Groq LLM Signal Detector (powered by `llama-3.1-8b-instant`)
- **Purpose**: Analyzes chat messages to detect:
  - Performance signals (struggling, bored, ready for challenge)
  - Specific knowledge gaps (e.g., "confused about Bohr's atomic model")
  - Topic preferences and learning patterns
- **Output**: Performance signals that influence ranking in later stages

**STAGE 1: Candidate Generation (Green) - ⚡ Fast with Redis Cache**
- **NEW: Redis LRU Cache** (highlighted in red)
  - Caches the last 10 query results for instant retrieval
  - **Cache Hit Path**: If query exists in cache → results sent directly to Multi-Signal Ranker (0ms latency)
  - **Cache Miss Path**: Falls back to Qdrant Vector DB
- **Qdrant Vector DB**: 
  - Performs semantic similarity search using embeddings
  - Returns Top 50 candidate questions with metadata
  - Results are cached in Redis for future requests
- **Performance**: Cache hits achieve near-zero latency (~0ms), cache misses take ~10-150ms

**STAGE 2: Intelligent Ranking (Orange) - Precise**
- **Component**: Multi-Signal Ranker
- **Purpose**: Scores each of the 50 candidates using four weighted signals:
  1. ✓ **Relevance Score** (30%): Vector similarity + keyword matching + gap alignment
  2. ✓ **Difficulty Calibration** (35%): Matches question difficulty to student's Zone of Proximal Development
  3. ✓ **Personalization** (20%): Boosts weak topics, penalizes strong topics, adapts to recent performance
  4. ✓ **Diversity** (15%): Ensures variety across subtopics, prefers recent exam questions
- **Output**: Scored candidates ranked by composite score
- **Performance**: <5ms (pure Python, no external calls)

**STAGE 3: LLM Refinement (Purple) - Optional**
- **Component**: Groq LLM Re-ranker
- **Purpose**: Provides deep contextual understanding for the top 10 candidates
- **Process**: LLM acts as a "tutor" to evaluate why each question fits the specific student profile
- **Output**: Final top 3 recommendations with human-readable reasoning
- **Performance**: ~100-200ms per batch (leverages Groq's LPU for speed)

#### Return Path
- Final recommendations flow back through the orchestrator
- JSON response includes:
  - Top 3 questions with full details
  - Reasoning for each recommendation
  - Identified knowledge gaps
  - Pipeline metadata (latency breakdown, cache status)

## 2. Core Components

### 2.1 Data Ingestion (`src/ingestion.py`)
- **Source**: JSON dataset containing questions with rich metadata (topic, subtopic, difficulty, exam year, etc.).
- **Embedding**: Uses a local **SentenceTransformer (`all-MiniLM-L6-v2`)** model to generate 384-dimensional vectors for question text + options + explanations.
- **Storage**: **Qdrant** vector database is used for high-performance similarity search. Payload indexes are created on `subject`, `topic`, `difficulty`, and `exam_type` for efficient filtering.

### 2.2 Retrieval Pipeline (`src/retrieval.py`)
The pipeline executes in three stages to balance latency and quality:

**Stage 1: Candidate Generation (<200ms)**
- **Redis Cache**: Checks for recently accessed queries (LRU, limit 10) for instant retrieval.
- **Vector Search**: Retrieves top 50 candidates using cosine similarity (if cache miss).
- **Metadata Filtering**: Hard filters for `Subject` and `Exam Target` (e.g., "Physics", "JEE Mains").
- **Topic Filtering**: If specific topics are detected in chat history (e.g., "thermodynamics"), a `should` filter boosts questions from those topics.

**Stage 2: Intelligent Ranking (<100ms)**
A heuristic scoring function ranks candidates based on four weighted signals:
1.  **Relevance (30%)**: Vector similarity score + keyword boosting from chat context.
2.  **Difficulty Calibration (35%)**: Calculates the "optimal difficulty" based on user expertise and recent performance. Questions closer to this target score higher (Vygotsky's ZPD).
3.  **Personalization (20%)**: Boosts questions from user's `weak_topics` and penalizes `strong_topics`. Adapts to recent performance (e.g., if struggling, lower difficulty).
4.  **Diversity (15%)**: Penalizes questions from the same subtopic to ensure a balanced set. Boosts recent exam questions (last 2 years).

**Stage 3: LLM Re-ranking (Optional, ~100-200ms)**
- Uses **Groq (Llama-3.1-8b-instant)** to analyze the top 10 candidates.
- The LLM acts as a "Tutor" to evaluate *why* a question fits the specific student profile and chat context.
- Returns a refined score and human-readable reasoning.

### 2.3 Orchestrator (`src/orchestrator.py`)
- Coordinates the pipeline.
- Generates **Tutor Context**:
    - **Identified Gaps**: Combines explicit weak topics with LLM-detected struggles.
    - **Study Sequence**: Recommends a path (e.g., "Build Foundation" -> "Solve Direct Problems").
    - **Next Milestone**: Sets a concrete goal based on current expertise.

## 3. Latency Analysis

The system is designed to meet a strict **<500ms** end-to-end latency budget, with Redis caching providing near-instant retrieval for repeated queries.

| Component | Technology | Avg Latency | Optimization Strategy |
|-----------|------------|-------------|-----------------------|
| **Redis Cache (Stage 1)** | Redis LRU | **~0ms** (hit) / N/A (miss) | LRU cache with limit of 10 queries. Instant retrieval on cache hit. |
| **Embedding** | `all-MiniLM-L6-v2` (Local) | ~20-40ms | Run in thread pool to avoid blocking async loop. Small model size. |
| **Vector Search (Stage 1)** | Qdrant (Local/Docker) | ~10-150ms | HNSW index, payload indexing, efficient filtering. Only runs on cache miss. |
| **Signal Detection (Stage 0)** | Groq (`llama-3.1-8b`) | ~150-200ms | Async call, runs parallel to Stage 1 retrieval. |
| **Ranking (Stage 2)** | Python (NumPy/Native) | <5ms | Vectorized operations, efficient in-memory processing. |
| **LLM Re-ranking (Stage 3)** | Groq (`llama-3.1-8b`) | ~100-200ms | **Optional**. Parallel execution for multiple candidates using `asyncio.gather`. |
| **Total (Cache Hit)** | | **~0-300ms** | Redis cache eliminates Stage 1 latency for repeated queries. |
| **Total (Cache Miss)** | | **~200-450ms** | Async architecture, fast local embeddings, Groq LPU inference. |

## 4. Difficulty Calibration Logic

The system implements **Adaptive Difficulty** based on the Zone of Proximal Development (ZPD).

**Formula:**
`Target Difficulty = User Expertise + 0.7`

- **Base Logic**: Students learn best when challenged slightly beyond their current ability (+0.7 on a 1-5 scale).
- **Dynamic Adjustment**:
    - If `confidence_score < 2.0` (Struggling): `Target -= 0.3` (Ease off to build confidence).
    - If `confidence_score > 4.5` (Bored/Mastery): `Target += 0.5` (Push harder).
- **Matching Score**: `1.0 / (1.0 + abs(Question_Difficulty - Target))`
    - Questions exactly at the target difficulty get a score of 1.0.
    - The score decays as the difficulty gap increases.

## 5. Trade-offs and Scalability

### Trade-offs
1.  **Local vs. API Embeddings**:
    - *Decision*: Used local `sentence-transformers`.
    - *Why*: Removes network latency for embedding generation (saving ~100-200ms vs OpenAI).
    - *Trade-off*: Slightly lower semantic understanding than large models (e.g., OpenAI `text-embedding-3-small`), but sufficient for question retrieval.

2.  **Heuristic vs. Full LLM Ranking**:
    - *Decision*: Hybrid approach. Stage 2 is heuristic (fast), Stage 3 is LLM (slow/optional).
    - *Why*: Ranking 50 candidates with an LLM is too slow (>2s). Heuristics filter the list efficiently, allowing the LLM to focus on the top 10.

3.  **Groq vs. OpenAI**:
    - *Decision*: Used Groq.
    - *Why*: Inference speed. Groq delivers tokens at >500 T/s, essential for keeping LLM calls under 200ms.

### Scalability
- **Vector DB**: Qdrant is cloud-native and scales horizontally.
- **Stateless API**: The orchestrator is stateless; user context is passed in the request. This allows easy scaling behind a load balancer.
- **Async I/O**: Python `asyncio` ensures the server can handle many concurrent requests without blocking on I/O (DB or LLM calls).

## 6. Known Limitations
- **Context Window**: Chat history analysis is limited to the last few turns to maintain speed.
- **Cold Start**: New users with no history rely solely on the static profile until they interact.
- **Data Volume**: Current ingestion is for a demo dataset (~200 questions). Production would require batch ingestion pipelines.
