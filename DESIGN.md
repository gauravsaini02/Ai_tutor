# AI Tutor - System Design & Architecture

## 1. System Overview

The AI Tutor is an adaptive learning system that provides personalized question recommendations to students. The core component is the **Intelligent RAG (Retrieval-Augmented Generation) Pipeline**, which retrieves relevant questions based on student profile, chat history, and recent performance.

### Key Goals
- **Personalization**: Recommend questions that match the student's expertise and weak areas.
- **Adaptivity**: Adjust difficulty dynamically using Vygotsky's Zone of Proximal Development (ZPD).
- **Performance**: End-to-end latency **<500ms**.
- **Scalability**: Support concurrent users with efficient vector search.

---

## 2. Architecture

The system follows a modular architecture with three main stages:

```mermaid
graph TD
    User[Student] -->|Profile + Chat| Orchestrator[Tutor Orchestrator]
    Orchestrator -->|Context| Pipeline[Retrieval Pipeline]
    
    subgraph "Retrieval Pipeline"
        Pipeline -->|Query Embedding| Stage1[Stage 1: Candidate Generation]
        Stage1 -->|Vector Search| Qdrant[(Qdrant Vector DB)]
        Qdrant -->|50 Candidates| Stage2[Stage 2: Intelligent Ranking]
        Stage2 -->|Multi-Signal Score| Stage3[Stage 3: LLM Ranking (Optional)]
    end
    
    Stage3 -->|Top 10 Questions| Orchestrator
    Orchestrator -->|JSON Output| User
```

### Components

1.  **Tutor Orchestrator (`tutor_orchestrator.py`)**
    -   **Role**: Main entry point and controller.
    -   **Responsibilities**:
        -   Parses input (Profile, Chat, Performance).
        -   Invokes Retrieval Pipeline.
        -   Generates Tutor Context (Gaps, Study Sequence, Milestones).
        -   Formats final JSON output.

2.  **Retrieval Pipeline (`retrieval_pipeline.py`)**
    -   **Role**: Core logic for finding and ranking questions.
    -   **Stage 1: Candidate Generation**
        -   **Method**: Vector Similarity Search (Cosine Distance).
        -   **Model**: `all-MiniLM-L6-v2` (384 dimensions).
        -   **Filters**: Subject, Exam Type.
        -   **Output**: Top 50 candidates.
    -   **Stage 2: Intelligent Ranking**
        -   **Scoring**: Weighted sum of 4 signals:
            -   **Relevance (30%)**: Vector similarity + Chat topic match.
            -   **Difficulty (35%)**: ZPD calibration (Target = Expertise + 0.7).
            -   **Personalization (20%)**: Boost weak topics, penalize strong ones.
            -   **Diversity (15%)**: Penalize repeated subtopics, boost recent years.
    -   **Stage 3: LLM Ranking (Optional)**
        -   **Method**: GPT-4o-mini re-ranking for deep reasoning.
        -   **Status**: Disabled by default for latency (<500ms).

3.  **Data Ingestion (`ingest_data.py`)**
    -   **Role**: Pre-processing and indexing.
    -   **Process**:
        -   Load questions from JSON.
        -   Generate embeddings using `all-MiniLM-L6-v2`.
        -   Create Qdrant points with rich metadata.
        -   Upsert to Qdrant collection `questions`.

4.  **Vector Database (Qdrant)**
    -   **Role**: Storage and fast retrieval.
    -   **Config**:
        -   Collection: `questions`
        -   Vector Size: 384
        -   Distance: Cosine
        -   Indexes: subject, topic, difficulty, exam_type, year.

---

## 3. Latency Analysis & Optimization

The primary technical challenge was meeting the **<500ms** latency target.

### Evolution of Performance

| Iteration | Embedding Model | Avg Latency | Bottleneck | Status |
|-----------|----------------|-------------|------------|--------|
| **v1** | OpenAI `text-embedding-3-large` | ~1650ms | API Network Call (~1.5s) | ❌ Failed |
| **v2** | OpenAI `text-embedding-3-small` | ~1320ms | API Network Call (~1.2s) | ❌ Failed |
| **v3** | Local `all-MiniLM-L6-v2` | **~370ms** | Model Inference (~300ms) | ✅ **Passed** |

### Final Performance Breakdown (Warm Cache)

-   **Embedding Generation**: ~300-350ms (Local CPU)
-   **Vector Search**: ~2-5ms (Qdrant HNSW)
-   **Ranking Logic**: ~1-2ms (Python)
-   **Total**: **~310-420ms**

### Optimization Strategy
We moved from API-based embeddings to **local inference** using `sentence-transformers`. This eliminated network latency and per-query costs, allowing us to consistently hit the sub-500ms target on standard hardware.

---

## 4. Adaptive Difficulty Algorithm

The system implements **Vygotsky's Zone of Proximal Development (ZPD)** to ensure questions are "challenging but achievable."

```python
# Target Difficulty Calculation
target_difficulty = student_expertise + 0.7  # Slight stretch

# Adjustments
if confidence < 2.0:
    target_difficulty -= 0.3  # Build confidence
if success_rate < 0.4:
    target_difficulty -= 0.4  # Ease up
if success_rate > 0.9:
    target_difficulty += 0.3  # Push harder
```

This ensures a personalized learning curve for every student.

---

## 5. Trade-offs and Limitations

1.  **Local vs. API Embeddings**
    -   **Trade-off**: Switched to `all-MiniLM-L6-v2` (384d) from OpenAI (3072d).
    -   **Impact**: Massive speedup (4x) and cost reduction ($0). Slight theoretical loss in semantic nuance, but testing showed relevance remained high for this domain.

2.  **Cold Start Latency**
    -   **Issue**: First run takes ~1.5s to load model into memory.
    -   **Mitigation**: In production, the model stays loaded in RAM (warm cache), ensuring consistent <400ms performance.

3.  **Dataset Scale**
    -   **Current**: 136 mock questions.
    -   **Scalability**: Qdrant handles millions of vectors easily. Latency impact of scale is minimal due to HNSW indexing.

---

## 6. Future Improvements

1.  **GPU Acceleration**: Deploying on a GPU server would drop embedding latency to **<50ms**.
2.  **ONNX Runtime**: Optimizing the local model with ONNX could improve CPU performance by 2-3x.
3.  **Hybrid Search**: Combining dense vectors with sparse vectors (BM25) for better keyword matching.
