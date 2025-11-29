# AI Tutor - Intelligent RAG Pipeline

An adaptive tutoring system that uses Retrieval-Augmented Generation (RAG) to recommend personalized practice questions based on student profile, chat history, and performance.

## 🚀 Features

- **Personalized Recommendations**: Tailors questions to student's expertise and weak topics.
- **Adaptive Difficulty**: Uses Vygotsky's ZPD to find the "sweet spot" of difficulty.
- **Intelligent RAG Pipeline**:
    - **Stage 1**: Fast candidate generation using Qdrant vector search.
    - **Stage 2**: Multi-signal ranking (Relevance, Difficulty, Personalization, Diversity).
    - **Stage 3**: Optional LLM-based re-ranking.
- **High Performance**: End-to-end latency **<500ms** using local embeddings.
- **Tutor Orchestrator**: Generates study plans, identifies gaps, and estimates session duration.

## 🛠️ Setup

### Prerequisites
- Python 3.8+
- Docker (for Qdrant)

### 1. Install Dependencies
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment
Create a `.env` file:
```env
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your_api_key_here
QDRANT_COLLECTION_NAME=questions
OPENAI_API_KEY=your_openai_key_here  # Optional (only for LLM ranking)
```

### 3. Start Qdrant
```bash
docker run -p 6333:6333 qdrant/qdrant
```

### 4. Ingest Data
Load the mock dataset (136 questions) into Qdrant:
```bash
python -m src.ingestion
```

## 🏃‍♂️ Usage

### Run the Demo
See the Tutor Orchestrator in action:
```bash
python -m src.orchestrator
```

### Run the API
Start the FastAPI server:
```bash
uvicorn src.api:app --reload
```
Access docs at `http://localhost:8000/docs`.

### Run Tests
Verify the pipeline performance and logic:
```bash
export PYTHONPATH=$PYTHONPATH:.
python tests/test_retrieval.py
```

## 📂 Project Structure

```
src/
  config.py          # Shared configuration
  api.py             # FastAPI application
  ingestion.py       # Data ingestion & indexing
  retrieval.py       # Core RAG pipeline logic
  orchestrator.py    # Main application logic

scripts/
  reset_db.py        # Reset Qdrant collection
  view_db.py         # Inspect database content

tests/
  test_retrieval.py  # Comprehensive test suite

docs/
  DESIGN.md          # Architecture & latency analysis
  FINAL_PERFORMANCE.md # Benchmark results
```

## ⚡ Performance

The system is optimized for low latency using local embeddings:

| Component | Latency (Warm) |
|-----------|----------------|
| Embedding Generation | ~300ms |
| Vector Search | ~5ms |
| Ranking Logic | ~2ms |
| **Total** | **~350-400ms** |

*Target: <500ms (Achieved ✅)*
