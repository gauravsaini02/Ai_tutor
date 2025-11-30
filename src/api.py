"""
AI Tutor API - FastAPI Interface

Exposes the Tutor Orchestrator via a REST API.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uvicorn

from .orchestrator import TutorOrchestrator
from .logger import setup_logger

logger = setup_logger(__name__)

from contextlib import asynccontextmanager

# Initialize Orchestrator (Global singleton to keep model in memory)
orchestrator = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    global orchestrator
    logger.info("Initializing Tutor Orchestrator...")
    orchestrator = TutorOrchestrator()
    logger.info("✅ Tutor Orchestrator ready!")
    yield
    # Clean up resources if needed
    logger.info("Shutting down Tutor Orchestrator...")

# Initialize FastAPI app
app = FastAPI(
    title="AI Tutor API",
    description="Adaptive RAG-based tutoring system API",
    version="1.0.0",
    lifespan=lifespan
)

# --- Request Models ---

class ChatMessageModel(BaseModel):
    role: str
    message: str

class UserProfileModel(BaseModel):
    grade: str
    exam_target: str
    subject: str
    expertise_level: float = Field(..., ge=1.0, le=5.0)
    weak_topics: List[str] = []
    strong_topics: List[str] = []

class RecentPerformanceModel(BaseModel):
    topic: str
    questions_attempted: int
    correct: int
    avg_time_seconds: int
    confidence_score: float = Field(..., ge=1.0, le=5.0)

class RecommendationRequest(BaseModel):
    user_profile: UserProfileModel
    chat_history: List[ChatMessageModel]
    recent_performance: Optional[RecentPerformanceModel] = None

# --- Endpoints ---

@app.get("/")
async def root():
    return {"status": "online", "message": "AI Tutor API is running"}

@app.post("/recommend")
async def get_recommendations(request: RecommendationRequest):
    """
    Get personalized question recommendations based on student profile and context.
    """
    if not orchestrator:
        raise HTTPException(status_code=503, detail="System initializing, please try again")
    
    try:
        # Convert Pydantic models to dicts for the orchestrator
        user_profile_dict = request.user_profile.dict()
        chat_history_dict = [msg.dict() for msg in request.chat_history]
        recent_performance_dict = request.recent_performance.dict() if request.recent_performance else None
        
        # Get recommendations
        result = await orchestrator.recommend(
            user_profile_dict,
            chat_history_dict,
            recent_performance_dict
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
