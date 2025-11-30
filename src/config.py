"""
Shared Configuration for AI Tutor Application.
"""
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Centralized configuration settings."""
    
    def __init__(self):
        # Qdrant Configuration
        # Qdrant Configuration
        # self.QDRANT_URL: str = os.getenv("QDRANT_URL", "")
        # self.QDRANT_URL: str = ":memory:"  # Volatile
        self.QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
        self.QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")
        self.COLLECTION_NAME: str = os.getenv("QDRANT_COLLECTION_NAME", "questions")
        
        # Embedding Configuration
        self.EMBEDDING_SERVICE_URL: str = os.getenv("EMBEDDING_SERVICE_URL", "http://localhost:7997")
        self.EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"  # Local sentence-transformer model
        self.VECTOR_SIZE: int = 384  # Dimension for all-MiniLM-L6-v2
        
        # Redis Configuration
        self.REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
        self.REDIS_PORT: int = int(os.getenv("REDIS_PORT", 6379))
        self.CACHE_TTL: int = 3600  # 1 hour
        
        # Optional LLM Configuration
        self.ENABLE_LLM_RANKING: bool = False
        
        # Groq Configuration (Fast Signal Detection)
        self.GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
        self.GROQ_MODEL: str = "llama-3.1-8b-instant"
        
        # Retrieval Parameters
        self.CANDIDATE_LIMIT: int = 50
        self.FINAL_RESULTS_LIMIT: int = 10
        self.RECENT_YEAR_THRESHOLD: int = 2
        
        # Ranking Weights (Must sum to 1.0)
        self.RELEVANCE_WEIGHT: float = 0.30
        self.DIFFICULTY_WEIGHT: float = 0.35
        self.PERSONALIZATION_WEIGHT: float = 0.20
        self.DIVERSITY_WEIGHT: float = 0.15
        
        # Latency Constraints (ms)
        self.MAX_RETRIEVAL_LATENCY_MS: int = 200
        self.MAX_RANKING_LATENCY_MS: int = 200
        self.MAX_LLM_RANKING_LATENCY_MS: int = 100
        self.MAX_TOTAL_LATENCY_MS: int = 500

    def validate(self):
        """Validate critical configuration."""
        if not self.QDRANT_URL:
            raise ValueError("QDRANT_URL is not set in environment variables.")
        if not self.QDRANT_URL:
            raise ValueError("QDRANT_URL is not set.")
