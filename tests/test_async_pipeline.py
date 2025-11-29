import asyncio
import sys
import os
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.orchestrator import TutorOrchestrator
from src.config import Config

async def test_pipeline():
    print("Initializing Orchestrator...")
    config = Config()
    
    # Override for testing if needed, but defaults should work with Docker
    print(f"Qdrant URL: {config.QDRANT_URL}")
    print(f"Embedding URL: {config.EMBEDDING_SERVICE_URL}")
    
    try:
        orchestrator = TutorOrchestrator(config)
        
        # Test Data
        user_profile = {
            "grade": "11",
            "exam_target": "neet",
            "subject": "Biology",
            "expertise_level": 2.0,
            "weak_topics": ["photosynthesis"],
            "strong_topics": []
        }
        
        chat_history = [
            {"role": "student", "message": "I need help with photosynthesis."}
        ]
        
        print("\nRunning Async Recommendation...")
        start = time.time()
        result = await orchestrator.recommend(user_profile, chat_history)
        duration = (time.time() - start) * 1000
        
        print(f"\n✅ Success! Pipeline finished in {duration:.2f}ms")
        print(f"Retrieved {len(result['recommended_questions'])} questions.")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Ensure Docker containers are running: docker-compose up -d")
        # raise e

if __name__ == "__main__":
    asyncio.run(test_pipeline())
