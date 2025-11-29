"""
Retrieval Pipeline for Adaptive Tutor System

Implements a 3-stage RAG pipeline:
1. Candidate Generation: Fast vector search + metadata filtering
2. Intelligent Ranking: Multi-signal scoring (relevance, difficulty, personalization, diversity)
3. Optional LLM Ranking: Deep reasoning about question fit

Target latency: <500ms end-to-end
"""

import os
import time
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
from dataclasses import dataclass

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
from openai import OpenAI  # Still needed for optional LLM ranking

load_dotenv()


@dataclass
class UserProfile:
    """Student profile information."""
    grade: str
    exam_target: str
    subject: str
    expertise_level: float  # 1-5 scale
    weak_topics: List[str]
    strong_topics: List[str]


@dataclass
class RecentPerformance:
    """Recent performance metrics."""
    topic: str
    questions_attempted: int
    correct: int
    avg_time_seconds: float
    confidence_score: float  # 1-5 scale


@dataclass
class ChatMessage:
    """Chat history message."""
    role: str  # 'student' or 'tutor'
    message: str


@dataclass
class RetrievalResult:
    """Single retrieved question with scores."""
    question_id: str
    topic: str
    subtopic: str
    difficulty_score: float
    question_text: str
    options: List[str]
    source: str
    explanation: str
    time_estimate_seconds: int
    relevance_score: float
    reasoning: str
    # Detailed scoring breakdown
    vector_similarity: float
    difficulty_match_score: float
    personalization_score: float
    diversity_score: float
    final_score: float


class Config:
    """Pipeline configuration."""
    
    def __init__(self):
        self.QDRANT_URL = os.getenv("QDRANT_URL", "")
        self.QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
        self.COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "questions")
        self.EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Local sentence-transformer model
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")  # Only for optional LLM ranking
        
        # Retrieval parameters
        self.CANDIDATE_LIMIT = 50  # Stage 1: candidate generation
        self.FINAL_RESULTS_LIMIT = 10  # Final top-K to return
        self.RECENT_YEAR_THRESHOLD = 2  # Prioritize questions from last N years
        
        # Ranking weights (must sum to 1.0)
        self.RELEVANCE_WEIGHT = 0.30
        self.DIFFICULTY_WEIGHT = 0.35
        self.PERSONALIZATION_WEIGHT = 0.20
        self.DIVERSITY_WEIGHT = 0.15
        
        # LLM ranking
        self.ENABLE_LLM_RANKING = False  # Toggle for optional Stage 3
        self.LLM_MODEL = "gpt-4o-mini"  # Fast model for ranking
        
        # Latency constraints (ms)
        self.MAX_RETRIEVAL_LATENCY_MS = 200
        self.MAX_RANKING_LATENCY_MS = 200
        self.MAX_LLM_RANKING_LATENCY_MS = 100
        self.MAX_TOTAL_LATENCY_MS = 500


class ChatHistoryParser:
    """Extract intent and context from chat history."""
    
    @staticmethod
    def extract_topics(chat_history: List[ChatMessage]) -> List[str]:
        """Extract mentioned topics from chat."""
        topics = []
        for msg in chat_history:
            if msg.role == "student":
                text = msg.message.lower()
                # Simple keyword extraction (can be enhanced with NLP)
                # Look for topic keywords
                keywords = ["shm", "oscillation", "vector", "kinematics", "thermodynamics",
                           "mechanics", "electromagnetism", "genetics", "mendel", "inheritance",
                           "photosynthesis", "respiration", "circular motion", "damped"]
                for keyword in keywords:
                    if keyword in text:
                        topics.append(keyword)
        return list(set(topics))
    
    @staticmethod
    def detect_performance_signals(chat_history: List[ChatMessage]) -> Dict[str, str]:
        """Detect performance indicators."""
        signals = {}
        for msg in chat_history:
            if msg.role == "student":
                text = msg.message.lower()
                if any(word in text for word in ["struggled", "wrong", "don't understand", "confused"]):
                    signals["performance"] = "struggling"
                elif any(word in text for word in ["boring", "easy", "too simple"]):
                    signals["performance"] = "bored"
                elif any(word in text for word in ["harder", "challenge", "push me"]):
                    signals["performance"] = "ready_for_more"
        return signals
    
    @staticmethod
    def build_context_query(user_profile: UserProfile, chat_history: List[ChatMessage], 
                           recent_performance: Optional[RecentPerformance]) -> str:
        """Build query string combining all context."""
        query_parts = []
        
        # Add weak topics (high priority)
        if user_profile.weak_topics:
            query_parts.extend([f"{topic} practice problems" for topic in user_profile.weak_topics])
        
        # Add chat-mentioned topics
        chat_topics = ChatHistoryParser.extract_topics(chat_history)
        query_parts.extend(chat_topics)
        
        # Add recent performance topic
        if recent_performance:
            query_parts.append(f"{recent_performance.topic} questions")
        
        # Add subject context
        query_parts.append(f"{user_profile.subject} {user_profile.exam_target}")
        
        return " ".join(query_parts)


class DifficultyCalibrator:
    """Adaptive difficulty calibration based on Vygotsky's ZPD."""
    
    @staticmethod
    def compute_target_difficulty(user_profile: UserProfile, 
                                  recent_performance: Optional[RecentPerformance]) -> float:
        """
        Compute optimal difficulty for learning.
        Vygotsky's Zone of Proximal Development: expertise + 0.5 to expertise + 1.0
        """
        target = user_profile.expertise_level + 0.7  # Slight challenge
        
        if recent_performance:
            # Adjust based on confidence
            if recent_performance.confidence_score < 2.0:
                target -= 0.3  # Build confidence first
            elif recent_performance.confidence_score > 4.5:
                target += 0.5  # Push harder for confident students
            
            # Adjust based on success rate
            success_rate = recent_performance.correct / max(recent_performance.questions_attempted, 1)
            if success_rate < 0.4:
                target -= 0.4  # They need easier questions
            elif success_rate > 0.9:
                target += 0.3  # They can handle harder
        
        # Clamp to valid range [1, 5]
        return max(1.0, min(5.0, target))
    
    @staticmethod
    def score_difficulty_match(question_difficulty: float, target_difficulty: float) -> float:
        """
        Score how well a question's difficulty matches the target.
        Returns score in [0, 1] where 1 is perfect match.
        """
        difficulty_gap = abs(question_difficulty - target_difficulty)
        # Use exponential decay: perfect match = 1.0, gap of 1.0 = ~0.37, gap of 2.0 = ~0.14
        return 1.0 / (1.0 + difficulty_gap ** 2)


class RetrievalPipeline:
    """Main retrieval pipeline orchestrator."""
    
    def __init__(self, config: Config):
        self.config = config
        self.qdrant_client = QdrantClient(
            url=config.QDRANT_URL,
            api_key=config.QDRANT_API_KEY
        )
        print(f"Loading local embedding model: {config.EMBEDDING_MODEL}...")
        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
        print(f"✅ Embedding model loaded!")
        
        # Only initialize OpenAI if LLM ranking is enabled
        if config.ENABLE_LLM_RANKING and config.OPENAI_API_KEY:
            self.openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
        else:
            self.openai_client = None
            
        self.chat_parser = ChatHistoryParser()
        self.difficulty_calibrator = DifficultyCalibrator()
    
    def _get_embedding(self, text: str) -> List[float]:
        """Generate embedding for query text using local model."""
        embedding = self.embedding_model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    def _stage1_candidate_generation(
        self,
        user_profile: UserProfile,
        chat_history: List[ChatMessage],
        recent_performance: Optional[RecentPerformance]
    ) -> Tuple[List[Dict[str, Any]], float]:
        """
        Stage 1: Fast candidate generation using vector search + metadata filters.
        Returns: (candidates, latency_ms)
        """
        start_time = time.time()
        
        # Build query from context
        query_text = self.chat_parser.build_context_query(
            user_profile, chat_history, recent_performance
        )
        
        # Generate query embedding
        query_embedding = self._get_embedding(query_text)
        
        # Build metadata filters
        must_conditions = []
        
        # Filter by subject
        if user_profile.subject:
            must_conditions.append(
                models.FieldCondition(
                    key="subject",
                    match=models.MatchValue(value=user_profile.subject.capitalize())
                )
            )
        
        # Filter by exam type
        if user_profile.exam_target:
            must_conditions.append(
                models.FieldCondition(
                    key="exam_type",
                    match=models.MatchValue(value=user_profile.exam_target.lower())
                )
            )
        
        # Search with filters using query_points
        search_results = self.qdrant_client.query_points(
            collection_name=self.config.COLLECTION_NAME,
            query=query_embedding,
            query_filter=models.Filter(must=must_conditions) if must_conditions else None,
            limit=self.config.CANDIDATE_LIMIT,
            with_payload=True,
            with_vectors=False
        )
        
        # Convert to dict format
        candidates = []
        for result in search_results.points:
            candidate = dict(result.payload)
            candidate["vector_similarity"] = result.score
            candidates.append(candidate)
        
        latency_ms = (time.time() - start_time) * 1000
        return candidates, latency_ms
    
    def _stage2_intelligent_ranking(
        self,
        candidates: List[Dict[str, Any]],
        user_profile: UserProfile,
        chat_history: List[ChatMessage],
        recent_performance: Optional[RecentPerformance]
    ) -> Tuple[List[RetrievalResult], float]:
        """
        Stage 2: Multi-signal ranking with relevance, difficulty, personalization, diversity.
        Returns: (ranked_results, latency_ms)
        """
        start_time = time.time()
        
        # Extract context
        chat_topics = self.chat_parser.extract_topics(chat_history)
        performance_signals = self.chat_parser.detect_performance_signals(chat_history)
        target_difficulty = self.difficulty_calibrator.compute_target_difficulty(
            user_profile, recent_performance
        )
        
        # Track subtopic distribution for diversity
        subtopic_counts = Counter()
        
        scored_results = []
        
        for candidate in candidates:
            # 1. Relevance Score (30%)
            relevance_score = candidate.get("vector_similarity", 0.0)
            
            # Boost if topic mentioned in chat
            topic = candidate.get("topic", "").lower()
            subtopic = candidate.get("sub_topic", "").lower()
            if any(chat_topic in topic or chat_topic in subtopic for chat_topic in chat_topics):
                relevance_score *= 1.2  # 20% boost
            
            relevance_score = min(1.0, relevance_score)  # Clamp to [0, 1]
            
            # 2. Difficulty Calibration Score (35%)
            question_difficulty = candidate.get("difficulty", 3.0)
            difficulty_score = self.difficulty_calibrator.score_difficulty_match(
                question_difficulty, target_difficulty
            )
            
            # Adjust based on performance signals
            if performance_signals.get("performance") == "struggling":
                # Penalize hard questions
                if question_difficulty > user_profile.expertise_level:
                    difficulty_score *= 0.7
            elif performance_signals.get("performance") == "ready_for_more":
                # Reward harder questions
                if question_difficulty > user_profile.expertise_level:
                    difficulty_score *= 1.3
            
            # 3. Personalization Score (20%)
            personalization_score = 0.5  # Base score
            
            # Boost if in weak topics
            if any(weak in topic or weak in subtopic for weak in user_profile.weak_topics):
                personalization_score += 0.3
            
            # Slight penalty if in strong topics (focus on gaps)
            if any(strong in topic or strong in subtopic for strong in user_profile.strong_topics):
                personalization_score -= 0.1
            
            # Boost if related to recent poor performance
            if recent_performance and recent_performance.correct / max(recent_performance.questions_attempted, 1) < 0.5:
                if recent_performance.topic.lower() in topic or recent_performance.topic.lower() in subtopic:
                    personalization_score += 0.2
            
            personalization_score = max(0.0, min(1.0, personalization_score))
            
            # 4. Diversity Score (15%)
            # Penalize repeated subtopics
            subtopic_key = candidate.get("sub_topic", "unknown")
            repeat_count = subtopic_counts[subtopic_key]
            diversity_score = 1.0 / (1.0 + repeat_count * 0.5)  # Decay with repetition
            subtopic_counts[subtopic_key] += 1
            
            # Temporal boost: prioritize recent exams
            year = candidate.get("year", 2000)
            current_year = 2025
            if year >= current_year - self.config.RECENT_YEAR_THRESHOLD:
                diversity_score *= 1.1
            
            diversity_score = min(1.0, diversity_score)
            
            # Compute final weighted score
            final_score = (
                self.config.RELEVANCE_WEIGHT * relevance_score +
                self.config.DIFFICULTY_WEIGHT * difficulty_score +
                self.config.PERSONALIZATION_WEIGHT * personalization_score +
                self.config.DIVERSITY_WEIGHT * diversity_score
            )
            
            # Generate reasoning
            reasoning = self._generate_reasoning(
                candidate, target_difficulty, user_profile, 
                relevance_score, difficulty_score, personalization_score, diversity_score
            )
            
            # Build result object
            result = RetrievalResult(
                question_id=candidate.get("question_id", ""),
                topic=candidate.get("topic", ""),
                subtopic=candidate.get("sub_topic", ""),
                difficulty_score=candidate.get("difficulty", 3.0),
                question_text=candidate.get("question_text", ""),
                options=candidate.get("options", []),
                source=f"{candidate.get('exam_type', '').upper()} {candidate.get('year', '')}",
                explanation=candidate.get("explanation", ""),
                time_estimate_seconds=candidate.get("time_estimate", 120),
                relevance_score=final_score,
                reasoning=reasoning,
                vector_similarity=candidate.get("vector_similarity", 0.0),
                difficulty_match_score=difficulty_score,
                personalization_score=personalization_score,
                diversity_score=diversity_score,
                final_score=final_score
            )
            
            scored_results.append(result)
        
        # Sort by final score descending
        ranked_results = sorted(scored_results, key=lambda x: x.final_score, reverse=True)
        
        # Take top K
        top_results = ranked_results[:self.config.FINAL_RESULTS_LIMIT]
        
        latency_ms = (time.time() - start_time) * 1000
        return top_results, latency_ms
    
    def _generate_reasoning(
        self, 
        candidate: Dict[str, Any], 
        target_difficulty: float,
        user_profile: UserProfile,
        relevance_score: float,
        difficulty_score: float,
        personalization_score: float,
        diversity_score: float
    ) -> str:
        """Generate human-readable reasoning for why this question was selected."""
        reasons = []
        
        topic = candidate.get("topic", "")
        difficulty = candidate.get("difficulty", 3.0)
        
        # Difficulty reasoning
        if abs(difficulty - target_difficulty) < 0.5:
            reasons.append(f"Optimally calibrated difficulty ({difficulty}/5) matches your current level")
        elif difficulty < user_profile.expertise_level:
            reasons.append(f"Foundation-building question to strengthen basics")
        else:
            reasons.append(f"Challenging question ({difficulty}/5) to push your growth")
        
        # Weak topics
        if any(weak in topic.lower() or weak in candidate.get("sub_topic", "").lower() 
               for weak in user_profile.weak_topics):
            reasons.append(f"Addresses your weak area: {topic}")
        
        # High relevance
        if relevance_score > 0.7:
            reasons.append(f"Highly relevant to your study context")
        
        # Recent exam
        year = candidate.get("year", 2000)
        if year >= 2023:
            reasons.append(f"Recent {candidate.get('exam_type', '').upper()} question ({year})")
        
        return "; ".join(reasons) if reasons else "Good practice question for your level"
    
    def _stage3_llm_ranking(
        self,
        results: List[RetrievalResult],
        user_profile: UserProfile,
        chat_history: List[ChatMessage]
    ) -> Tuple[List[RetrievalResult], float]:
        """
        Stage 3 (Optional): LLM-based ranking for deeper reasoning.
        Returns: (re_ranked_results, latency_ms)
        """
        if not self.config.ENABLE_LLM_RANKING:
            return results, 0.0
        
        start_time = time.time()
        
        # Build prompt for LLM
        profile_summary = (
            f"Student Profile: Grade {user_profile.grade}, "
            f"targeting {user_profile.exam_target}, "
            f"expertise level {user_profile.expertise_level}/5, "
            f"weak topics: {', '.join(user_profile.weak_topics)}"
        )
        
        recent_chat = chat_history[-3:] if len(chat_history) > 3 else chat_history
        chat_summary = "\n".join([f"{msg.role}: {msg.message}" for msg in recent_chat])
        
        # Score each question (in practice, batch this for efficiency)
        for result in results:
            prompt = f"""Given this student profile and recent conversation, score how well this question fits their learning needs on a scale of 0-100.

{profile_summary}

Recent conversation:
{chat_summary}

Question:
Topic: {result.topic} - {result.subtopic}
Difficulty: {result.difficulty_score}/5
Text: {result.question_text[:200]}...

Provide only a numeric score (0-100) and brief one-line reasoning."""

            try:
                response = self.openai_client.chat.completions.create(
                    model=self.config.LLM_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=50,
                    temperature=0.3
                )
                
                llm_output = response.choices[0].message.content.strip()
                # Parse score (simple extraction)
                llm_score = float(llm_output.split()[0]) / 100.0
                
                # Blend with existing score (70% existing, 30% LLM)
                result.final_score = 0.7 * result.final_score + 0.3 * llm_score
                
            except Exception as e:
                print(f"LLM ranking error for {result.question_id}: {e}")
                # Keep original score on error
        
        # Re-sort by updated scores
        re_ranked = sorted(results, key=lambda x: x.final_score, reverse=True)
        
        latency_ms = (time.time() - start_time) * 1000
        return re_ranked, latency_ms
    
    def retrieve(
        self,
        user_profile: UserProfile,
        chat_history: List[ChatMessage],
        recent_performance: Optional[RecentPerformance] = None
    ) -> Tuple[List[RetrievalResult], Dict[str, float]]:
        """
        Main retrieval method. Executes full 3-stage pipeline.
        
        Returns:
            (results, latency_metadata)
        """
        total_start = time.time()
        
        # Stage 1: Candidate Generation
        candidates, retrieval_latency = self._stage1_candidate_generation(
            user_profile, chat_history, recent_performance
        )
        print(f"Stage 1: Retrieved {len(candidates)} candidates in {retrieval_latency:.2f}ms")
        
        # Stage 2: Intelligent Ranking
        ranked_results, ranking_latency = self._stage2_intelligent_ranking(
            candidates, user_profile, chat_history, recent_performance
        )
        print(f"Stage 2: Ranked to top {len(ranked_results)} in {ranking_latency:.2f}ms")
        
        # Stage 3: Optional LLM Ranking
        final_results, llm_latency = self._stage3_llm_ranking(
            ranked_results, user_profile, chat_history
        )
        if self.config.ENABLE_LLM_RANKING:
            print(f"Stage 3: LLM re-ranking in {llm_latency:.2f}ms")
        
        total_latency = (time.time() - total_start) * 1000
        
        latency_metadata = {
            "retrieval_latency_ms": round(retrieval_latency, 2),
            "ranking_latency_ms": round(ranking_latency, 2),
            "llm_ranking_latency_ms": round(llm_latency, 2),
            "total_latency_ms": round(total_latency, 2)
        }
        
        print(f"\n✅ Total pipeline latency: {total_latency:.2f}ms")
        
        return final_results, latency_metadata


# Example usage
if __name__ == "__main__":
    # Validate configuration
    config = Config()
    
    # Check if API keys are set
    if not config.OPENAI_API_KEY or not config.QDRANT_API_KEY:
        print("❌ Error: Missing API keys in .env file")
        print("   Please ensure OPENAI_API_KEY and QDRANT_API_KEY are set")
        exit(1)
    
    try:
        # Initialize
        pipeline = RetrievalPipeline(config)
        
        # Test case: Struggling beginner
        user_profile = UserProfile(
            grade="11",
            exam_target="neet",
            subject="Biology",
            expertise_level=2.0,
            weak_topics=["photosynthesis", "respiration"],
            strong_topics=[]
        )
        
        chat_history = [
            ChatMessage(role="student", message="I don't understand photosynthesis well. Can we practice?"),
            ChatMessage(role="tutor", message="Sure! Let me find some questions for you.")
        ]
        
        recent_performance = RecentPerformance(
            topic="photosynthesis",
            questions_attempted=5,
            correct=2,
            avg_time_seconds=180,
            confidence_score=1.5
        )
        
        print("\n" + "="*80)
        print("RETRIEVAL PIPELINE TEST - Struggling Beginner in Photosynthesis")
        print("="*80)
        print(f"User Profile: Grade {user_profile.grade}, {user_profile.exam_target.upper()}, {user_profile.subject}")
        print(f"Expertise Level: {user_profile.expertise_level}/5")
        print(f"Weak Topics: {', '.join(user_profile.weak_topics)}")
        print(f"Recent Performance: {recent_performance.correct}/{recent_performance.questions_attempted} correct, confidence {recent_performance.confidence_score}/5")
        print("="*80)
        
        # Retrieve
        results, latency = pipeline.retrieve(user_profile, chat_history, recent_performance)
        
        # Display results
        print("\n" + "="*80)
        print("TOP RECOMMENDED QUESTIONS")
        print("="*80)
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result.question_id}")
            print(f"   Topic: {result.topic} - {result.subtopic}")
            print(f"   Difficulty: {result.difficulty_score}/5")
            print(f"   Score: {result.final_score:.3f}")
            print(f"   Component Scores:")
            print(f"      - Vector Similarity: {result.vector_similarity:.3f}")
            print(f"      - Difficulty Match: {result.difficulty_match_score:.3f}")
            print(f"      - Personalization: {result.personalization_score:.3f}")
            print(f"      - Diversity: {result.diversity_score:.3f}")
            print(f"   Reasoning: {result.reasoning}")
            print(f"   Source: {result.source}")
            print(f"   Question: {result.question_text[:150]}...")
        
        print("\n" + "="*80)
        print("LATENCY BREAKDOWN")
        print("="*80)
        for key, value in latency.items():
            print(f"  {key}: {value}ms")
        
        # Check latency constraint
        if latency["total_latency_ms"] > config.MAX_TOTAL_LATENCY_MS:
            print(f"\n⚠️  WARNING: Total latency ({latency['total_latency_ms']:.2f}ms) exceeds target ({config.MAX_TOTAL_LATENCY_MS}ms)")
        else:
            print(f"\n✅ Latency constraint satisfied: {latency['total_latency_ms']:.2f}ms < {config.MAX_TOTAL_LATENCY_MS}ms")
        
    except Exception as e:
        print(f"\n❌ Error running pipeline: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
