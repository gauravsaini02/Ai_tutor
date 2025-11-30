"""
Retrieval Pipeline for Adaptive Tutor System

Implements a 3-stage RAG pipeline (Async):
1. Candidate Generation: Fast vector search + metadata filtering
2. Intelligent Ranking: Multi-signal scoring (relevance, difficulty, personalization, diversity)
3. Optional LLM Ranking: Deep reasoning about question fit

Target latency: <500ms end-to-end
"""

import os
import time
import asyncio
import json
import re
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
from dataclasses import dataclass, asdict

from dotenv import load_dotenv
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
from groq import AsyncGroq

from .config import Config
from .logger import setup_logger
from .cache import RedisCache

logger = setup_logger(__name__)

load_dotenv()


@dataclass
class UserProfile:
    """Student profile information."""
    grade: str
    exam_target: str
    subject: str
    expertise_level: float
    weak_topics: List[str]
    strong_topics: List[str]


@dataclass
class RecentPerformance:
    """Recent performance metrics."""
    topic: str
    questions_attempted: int
    correct: int
    avg_time_seconds: float
    confidence_score: float


@dataclass
class ChatMessage:
    """Chat history message."""
    role: str
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
    vector_similarity: float
    difficulty_match_score: float
    personalization_score: float
    diversity_score: float
    final_score: float


class ChatHistoryParser:
    """Extract intent and context from chat history."""
    
    TOPIC_KEYWORDS = [
        "shm", "oscillation", "vector", "kinematics", "thermodynamics",
        "mechanics", "electromagnetism", "genetics", "mendel", "inheritance",
        "photosynthesis", "respiration", "circular motion", "damped"
    ]
    
    @staticmethod
    def extract_topics(chat_history: List[ChatMessage]) -> List[str]:
        """Extract mentioned topics from chat."""
        topics = []
        for msg in chat_history:
            if msg.role == "student":
                text = msg.message.lower()
                for keyword in ChatHistoryParser.TOPIC_KEYWORDS:
                    if keyword in text:
                        topics.append(keyword)
        return list(set(topics))
    

    
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
        """Compute optimal difficulty for learning."""
        target = user_profile.expertise_level + 0.7  # Slight challenge
        
        if recent_performance:
            if recent_performance.confidence_score < 2.0:
                target -= 0.3
            elif recent_performance.confidence_score > 4.5:
                target += 0.5
        
        return target
    
    @staticmethod
    def score_difficulty_match(question_difficulty: float, target_difficulty: float) -> float:
        """Score how well a question's difficulty matches the target."""
        difficulty_gap = abs(question_difficulty - target_difficulty)
        return 1.0 / (1.0 + difficulty_gap)


class RetrievalPipeline:
    """Main retrieval pipeline orchestrator (Async)."""
    
    def __init__(self, config: Config):
        self.config = config
        
        # Async Qdrant Client
        self.qdrant_client = AsyncQdrantClient(
            url=config.QDRANT_URL,
            api_key=config.QDRANT_API_KEY
        )
        
        # Local Embedding Model (Loaded in memory)
        logger.info(f"Loading local embedding model: {self.config.EMBEDDING_MODEL}...")
        self.embedding_model = SentenceTransformer(self.config.EMBEDDING_MODEL)
        logger.info("✅ Embedding model loaded!")
        
        # Initialize Cache
        self.cache = RedisCache(config)



        # Initialize Groq Client
        self.groq_client = None
        if self.config.GROQ_API_KEY:
            self.groq_client = AsyncGroq(api_key=self.config.GROQ_API_KEY)
            
        self.chat_parser = ChatHistoryParser()
        self.difficulty_calibrator = DifficultyCalibrator()
        
    async def detect_performance_signals_groq(self, chat_history: List[ChatMessage]) -> Dict[str, Any]:
        """
        Detect performance signals using Groq LLM (Fast & Accurate).
        Returns: signal, confidence, gap, action.
        """
        if not self.groq_client or not chat_history:
            return {"performance": "neutral", "scores": {}}
            
        # Get last 3 messages for context
        recent_msgs = chat_history[-3:]
        conversation_text = "\n".join([f"{msg.role}: {msg.message}" for msg in recent_msgs])
        
        prompt = f"""Analyze the student's LATEST message (using recent conversation as context) for performance signals.
        
        Categories:
        - struggling: Confused, stuck, wrong answer, asking for help, "don't understand".
        - bored: Too easy, already knows, wants to skip, "boring".
        - ready_for_more: Wants challenge, "harder", "push me", "next level".
        - neutral: Just answering, asking for clarification, "next", "okay".
        
        Conversation:
        {conversation_text}
        
        Return JSON only: 
        {{
            "signal": "category_name", 
            "confidence": 0.0-1.0,
            "gap": "Specific concept/topic missing or causing confusion (e.g. 'Bohr radius formula', '5 kingdoms criteria'). Null if none.",
            "action": "Brief recommended action (e.g. 'Review basics', 'Increase difficulty')."
        }}
        """
        
        try:
            start = time.time()
            completion = await self.groq_client.chat.completions.create(
                model=self.config.GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=100
            )
            latency = (time.time() - start) * 1000
            
            result = json.loads(completion.choices[0].message.content)
            signal = result.get("signal", "neutral")
            
            # Map to internal format
            return {
                "performance": signal, 
                "gap": result.get("gap"),
                "action": result.get("action"),
                "scores": {signal: result.get("confidence", 1.0)},
                "latency_ms": latency
            }
            
        except Exception as e:
            logger.error(f"Groq Signal Error: {e}", exc_info=True)
            return {"performance": "neutral", "scores": {}}

    
    async def _get_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for query text.
        Runs in a separate thread to avoid blocking the async event loop.
        """
        loop = asyncio.get_running_loop()
        # Offload the CPU-bound encode method to a thread
        embedding = await loop.run_in_executor(
            None, 
            lambda: self.embedding_model.encode(text, convert_to_numpy=True)
        )
        return embedding.tolist()
    
    async def _stage1_candidate_generation(
        self,
        user_profile: UserProfile,
        chat_history: List[ChatMessage],
        recent_performance: Optional[RecentPerformance] = None
    ) -> Tuple[List[RetrievalResult], float]:
        """Stage 1: Candidate Generation (Vector Search) with Caching."""
        
        # 1. Generate Cache Key
        cache_key = self.cache.generate_key(
            stage="stage1",
            grade=user_profile.grade,
            subject=user_profile.subject,
            target=user_profile.exam_target,
            weak=sorted(user_profile.weak_topics),
            chat_len=len(chat_history),
            last_msg=chat_history[-1].message if chat_history else ""
        )
        
        # 2. Check Cache
        cached_data = await self.cache.get_lru(cache_key)
        if cached_data:
            logger.info("⚡ Cache Hit for Stage 1 Retrieval (LRU Promoted)")
            results = []
            for item in cached_data:
                # Reconstruct RetrievalResult objects
                results.append(RetrievalResult(**item))
            return results, 0.0  # 0 latency for cache hit

        # 3. Embed Query
        query_text = ChatHistoryParser.build_context_query(
            user_profile, chat_history, recent_performance
        )
        query_vector = self.embedding_model.encode(query_text).tolist()
        
        # 4. Filter Construction
        filter_conditions = [
            models.FieldCondition(key="subject", match=models.MatchValue(value=user_profile.subject)),
            models.FieldCondition(key="exam_type", match=models.MatchValue(value=user_profile.exam_target)),
        ]
        
        # Add topic filter if detected
        detected_topics = ChatHistoryParser.extract_topics(chat_history)
        if detected_topics:
            logger.info(f"Filtering by detected topics: {detected_topics}")
            filter_conditions.append(
                models.FieldCondition(key="topic", match=models.MatchAny(any=detected_topics))
            )
            
        search_filter = models.Filter(must=filter_conditions)
        
        # 5. Vector Search
        start_time = time.time()
        search_results = await self.qdrant_client.search(
            collection_name=self.config.COLLECTION_NAME,
            query_vector=query_vector,
            query_filter=search_filter,
            limit=self.config.CANDIDATE_LIMIT * 2  # Fetch more for diversity
        )
        
        # 6. Convert to RetrievalResult
        results = []
        for hit in search_results:
            payload = hit.payload
            results.append(RetrievalResult(
                question_id=payload.get("question_id"),
                topic=payload.get("topic"),
                subtopic=payload.get("subtopic"),
                difficulty_score=payload.get("difficulty_score") or 3.0,
                question_text=payload.get("question_text"),
                options=payload.get("options", []),
                source=payload.get("source"),
                explanation=payload.get("explanation"),
                time_estimate_seconds=payload.get("time_estimate_seconds") or 120,
                relevance_score=0.0, # Placeholder
                reasoning="",
                vector_similarity=hit.score,
                difficulty_match_score=0.0,
                personalization_score=0.0,
                diversity_score=0.0,
                final_score=0.0
            ))
            
        latency_ms = (time.time() - start_time) * 1000
        
        # 7. Cache Results (Last 10 queries)
        if results:
            # Convert to dicts for JSON serialization
            results_dicts = [asdict(res) for res in results]
            await self.cache.set_lru(cache_key, results_dicts, limit=10)
            
        return results, latency_ms
    
    def _calculate_candidate_score(
        self,
        candidate: RetrievalResult,
        user_profile: UserProfile,
        chat_topics: List[str],
        target_difficulty: float,
        performance_signals: Dict[str, Any],
        recent_performance: Optional[RecentPerformance],
        subtopic_counts: Counter
    ) -> RetrievalResult:
        """Calculate scores for a single candidate."""
        try:
            # 1. Relevance Score
            relevance_score = candidate.vector_similarity
            if relevance_score is None: relevance_score = 0.0
            
            topic = (candidate.topic or "").lower()
            subtopic = (candidate.subtopic or "").lower()
            question_text = (candidate.question_text or "").lower()
        
            # Boost based on chat topics
            if any(chat_topic in topic or chat_topic in subtopic for chat_topic in chat_topics):
                relevance_score *= 1.2
                
            # Boost based on specific GAP
            gap = performance_signals.get("gap")
            if gap and isinstance(gap, str):
                gap_lower = gap.lower()
                if gap_lower in topic or gap_lower in subtopic or gap_lower in question_text:
                    relevance_score *= 1.5
                    
            relevance_score = min(1.0, relevance_score)
            
            # 2. Difficulty Score
            question_difficulty = candidate.difficulty_score
            difficulty_score = self.difficulty_calibrator.score_difficulty_match(
                question_difficulty, target_difficulty
            )
            
            signal = performance_signals.get("performance")
            if signal == "struggling":
                if question_difficulty > user_profile.expertise_level:
                    difficulty_score *= 0.7
            elif signal == "ready_for_more":
                if question_difficulty > user_profile.expertise_level:
                    difficulty_score *= 1.3
            
            # 3. Personalization Score
            personalization_score = 0.5
            if any(weak in topic or weak in subtopic for weak in user_profile.weak_topics):
                personalization_score += 0.3
            if any(strong in topic or strong in subtopic for strong in user_profile.strong_topics):
                personalization_score -= 0.1
            
            if recent_performance and recent_performance.correct / max(recent_performance.questions_attempted, 1) < 0.5:
                if recent_performance.topic.lower() in topic or recent_performance.topic.lower() in subtopic:
                    personalization_score += 0.2
                    
            personalization_score = max(0.0, min(1.0, personalization_score))
            
            # 4. Diversity Score
            subtopic_key = candidate.subtopic
            repeat_count = subtopic_counts[subtopic_key]
            diversity_score = 1.0 / (1.0 + repeat_count * 0.5)
            subtopic_counts[subtopic_key] += 1
            
            # Year check (assuming source format "EXAM YEAR")
            year = 2000
            try:
                year_match = re.search(r'\d{4}', candidate.source)
                if year_match:
                    year = int(year_match.group(0))
            except:
                pass
                
            if year >= 2025 - self.config.RECENT_YEAR_THRESHOLD:
                diversity_score *= 1.1
            diversity_score = min(1.0, diversity_score)
            
            # Final Score
            final_score = (
                self.config.RELEVANCE_WEIGHT * relevance_score +
                self.config.DIFFICULTY_WEIGHT * difficulty_score +
                self.config.PERSONALIZATION_WEIGHT * personalization_score +
                self.config.DIVERSITY_WEIGHT * diversity_score
            )
            
            reasoning = self._generate_reasoning(
                candidate, target_difficulty, user_profile, 
                relevance_score, difficulty_score, personalization_score, diversity_score
            )
            
            # Update and return the object
            candidate.relevance_score = final_score # Store final in relevance for now or just update final
            candidate.difficulty_match_score = difficulty_score
            candidate.personalization_score = personalization_score
            candidate.diversity_score = diversity_score
            candidate.final_score = final_score
            candidate.reasoning = reasoning
            
            return candidate

        except Exception as e:
            logger.error(f"Scoring Error: {e}", exc_info=True)
            return candidate

    def _stage2_intelligent_ranking(
        self,
        candidates: List[RetrievalResult],
        user_profile: UserProfile,
        chat_history: List[ChatMessage],
        recent_performance: Optional[RecentPerformance],
        performance_signals: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[RetrievalResult], float]:
        """
        Stage 2: Multi-signal ranking.
        (CPU-bound but fast, so kept synchronous for now, wrapped in async pipeline)
        """
        start_time = time.time()
        
        chat_topics = self.chat_parser.extract_topics(chat_history)
        performance_signals = performance_signals or {}
        
        target_difficulty = self.difficulty_calibrator.compute_target_difficulty(
            user_profile, recent_performance
        )
        
        subtopic_counts = Counter()
        scored_results = []
        
        for candidate in candidates:
            result = self._calculate_candidate_score(
                candidate, user_profile, chat_topics, target_difficulty,
                performance_signals, recent_performance, subtopic_counts
            )
            scored_results.append(result)
        
        ranked_results = sorted(scored_results, key=lambda x: x.final_score, reverse=True)
        top_results = ranked_results[:self.config.FINAL_RESULTS_LIMIT]
        
        latency_ms = (time.time() - start_time) * 1000
        return top_results, latency_ms
    
    def _generate_reasoning(self, candidate: RetrievalResult, target_difficulty, user_profile, relevance, difficulty, personalization, diversity):
        """Generate human-readable reasoning."""
        reasons = []
        diff = candidate.difficulty_score
        if abs(diff - target_difficulty) < 0.5:
            reasons.append(f"Optimally calibrated difficulty ({diff}/5)")
        elif diff < user_profile.expertise_level:
            reasons.append(f"Foundation-building question")
        else:
            reasons.append(f"Challenging question ({diff}/5)")
            
        if any(weak in candidate.topic.lower() for weak in user_profile.weak_topics):
            reasons.append(f"Addresses weak area")
            
        return "; ".join(reasons) if reasons else "Good practice question"

    async def _stage3_llm_ranking(
        self,
        results: List[RetrievalResult],
        user_profile: UserProfile,
        chat_history: List[ChatMessage]
    ) -> Tuple[List[RetrievalResult], float]:
        """Stage 3 (Optional): LLM-based ranking (Async) using Groq."""
        if not self.config.ENABLE_LLM_RANKING:
            return results, 0.0
            
        if not self.groq_client:
            logger.warning("LLM Ranking enabled but Groq client not initialized.")
            return results, 0.0
        
        start_time = time.time()
        
        profile_summary = (
            f"Student Profile: Grade {user_profile.grade}, "
            f"targeting {user_profile.exam_target}, "
            f"expertise {user_profile.expertise_level}/5"
        )
        recent_chat = chat_history[-3:] if len(chat_history) > 3 else chat_history
        chat_summary = "\n".join([f"{msg.role}: {msg.message}" for msg in recent_chat])
        
        # Prepare Batch Prompt
        candidates_text = ""
        for i, res in enumerate(results):
            candidates_text += f"\nCandidate {i+1} (ID: {res.question_id}):\n"
            candidates_text += f"Topic: {res.topic} | Subtopic: {res.subtopic} | Difficulty: {res.difficulty_score}/5\n"
            candidates_text += f"Question: {res.question_text[:200]}...\n"

        prompt = f"""You are an expert tutor. Rank these {len(results)} candidate questions for this student:
        
        Student Profile:
        {profile_summary}
        
        Recent Chat Context:
        {chat_summary}
        
        Candidates:
        {candidates_text}
        
        Task:
        For each candidate, provide ONLY a relevance score (0-100) based on how well it fits the student's needs.
        
        Return a JSON object where keys are the Candidate IDs and values are the integer scores.
        Example:
        {{
            "Q123": 85,
            "Q124": 60,
            ...
        }}
        """

        try:
            response = await self.groq_client.chat.completions.create(
                model=self.config.GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                max_tokens=200, # Reduced max tokens
                temperature=0.1
            )
            content = response.choices[0].message.content.strip()
            batch_scores = json.loads(content)
            
            # Update results with batch scores
            for result in results:
                if result.question_id in batch_scores:
                    # Handle both direct int and dict (just in case)
                    val = batch_scores[result.question_id]
                    score = 0
                    if isinstance(val, (int, float)):
                        score = float(val)
                    elif isinstance(val, dict):
                        score = float(val.get("score", 0))
                        
                    llm_score = score / 100.0
                    
                    # Update Score (Weighted average)
                    result.final_score = 0.7 * result.final_score + 0.3 * llm_score
                    
                    # No reasoning to append
        
        except Exception as e:
            logger.error(f"Groq Batch Ranking error: {e}", exc_info=True)
            # On error, return original results
        
        re_ranked = sorted(results, key=lambda x: x.final_score, reverse=True)
        latency_ms = (time.time() - start_time) * 1000
        return re_ranked, latency_ms
    
    async def retrieve(
        self, 
        user_profile: UserProfile, 
        chat_history: List[ChatMessage],
        recent_performance: Optional[RecentPerformance] = None
    ) -> Tuple[List[RetrievalResult], Dict[str, Any], Dict[str, Any]]:
        """
        Main retrieval method.
        Returns: (results, latency_metadata, performance_signals)
        """
        start_total = time.time()
        
        # Stage 0: Detect Signals (Groq LLM)
        signals_start = time.time()
        performance_signals = await self.detect_performance_signals_groq(chat_history)
        signals_latency = (time.time() - signals_start) * 1000
        
        # Stage 1: Candidate Retrieval
        candidates, retrieval_latency = await self._stage1_candidate_generation(
            user_profile, chat_history, recent_performance
        )
        
        # Stage 2: Intelligent Ranking
        ranked_candidates, ranking_latency = self._stage2_intelligent_ranking(
            candidates, user_profile, chat_history, recent_performance, performance_signals
        )
        
        # Stage 3: LLM Re-ranking
        final_results, llm_ranking_latency = await self._stage3_llm_ranking(
            ranked_candidates, user_profile, chat_history
        )
        
        total_latency = (time.time() - start_total) * 1000
        
        # Determine ranker used and total ranking time
        ranker_used = "heuristic_scoring"
        total_ranking_latency = ranking_latency
        
        if self.config.ENABLE_LLM_RANKING and self.groq_client:
            ranker_used = "llm_reranking"
            total_ranking_latency += llm_ranking_latency

        latency_metadata = {
            "retrieval_latency_ms": retrieval_latency,
            "ranking_latency_ms": total_ranking_latency,
            "total_latency_ms": total_latency,
            "retriever_used": "vector_search",
            "ranker_used": ranker_used
        }
        
        return final_results, latency_metadata, performance_signals
