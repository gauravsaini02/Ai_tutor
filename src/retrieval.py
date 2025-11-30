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
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter

from dotenv import load_dotenv
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer, util
from openai import AsyncOpenAI

load_dotenv()


class UserProfile:
    """Student profile information."""
    def __init__(self, grade: str, exam_target: str, subject: str, expertise_level: float, weak_topics: List[str], strong_topics: List[str]):
        self.grade = grade
        self.exam_target = exam_target
        self.subject = subject
        self.expertise_level = expertise_level
        self.weak_topics = weak_topics
        self.strong_topics = strong_topics


class RecentPerformance:
    """Recent performance metrics."""
    def __init__(self, topic: str, questions_attempted: int, correct: int, avg_time_seconds: float, confidence_score: float):
        self.topic = topic
        self.questions_attempted = questions_attempted
        self.correct = correct
        self.avg_time_seconds = avg_time_seconds
        self.confidence_score = confidence_score


class ChatMessage:
    """Chat history message."""
    def __init__(self, role: str, message: str):
        self.role = role
        self.message = message


class RetrievalResult:
    """Single retrieved question with scores."""
    def __init__(self, question_id: str, topic: str, subtopic: str, difficulty_score: float, question_text: str, options: List[str], source: str, explanation: str, time_estimate_seconds: int, relevance_score: float, reasoning: str, vector_similarity: float, difficulty_match_score: float, personalization_score: float, diversity_score: float, final_score: float):
        self.question_id = question_id
        self.topic = topic
        self.subtopic = subtopic
        self.difficulty_score = difficulty_score
        self.question_text = question_text
        self.options = options
        self.source = source
        self.explanation = explanation
        self.time_estimate_seconds = time_estimate_seconds
        self.relevance_score = relevance_score
        self.reasoning = reasoning
        self.vector_similarity = vector_similarity
        self.difficulty_match_score = difficulty_match_score
        self.personalization_score = personalization_score
        self.diversity_score = diversity_score
        self.final_score = final_score


from .config import Config


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
    async def detect_performance_signals_llm(chat_history: List[ChatMessage], client: AsyncOpenAI, model: str) -> Dict[str, str]:
        """Detect performance indicators using LLM."""
        if not client or not chat_history:
            return {}

        recent_chat = chat_history[-3:]
        chat_text = "\n".join([f"{msg.role}: {msg.message}" for msg in recent_chat])
        
        prompt = f"""Analyze the student's messages in this chat history for performance signals.
                    Chat:
                    {chat_text}

                    Classify the student's state into exactly one of these categories:
                    - struggling (if they are confused, getting things wrong, asking for help)
                    - bored (if they find it too easy, unengaging)
                    - ready_for_more (if they explicitly ask for harder questions or challenge)
                    - neutral (default)

                    Output only the category name."""

        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0.0
            )
            signal = response.choices[0].message.content.strip().lower()
            if signal in ["struggling", "bored", "ready_for_more"]:
                return {"performance": signal}
            return {}
        except Exception as e:
            print(f"Signal detection error: {e}")
            return {}

    @staticmethod
    def detect_performance_signals_semantic(chat_history: List[ChatMessage], embedding_model: SentenceTransformer) -> Dict[str, str]:
        """Detect performance indicators using semantic similarity (Fast Local)."""
        if not chat_history:
            return {}

        # Anchor phrases for each category
        anchors = {
            "struggling": [
                "I don't understand", "I am confused", "I keep getting it wrong", 
                "Explain it again", "I'm lost", "I am stuck", "I don't know the answer",
                "This makes no sense", "I'm failing", "It's too complicated", "I can't follow",
                "I'm hitting a wall", "It's not clicking", "Like speaking another language", "Going in circles",
                "Going over my head", "Break it down simpler", "Gibberish", "Why is this so hard",
                "About to give up", "Too advanced", "Need a hint", "Not sinking in", "Feel dumb",
                "Back to basics", "Drowning in equations", "My mind is blank", "Confused by the formula",
                "Frustratingly difficult", "Explain like I'm 5", "Concept is alien", "Don't know where to start",
                "Above my pay grade", "I can't wrap my head around this", "Lost in details", "Struggling to keep up",
                "Not getting anywhere", "Help I'm stuck", "I don't have a clue"
            ],
            "bored": [
                "This is boring", "Can we move on", "I already know this", 
                "Not challenging enough", "This is too easy", "Seen this before", 
                "Repetitive", "Yawn", "Too basic",
                "I could do this in my sleep", "Nothing new here", "Skip this",
                "Learned this in grade school", "Skip to the good part", "Waste of time",
                "Next topic please", "Trivial", "Basic stuff", "Done this a thousand times",
                "Move faster", "Too slow", "Bored out of my mind", "Is this a joke",
                "Tell me something I don't know", "Elementary", "Skip the basics",
                "Way past this level", "Not challenging at all", "Can we do something else",
                "Checking out dull", "Give me something I don't know"
            ],
            "ready_for_more": [
                "Give me a challenge", "I want harder questions", "Push me", "Test my limits", 
                "Next level please", "Something more advanced", "Give me something hard",
                "Make it difficult", "I want to sweat", "Is that all you got", "Crank up the difficulty",
                "JEE Advanced level", "Something complex", "Real work", "Toughest problem",
                "This is child's play give me real work", "Brain teaser", "Push my limits",
                "Ramp it up", "Hardest one", "Bored of easy give me hard", "Test me",
                "Go deeper", "Complex scenario", "Ready to level up", "Throw everything at me",
                "Solve the impossible", "Olympiad level", "Struggle in a good way", "Test my mastery"
            ],
            "neutral": [
                "What is the answer", "Explain this concept", "I have a test", "Let's solve this",
                "Okay", "Thanks", "How many questions left", "Is the answer X", "Wait a minute",
                "Writing this down", "Repeat the question", "I think the answer is", "What topic is this",
                "Let me think", "Hold on", "Interesting", "Go on", "I see", "Got it", "Next",
                "Hello", "How are you"
            ]
        }
        
        last_msg = chat_history[-1].message
        
        # Encode user message and anchors
        user_emb = embedding_model.encode(last_msg, convert_to_tensor=True)
        
        best_score = 0.0
        best_signal = None
        
        for signal, phrases in anchors.items():
            anchor_embs = embedding_model.encode(phrases, convert_to_tensor=True)
            scores = util.cos_sim(user_emb, anchor_embs)[0]
            scores = util.cos_sim(user_emb, anchor_embs)[0]
            max_score = float(scores.max())
            
            if max_score > best_score:
                best_score = max_score
                best_signal = signal
        
        # Threshold to avoid false positives
        if best_score > 0.35:  # Slightly lowered threshold since we have explicit neutral
            if best_signal == "neutral":
                return {}
            return {"performance": best_signal}
            
        return {}
    
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
            
            success_rate = recent_performance.correct / max(recent_performance.questions_attempted, 1)
            if success_rate < 0.4:
                target -= 0.4
            elif success_rate > 0.9:
                target += 0.3
        
        return max(1.0, min(5.0, target))
    
    @staticmethod
    def score_difficulty_match(question_difficulty: float, target_difficulty: float) -> float:
        """Score how well a question's difficulty matches the target."""
        difficulty_gap = abs(question_difficulty - target_difficulty)
        return 1.0 / (1.0 + difficulty_gap ** 2)


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
        print(f"Loading local embedding model: {config.EMBEDDING_MODEL}...")
        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
        print(f"✅ Embedding model loaded!")
        
        # Async OpenAI Client
        if config.OPENAI_API_KEY:
            self.openai_client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
        else:
            self.openai_client = None
            
        self.chat_parser = ChatHistoryParser()
        self.difficulty_calibrator = DifficultyCalibrator()
    
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
        recent_performance: Optional[RecentPerformance]
    ) -> Tuple[List[Dict[str, Any]], float]:
        """Stage 1: Fast candidate generation using vector search + metadata filters."""
        start_time = time.time()
        
        # Build query from context
        query_text = self.chat_parser.build_context_query(
            user_profile, chat_history, recent_performance
        )
        
        # Generate query embedding (Async wrapper around sync call)
        embed_start = time.time()
        query_embedding = await self._get_embedding(query_text)
        embed_time = (time.time() - embed_start) * 1000
        print(f"  - Embedding generation: {embed_time:.2f}ms")
        
        # Build metadata filters
        must_conditions = []
        if user_profile.subject:
            must_conditions.append(
                models.FieldCondition(
                    key="subject",
                    match=models.MatchValue(value=user_profile.subject.capitalize())
                )
            )
        if user_profile.exam_target:
            must_conditions.append(
                models.FieldCondition(
                    key="exam_type",
                    match=models.MatchValue(value=user_profile.exam_target.lower())
                )
            )
        
        # Search with filters (Async)
        search_start = time.time()
        search_results = await self.qdrant_client.query_points(
            collection_name=self.config.COLLECTION_NAME,
            query=query_embedding,
            query_filter=models.Filter(must=must_conditions) if must_conditions else None,
            limit=self.config.CANDIDATE_LIMIT,
            with_payload=True,
            with_vectors=False
        )
        search_time = (time.time() - search_start) * 1000
        print(f"  - Qdrant search: {search_time:.2f}ms")
        
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
        recent_performance: Optional[RecentPerformance],
        performance_signals: Optional[Dict[str, str]] = None
    ) -> Tuple[List[RetrievalResult], float]:
        """
        Stage 2: Multi-signal ranking.
        (CPU-bound but fast, so kept synchronous for now, wrapped in async pipeline)
        """
        start_time = time.time()
        
        chat_topics = self.chat_parser.extract_topics(chat_history)
        # performance_signals = self.chat_parser.detect_performance_signals(chat_history) # Deprecated
        # Use passed signals or default to empty
        performance_signals = performance_signals or {}
        
        target_difficulty = self.difficulty_calibrator.compute_target_difficulty(
            user_profile, recent_performance
        )
        
        subtopic_counts = Counter()
        scored_results = []
        
        for candidate in candidates:
            # 1. Relevance Score
            relevance_score = candidate.get("vector_similarity", 0.0)
            topic = candidate.get("topic", "").lower()
            subtopic = candidate.get("sub_topic", "").lower()
            if any(chat_topic in topic or chat_topic in subtopic for chat_topic in chat_topics):
                relevance_score *= 1.2
            relevance_score = min(1.0, relevance_score)
            
            # 2. Difficulty Score
            question_difficulty = candidate.get("difficulty", 3.0)
            difficulty_score = self.difficulty_calibrator.score_difficulty_match(
                question_difficulty, target_difficulty
            )
            
            if performance_signals.get("performance") == "struggling":
                if question_difficulty > user_profile.expertise_level:
                    difficulty_score *= 0.7
            elif performance_signals.get("performance") == "ready_for_more":
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
            subtopic_key = candidate.get("sub_topic", "unknown")
            repeat_count = subtopic_counts[subtopic_key]
            diversity_score = 1.0 / (1.0 + repeat_count * 0.5)
            subtopic_counts[subtopic_key] += 1
            
            year = candidate.get("year", 2000)
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
        
        ranked_results = sorted(scored_results, key=lambda x: x.final_score, reverse=True)
        top_results = ranked_results[:self.config.FINAL_RESULTS_LIMIT]
        
        latency_ms = (time.time() - start_time) * 1000
        return top_results, latency_ms
    
    def _generate_reasoning(self, candidate, target_difficulty, user_profile, relevance, difficulty, personalization, diversity):
        """Generate human-readable reasoning."""
        reasons = []
        diff = candidate.get("difficulty", 3.0)
        if abs(diff - target_difficulty) < 0.5:
            reasons.append(f"Optimally calibrated difficulty ({diff}/5)")
        elif diff < user_profile.expertise_level:
            reasons.append(f"Foundation-building question")
        else:
            reasons.append(f"Challenging question ({diff}/5)")
            
        if any(weak in candidate.get("topic", "").lower() for weak in user_profile.weak_topics):
            reasons.append(f"Addresses weak area")
            
        return "; ".join(reasons) if reasons else "Good practice question"

    async def _stage3_llm_ranking(
        self,
        results: List[RetrievalResult],
        user_profile: UserProfile,
        chat_history: List[ChatMessage]
    ) -> Tuple[List[RetrievalResult], float]:
        """Stage 3 (Optional): LLM-based ranking (Async)."""
        if not self.config.ENABLE_LLM_RANKING:
            return results, 0.0
        
        start_time = time.time()
        
        profile_summary = (
            f"Student Profile: Grade {user_profile.grade}, "
            f"targeting {user_profile.exam_target}, "
            f"expertise {user_profile.expertise_level}/5"
        )
        recent_chat = chat_history[-3:] if len(chat_history) > 3 else chat_history
        chat_summary = "\n".join([f"{msg.role}: {msg.message}" for msg in recent_chat])
        
        # Async LLM calls
        # Note: In a real app, we would use asyncio.gather to run these in parallel
        # For simplicity, we loop (but await yields control)
        for result in results:
            prompt = f"""Score question fit (0-100) for student.
                        {profile_summary}
                        Chat: {chat_summary}
                        Question: {result.topic} - {result.difficulty_score}/5
                        {result.question_text[:200]}...
                        Output only number."""

            try:
                response = await self.openai_client.chat.completions.create(
                    model=self.config.LLM_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=10,
                    temperature=0.3
                )
                score = float(response.choices[0].message.content.strip()) / 100.0
                result.final_score = 0.7 * result.final_score + 0.3 * score
            except Exception as e:
                print(f"LLM error: {e}")
        
        re_ranked = sorted(results, key=lambda x: x.final_score, reverse=True)
        latency_ms = (time.time() - start_time) * 1000
        return re_ranked, latency_ms
    
    async def retrieve(
        self,
        user_profile: UserProfile,
        chat_history: List[ChatMessage],
        recent_performance: Optional[RecentPerformance] = None
    ) -> Tuple[List[RetrievalResult], Dict[str, float]]:
        """Main retrieval method (Async)."""
        total_start = time.time()
        
        # Stage 1 (Async Vector Search)
        candidates, retrieval_latency = await self._stage1_candidate_generation(
            user_profile, chat_history, recent_performance
        )
        print(f"Stage 1: Retrieved {len(candidates)} candidates in {retrieval_latency:.2f}ms")

        # Signal Detection (Fast Local - Synchronous)
        # No need for async/await or parallel task, this takes ~10ms
        signal_start = time.time()
        performance_signals = self.chat_parser.detect_performance_signals_semantic(
            chat_history, self.embedding_model
        )
        signal_time = (time.time() - signal_start) * 1000
        if performance_signals:
            print(f"  - Detected signals: {performance_signals} ({signal_time:.2f}ms)")
        
        # Stage 2 (CPU bound, synchronous but fast)
        ranked_results, ranking_latency = self._stage2_intelligent_ranking(
            candidates, user_profile, chat_history, recent_performance, performance_signals
        )
        print(f"Stage 2: Ranked to top {len(ranked_results)} in {ranking_latency:.2f}ms")
        
        # Stage 3
        final_results, llm_latency = await self._stage3_llm_ranking(
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
