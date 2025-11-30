"""
Tutor Orchestrator - Main entry point for the adaptive tutoring system

Takes student profile, chat history, and recent performance.
Returns intelligent question recommendations with tutor context and metadata.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from .retrieval import (
    RetrievalPipeline, UserProfile, ChatMessage, 
    RecentPerformance, RetrievalResult, RevisionRecord
)
from .config import Config
from .logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class TutorContext:
    """Context and guidance for the tutor."""
    identified_gaps: List[str]
    recommended_study_sequence: List[str]
    estimated_session_duration_minutes: int
    next_milestone: str


@dataclass
class RecommendedQuestion:
    """Single recommended question with full details."""
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


@dataclass
class OrchestratorOutput:
    """Complete output from the tutor orchestrator."""
    recommended_questions: List[Dict[str, Any]]
    tutor_context: Dict[str, Any]
    pipeline_metadata: Dict[str, Any]


class TutorOrchestrator:
    """
    Main orchestrator for the adaptive tutoring system.
    
    Coordinates retrieval, generates tutor context, and formats complete output.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize orchestrator with retrieval pipeline."""
        self.config = config or Config()
        self.pipeline = RetrievalPipeline(self.config)
    
    def _identify_gaps(
        self, 
        user_profile: UserProfile,
        chat_history: List[ChatMessage],
        recent_performance: Optional[RecentPerformance],
        performance_signals: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """Identify learning gaps from profile, performance, and LLM signals."""
        gaps = []
        
        # 1. LLM Detected Gap (Highest Priority)
        if performance_signals:
            llm_gap = performance_signals.get("gap")
            if llm_gap and isinstance(llm_gap, str) and llm_gap.lower() != "none":
                gaps.append(llm_gap)
        
        # 2. Weak topics from profile
        for topic in user_profile.weak_topics:
            if topic not in gaps:
                gaps.append(topic)
        
        # 3. Add topics from poor recent performance
        if recent_performance and recent_performance.correct / max(recent_performance.questions_attempted, 1) < 0.5:
            if recent_performance.topic not in gaps:
                gaps.append(recent_performance.topic)
        
        return gaps[:5]  # Limit to top 5 gaps
    
    def _generate_study_sequence(
        self,
        user_profile: UserProfile,
        identified_gaps: List[str],
        recent_performance: Optional[RecentPerformance]
    ) -> List[str]:
        """Generate recommended study sequence based on ZPD."""
        sequence = []
        
        # Determine starting point based on expertise and confidence
        if user_profile.expertise_level < 2.0:
            sequence.append("build_foundation")
            sequence.append("practice_basics")
            sequence.append("solve_direct_problems")
        elif user_profile.expertise_level < 4.0:
            # Check confidence
            if recent_performance and recent_performance.confidence_score < 2.5:
                sequence.append("build_confidence")
                sequence.append("strengthen_fundamentals")
            else:
                sequence.append("solve_direct_problems")
            sequence.append("tackle_application_problems")
        else:
            # Advanced students
            sequence.append("solve_complex_problems")
            sequence.append("practice_edge_cases")
            sequence.append("attempt_multi_concept_questions")
        
        # Add topic-specific steps if there are gaps
        if identified_gaps:
            sequence.append(f"focus_on_{identified_gaps[0].replace(' ', '_')}")
        
        return sequence[:4]  # Limit to 4 steps
    
    def _generate_next_milestone(
        self,
        user_profile: UserProfile,
        identified_gaps: List[str],
        recent_performance: Optional[RecentPerformance]
    ) -> str:
        """Generate next learning milestone for the student."""
        exam = user_profile.exam_target.upper()
        
        # Based on expertise level and gaps
        if user_profile.expertise_level < 2.0:
            if identified_gaps:
                return f"Build strong foundation in {identified_gaps[0]} for {exam}"
            return f"Master basic concepts for {exam}"
        
        elif user_profile.expertise_level < 3.5:
            if identified_gaps:
                return f"Close knowledge gaps in {', '.join(identified_gaps[:2])} for {exam} cutoff"
            return f"Reach {exam} cutoff level in {user_profile.subject}"
        
        else:
            if recent_performance and recent_performance.confidence_score < 4.0:
                return f"Build confidence for {exam} advanced problems"
            return f"Master {exam} Advanced level in {user_profile.subject}"
    
    def _format_questions(self, results: List[RetrievalResult]) -> List[Dict[str, Any]]:
        """Format retrieval results into output question format."""
        return [
            {
                "question_id": r.question_id,
                "topic": r.topic,
                "subtopic": r.subtopic,
                "difficulty_score": r.difficulty_score,
                "question_text": r.question_text,
                "options": r.options,
                "source": r.source,
                "explanation": r.explanation,
                "time_estimate_seconds": int(r.time_estimate_seconds) if isinstance(r.time_estimate_seconds, str) else r.time_estimate_seconds,
                "relevance_score": round(r.relevance_score, 3),
                "reasoning": r.reasoning
            }
            for r in results
        ]
    
    async def recommend(
        self,
        user_profile_dict: Dict[str, Any],
        chat_history_dict: List[Dict[str, str]],
        recent_performance_dict: Optional[Dict[str, Any]] = None,
        revision_history_dict: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Main orchestration method - returns complete tutor recommendation.
        
        Args:
            user_profile_dict: User profile as dict
            chat_history_dict: Chat history as list of dicts
            recent_performance_dict: Recent performance as dict (optional)
            revision_history_dict: Spaced repetition history as list of dicts (optional)
        
        Returns:
            Complete recommendation with questions, context, and metadata
        """
        # Convert dicts to dataclasses
        user_profile = UserProfile(**user_profile_dict)
        
        chat_history = [
            ChatMessage(role=msg["role"], message=msg["message"]) 
            for msg in chat_history_dict
        ]
        
        recent_performance = None
        if recent_performance_dict:
            recent_performance = RecentPerformance(**recent_performance_dict)
        
        revision_history = None
        if revision_history_dict:
            revision_history = [
                RevisionRecord(**record) for record in revision_history_dict
            ]
        
        # Run retrieval pipeline (with SR support)
        results, latency_metadata, performance_signals = await self.pipeline.retrieve(
            user_profile, chat_history, recent_performance, revision_history
        )
        
        # Generate tutor context
        identified_gaps = self._identify_gaps(
            user_profile, chat_history, recent_performance, performance_signals
        )
        
        study_sequence = self._generate_study_sequence(
            user_profile, identified_gaps, recent_performance
        )
        
        # Calculate session duration
        total_seconds = sum(r.time_estimate_seconds for r in results)
        session_duration = max(5, int(total_seconds / 60))

        
        next_milestone = self._generate_next_milestone(
            user_profile, identified_gaps, recent_performance
        )
        
        # Format output
        output = {
            "recommended_questions": self._format_questions(results),
            "tutor_context": {
                "identified_gaps": identified_gaps,
                "recommended_study_sequence": study_sequence,
                "estimated_session_duration_minutes": session_duration,
                "next_milestone": next_milestone
            },
            "pipeline_metadata": latency_metadata
        }
        
        return output


