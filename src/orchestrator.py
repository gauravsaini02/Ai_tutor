"""
Tutor Orchestrator - Main entry point for the adaptive tutoring system

Takes student profile, chat history, and recent performance.
Returns intelligent question recommendations with tutor context and metadata.
"""

import json
from typing import List, Dict, Any, Optional


from .retrieval import (
    RetrievalPipeline, UserProfile, ChatMessage, 
    RecentPerformance, RetrievalResult
)
from .config import Config


class TutorContext:
    """Context and guidance for the tutor."""
    def __init__(self, identified_gaps: List[str], recommended_study_sequence: List[str], estimated_session_duration_minutes: int, next_milestone: str):
        self.identified_gaps = identified_gaps
        self.recommended_study_sequence = recommended_study_sequence
        self.estimated_session_duration_minutes = estimated_session_duration_minutes
        self.next_milestone = next_milestone


class RecommendedQuestion:
    """Single recommended question with full details."""
    def __init__(self, question_id: str, topic: str, subtopic: str, difficulty_score: float, question_text: str, options: List[str], source: str, explanation: str, time_estimate_seconds: int, relevance_score: float, reasoning: str):
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


class OrchestratorOutput:
    """Complete output from the tutor orchestrator."""
    def __init__(self, recommended_questions: List[Dict[str, Any]], tutor_context: Dict[str, Any], pipeline_metadata: Dict[str, Any]):
        self.recommended_questions = recommended_questions
        self.tutor_context = tutor_context
        self.pipeline_metadata = pipeline_metadata


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
        recent_performance: Optional[RecentPerformance]
    ) -> List[str]:
        """Identify learning gaps from profile and performance."""
        gaps = []
        
        # Weak topics are primary gaps
        gaps.extend(user_profile.weak_topics)
        
        # Add topics from poor recent performance
        if recent_performance and recent_performance.correct / max(recent_performance.questions_attempted, 1) < 0.5:
            if recent_performance.topic not in gaps:
                gaps.append(recent_performance.topic)
        
        # Parse chat for mentioned struggles
        for msg in chat_history:
            if msg.role == "student":
                text = msg.message.lower()
                # Look for struggle indicators
                if any(word in text for word in ["struggling", "don't understand", "confused", "difficult"]):
                    # Extract topic keywords (simplified - could use NLP)
                    for topic in ["photosynthesis", "respiration", "genetics", "mechanics", 
                                 "thermodynamics", "kinematics", "shm", "vectors"]:
                        if topic in text and topic not in gaps:
                            gaps.append(topic)
        
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
    
    def _estimate_session_duration(
        self,
        num_questions: int,
        avg_time_per_question: int,
        user_profile: UserProfile
    ) -> int:
        """Estimate study session duration in minutes."""
        # Base time from questions
        base_time_seconds = num_questions * avg_time_per_question
        
        # Add buffer for review and explanation reading (30%)
        total_time_seconds = base_time_seconds * 1.3
        
        # Add setup time
        setup_time_seconds = 5 * 60  # 5 minutes
        
        total_minutes = int((total_time_seconds + setup_time_seconds) / 60)
        
        # Round to nearest 5 minutes
        return ((total_minutes + 4) // 5) * 5
    
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
        questions = []
        
        for result in results:
            # Convert time_estimate to int if it's a string
            time_estimate = result.time_estimate_seconds
            if isinstance(time_estimate, str):
                time_estimate = int(time_estimate)
            
            question = {
                "question_id": result.question_id,
                "topic": result.topic,
                "subtopic": result.subtopic,
                "difficulty_score": result.difficulty_score,
                "question_text": result.question_text,
                "options": result.options,
                "source": result.source,
                "explanation": result.explanation,
                "time_estimate_seconds": time_estimate,
                "relevance_score": round(result.relevance_score, 3),
                "reasoning": result.reasoning
            }
            questions.append(question)
        
        return questions
    
    async def recommend(
        self,
        user_profile_dict: Dict[str, Any],
        chat_history_dict: List[Dict[str, str]],
        recent_performance_dict: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Main orchestration method - returns complete tutor recommendation.
        
        Args:
            user_profile_dict: User profile as dict
            chat_history_dict: Chat history as list of dicts
            recent_performance_dict: Recent performance as dict (optional)
        
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
        
        # Run retrieval pipeline
        results, latency_metadata = await self.pipeline.retrieve(
            user_profile, chat_history, recent_performance
        )
        
        # Generate tutor context
        identified_gaps = self._identify_gaps(user_profile, chat_history, recent_performance)
        
        study_sequence = self._generate_study_sequence(
            user_profile, identified_gaps, recent_performance
        )
        
        # Calculate session duration
        avg_time = sum(
            int(r.time_estimate_seconds) if isinstance(r.time_estimate_seconds, str) 
            else r.time_estimate_seconds 
            for r in results
        ) / max(len(results), 1)
        session_duration = self._estimate_session_duration(
            len(results), int(avg_time), user_profile
        )
        
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


import asyncio

async def main():
    """Example usage of the tutor orchestrator."""
    
    # Initialize orchestrator
    orchestrator = TutorOrchestrator()
    
    # Example input: Struggling beginner in photosynthesis
    user_profile = {
        "grade": "11",
        "exam_target": "neet",
        "subject": "Biology",
        "expertise_level": 1.5,
        "weak_topics": ["photosynthesis", "calvin cycle"],
        "strong_topics": []
    }
    
    chat_history = [
        {
            "role": "student",
            "message": "I don't understand photosynthesis. Can we start easy?"
        },
        {
            "role": "tutor",
            "message": "Sure! Let me find some foundational questions for you."
        }
    ]
    
    recent_performance = {
        "topic": "photosynthesis",
        "questions_attempted": 3,
        "correct": 1,
        "avg_time_seconds": 200,
        "confidence_score": 1.2
    }
    
    # Get recommendations
    print("="*80)
    print("ADAPTIVE TUTOR ORCHESTRATOR - DEMO")
    print("="*80)
    print(f"\nStudent: Grade {user_profile['grade']}, {user_profile['exam_target'].upper()}, {user_profile['subject']}")
    print(f"Expertise: {user_profile['expertise_level']}/5")
    print(f"Weak Topics: {', '.join(user_profile['weak_topics'])}")
    print("="*80)
    
    result = await orchestrator.recommend(user_profile, chat_history, recent_performance)
    
    # Display output
    print(f"\n📚 RECOMMENDED QUESTIONS ({len(result['recommended_questions'])} total)")
    print("-"*80)
    for i, q in enumerate(result['recommended_questions'][:3], 1):
        print(f"\n{i}. [{q['question_id']}] {q['topic']} - {q['subtopic']}")
        print(f"   Difficulty: {q['difficulty_score']}/5 | Relevance: {q['relevance_score']:.2f}")
        print(f"   Reasoning: {q['reasoning']}")
        print(f"   Question: {q['question_text'][:100]}...")
    
    print(f"\n\n🎯 TUTOR CONTEXT")
    print("-"*80)
    tc = result['tutor_context']
    print(f"Identified Gaps: {', '.join(tc['identified_gaps'])}")
    print(f"Study Sequence: {' → '.join(tc['recommended_study_sequence'])}")
    print(f"Session Duration: {tc['estimated_session_duration_minutes']} minutes")
    print(f"Next Milestone: {tc['next_milestone']}")
    
    print(f"\n\n⚡ PERFORMANCE METRICS")
    print("-"*80)
    pm = result['pipeline_metadata']
    print(f"Retrieval: {pm['retrieval_latency_ms']:.0f}ms")
    print(f"Ranking: {pm['ranking_latency_ms']:.1f}ms")
    print(f"Total: {pm['total_latency_ms']:.0f}ms")
    
    # Save to file
    with open('example_output.json', 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n✅ Full output saved to: example_output.json")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())
