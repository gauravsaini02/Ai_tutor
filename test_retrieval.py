"""
Test the retrieval pipeline with various scenarios.
"""

from retrieval_pipeline import (
    RetrievalPipeline, Config, UserProfile, ChatMessage, RecentPerformance
)


def test_case_1_struggling_beginner():
    """Test Case 1: Struggling beginner in Biology."""
    print("\n" + "="*100)
    print("TEST CASE 1: Struggling Beginner - Biology (Photosynthesis)")
    print("="*100)
    
    user_profile = UserProfile(
        grade="11",
        exam_target="neet",
        subject="Biology",
        expertise_level=1.5,
        weak_topics=["photosynthesis", "calvin cycle"],
        strong_topics=[]
    )
    
    chat_history = [
        ChatMessage(role="student", message="I don't understand photosynthesis. Can we start easy?")
    ]
    
    recent_performance = RecentPerformance(
        topic="photosynthesis",
        questions_attempted=3,
        correct=1,
        avg_time_seconds=200,
        confidence_score=1.2
    )
    
    return user_profile, chat_history, recent_performance


def test_case_2_advanced_challenge():
    """Test Case 2: Advanced student wanting challenge."""
    print("\n" + "="*100)
    print("TEST CASE 2: Advanced Student - Physics (Mechanics), Challenge Mode")
    print("="*100)
    
    user_profile = UserProfile(
        grade="12",
        exam_target="neet",
        subject="Physics",
        expertise_level=4.5,
        weak_topics=[],
        strong_topics=["mechanics", "gravitation"]
    )
    
    chat_history = [
        ChatMessage(role="student", message="I want harder problems now. Push me."),
        ChatMessage(role="tutor", message="Great! I'll find challenging questions for you.")
    ]
    
    recent_performance = RecentPerformance(
        topic="gravitation",
        questions_attempted=10,
        correct=9,
        avg_time_seconds=150,
        confidence_score=4.8
    )
    
    return user_profile, chat_history, recent_performance


def test_case_3_mid_session_genetics():
    """Test Case 3: Mid-session with chat context on genetics."""
    print("\n" + "="*100)
    print("TEST CASE 3: Mid-Session - Biology (Genetics), Struggling with Inheritance")
    print("="*100)
    
    user_profile = UserProfile(
        grade="12",
        exam_target="neet",
        subject="Biology",
        expertise_level=3.0,
        weak_topics=["genetics"],
        strong_topics=["anatomy"]
    )
    
    chat_history = [
        ChatMessage(role="student", message="I got Mendel's laws wrong. Struggling with inheritance patterns."),
        ChatMessage(role="tutor", message="Let me find questions on Mendel's laws."),
        ChatMessage(role="student", message="I also don't understand homozygous vs heterozygous well.")
    ]
    
    recent_performance = RecentPerformance(
        topic="genetics",
        questions_attempted=5,
        correct=2,
        avg_time_seconds=210,
        confidence_score=2.0
    )
    
    return user_profile, chat_history, recent_performance


def test_case_4_building_confidence():
    """Test Case 4: Student needs confidence building."""
    print("\n" + "="*100)
    print("TEST CASE 4: Confidence Building - Biology (Respiration)")
    print("="*100)
    
    user_profile = UserProfile(
        grade="11",
        exam_target="neet",
        subject="Biology",
        expertise_level=2.5,
        weak_topics=["respiration", "glycolysis"],
        strong_topics=[]
    )
    
    chat_history = [
        ChatMessage(role="student", message="I keep getting these wrong. I'm losing confidence."),
        ChatMessage(role="tutor", message="Don't worry! Let's practice some easier questions first.")
    ]
    
    recent_performance = RecentPerformance(
        topic="respiration",
        questions_attempted=6,
        correct=2,
        avg_time_seconds=250,
        confidence_score=1.5
    )
    
    return user_profile, chat_history, recent_performance


def run_test(pipeline, test_func):
    """Run a single test case."""
    user_profile, chat_history, recent_performance = test_func()
    
    print(f"Profile: Grade {user_profile.grade}, {user_profile.exam_target.upper()}, {user_profile.subject}")
    print(f"Expertise: {user_profile.expertise_level}/5")
    print(f"Weak Topics: {', '.join(user_profile.weak_topics) if user_profile.weak_topics else 'None'}")
    print(f"Recent Performance: {recent_performance.correct}/{recent_performance.questions_attempted} correct, confidence {recent_performance.confidence_score}/5")
    print("-"*100)
    
    # Retrieve
    results, latency = pipeline.retrieve(user_profile, chat_history, recent_performance)
    
    # Display top 3 results
    print(f"\nTop {min(3, len(results))} Recommended Questions:")
    for i, result in enumerate(results[:3], 1):
        print(f"\n  {i}. [{result.question_id}] {result.topic} - {result.subtopic}")
        print(f"     Difficulty: {result.difficulty_score}/5 | Score: {result.final_score:.3f}")
        print(f"     Reasoning: {result.reasoning}")
        print(f"     Question: {result.question_text[:120]}...")
    
    print(f"\n  Latency: {latency['total_latency_ms']:.0f}ms (retrieval: {latency['retrieval_latency_ms']:.0f}ms, ranking: {latency['ranking_latency_ms']:.0f}ms)")
    
    return latency['total_latency_ms']


def main():
    """Run all test cases."""
    print("\n" + "🧪 "*40)
    print("RETRIEVAL PIPELINE TEST SUITE")
    print("🧪 "*40)
    
    # Initialize pipeline
    config = Config()
    
    if not config.OPENAI_API_KEY or not config.QDRANT_API_KEY:
        print("❌ Error: Missing API keys in .env file")
        exit(1)
    
    try:
        pipeline = RetrievalPipeline(config)
        
        # Run all test cases
        test_cases = [
            test_case_1_struggling_beginner,
            test_case_2_advanced_challenge,
            test_case_3_mid_session_genetics,
            test_case_4_building_confidence
        ]
        
        latencies = []
        for test_func in test_cases:
            latency = run_test(pipeline, test_func)
            latencies.append(latency)
        
        # Summary
        print("\n" + "="*100)
        print("TEST SUMMARY")
        print("="*100)
        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)
        
        print(f"Total test cases: {len(test_cases)}")
        print(f"Average latency: {avg_latency:.2f}ms")
        print(f"Max latency: {max_latency:.2f}ms")
        print(f"Target latency: {config.MAX_TOTAL_LATENCY_MS}ms")
        
        if max_latency <= config.MAX_TOTAL_LATENCY_MS:
            print(f"\n✅ All tests passed! Latency constraint satisfied.")
        else:
            print(f"\n⚠️  Warning: Max latency ({max_latency:.2f}ms) exceeds target ({config.MAX_TOTAL_LATENCY_MS}ms)")
        
        print("="*100)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
