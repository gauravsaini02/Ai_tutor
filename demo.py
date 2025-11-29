import time
from src.orchestrator import TutorOrchestrator
from src.retrieval import UserProfile, ChatMessage, RecentPerformance

def run_test_case(name, profile_dict, chat_list, perf_dict):
    print(f"\n{'#'*80}")
    print(f"TEST CASE: {name}")
    print(f"{'#'*80}")
    
    orchestrator = TutorOrchestrator()
    
    # Convert dicts to objects for display (orchestrator handles conversion internally too)
    print(f"User: Grade {profile_dict['grade']}, {profile_dict['exam_target']}, {profile_dict['subject']}")
    print(f"Expertise: {profile_dict['expertise_level']}/5 | Weak: {profile_dict['weak_topics']}")
    print(f"Chat Context: \"{chat_list[-1]['message']}\"")
    
    start = time.time()
    result = orchestrator.recommend(profile_dict, chat_list, perf_dict)
    latency = (time.time() - start) * 1000
    
    print(f"\n📚 Top 3 Recommendations:")
    for i, q in enumerate(result['recommended_questions'][:3], 1):
        print(f"{i}. [{q['difficulty_score']}/5] {q['topic']} - {q['subtopic']}")
        print(f"   Reasoning: {q['reasoning']}")
    
    print(f"\n⚡ Latency: {result['pipeline_metadata']['total_latency_ms']}ms")
    return result

def main():
    # Case 1: Struggling Beginner (Biological Classification)
    case1_profile = {
        "grade": "11",
        "exam_target": "neet",
        "subject": "Biology",
        "expertise_level": 1.5,
        "weak_topics": ["Biological Classification", "Monera"],
        "strong_topics": []
    }
    case1_chat = [{"role": "student", "message": "I'm confused about the 5 kingdoms. Can we start easy?"}]
    case1_perf = {
        "topic": "Biological Classification",
        "questions_attempted": 3,
        "correct": 1,
        "avg_time_seconds": 120,
        "confidence_score": 1.2
    }

    # Case 2: Advanced Student (Photosynthesis Challenge)
    case2_profile = {
        "grade": "12",
        "exam_target": "neet",
        "subject": "Biology",
        "expertise_level": 4.5,
        "weak_topics": [],
        "strong_topics": ["Photosynthesis", "Plant Physiology"]
    }
    case2_chat = [{"role": "student", "message": "I want harder problems on Calvin Cycle. Push me."}]
    case2_perf = {
        "topic": "Photosynthesis",
        "questions_attempted": 10,
        "correct": 9,
        "avg_time_seconds": 300,
        "confidence_score": 4.8
    }

    # Case 3: Mid-Session Context (Respiration)
    case3_profile = {
        "grade": "11",
        "exam_target": "neet",
        "subject": "Biology",
        "expertise_level": 3.0,
        "weak_topics": ["Respiration in Plants"],
        "strong_topics": ["Morphology"]
    }
    case3_chat = [
        {"role": "student", "message": "I keep forgetting the enzymes in Glycolysis."},
        {"role": "tutor", "message": "Let's review the investment phase."},
        {"role": "student", "message": "Also the ATP yield calculation confuses me."}
    ]
    case3_perf = {
        "topic": "Respiration in Plants",
        "questions_attempted": 5,
        "correct": 2,
        "avg_time_seconds": 150,
        "confidence_score": 2.2
    }

    run_test_case("Struggling Beginner (Bio Classification)", case1_profile, case1_chat, case1_perf)
    run_test_case("Advanced Challenge (Photosynthesis)", case2_profile, case2_chat, case2_perf)
    run_test_case("Mid-Session Context (Respiration)", case3_profile, case3_chat, case3_perf)

if __name__ == "__main__":
    main()
