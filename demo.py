import time
import asyncio
from src.orchestrator import TutorOrchestrator
from src.retrieval import UserProfile, ChatMessage, RecentPerformance

async def run_test_case(orchestrator, name, profile_dict, chat_list, perf_dict):
    print(f"\n{'#'*80}")
    print(f"TEST CASE: {name}")
    print(f"{'#'*80}")
    
    # Convert dicts to objects for display (orchestrator handles conversion internally too)
    print(f"User: Grade {profile_dict['grade']}, {profile_dict['exam_target']}, {profile_dict['subject']}")
    print(f"Expertise: {profile_dict['expertise_level']}/5 | Weak: {profile_dict['weak_topics']}")
    print(f"Chat Context: \"{chat_list[-1]['message']}\"")
    
    start = time.time()
    result = await orchestrator.recommend(profile_dict, chat_list, perf_dict)
    latency = (time.time() - start) * 1000
    
    print(f"\n📚 Top 3 Recommendations:")
    for i, q in enumerate(result['recommended_questions'][:3], 1):
        print(f"{i}. [{q['difficulty_score']}/5] {q['topic']} - {q['subtopic']}")
        print(f"   Reasoning: {q['reasoning']}")
    
    print(f"\n⚡ Latency: {result['pipeline_metadata']['total_latency_ms']}ms")
    return result

async def main():
    print("Initializing Orchestrator (loading models)...")
    orchestrator = TutorOrchestrator()
    print("Orchestrator ready!\n")

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

    # Case 3: Chemistry Concept (Structure of Atom)
    case3_profile = {
        "grade": "11",
        "exam_target": "neet",
        "subject": "Chemistry",
        "expertise_level": 2.5,
        "weak_topics": ["Structure of Atom"],
        "strong_topics": []
    }
    case3_chat = [
        {"role": "student", "message": "I keep getting the Bohr radius and energy formulas mixed up for ions like He+."}
    ]
    case3_perf = {
        "topic": "Structure of Atom",
        "questions_attempted": 6,
        "correct": 2,
        "avg_time_seconds": 180,
        "confidence_score": 2.0
    }

    # Case 4: Physics Challenge (Gravitation)
    case4_profile = {
        "grade": "11",
        "exam_target": "neet",
        "subject": "Physics",
        "expertise_level": 3.5,
        "weak_topics": ["Gravitation"],
        "strong_topics": ["Kinematics"]
    }
    case4_chat = [{"role": "student", "message": "I keep getting confused with the variation of g with height and depth."}]
    case4_perf = {
        "topic": "Gravitation",
        "questions_attempted": 8,
        "correct": 4,
        "avg_time_seconds": 200,
        "confidence_score": 2.5
    }

    # Case 5: Chemistry Practice (Solutions)
    case5_profile = {
        "grade": "12",
        "exam_target": "neet",
        "subject": "Chemistry",
        "expertise_level": 4.0,
        "strong_topics": ["Solutions", "Electrochemistry"],
        "weak_topics": []
    }
    case5_chat = [{"role": "student", "message": "Let's solve some numericals on Colligative Properties."}]
    case5_perf = {
        "topic": "Solutions",
        "questions_attempted": 12,
        "correct": 10,
        "avg_time_seconds": 250,
        "confidence_score": 4.0
    }

    await run_test_case(orchestrator, "Struggling Beginner (Bio Classification)", case1_profile, case1_chat, case1_perf)
    await run_test_case(orchestrator, "Advanced Challenge (Photosynthesis)", case2_profile, case2_chat, case2_perf)
    await run_test_case(orchestrator, "Chemistry Concept (Structure of Atom)", case3_profile, case3_chat, case3_perf)
    await run_test_case(orchestrator, "Physics Challenge (Gravitation)", case4_profile, case4_chat, case4_perf)
    await run_test_case(orchestrator, "Chemistry Practice (Solutions)", case5_profile, case5_chat, case5_perf)

if __name__ == "__main__":
    asyncio.run(main())
