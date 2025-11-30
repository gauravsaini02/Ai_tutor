import asyncio
import httpx
import json
import time
import os

API_URL = "http://localhost:8000/recommend"

async def run_test_case(client, name, profile_dict, chat_list, perf_dict):
    print(f"\n{'#'*80}")
    print(f"TEST CASE: {name}")
    print(f"{'#'*80}")
    
    print(f"User: Grade {profile_dict['grade']}, {profile_dict['exam_target']}, {profile_dict['subject']}")
    print(f"Expertise: {profile_dict['expertise_level']}/5 | Weak: {profile_dict['weak_topics']}")
    print(f"Chat Context: \"{chat_list[-1]['message']}\"")
    
    payload = {
        "user_profile": profile_dict,
        "chat_history": chat_list,
        "recent_performance": perf_dict
    }
    
    start = time.time()
    try:
        response = await client.post(API_URL, json=payload, timeout=30.0)
        response.raise_for_status()
        result = response.json()
        latency = (time.time() - start) * 1000
        
        print(f"\n📚 Top 3 Recommendations:")
        for i, q in enumerate(result['recommended_questions'][:3], 1):
            print(f"{i}. [{q['difficulty_score']}/5] {q['topic']} - {q['subtopic']}")
            print(f"   Reasoning: {q['reasoning']}")
        
        print(f"\n🧠 Identified Gaps: {result['tutor_context']['identified_gaps']}")
        
        pm = result.get('pipeline_metadata', {})
        print(f"⚡ Pipeline Metadata:")
        print(f"   Retrieval Latency: {pm.get('retrieval_latency_ms', 0):.2f}ms")
        print(f"   Ranking Latency: {pm.get('ranking_latency_ms', 0):.2f}ms")
        print(f"   Total Latency: {pm.get('total_latency_ms', 0):.2f}ms")
        print(f"   Retriever: {pm.get('retriever_used', 'unknown')}")
        print(f"   Ranker: {pm.get('ranker_used', 'unknown')}")
        
        # Add test case name to result for clarity in JSON
        result['test_case_name'] = name
        return result
        
    except httpx.HTTPStatusError as e:
        print(f"❌ API Error: {e.response.status_code} - {e.response.text}")
    except httpx.RequestError as e:
        print(f"❌ Connection Error: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")
    return None

async def clear_redis_cache():
    """Clear all keys from Redis cache"""
    import redis.asyncio as redis
    r = redis.Redis(host='localhost', port=6379, decode_responses=True)
    try:
        await r.flushdb()
        print("✅ Redis cache cleared!")
        await r.close()
    except Exception as e:
        print(f"⚠️  Could not clear Redis: {e}")

async def run_all_test_cases(client):
    """Run all 5 test cases and return results"""
    results = []
    
    # 1. Struggling Beginner (Biology - Classification)
    res1 = await run_test_case(
        client,
        "Struggling Beginner (Bio Classification)",
        {
            "grade": "11",
            "exam_target": "neet",
            "subject": "Biology",
            "expertise_level": 1.5,
            "weak_topics": ["Biological Classification", "Monera"],
            "strong_topics": []
        },
        [
            {"role": "tutor", "message": "Let's start with the basics of classification."},
            {"role": "student", "message": "I tried reading the NCERT chapter yesterday."},
            {"role": "student", "message": "But I'm confused about the 5 kingdoms. Can we start easy?"}
        ],
        {
            "topic": "Biological Classification",
            "questions_attempted": 3,
            "correct": 1,
            "avg_time_seconds": 120,
            "confidence_score": 1.2
        }
    )
    if res1: results.append(res1)

    # 2. Advanced Student (Biology - Photosynthesis)
    res2 = await run_test_case(
        client,
        "Advanced Challenge (Photosynthesis)",
        {
            "grade": "12",
            "exam_target": "neet",
            "subject": "Biology",
            "expertise_level": 4.5,
            "weak_topics": [],
            "strong_topics": ["Photosynthesis", "Plant Physiology"]
        },
        [
            {"role": "tutor", "message": "Here is a standard question on C3 plants."},
            {"role": "student", "message": "Solved it. That was trivial."},
            {"role": "student", "message": "I want harder problems on Calvin Cycle. Push me."}
        ],
        {
            "topic": "Photosynthesis",
            "questions_attempted": 10,
            "correct": 9,
            "avg_time_seconds": 300,
            "confidence_score": 4.8
        }
    )
    if res2: results.append(res2)

    # 3. Chemistry Concept (Structure of Atom)
    res3 = await run_test_case(
        client,
        "Chemistry Concept (Structure of Atom)",
        {
            "grade": "11",
            "exam_target": "neet",
            "subject": "Chemistry",
            "expertise_level": 2.5,
            "weak_topics": ["Structure of Atom"],
            "strong_topics": []
        },
        [
            {"role": "tutor", "message": "Do you remember the formula for Hydrogen atom radius?"},
            {"role": "student", "message": "Yes, for Hydrogen it's simple."},
            {"role": "student", "message": "But I keep getting the Bohr radius and energy formulas mixed up for ions like He+."}
        ],
        {
            "topic": "Structure of Atom",
            "questions_attempted": 6,
            "correct": 2,
            "avg_time_seconds": 180,
            "confidence_score": 2.0
        }
    )
    if res3: results.append(res3)

    # 4. Physics Challenge (Gravitation)
    res4 = await run_test_case(
        client,
        "Physics Challenge (Gravitation)",
        {
            "grade": "11",
            "exam_target": "neet",
            "subject": "Physics",
            "expertise_level": 3.5,
            "weak_topics": ["Gravitation"],
            "strong_topics": ["Kinematics"]
        },
        [
            {"role": "tutor", "message": "Let's move on to Gravitation."},
            {"role": "student", "message": "The basic formula F=GmM/r^2 is fine."},
            {"role": "student", "message": "I keep getting confused with the variation of g with height and depth."}
        ],
        {
            "topic": "Gravitation",
            "questions_attempted": 8,
            "correct": 4,
            "avg_time_seconds": 200,
            "confidence_score": 2.5
        }
    )
    if res4: results.append(res4)

    # 5. Chemistry Practice (Solutions)
    res5 = await run_test_case(
        client,
        "Chemistry Practice (Solutions)",
        {
            "grade": "12",
            "exam_target": "neet",
            "subject": "Chemistry",
            "expertise_level": 4.0,
            "weak_topics": [],
            "strong_topics": ["Solutions", "Electrochemistry"]
        },
        [
            {"role": "tutor", "message": "We have covered the theory of Colligative Properties."},
            {"role": "student", "message": "Okay, the concepts are clear."},
            {"role": "student", "message": "Let's solve some numericals on Colligative Properties."}
        ],
        {
            "topic": "Solutions",
            "questions_attempted": 12,
            "correct": 10,
            "avg_time_seconds": 250,
            "confidence_score": 4.0
        }
    )
    if res5: results.append(res5)
    
    return results

async def main():
    print(f"Connecting to AI Tutor API at {API_URL}...")
    
    # Create outputs directory if it doesn't exist
    os.makedirs("outputs", exist_ok=True)
    
    async with httpx.AsyncClient() as client:
        # Check if server is up
        try:
            resp = await client.get("http://localhost:8000/")
            print(f"Server Status: {resp.json()}")
        except Exception:
            print("❌ Could not connect to server. Make sure 'uvicorn src.api:app' is running!")
            return

        # === RUN 1: WITHOUT CACHE ===
        print("\n" + "="*80)
        print("PHASE 1: Running test cases WITHOUT cache (fresh start)")
        print("="*80)
        
        # Clear Redis cache
        await clear_redis_cache()
        
        # Run all test cases
        results_without_cache = await run_all_test_cases(client)
        
        # Save results WITHOUT cache
        output_without_cache = os.path.join("outputs", "output_without_cache.json")
        with open(output_without_cache, "w") as f:
            json.dump(results_without_cache, f, indent=2)
        
        print(f"\n✅ Results WITHOUT cache saved to: {output_without_cache}")
        
        # === RUN 2: WITH CACHE ===
        print("\n" + "="*80)
        print("PHASE 2: Running test cases WITH cache (should hit Redis)")
        print("="*80)
        
        # Run the same test cases again (should hit cache)
        results_with_cache = await run_all_test_cases(client)
        
        # Save results WITH cache
        output_with_cache = os.path.join("outputs", "output_with_cache.json")
        with open(output_with_cache, "w") as f:
            json.dump(results_with_cache, f, indent=2)
        
        print(f"\n✅ Results WITH cache saved to: {output_with_cache}")
        
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"📁 Output without cache: {output_without_cache}")
        print(f"📁 Output with cache: {output_with_cache}")
        print("\nYou can now compare both files to see the difference in latency!")
        print("="*80)

if __name__ == "__main__":
    asyncio.run(main())
