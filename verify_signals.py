import asyncio
import os
from dotenv import load_dotenv
from src.retrieval import ChatHistoryParser, ChatMessage
from sentence_transformers import SentenceTransformer
from src.config import Config

load_dotenv()

def test_signals():
    config = Config()
    
    print(f"Loading model: {config.EMBEDDING_MODEL}...")
    model = SentenceTransformer(config.EMBEDDING_MODEL)
    print("Model loaded.")
    
    test_cases = [
        # --- Struggling (30 cases) ---
        ([ChatMessage("student", "I am completely lost.")], "struggling"),
        ([ChatMessage("student", "This is going over my head.")], "struggling"),
        ([ChatMessage("student", "I don't have a clue what's happening.")], "struggling"),
        ([ChatMessage("student", "Can you break this down simpler?")], "struggling"),
        ([ChatMessage("student", "I'm stuck on the first step.")], "struggling"),
        ([ChatMessage("student", "This is gibberish to me.")], "struggling"),
        ([ChatMessage("student", "I keep getting the wrong answer.")], "struggling"),
        ([ChatMessage("student", "Why is this so hard?")], "struggling"),
        ([ChatMessage("student", "I'm about to give up.")], "struggling"),
        ([ChatMessage("student", "My mind is blank.")], "struggling"),
        ([ChatMessage("student", "I don't get the logic here.")], "struggling"),
        ([ChatMessage("student", "This is too advanced for me.")], "struggling"),
        ([ChatMessage("student", "I need a hint, please.")], "struggling"),
        ([ChatMessage("student", "I'm confused by the formula.")], "struggling"),
        ([ChatMessage("student", "It's not sinking in.")], "struggling"),
        ([ChatMessage("student", "I feel dumb right now.")], "struggling"),
        ([ChatMessage("student", "Can we go back to basics?")], "struggling"),
        ([ChatMessage("student", "I'm drowning in these equations.")], "struggling"),
        ([ChatMessage("student", "Nothing makes sense.")], "struggling"),
        ([ChatMessage("student", "I'm hitting a brick wall.")], "struggling"),
        ([ChatMessage("student", "I can't wrap my head around this.")], "struggling"),
        ([ChatMessage("student", "This is frustratingly difficult.")], "struggling"),
        ([ChatMessage("student", "I'm lost in the details.")], "struggling"),
        ([ChatMessage("student", "Can you explain it like I'm 5?")], "struggling"),
        ([ChatMessage("student", "I'm struggling to keep up.")], "struggling"),
        ([ChatMessage("student", "This concept is alien to me.")], "struggling"),
        ([ChatMessage("student", "I'm not getting anywhere with this.")], "struggling"),
        ([ChatMessage("student", "Help, I'm stuck.")], "struggling"),
        ([ChatMessage("student", "I don't know where to start.")], "struggling"),
        ([ChatMessage("student", "This is way above my pay grade.")], "struggling"),

        # --- Bored (25 cases) ---
        ([ChatMessage("student", "This is too easy, honestly.")], "bored"),
        ([ChatMessage("student", "I learned this in 5th grade.")], "bored"),
        ([ChatMessage("student", "Can we skip to the good part?")], "bored"),
        ([ChatMessage("student", "Boring.")], "bored"),
        ([ChatMessage("student", "I'm falling asleep here.")], "bored"),
        ([ChatMessage("student", "Give me something I don't know.")], "bored"),
        ([ChatMessage("student", "This is trivial.")], "bored"),
        ([ChatMessage("student", "Waste of time.")], "bored"),
        ([ChatMessage("student", "Next topic, please.")], "bored"),
        ([ChatMessage("student", "I'm not learning anything new.")], "bored"),
        ([ChatMessage("student", "This is basic stuff.")], "bored"),
        ([ChatMessage("student", "I've done this a thousand times.")], "bored"),
        ([ChatMessage("student", "Can we move faster?")], "bored"),
        ([ChatMessage("student", "This is too slow.")], "bored"),
        ([ChatMessage("student", "I'm bored out of my mind.")], "bored"),
        ([ChatMessage("student", "Is this a joke? It's so easy.")], "bored"),
        ([ChatMessage("student", "Tell me something I don't know.")], "bored"),
        ([ChatMessage("student", "This is elementary.")], "bored"),
        ([ChatMessage("student", "Let's skip the basics.")], "bored"),
        ([ChatMessage("student", "I'm way past this level.")], "bored"),
        ([ChatMessage("student", "This is not challenging at all.")], "bored"),
        ([ChatMessage("student", "Yawn.")], "bored"),
        ([ChatMessage("student", "Can we do something else?")], "bored"),
        ([ChatMessage("student", "This is repetitive.")], "bored"),
        ([ChatMessage("student", "I'm checking out, this is dull.")], "bored"),

        # --- Ready for More (25 cases) ---
        ([ChatMessage("student", "Hit me with a hard one.")], "ready_for_more"),
        ([ChatMessage("student", "I want to test my mastery.")], "ready_for_more"),
        ([ChatMessage("student", "Give me a JEE Advanced level question.")], "ready_for_more"),
        ([ChatMessage("student", "Too simple, increase the level.")], "ready_for_more"),
        ([ChatMessage("student", "I need a challenge.")], "ready_for_more"),
        ([ChatMessage("student", "Let's do something complex.")], "ready_for_more"),
        ([ChatMessage("student", "Show me the toughest problem you have.")], "ready_for_more"),
        ([ChatMessage("student", "I'm ready for the next level.")], "ready_for_more"),
        ([ChatMessage("student", "Challenge me.")], "ready_for_more"),
        ([ChatMessage("student", "This is child's play, give me real work.")], "ready_for_more"),
        ([ChatMessage("student", "I want to sweat.")], "ready_for_more"),
        ([ChatMessage("student", "Give me a brain teaser.")], "ready_for_more"),
        ([ChatMessage("student", "I want to push my limits.")], "ready_for_more"),
        ([ChatMessage("student", "Let's ramp it up.")], "ready_for_more"),
        ([ChatMessage("student", "Give me the hardest one.")], "ready_for_more"),
        ([ChatMessage("student", "I'm bored of easy questions, give me hard ones.")], "ready_for_more"), # Tricky: contains "bored" but means ready_for_more
        ([ChatMessage("student", "Test me.")], "ready_for_more"),
        ([ChatMessage("student", "I want to go deeper.")], "ready_for_more"),
        ([ChatMessage("student", "Give me a complex scenario.")], "ready_for_more"),
        ([ChatMessage("student", "I'm ready to level up.")], "ready_for_more"),
        ([ChatMessage("student", "Throw everything you've got at me.")], "ready_for_more"),
        ([ChatMessage("student", "Make it difficult.")], "ready_for_more"),
        ([ChatMessage("student", "I want to solve the impossible.")], "ready_for_more"),
        ([ChatMessage("student", "Give me an Olympiad level problem.")], "ready_for_more"),
        ([ChatMessage("student", "I want to struggle (in a good way).")], "ready_for_more"),

        # --- Neutral (20 cases) ---
        ([ChatMessage("student", "What is the atomic weight of Carbon?")], "neutral"),
        ([ChatMessage("student", "Explain the second law of thermodynamics.")], "neutral"),
        ([ChatMessage("student", "I have a test on Monday.")], "neutral"),
        ([ChatMessage("student", "Let's solve this.")], "neutral"),
        ([ChatMessage("student", "Okay.")], "neutral"),
        ([ChatMessage("student", "Thanks.")], "neutral"),
        ([ChatMessage("student", "How many questions are left?")], "neutral"),
        ([ChatMessage("student", "Is the answer 42?")], "neutral"),
        ([ChatMessage("student", "Wait a minute.")], "neutral"),
        ([ChatMessage("student", "I am writing this down.")], "neutral"),
        ([ChatMessage("student", "Can you repeat the question?")], "neutral"),
        ([ChatMessage("student", "I think the answer is B.")], "neutral"),
        ([ChatMessage("student", "What topic is this?")], "neutral"),
        ([ChatMessage("student", "Let me think.")], "neutral"),
        ([ChatMessage("student", "Hold on.")], "neutral"),
        ([ChatMessage("student", "Interesting.")], "neutral"),
        ([ChatMessage("student", "Go on.")], "neutral"),
        ([ChatMessage("student", "I see.")], "neutral"),
        ([ChatMessage("student", "Got it.")], "neutral"),
        ([ChatMessage("student", "Next.")], "neutral"), # Ambiguous, could be bored, but usually neutral navigation
    ]

    print(f"Testing Semantic Signal Detection (100 Cases)")
    print("-" * 60)
    print(f"{'Input Message':<50} | {'Expected':<15} | {'Detected':<15} | {'Result'}")
    print("-" * 60)

    correct_count = 0
    for history, expected in test_cases:
        # Synchronous call
        signals = ChatHistoryParser.detect_performance_signals_semantic(history, model)
        detected = signals.get("performance", "neutral")
        
        is_correct = detected == expected
        if is_correct:
            correct_count += 1
            result_icon = "✅"
        else:
            result_icon = "❌"
            
        print(f"{history[0].message[:47]:<50} | {expected:<15} | {detected:<15} | {result_icon}")

    print("-" * 60)
    print(f"Accuracy: {correct_count}/{len(test_cases)} ({correct_count/len(test_cases)*100:.1f}%)")

if __name__ == "__main__":
    test_signals()
