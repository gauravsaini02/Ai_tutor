import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Load environment variables
load_dotenv()

def main():
    """Create payload indexes for efficient filtering at scale."""
    
    # Get Qdrant connection details
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    collection_name = os.getenv("QDRANT_COLLECTION_NAME", "questions")
    
    if not qdrant_url or not qdrant_api_key:
        print("Error: QDRANT_URL or QDRANT_API_KEY not set in .env file")
        return
    
    # Connect to Qdrant
    client = QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key,
    )
    
    print("=" * 80)
    print("CREATING PAYLOAD INDEXES FOR SCALE (1M+ QUESTIONS)")
    print("=" * 80)
    print(f"\nCollection: {collection_name}\n")
    
    # Define indexes to create
    indexes = [
        {
            "field_name": "subject",
            "field_schema": models.PayloadSchemaType.KEYWORD,
            "description": "Subject name (Biology, Physics, Chemistry)"
        },
        {
            "field_name": "topic",
            "field_schema": models.PayloadSchemaType.KEYWORD,
            "description": "Topic name (Gravitation, Photorespiration, etc.)"
        },
        {
            "field_name": "sub_topic",
            "field_schema": models.PayloadSchemaType.KEYWORD,
            "description": "Sub-topic name"
        },
        {
            "field_name": "difficulty",
            "field_schema": models.PayloadSchemaType.INTEGER,
            "description": "Difficulty level (1-5)"
        },
        {
            "field_name": "exam_type",
            "field_schema": models.PayloadSchemaType.KEYWORD,
            "description": "Exam type (neet, etc.)"
        },
        {
            "field_name": "year",
            "field_schema": models.PayloadSchemaType.INTEGER,
            "description": "Question year"
        }
    ]
    
    # Create each index
    created_count = 0
    skipped_count = 0
    
    for idx in indexes:
        field_name = idx["field_name"]
        field_schema = idx["field_schema"]
        description = idx["description"]
        
        try:
            print(f"Creating index on '{field_name}' ({description})...")
            
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=field_schema
            )
            
            print(f"  ✅ Index created successfully\n")
            created_count += 1
            
        except Exception as e:
            error_msg = str(e)
            if "already exists" in error_msg.lower():
                print(f"  ℹ️  Index already exists, skipping\n")
                skipped_count += 1
            else:
                print(f"  ❌ Error: {error_msg}\n")
    
    print("=" * 80)
    print(f"INDEXING COMPLETE")
    print(f"  - Created: {created_count}")
    print(f"  - Skipped (already exist): {skipped_count}")
    print(f"  - Total: {len(indexes)}")
    print("=" * 80)
    print("\n✅ Your collection is now optimized for 1M+ questions!")
    print("   Fast filtering enabled on: subject, topic, sub_topic, difficulty, exam_type, year")

if __name__ == "__main__":
    main()
