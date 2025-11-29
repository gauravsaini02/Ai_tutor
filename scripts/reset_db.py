"""
Reset and re-create the Qdrant collection with correct vector dimensions.
This script will delete the existing collection and create a new one with proper OpenAI embedding dimensions.
"""

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models

load_dotenv()

def main():
    # Get configuration
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    collection_name = os.getenv("QDRANT_COLLECTION_NAME", "questions")
    
    # Using all-MiniLM-L6-v2 (384 dimensions) - local model, much faster!
    # No API calls, ~50-100ms latency vs 1200ms+ for OpenAI
    vector_size = 384  # all-MiniLM-L6-v2
    
    if not qdrant_url or not qdrant_api_key:
        print("❌ Error: QDRANT_URL or QDRANT_API_KEY not set")
        exit(1)
    
    # Connect to Qdrant
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    
    print("="*80)
    print("RESETTING QDRANT COLLECTION")
    print("="*80)
    print(f"Collection: {collection_name}")
    print(f"New vector size: {vector_size} (sentence-transformers all-MiniLM-L6-v2)")
    print("="*80)
    
    # Check if collection exists
    collections = client.get_collections()
    collection_names = [c.name for c in collections.collections]
    
    if collection_name in collection_names:
        # Get current info
        info = client.get_collection(collection_name)
        current_size = info.config.params.vectors.size
        current_count = info.points_count
        
        print(f"\n⚠️  Existing collection found:")
        print(f"   - Current vector size: {current_size}")
        print(f"   - Current points: {current_count}")
        
        # Confirm deletion
        response = input(f"\n❓ Delete and recreate '{collection_name}'? (yes/no): ")
        if response.lower() != 'yes':
            print("❌ Aborted.")
            exit(0)
        
        # Delete collection
        print(f"\n🗑️  Deleting collection '{collection_name}'...")
        client.delete_collection(collection_name)
        print("✅ Deleted.")
    
    # Create new collection with correct dimensions
    print(f"\n📦 Creating collection '{collection_name}' with vector size {vector_size}...")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=models.Distance.COSINE
        )
    )
    print("✅ Collection created successfully!")
    
    # Create indexes for efficient filtering
    print("\n🔍 Creating payload indexes...")
    indexes = [
        ("subject", models.PayloadSchemaType.KEYWORD),
        ("topic", models.PayloadSchemaType.KEYWORD),
        ("sub_topic", models.PayloadSchemaType.KEYWORD),
        ("difficulty", models.PayloadSchemaType.INTEGER),
        ("exam_type", models.PayloadSchemaType.KEYWORD),
        ("year", models.PayloadSchemaType.INTEGER)
    ]
    
    for field_name, field_schema in indexes:
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=field_schema
            )
            print(f"  ✅ Created index on '{field_name}'")
        except Exception as e:
            print(f"  ⚠️  Index '{field_name}': {e}")
    
    print("\n" + "="*80)
    print("✅ COLLECTION RESET COMPLETE!")
    print("="*80)
    print(f"\nNext steps:")
    print(f"1. Run: python ingest_data.py")
    print(f"   This will populate the collection with embeddings using all-MiniLM-L6-v2 (local, fast!)")
    print("="*80)

if __name__ == "__main__":
    main()
