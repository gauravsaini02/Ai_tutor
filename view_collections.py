import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

# Load environment variables
load_dotenv()

def main():
    """View all collections and their data from Qdrant database."""
    
    # Get Qdrant connection details from environment
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    
    if not qdrant_url or not qdrant_api_key:
        print("Error: QDRANT_URL or QDRANT_API_KEY not set in .env file")
        return
    
    # Connect to Qdrant
    client = QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key,
    )
    
    try:
        # Get all collections
        collections = client.get_collections()
        
        print("=" * 80)
        print("QDRANT DATABASE COLLECTIONS")
        print("=" * 80)
        print(f"\nTotal Collections: {len(collections.collections)}\n")
        
        if not collections.collections:
            print("No collections found in the database.")
            return
        
        # Iterate through each collection
        for collection in collections.collections:
            collection_name = collection.name
            print(f"\n{'=' * 80}")
            print(f"COLLECTION: {collection_name}")
            print(f"{'=' * 80}")
            
            # Get collection info
            collection_info = client.get_collection(collection_name)
            print(f"\nCollection Info:")
            print(f"  - Points Count: {collection_info.points_count}")
            
            # Try to get vector config details
            try:
                vector_config = collection_info.config.params.vectors
                if hasattr(vector_config, 'size'):
                    print(f"  - Vector Size: {vector_config.size}")
                    print(f"  - Distance: {vector_config.distance}")
            except AttributeError:
                pass
            
            # Get sample data from the collection
            print(f"\n{'─' * 80}")
            print(f"SAMPLE DATA (First 5 points):")
            print(f"{'─' * 80}\n")
            
            # Scroll to get points (limit to 5 for preview)
            points, _ = client.scroll(
                collection_name=collection_name,
                limit=5,
                with_payload=True,
                with_vectors=False  # Don't show vectors to keep output clean
            )
            
            if not points:
                print("  No data found in this collection.")
            else:
                for idx, point in enumerate(points, 1):
                    print(f"Point #{idx}")
                    print(f"  ID: {point.id}")
                    print(f"  Payload:")
                    
                    # Print each payload field
                    for key, value in point.payload.items():
                        # Truncate long values for readability
                        if isinstance(value, str) and len(value) > 100:
                            display_value = value[:100] + "..."
                        elif isinstance(value, list) and len(str(value)) > 100:
                            display_value = str(value)[:100] + "..."
                        else:
                            display_value = value
                        print(f"    - {key}: {display_value}")
                    print()
            
            # Get total count
            total_count = client.count(collection_name=collection_name)
            print(f"{'─' * 80}")
            print(f"Total points in '{collection_name}': {total_count.count}")
            print(f"{'─' * 80}\n")
        
        print("\n" + "=" * 80)
        print("END OF COLLECTIONS VIEW")
        print("=" * 80)
        
    except Exception as e:
        print(f"Error accessing Qdrant database: {e}")

if __name__ == "__main__":
    main()
