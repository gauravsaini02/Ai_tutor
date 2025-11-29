import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

# Load environment variables
load_dotenv()

def extract_subject_from_id(question_id: str) -> str:
    """Extracts subject name from question_id prefix.
    
    Args:
        question_id: Question ID in format 'SUBJECT_TOPIC_YEAR_NUM'
        
    Returns:
        Full subject name (e.g., 'Biology', 'Physics', 'Chemistry')
    """
    if not question_id:
        return ""
    
    # Extract prefix before first underscore
    prefix = question_id.split('_')[0].upper()
    
    # Map prefix to full subject name
    subject_mapping = {
        'BIO': 'Biology',
        'PHY': 'Physics',
        'CHE': 'Chemistry',
        'CHEM': 'Chemistry',
        'MAT': 'Mathematics',
        'MATH': 'Mathematics'
    }
    
    return subject_mapping.get(prefix, "")

def main():
    """Update all existing questions in Qdrant with subject field extracted from question_id."""
    
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
    
    try:
        print("=" * 80)
        print("UPDATING SUBJECT FIELDS IN QDRANT")
        print("=" * 80)
        
        # Get total count
        total_count = client.count(collection_name=collection_name)
        print(f"\nTotal questions to update: {total_count.count}\n")
        
        # Scroll through all points
        offset = None
        updated_count = 0
        batch_size = 100
        
        while True:
            # Fetch a batch of points
            points, next_offset = client.scroll(
                collection_name=collection_name,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
            
            if not points:
                break
            
            # Update each point with subject
            for point in points:
                question_id = point.payload.get('question_id', '')
                current_subject = point.payload.get('subject', '')
                
                # Extract subject from question_id
                new_subject = extract_subject_from_id(question_id)
                
                # Only update if subject is missing or different
                if new_subject and new_subject != current_subject:
                    # Update the payload
                    client.set_payload(
                        collection_name=collection_name,
                        payload={"subject": new_subject},
                        points=[point.id]
                    )
                    updated_count += 1
                    print(f"Updated {question_id}: subject = '{new_subject}'")
                elif current_subject:
                    print(f"Skipped {question_id}: subject already set to '{current_subject}'")
                else:
                    print(f"Warning {question_id}: could not extract subject from ID")
            
            # Check if we've processed all points
            if next_offset is None:
                break
            
            offset = next_offset
        
        print("\n" + "=" * 80)
        print(f"UPDATE COMPLETE: {updated_count} questions updated")
        print("=" * 80)
        
    except Exception as e:
        print(f"Error updating Qdrant database: {e}")

if __name__ == "__main__":
    main()
