import json
import os
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer

from .config import Config
from .logger import setup_logger

logger = setup_logger(__name__)

# Load environment variables
load_dotenv()

class VectorDBManager:
    """Manages interactions with the Qdrant Vector Database."""
    
    def __init__(self, config: Config):
        self.config = config
        if self.config.QDRANT_URL.startswith("path:"):
            path = self.config.QDRANT_URL.split(":", 1)[1]
            self.client = QdrantClient(path=path)
        else:
            self.client = QdrantClient(
                url=self.config.QDRANT_URL,
                api_key=self.config.QDRANT_API_KEY,
            )

    def ensure_collection_exists(self) -> None:
        """Creates the collection if it does not already exist."""
        collections = self.client.get_collections()
        collection_names = [c.name for c in collections.collections]
        
        if self.config.COLLECTION_NAME in collection_names:
            logger.info(f"Deleting existing collection: {self.config.COLLECTION_NAME}")
            self.client.delete_collection(self.config.COLLECTION_NAME)
        
        logger.info(f"Creating collection: {self.config.COLLECTION_NAME}")
        self.client.create_collection(
            collection_name=self.config.COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=self.config.VECTOR_SIZE,
                distance=models.Distance.COSINE
            )
        )

    def upsert_points(self, points: List[models.PointStruct], batch_size: int = 100) -> None:
        """Upserts points into the collection in batches."""
        total_points = len(points)
        logger.info(f"Starting upsert of {total_points} points...")
        
        for i in range(0, total_points, batch_size):
            batch = points[i:i+batch_size]
            logger.info(f"Upserting batch {i//batch_size + 1}/{(total_points + batch_size - 1)//batch_size}...")
            self.client.upsert(
                collection_name=self.config.COLLECTION_NAME,
                points=batch
            )
        logger.info("Upsert complete.")

    def create_indexes(self) -> None:
        """Creates payload indexes for efficient filtering."""
        logger.info("CREATING PAYLOAD INDEXES")
        
        indexes = [
            {
                "field_name": "subject",
                "field_schema": models.PayloadSchemaType.KEYWORD,
                "description": "Subject name"
            },
            {
                "field_name": "topic",
                "field_schema": models.PayloadSchemaType.TEXT,
                "description": "Topic name"
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
                "description": "Exam type"
            },
            {
                "field_name": "year",
                "field_schema": models.PayloadSchemaType.INTEGER,
                "description": "Question year"
            }
        ]
        
        for idx in indexes:
            field_name = idx["field_name"]
            field_schema = idx["field_schema"]
            description = idx["description"]
            
            try:
                logger.info(f"Creating index on '{field_name}' ({description})...")
                self.client.create_payload_index(
                    collection_name=self.config.COLLECTION_NAME,
                    field_name=field_name,
                    field_schema=field_schema
                )
                logger.info(f"  ✅ Index created successfully")
            except Exception as e:
                error_msg = str(e)
                if "already exists" in error_msg.lower():
                    logger.info(f"  ℹ️  Index already exists, skipping")
                else:
                    logger.error(f"  ❌ Error: {error_msg}")
        logger.info("Index creation complete.")

class DataProcessor:
    """Handles data loading and processing with local embeddings."""
    
    def __init__(self, model_name: str):
        logger.info(f"Loading local embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        logger.info(f"✅ Model loaded successfully!")
    
    SUBJECT_MAPPING = {
        'BIO': 'Biology',
        'PHY': 'Physics',
        'CHE': 'Chemistry',
        'CHEM': 'Chemistry',
        'MAT': 'Mathematics',
        'MATH': 'Mathematics'
    }

    @staticmethod
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
        
        return DataProcessor.SUBJECT_MAPPING.get(prefix, "")

    def load_data(self, file_path: str) -> List[Dict[str, Any]]:
        """Loads JSON data from a file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        with open(file_path, 'r') as f:
            return json.load(f)

    def get_embedding(self, text: str) -> List[float]:
        """Generates embedding using local sentence-transformers model."""
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def process_item(self, item: Dict[str, Any]) -> models.PointStruct:
        """Processes a single data item into a Qdrant PointStruct."""
        # Construct text to embed
        options_str = ", ".join(item.get('options', []))
        text_to_embed = (
            f"{item.get('question_text', '')}\n"
            f"Options: {options_str}\n"
            f"Explanation: {item.get('explanation', '')}\n"
            f"Topic: {item.get('topic', '')}\n"
            f"Subtopic: {item.get('sub_topic', '')}"
        )
        
        # Generate embedding
        embedding = self.get_embedding(text_to_embed)
        
        # Extract subject from question_id
        question_id = item.get('question_id', '')
        subject = self.extract_subject_from_id(question_id)
        
        # Prepare metadata
        payload = {
            "question_id": question_id,
            "subject": subject,
            "topic": item.get('topic'),
            "sub_topic": item.get('sub_topic'),
            "year": item.get('year'),
            "difficulty": item.get('difficulty'),
            "exam_type": item.get('exam_type'),
            "time_estimate": item.get('time_estimate'),
            "prerequisites": item.get('prerequisites'),
            "options": item.get('options'),
            "explanation": item.get('explanation'),
            "source_pdf": item.get('source_pdf', ''),
            "source_chapter": item.get('source_chapter', ''),
            "question_text": item.get('question_text')
        }
        
        # Generate UUID from question_id for consistent point IDs
        import uuid
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, question_id))
        
        return models.PointStruct(
            id=point_id,
            vector=embedding,
            payload=payload
        )

    def prepare_points(self, data: List[Dict[str, Any]]) -> List[models.PointStruct]:
        """Converts raw data into a list of PointStructs."""
        logger.info(f"Processing {len(data)} items...")
        return [self.process_item(item) for item in data]

def main():
    # Initialize Config
    config = Config()
    try:
        config.validate()
    except ValueError as e:
        logger.error(f"Configuration Error: {e}")
        return

    # Initialize Components
    db_manager = VectorDBManager(config)
    processor = DataProcessor(config.EMBEDDING_MODEL)

    # Execution Flow
    try:
        db_manager.ensure_collection_exists()
        db_manager.create_indexes()  # Create indexes before upserting
        
        data = processor.load_data('data/data.json')
        points = processor.prepare_points(data)
        
        db_manager.upsert_points(points)
        
        db_manager.upsert_points(points)
        
        logger.info("Ingestion pipeline finished successfully.")
        
    except Exception as e:
        logger.error(f"An error occurred during ingestion: {e}", exc_info=True)

if __name__ == "__main__":
    main()
