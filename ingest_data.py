import json
import os
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models
from openai import OpenAI

# Load environment variables
load_dotenv()

class Config:
    """Configuration settings loaded from environment variables."""
    
    def __init__(self):
        self.QDRANT_URL: str = os.getenv("QDRANT_URL", "")
        self.QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")
        self.COLLECTION_NAME: str = os.getenv("QDRANT_COLLECTION_NAME", "questions_dataset")
        self.OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
        self.EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
        self.VECTOR_SIZE: int = 3072  # Dimension for text-embedding-3-large

    def validate(self):
        if not self.QDRANT_URL:
            raise ValueError("QDRANT_URL is not set in environment variables.")
        if not self.QDRANT_API_KEY:
            raise ValueError("QDRANT_API_KEY is not set in environment variables.")
        if not self.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set in environment variables.")

class VectorDBManager:
    """Manages interactions with the Qdrant Vector Database."""
    
    def __init__(self, config: Config):
        self.config = config
        self.client = QdrantClient(
            url=self.config.QDRANT_URL,
            api_key=self.config.QDRANT_API_KEY,
        )

    def ensure_collection_exists(self) -> None:
        """Creates the collection if it does not already exist."""
        collections = self.client.get_collections()
        collection_names = [c.name for c in collections.collections]
        
        if self.config.COLLECTION_NAME not in collection_names:
            print(f"Creating collection: {self.config.COLLECTION_NAME}")
            self.client.create_collection(
                collection_name=self.config.COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=self.config.VECTOR_SIZE,
                    distance=models.Distance.COSINE
                )
            )
        else:
            print(f"Collection '{self.config.COLLECTION_NAME}' already exists.")

    def upsert_points(self, points: List[models.PointStruct], batch_size: int = 100) -> None:
        """Upserts points into the collection in batches."""
        total_points = len(points)
        print(f"Starting upsert of {total_points} points...")
        
        for i in range(0, total_points, batch_size):
            batch = points[i:i+batch_size]
            print(f"Upserting batch {i//batch_size + 1}/{(total_points + batch_size - 1)//batch_size}...")
            self.client.upsert(
                collection_name=self.config.COLLECTION_NAME,
                points=batch
            )
        print("Upsert complete.")

class DataProcessor:
    """Handles data loading and processing."""
    
    def __init__(self, api_key: str, model_name: str):
        print(f"Initializing OpenAI client with model: {model_name}...")
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    def load_data(self, file_path: str) -> List[Dict[str, Any]]:
        """Loads JSON data from a file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        with open(file_path, 'r') as f:
            return json.load(f)

    def get_embedding(self, text: str) -> List[float]:
        """Generates embedding using OpenAI API."""
        text = text.replace("\n", " ")
        return self.client.embeddings.create(input=[text], model=self.model_name).data[0].embedding

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
        
        # Prepare metadata
        payload = {
            "question_id": item.get('question_id'),
            "subject": item.get('subject', ''),
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
        
        # Use question_id directly as point_id
        point_id = str(item.get('question_id'))
        
        return models.PointStruct(
            id=point_id,
            vector=embedding,
            payload=payload
        )

    def prepare_points(self, data: List[Dict[str, Any]]) -> List[models.PointStruct]:
        """Converts raw data into a list of PointStructs."""
        print(f"Processing {len(data)} items...")
        return [self.process_item(item) for item in data]

def main():
    # Initialize Config
    config = Config()
    try:
        config.validate()
    except ValueError as e:
        print(f"Configuration Error: {e}")
        return

    # Initialize Components
    db_manager = VectorDBManager(config)
    processor = DataProcessor(config.OPENAI_API_KEY, config.EMBEDDING_MODEL)

    # Execution Flow
    try:
        db_manager.ensure_collection_exists()
        
        data = processor.load_data('data.json')
        points = processor.prepare_points(data)
        
        db_manager.upsert_points(points)
        
        print("Ingestion pipeline finished successfully.")
        
    except Exception as e:
        print(f"An error occurred during ingestion: {e}")

if __name__ == "__main__":
    main()
