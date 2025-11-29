import asyncio
from qdrant_client import AsyncQdrantClient
from src.config import Config

async def reset_db():
    config = Config()
    client = AsyncQdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)
    
    print(f"Deleting collection: {config.COLLECTION_NAME}")
    await client.delete_collection(config.COLLECTION_NAME)
    print("Collection deleted.")

if __name__ == "__main__":
    asyncio.run(reset_db())
