import json
import hashlib
import redis.asyncio as redis
from typing import Optional, Any, Callable
from functools import wraps
from .config import Config
from .logger import setup_logger

logger = setup_logger(__name__)

class RedisCache:
    def __init__(self, config: Config):
        self.redis = redis.Redis(
            host=config.REDIS_HOST,
            port=config.REDIS_PORT,
            decode_responses=True
        )
        self.ttl = config.CACHE_TTL
        logger.info(f"Redis Cache initialized at {config.REDIS_HOST}:{config.REDIS_PORT}")

    async def get(self, key: str) -> Optional[Any]:
        try:
            data = await self.redis.get(key)
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            logger.error(f"Redis GET error: {e}")
            return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        try:
            await self.redis.set(
                key, 
                json.dumps(value), 
                ex=ttl or self.ttl
            )
        except Exception as e:
            logger.error(f"Redis SET error: {e}")

    async def get_lru(self, key: str) -> Optional[Any]:
        """Get value and promote key to most recent (LRU logic)."""
        try:
            data = await self.redis.get(key)
            if data:
                # Promote to tail (most recent)
                list_key = "recent_keys_tracker"
                await self.redis.lrem(list_key, 0, key)
                await self.redis.rpush(list_key, key)
                return json.loads(data)
            return None
        except Exception as e:
            logger.error(f"Redis GET LRU error: {e}")
            return None

    async def set_lru(self, key: str, value: Any, limit: int = 10):
        """Set value and maintain LRU list size."""
        try:
            # 1. Set the actual data
            await self.set(key, value)
            
            # 2. Update tracker
            list_key = "recent_keys_tracker"
            await self.redis.lrem(list_key, 0, key) # Remove if exists
            await self.redis.rpush(list_key, key)   # Add to tail (most recent)
            
            # 3. Check size and evict
            count = await self.redis.llen(list_key)
            while count > limit:
                # Remove oldest (head)
                oldest_key = await self.redis.lpop(list_key)
                if oldest_key:
                    await self.redis.delete(oldest_key)
                count -= 1
                    
        except Exception as e:
            logger.error(f"Redis SET LRU error: {e}")

    @staticmethod
    def generate_key(*args, **kwargs) -> str:
        """Generate a stable hash key from arguments."""
        key_str = json.dumps(
            {"args": args, "kwargs": kwargs}, 
            sort_keys=True, 
            default=str
        )
        return hashlib.md5(key_str.encode()).hexdigest()

    async def close(self):
        await self.redis.close()
