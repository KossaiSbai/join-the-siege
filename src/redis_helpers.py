
import json
import logging
import os
from typing import List

import redis

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)


REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)

redis_client = None
try:
    redis_client = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        password=REDIS_PASSWORD,
        decode_responses=True
    )
    redis_client.ping()
    logging.info(f"Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
except redis.exceptions.ConnectionError as e:
    logging.error(f"Could not connect to Redis: {e}")
    raise e


def get_cached_value(redis_client, redis_key: str) -> List[str]:
    try:
        cached_descriptions = redis_client.get(redis_key)
        if cached_descriptions:
            return json.loads(cached_descriptions)
    except redis.exceptions.RedisError as e:
        logging.error(f"Redis error while accessing '{redis_key}': {e}")
    return []

def cache_value(redis_client, redis_key: str, value: List[str]) -> None:
    try:
        redis_client.set(redis_key, json.dumps(value))
        logging.info(f"Cached value for key '{redis_key}'.")
    except redis.exceptions.RedisError as e:
        logging.error(f"Redis error while caching descriptions for '{redis_key}': {e}")

