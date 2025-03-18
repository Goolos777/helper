"""
Caching utilities for improved performance.
Supports both in-memory and Redis-based caching strategies.
"""
import json
import time
import hashlib
from typing import Any, Dict, Optional, Callable, Union, TypeVar, Tuple
from functools import wraps
import asyncio
from contextlib import suppress

from ..config import settings
from ..utils.logger import get_logger

logger = get_logger("cache")

# Type variable for generic function return types
T = TypeVar('T')

# Simple in-memory cache
_memory_cache: Dict[str, Tuple[Any, float]] = {}

# Optional Redis integration
redis_client = None
if settings.CACHE_TYPE == "redis" and settings.REDIS_URL:
    try:
        import redis
        from redis.exceptions import RedisError

        logger.info(f"Initializing Redis cache with URL: {settings.REDIS_URL}")
        redis_client = redis.from_url(settings.REDIS_URL)
        # Test connection
        redis_client.ping()
        logger.info("Successfully connected to Redis")
    except ImportError:
        logger.warning("Redis package not installed. Using in-memory cache instead.")
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {e}", exc_info=True)
        redis_client = None


def _generate_cache_key(prefix: str, *args, **kwargs) -> str:
    """
    Generate a unique cache key based on function arguments.

    Args:
        prefix: A prefix for the cache key (usually function name)
        *args: Positional arguments
        **kwargs: Keyword arguments

    Returns:
        A unique hash-based string to use as a cache key
    """
    # Convert args and kwargs to a stable string representation
    key_parts = [prefix]

    if args:
        key_parts.append(str(args))

    if kwargs:
        # Sort kwargs by key for consistent hashing
        sorted_kwargs = sorted(kwargs.items())
        key_parts.append(str(sorted_kwargs))

    # Join the parts and create a hash
    key_base = "::".join(key_parts)
    return f"{prefix}:{hashlib.md5(key_base.encode()).hexdigest()}"


def _get_from_cache(key: str) -> Optional[Any]:
    """
    Retrieve an item from the appropriate cache.

    Args:
        key: The cache key

    Returns:
        The cached value or None if not found/expired
    """
    # Try Redis first if configured
    if redis_client:
        try:
            data = redis_client.get(key)
            if data:
                logger.debug(f"Cache hit (Redis): {key}")
                return json.loads(data)
            logger.debug(f"Cache miss (Redis): {key}")
        except Exception as e:
            logger.warning(f"Redis error when retrieving key {key}: {e}")

    # Fallback to memory cache
    if key in _memory_cache:
        value, expiry = _memory_cache[key]
        # Check if expired
        if expiry == 0 or expiry > time.time():
            logger.debug(f"Cache hit (memory): {key}")
            return value
        # Remove expired item
        del _memory_cache[key]

    logger.debug(f"Cache miss (memory): {key}")
    return None


def _set_in_cache(key: str, value: Any, ttl: int = 0) -> None:
    """
    Store an item in the appropriate cache.

    Args:
        key: The cache key
        value: The value to store
        ttl: Time to live in seconds (0 means no expiry)
    """
    # JSON serialization may fail for complex objects
    try:
        serialized = json.dumps(value)
    except (TypeError, ValueError):
        logger.warning(f"Failed to serialize value for key {key}")
        return

    # Try Redis first if configured
    if redis_client:
        try:
            redis_client.set(key, serialized, ex=ttl if ttl > 0 else None)
            logger.debug(f"Stored in Redis cache: {key}, TTL: {ttl}s")
        except Exception as e:
            logger.warning(f"Redis error when storing key {key}: {e}")

    # Always store in memory cache as well for fastest retrieval
    expiry = time.time() + ttl if ttl > 0 else 0
    _memory_cache[key] = (value, expiry)
    logger.debug(f"Stored in memory cache: {key}, TTL: {ttl}s")


def _delete_from_cache(key: str) -> None:
    """
    Delete an item from all caches.

    Args:
        key: The cache key to delete
    """
    # Remove from Redis if configured
    if redis_client:
        try:
            redis_client.delete(key)
            logger.debug(f"Deleted from Redis cache: {key}")
        except Exception as e:
            logger.warning(f"Redis error when deleting key {key}: {e}")

    # Remove from memory cache
    with suppress(KeyError):
        del _memory_cache[key]
        logger.debug(f"Deleted from memory cache: {key}")


def cache(ttl: int = 300, prefix: Optional[str] = None, skip_args: int = 0):
    """
    Cache decorator for regular functions.

    Args:
        ttl: Time to live in seconds (default: 5 minutes)
        prefix: Custom prefix for cache keys (default: function name)
        skip_args: Number of initial args to skip when creating cache key (e.g., 'self')

    Returns:
        Decorated function with caching
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        cache_prefix = prefix or func.__name__

        @wraps(func)
        def wrapper(*args, **kwargs):
            if not settings.ENABLE_CACHE:
                return func(*args, **kwargs)

            # Skip specified args (e.g., 'self')
            cache_args = args[skip_args:] if skip_args else args

            # Generate cache key
            key = _generate_cache_key(cache_prefix, *cache_args, **kwargs)

            # Try to get from cache
            cached_value = _get_from_cache(key)
            if cached_value is not None:
                return cached_value

            # Not in cache, calculate the result
            start_time = time.time()
            result = func(*args, **kwargs)
            duration = time.time() - start_time

            # Store in cache
            _set_in_cache(key, result, ttl)
            logger.debug(f"Cached result for {func.__name__} (took {duration:.2f}s)")

            return result

        # Add clear cache method to the function
        def clear_cache_for(*args, **kwargs):
            cache_args = args[skip_args:] if skip_args else args
            key = _generate_cache_key(cache_prefix, *cache_args, **kwargs)
            _delete_from_cache(key)

        wrapper.clear_cache = clear_cache_for

        return wrapper

    return decorator


def async_cache(ttl: int = 300, prefix: Optional[str] = None, skip_args: int = 0):
    """
    Cache decorator for async functions.

    Args:
        ttl: Time to live in seconds (default: 5 minutes)
        prefix: Custom prefix for cache keys (default: function name)
        skip_args: Number of initial args to skip when creating cache key (e.g., 'self')

    Returns:
        Decorated async function with caching
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        cache_prefix = prefix or func.__name__

        @wraps(func)
        async def wrapper(*args, **kwargs):
            if not settings.ENABLE_CACHE:
                return await func(*args, **kwargs)

            # Skip specified args (e.g., 'self')
            cache_args = args[skip_args:] if skip_args else args

            # Generate cache key
            key = _generate_cache_key(cache_prefix, *cache_args, **kwargs)

            # Try to get from cache
            cached_value = _get_from_cache(key)
            if cached_value is not None:
                return cached_value

            # Not in cache, calculate the result
            start_time = time.time()
            result = await func(*args, **kwargs)
            duration = time.time() - start_time

            # Store in cache
            _set_in_cache(key, result, ttl)
            logger.debug(f"Cached result for {func.__name__} (took {duration:.2f}s)")

            return result

        # Add clear cache method to the function
        def clear_cache_for(*args, **kwargs):
            cache_args = args[skip_args:] if skip_args else args
            key = _generate_cache_key(cache_prefix, *cache_args, **kwargs)
            _delete_from_cache(key)

        wrapper.clear_cache = clear_cache_for

        return wrapper

    return decorator


def clear_all_cache():
    """Clear the entire cache."""
    global _memory_cache

    # Clear memory cache
    _memory_cache = {}
    logger.info("In-memory cache cleared")

    # Clear Redis cache if configured
    if redis_client:
        try:
            redis_client.flushdb()
            logger.info("Redis cache cleared")
        except Exception as e:
            logger.error(f"Failed to clear Redis cache: {e}", exc_info=True)


def get_cache_stats() -> Dict[str, Any]:
    """
    Get statistics about the current cache usage.

    Returns:
        Dictionary with cache statistics
    """
    stats = {
        "memory_cache_size": len(_memory_cache),
        "memory_cache_keys": list(_memory_cache.keys())[:10],  # First 10 keys only
    }

    if redis_client:
        try:
            stats["redis_keys_count"] = redis_client.dbsize()
            stats["redis_info"] = {
                k: v for k, v in redis_client.info().items()
                if k in ["used_memory_human", "connected_clients", "uptime_in_seconds"]
            }
        except Exception as e:
            logger.error(f"Failed to get Redis stats: {e}", exc_info=True)
            stats["redis_error"] = str(e)

    return stats