from redis import Redis

from config.settings import settings


class RedisConfig:
    _instance = None
    _redis = None

    @classmethod
    def get_instance(cls) -> Redis:
        """
        Get the singleton Redis instance.
        If it doesn't exist, create it.
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._redis

    def __init__(self):
        if RedisConfig._instance is not None:
            raise Exception("This class is a singleton!")
        else:
            RedisConfig._instance = self
            RedisConfig._redis = Redis.from_url(url=str(settings.redis_url))


# Create a global instance
redis_client = RedisConfig.get_instance()
