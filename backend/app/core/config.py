from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "AI-Driven Unified Data Platform"
    environment: str = "dev"

    # Database settings  
    DATABASE_URL: str = "sqlite:///./test.db"
    database_url: str = "sqlite:///./test.db"  # Keep for backward compatibility
    postgres_user: str = "ocean"
    postgres_password: str = "change-me"
    postgres_db: str = "ocean"
    postgres_host: str = "localhost"
    postgres_port: int = 5432

    # Authentication settings
    SECRET_KEY: str = "your-secret-key-change-this-in-production-very-long-and-random-key-123456789"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Keep global instance for backward compatibility
settings = Settings()
