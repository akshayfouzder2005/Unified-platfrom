from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "AI-Driven Unified Data Platform"
    environment: str = "dev"

    # Database settings
    database_url: str = "sqlite:///./test.db"
    postgres_user: str = "ocean"
    postgres_password: str = "change-me"
    postgres_db: str = "ocean"
    postgres_host: str = "localhost"
    postgres_port: int = 5432

    # Authentication settings
    SECRET_KEY: str = "your-secret-key-change-this-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()
