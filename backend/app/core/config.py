from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "AI-Driven Unified Data Platform"
    environment: str = "dev"

    postgres_user: str = "ocean"
    postgres_password: str = "change-me"
    postgres_db: str = "ocean"
    postgres_host: str = "localhost"
    postgres_port: int = 5432

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
