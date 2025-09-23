from sqlalchemy import create_engine, text
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.engine import Engine
import logging

from .config import get_settings

logger = logging.getLogger(__name__)

# Create declarative base for models
Base = declarative_base()

# Get settings
settings = get_settings()

# Use the DATABASE_URL from settings, fallback to constructed PostgreSQL URL
if hasattr(settings, 'DATABASE_URL') and settings.DATABASE_URL:
    DATABASE_URL = settings.DATABASE_URL
else:
    # Fallback to constructed PostgreSQL URL for backward compatibility
    DATABASE_URL = (
        f"postgresql://{settings.postgres_user}:"
        f"{settings.postgres_password}@{settings.postgres_host}:"
        f"{settings.postgres_port}/{settings.postgres_db}"
    )

# Create engine with appropriate configuration
if DATABASE_URL.startswith('sqlite'):
    # SQLite configuration
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
        echo=settings.environment == "dev"
    )
else:
    # PostgreSQL configuration
    engine = create_engine(
        DATABASE_URL,
        echo=settings.environment == "dev"
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_postgis(engine: Engine) -> None:
    """Initialize PostGIS extensions if using PostgreSQL."""
    try:
        # Only attempt PostGIS initialization for PostgreSQL
        if engine.url.get_backend_name() == "postgresql":
            with engine.connect() as conn:
                # Enable PostGIS extensions
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis;"))
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis_topology;"))
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS fuzzystrmatch;"))
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis_tiger_geocoder;"))
                conn.commit()
                logger.info("🗺️ PostGIS extensions initialized successfully")
        else:
            logger.info(f"🗺️ Skipping PostGIS initialization for {engine.url.get_backend_name()} backend")
            
    except Exception as e:
        logger.warning(f"⚠️ PostGIS initialization failed (this is normal for SQLite): {e}")


def create_tables() -> None:
    """Create all tables in the database."""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Database tables created successfully")
    except Exception as e:
        logger.error(f"❌ Failed to create database tables: {e}")


# Dependency to get database session
def get_db():
    """FastAPI dependency to get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Initialize PostGIS on module import (safe for both SQLite and PostgreSQL)
init_postgis(engine)
