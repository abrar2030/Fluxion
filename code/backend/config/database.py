"""
Enhanced database configuration for Fluxion backend
"""

import logging
from code.backend.config.settings import settings
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import (AsyncEngine, AsyncSession,
                                    async_sessionmaker, create_async_engine)
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.pool import QueuePool

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """Base class for all database models"""

    pass


class DatabaseManager:
    """Database connection manager"""

    def __init__(self):
        self._engine: Optional[AsyncEngine] = None
        self._session_factory: Optional[async_sessionmaker] = None
        self._read_engine: Optional[AsyncEngine] = None
        self._read_session_factory: Optional[async_sessionmaker] = None

    def create_engine(
        self, database_url: str, is_read_replica: bool = False
    ) -> AsyncEngine:
        """Create database engine with optimized settings"""
        engine_kwargs = {
            "url": database_url,
            "echo": settings.database.DB_ECHO and not is_read_replica,
            "poolclass": QueuePool,
            "pool_size": settings.database.DB_POOL_SIZE,
            "max_overflow": settings.database.DB_MAX_OVERFLOW,
            "pool_timeout": settings.database.DB_POOL_TIMEOUT,
            "pool_recycle": settings.database.DB_POOL_RECYCLE,
            "pool_pre_ping": True,
            "connect_args": {
                "server_settings": {
                    "application_name": f"fluxion_{'read' if is_read_replica else 'write'}",
                    "jit": "off",
                }
            },
        }

        engine = create_async_engine(**engine_kwargs)

        # Add connection event listeners
        @event.listens_for(engine.sync_engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            """Set database-specific optimizations"""
            if "postgresql" in str(engine.url):
                # PostgreSQL optimizations
                with dbapi_connection.cursor() as cursor:
                    cursor.execute("SET timezone TO 'UTC'")
                    cursor.execute("SET statement_timeout = '300s'")
                    cursor.execute("SET lock_timeout = '30s'")

        return engine

    async def init_database(self):
        """Initialize database connections"""
        try:
            # Create main engine
            self._engine = self.create_engine(str(settings.database.DATABASE_URL))
            self._session_factory = async_sessionmaker(
                bind=self._engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autoflush=True,
                autocommit=False,
            )

            # Create read replica engine if configured
            if settings.database.DATABASE_READ_URL:
                self._read_engine = self.create_engine(
                    str(settings.database.DATABASE_READ_URL), is_read_replica=True
                )
                self._read_session_factory = async_sessionmaker(
                    bind=self._read_engine,
                    class_=AsyncSession,
                    expire_on_commit=False,
                    autoflush=False,
                    autocommit=False,
                )

            # Test connections
            await self.test_connection()
            logger.info("Database connections initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise

    async def test_connection(self):
        """Test database connectivity"""
        try:
            async with self._engine.begin() as conn:
                result = await conn.execute(text("SELECT 1"))
                assert result.scalar() == 1

            if self._read_engine:
                async with self._read_engine.begin() as conn:
                    result = await conn.execute(text("SELECT 1"))
                    assert result.scalar() == 1

        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            raise

    async def close_database(self):
        """Close database connections"""
        try:
            if self._engine:
                await self._engine.dispose()
                logger.info("Main database connection closed")

            if self._read_engine:
                await self._read_engine.dispose()
                logger.info("Read replica database connection closed")

        except Exception as e:
            logger.error(f"Error closing database connections: {e}")

    @asynccontextmanager
    async def get_session(
        self, read_only: bool = False
    ) -> AsyncGenerator[AsyncSession, None]:
        """Get database session context manager"""
        if read_only and self._read_session_factory:
            session_factory = self._read_session_factory
        else:
            session_factory = self._session_factory

        if not session_factory:
            raise RuntimeError("Database not initialized")

        async with session_factory() as session:
            try:
                yield session
                if not read_only:
                    await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()

    async def get_async_session(self, read_only: bool = False) -> AsyncSession:
        """Get database session (for dependency injection)"""
        if read_only and self._read_session_factory:
            session_factory = self._read_session_factory
        else:
            session_factory = self._session_factory

        if not session_factory:
            raise RuntimeError("Database not initialized")

        return session_factory()

    @property
    def engine(self) -> AsyncEngine:
        """Get main database engine"""
        if not self._engine:
            raise RuntimeError("Database not initialized")
        return self._engine

    @property
    def read_engine(self) -> Optional[AsyncEngine]:
        """Get read replica database engine"""
        return self._read_engine


# Global database manager instance
db_manager = DatabaseManager()


async def init_database():
    """Initialize database connections"""
    await db_manager.init_database()


async def close_database():
    """Close database connections"""
    await db_manager.close_database()


async def get_async_session(
    read_only: bool = False,
) -> AsyncGenerator[AsyncSession, None]:
    """Dependency for getting database session"""
    async with db_manager.get_session(read_only=read_only) as session:
        yield session


async def get_read_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for getting read-only database session"""
    async with db_manager.get_session(read_only=True) as session:
        yield session


class DatabaseHealthCheck:
    """Database health check utility"""

    @staticmethod
    async def check_write_db() -> dict:
        """Check write database health"""
        try:
            async with db_manager.get_session() as session:
                result = await session.execute(text("SELECT 1"))
                latency_start = logger.time()
                await session.execute(text("SELECT pg_sleep(0.001)"))
                latency = (logger.time() - latency_start) * 1000

                return {
                    "status": "healthy",
                    "latency_ms": round(latency, 2),
                    "connection_pool": {
                        "size": db_manager.engine.pool.size(),
                        "checked_in": db_manager.engine.pool.checkedin(),
                        "checked_out": db_manager.engine.pool.checkedout(),
                        "overflow": db_manager.engine.pool.overflow(),
                    },
                }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    @staticmethod
    async def check_read_db() -> dict:
        """Check read database health"""
        if not db_manager.read_engine:
            return {"status": "not_configured"}

        try:
            async with db_manager.get_session(read_only=True) as session:
                result = await session.execute(text("SELECT 1"))
                return {
                    "status": "healthy",
                    "connection_pool": {
                        "size": db_manager.read_engine.pool.size(),
                        "checked_in": db_manager.read_engine.pool.checkedin(),
                        "checked_out": db_manager.read_engine.pool.checkedout(),
                        "overflow": db_manager.read_engine.pool.overflow(),
                    },
                }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
