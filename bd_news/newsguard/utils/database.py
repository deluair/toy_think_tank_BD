"""Database management utilities for NewsGuard Bangladesh simulation."""

import os
import json
from typing import Dict, Any, Optional, List, Union
from contextlib import contextmanager
from datetime import datetime, timezone
from dataclasses import asdict

# PostgreSQL
import psycopg2
from psycopg2.pool import ThreadedConnectionPool
from psycopg2.extras import RealDictCursor, Json
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float, DateTime, Boolean, Text, JSON
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.dialects.postgresql import UUID

# MongoDB
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

# Redis
import redis
from redis.connection import ConnectionPool

# Utilities
from .config import get_config
from .logger import get_logger


logger = get_logger(__name__)
Base = declarative_base()


class PostgreSQLManager:
    """PostgreSQL database manager for structured data."""
    
    def __init__(self, config: Optional[Dict[str, str]] = None):
        """Initialize PostgreSQL manager.
        
        Args:
            config: Database configuration override
        """
        self.config = config or get_config().get("database.postgresql", {})
        self.engine = None
        self.session_factory = None
        self.connection_pool = None
        self._setup_connection()
    
    def _setup_connection(self) -> None:
        """Setup database connection and session factory."""
        try:
            # Create SQLAlchemy engine
            db_url = (
                f"postgresql://{self.config['username']}:{self.config['password']}"
                f"@{self.config['host']}:{self.config['port']}/{self.config['database']}"
            )
            
            self.engine = create_engine(
                db_url,
                pool_size=int(self.config.get('pool_size', 10)),
                pool_pre_ping=True,
                echo=False
            )
            
            # Create session factory
            self.session_factory = sessionmaker(bind=self.engine)
            
            # Create connection pool for raw connections
            self.connection_pool = ThreadedConnectionPool(
                minconn=1,
                maxconn=int(self.config.get('pool_size', 10)),
                host=self.config['host'],
                port=self.config['port'],
                database=self.config['database'],
                user=self.config['username'],
                password=self.config['password']
            )
            
            logger.info("PostgreSQL connection established")
            
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise
    
    @contextmanager
    def get_session(self):
        """Get SQLAlchemy session with automatic cleanup.
        
        Yields:
            SQLAlchemy session
        """
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    @contextmanager
    def get_connection(self):
        """Get raw database connection with automatic cleanup.
        
        Yields:
            psycopg2 connection
        """
        conn = self.connection_pool.getconn()
        try:
            yield conn
        except Exception as e:
            conn.rollback()
            logger.error(f"Database connection error: {e}")
            raise
        finally:
            self.connection_pool.putconn(conn)
    
    def execute_query(self, query: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
        """Execute SQL query and return results.
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            List of result dictionaries
        """
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query, params)
                if cursor.description:
                    return [dict(row) for row in cursor.fetchall()]
                return []
    
    def execute_command(self, command: str, params: Optional[tuple] = None) -> int:
        """Execute SQL command and return affected rows.
        
        Args:
            command: SQL command string
            params: Command parameters
            
        Returns:
            Number of affected rows
        """
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(command, params)
                conn.commit()
                return cursor.rowcount
    
    def create_tables(self) -> None:
        """Create all database tables."""
        Base.metadata.create_all(self.engine)
        logger.info("Database tables created")
    
    def drop_tables(self) -> None:
        """Drop all database tables."""
        Base.metadata.drop_all(self.engine)
        logger.info("Database tables dropped")
    
    def close(self) -> None:
        """Close database connections."""
        if self.connection_pool:
            self.connection_pool.closeall()
        if self.engine:
            self.engine.dispose()
        logger.info("PostgreSQL connections closed")


class MongoDBManager:
    """MongoDB manager for document storage."""
    
    def __init__(self, config: Optional[Dict[str, str]] = None):
        """Initialize MongoDB manager.
        
        Args:
            config: Database configuration override
        """
        self.config = config or get_config().get("database.mongodb", {})
        self.client = None
        self.database = None
        self._setup_connection()
    
    def _setup_connection(self) -> None:
        """Setup MongoDB connection."""
        try:
            # Build connection string
            if self.config.get('username') and self.config.get('password'):
                connection_string = (
                    f"mongodb://{self.config['username']}:{self.config['password']}"
                    f"@{self.config['host']}:{self.config['port']}/{self.config['database']}"
                )
            else:
                connection_string = f"mongodb://{self.config['host']}:{self.config['port']}"
            
            self.client = MongoClient(connection_string)
            self.database = self.client[self.config['database']]
            
            # Test connection
            self.client.admin.command('ping')
            logger.info("MongoDB connection established")
            
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
    
    def get_collection(self, collection_name: str) -> Collection:
        """Get MongoDB collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            MongoDB collection
        """
        return self.database[collection_name]
    
    def insert_document(self, collection_name: str, document: Dict[str, Any]) -> str:
        """Insert document into collection.
        
        Args:
            collection_name: Name of the collection
            document: Document to insert
            
        Returns:
            Inserted document ID
        """
        collection = self.get_collection(collection_name)
        document['created_at'] = datetime.now(timezone.utc)
        result = collection.insert_one(document)
        return str(result.inserted_id)
    
    def insert_documents(self, collection_name: str, documents: List[Dict[str, Any]]) -> List[str]:
        """Insert multiple documents into collection.
        
        Args:
            collection_name: Name of the collection
            documents: List of documents to insert
            
        Returns:
            List of inserted document IDs
        """
        collection = self.get_collection(collection_name)
        for doc in documents:
            doc['created_at'] = datetime.now(timezone.utc)
        
        result = collection.insert_many(documents)
        return [str(id_) for id_ in result.inserted_ids]
    
    def find_documents(self, collection_name: str, query: Dict[str, Any], 
                      limit: Optional[int] = None, sort: Optional[List[tuple]] = None) -> List[Dict[str, Any]]:
        """Find documents in collection.
        
        Args:
            collection_name: Name of the collection
            query: MongoDB query
            limit: Maximum number of documents to return
            sort: Sort specification
            
        Returns:
            List of matching documents
        """
        collection = self.get_collection(collection_name)
        cursor = collection.find(query)
        
        if sort:
            cursor = cursor.sort(sort)
        if limit:
            cursor = cursor.limit(limit)
        
        return list(cursor)
    
    def update_document(self, collection_name: str, query: Dict[str, Any], 
                       update: Dict[str, Any], upsert: bool = False) -> int:
        """Update document in collection.
        
        Args:
            collection_name: Name of the collection
            query: Query to match documents
            update: Update operations
            upsert: Whether to insert if no match found
            
        Returns:
            Number of modified documents
        """
        collection = self.get_collection(collection_name)
        update['$set'] = update.get('$set', {})
        update['$set']['updated_at'] = datetime.now(timezone.utc)
        
        result = collection.update_many(query, update, upsert=upsert)
        return result.modified_count
    
    def delete_documents(self, collection_name: str, query: Dict[str, Any]) -> int:
        """Delete documents from collection.
        
        Args:
            collection_name: Name of the collection
            query: Query to match documents to delete
            
        Returns:
            Number of deleted documents
        """
        collection = self.get_collection(collection_name)
        result = collection.delete_many(query)
        return result.deleted_count
    
    def create_index(self, collection_name: str, index_spec: Union[str, List[tuple]], 
                    unique: bool = False) -> str:
        """Create index on collection.
        
        Args:
            collection_name: Name of the collection
            index_spec: Index specification
            unique: Whether index should be unique
            
        Returns:
            Index name
        """
        collection = self.get_collection(collection_name)
        return collection.create_index(index_spec, unique=unique)
    
    def close(self) -> None:
        """Close MongoDB connection."""
        if self.client:
            self.client.close()
        logger.info("MongoDB connection closed")


class RedisManager:
    """Redis manager for caching and real-time data."""
    
    def __init__(self, config: Optional[Dict[str, str]] = None):
        """Initialize Redis manager.
        
        Args:
            config: Redis configuration override
        """
        self.config = config or get_config().get("database.redis", {})
        self.client = None
        self.connection_pool = None
        self._setup_connection()
    
    def _setup_connection(self) -> None:
        """Setup Redis connection."""
        try:
            # Create connection pool
            self.connection_pool = ConnectionPool(
                host=self.config['host'],
                port=int(self.config['port']),
                db=int(self.config['database']),
                password=self.config.get('password') or None,
                decode_responses=True,
                max_connections=20
            )
            
            self.client = redis.Redis(connection_pool=self.connection_pool)
            
            # Test connection
            self.client.ping()
            logger.info("Redis connection established")
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    def set(self, key: str, value: Any, expire: Optional[int] = None) -> bool:
        """Set key-value pair in Redis.
        
        Args:
            key: Redis key
            value: Value to store
            expire: Expiration time in seconds
            
        Returns:
            True if successful
        """
        if isinstance(value, (dict, list)):
            value = json.dumps(value)
        
        return self.client.set(key, value, ex=expire)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from Redis.
        
        Args:
            key: Redis key
            default: Default value if key not found
            
        Returns:
            Stored value or default
        """
        value = self.client.get(key)
        if value is None:
            return default
        
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value
    
    def delete(self, *keys: str) -> int:
        """Delete keys from Redis.
        
        Args:
            *keys: Keys to delete
            
        Returns:
            Number of keys deleted
        """
        return self.client.delete(*keys)
    
    def exists(self, key: str) -> bool:
        """Check if key exists in Redis.
        
        Args:
            key: Redis key
            
        Returns:
            True if key exists
        """
        return bool(self.client.exists(key))
    
    def expire(self, key: str, seconds: int) -> bool:
        """Set expiration time for key.
        
        Args:
            key: Redis key
            seconds: Expiration time in seconds
            
        Returns:
            True if successful
        """
        return self.client.expire(key, seconds)
    
    def increment(self, key: str, amount: int = 1) -> int:
        """Increment numeric value.
        
        Args:
            key: Redis key
            amount: Amount to increment
            
        Returns:
            New value after increment
        """
        return self.client.incr(key, amount)
    
    def push_to_list(self, key: str, *values: Any) -> int:
        """Push values to list.
        
        Args:
            key: Redis key
            *values: Values to push
            
        Returns:
            New length of list
        """
        serialized_values = []
        for value in values:
            if isinstance(value, (dict, list)):
                serialized_values.append(json.dumps(value))
            else:
                serialized_values.append(str(value))
        
        return self.client.lpush(key, *serialized_values)
    
    def pop_from_list(self, key: str, timeout: int = 0) -> Optional[Any]:
        """Pop value from list.
        
        Args:
            key: Redis key
            timeout: Timeout in seconds (0 for non-blocking)
            
        Returns:
            Popped value or None
        """
        if timeout > 0:
            result = self.client.brpop(key, timeout=timeout)
            if result:
                _, value = result
            else:
                return None
        else:
            value = self.client.rpop(key)
        
        if value is None:
            return None
        
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value
    
    def publish(self, channel: str, message: Any) -> int:
        """Publish message to channel.
        
        Args:
            channel: Channel name
            message: Message to publish
            
        Returns:
            Number of subscribers that received the message
        """
        if isinstance(message, (dict, list)):
            message = json.dumps(message)
        
        return self.client.publish(channel, message)
    
    def close(self) -> None:
        """Close Redis connection."""
        if self.connection_pool:
            self.connection_pool.disconnect()
        logger.info("Redis connection closed")


class DatabaseManager:
    """Unified database manager for all database systems."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize database manager.
        
        Args:
            config: Database configuration override
        """
        self.config = config or get_config().get("database", {})
        self.postgresql = PostgreSQLManager(self.config.get("postgresql"))
        self.mongodb = MongoDBManager(self.config.get("mongodb"))
        self.redis = RedisManager(self.config.get("redis"))
        
        logger.info("Database manager initialized")
    
    def initialize_schema(self) -> None:
        """Initialize database schema and indexes."""
        # Create PostgreSQL tables
        self.postgresql.create_tables()
        
        # Create MongoDB indexes
        self._create_mongodb_indexes()
        
        logger.info("Database schema initialized")
    
    def _create_mongodb_indexes(self) -> None:
        """Create MongoDB indexes for optimal performance."""
        # Content collection indexes
        self.mongodb.create_index("content", "content_id", unique=True)
        self.mongodb.create_index("content", [("created_at", -1)])
        self.mongodb.create_index("content", "content_type")
        self.mongodb.create_index("content", "source_id")
        
        # Agent collection indexes
        self.mongodb.create_index("agents", "agent_id", unique=True)
        self.mongodb.create_index("agents", "agent_type")
        
        # Interactions collection indexes
        self.mongodb.create_index("interactions", [("timestamp", -1)])
        self.mongodb.create_index("interactions", "source_agent_id")
        self.mongodb.create_index("interactions", "target_agent_id")
        
        # Metrics collection indexes
        self.mongodb.create_index("metrics", [("timestamp", -1)])
        self.mongodb.create_index("metrics", "metric_type")
        
        logger.info("MongoDB indexes created")
    
    def health_check(self) -> Dict[str, bool]:
        """Check health of all database connections.
        
        Returns:
            Dictionary with health status of each database
        """
        health = {}
        
        # PostgreSQL health check
        try:
            self.postgresql.execute_query("SELECT 1")
            health["postgresql"] = True
        except Exception as e:
            logger.error(f"PostgreSQL health check failed: {e}")
            health["postgresql"] = False
        
        # MongoDB health check
        try:
            self.mongodb.client.admin.command('ping')
            health["mongodb"] = True
        except Exception as e:
            logger.error(f"MongoDB health check failed: {e}")
            health["mongodb"] = False
        
        # Redis health check
        try:
            self.redis.client.ping()
            health["redis"] = True
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            health["redis"] = False
        
        return health
    
    def close_all(self) -> None:
        """Close all database connections."""
        self.postgresql.close()
        self.mongodb.close()
        self.redis.close()
        logger.info("All database connections closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close_all()


# Global database manager instance
_db_manager = None


def get_database_manager() -> DatabaseManager:
    """Get global database manager instance.
    
    Returns:
        Global database manager
    """
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


def close_database_connections() -> None:
    """Close all database connections."""
    global _db_manager
    if _db_manager is not None:
        _db_manager.close_all()
        _db_manager = None