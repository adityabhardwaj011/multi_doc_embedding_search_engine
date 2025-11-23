import sqlite3
import json
import numpy as np
from datetime import datetime
from typing import Optional, List
from pathlib import Path


class CacheManager:
    
    def __init__(self, cache_db: str = "cache/embeddings_cache.db"):
        self.cache_db = Path(cache_db)
        self.cache_db.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        # Setting up the SQLite database table for storing embeddings
        conn = sqlite3.connect(self.cache_db)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                doc_id TEXT PRIMARY KEY,
                embedding TEXT NOT NULL,
                hash TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                filename TEXT,
                doc_length INTEGER
            )
        """)
        
        # Adding index on hash for faster lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_hash ON embeddings(hash)
        """)
        
        conn.commit()
        conn.close()
    
    def get_embedding(self, doc_id: str, doc_hash: str) -> Optional[np.ndarray]:
        # Checking if we have a cached embedding for this document
        conn = sqlite3.connect(self.cache_db)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT embedding, hash FROM embeddings
            WHERE doc_id = ?
        """, (doc_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            cached_embedding_json, cached_hash = result
            # Only returning cached embedding if document hasn't changed (hash matches)
            if cached_hash == doc_hash:
                embedding = np.array(json.loads(cached_embedding_json))
                return embedding
        
        return None
    
    def store_embedding(self, doc_id: str, embedding: np.ndarray, doc_hash: str,
                       filename: str = None, doc_length: int = None):
        # Saving the embedding to cache so we don't have to regenerate it
        conn = sqlite3.connect(self.cache_db)
        cursor = conn.cursor()
        
        # Converting numpy array to JSON string for storage
        embedding_json = json.dumps(embedding.tolist())
        updated_at = datetime.now().isoformat()
        
        cursor.execute("""
            INSERT OR REPLACE INTO embeddings
            (doc_id, embedding, hash, updated_at, filename, doc_length)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (doc_id, embedding_json, doc_hash, updated_at, filename, doc_length))
        
        conn.commit()
        conn.close()
    
    def get_all_cached_embeddings(self) -> List[tuple]:
        conn = sqlite3.connect(self.cache_db)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT doc_id, embedding, hash, filename, doc_length
            FROM embeddings
        """)
        
        results = cursor.fetchall()
        conn.close()
        
        embeddings_data = []
        for row in results:
            doc_id, embedding_json, doc_hash, filename, doc_length = row
            embedding = np.array(json.loads(embedding_json))
            embeddings_data.append((doc_id, embedding, doc_hash, filename, doc_length))
        
        return embeddings_data
    
    def clear_cache(self):
        conn = sqlite3.connect(self.cache_db)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM embeddings")
        conn.commit()
        conn.close()
        print("Cache cleared successfully")
    
    def get_cache_stats(self) -> dict:
        conn = sqlite3.connect(self.cache_db)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM embeddings")
        count = cursor.fetchone()[0]
        
        cursor.execute("SELECT MIN(updated_at), MAX(updated_at) FROM embeddings")
        date_range = cursor.fetchone()
        
        conn.close()
        
        return {
            "total_cached": count,
            "oldest_entry": date_range[0] if date_range[0] else None,
            "newest_entry": date_range[1] if date_range[1] else None
        }

