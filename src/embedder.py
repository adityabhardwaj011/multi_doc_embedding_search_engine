import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from .cache_manager import CacheManager
from .preprocessor import DocumentPreprocessor


class EmbeddingGenerator:
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 cache_db: str = "cache/embeddings_cache.db"):
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.cache_manager = CacheManager(cache_db)
        self.preprocessor = DocumentPreprocessor()
    
    def generate_embedding(self, text: str) -> np.ndarray:
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding
    
    def get_document_embedding(self, doc: Dict, use_cache: bool = True) -> np.ndarray:
        # checking if we already have this embedding cached
        if use_cache:
            cached_embedding = self.cache_manager.get_embedding(
                doc["doc_id"], doc["hash"]
            )
            if cached_embedding is not None:
                print(f"Using cached embedding for {doc['doc_id']}")
                return cached_embedding
        
        # If not cached, generate new embedding
        print(f"Generating new embedding for {doc['doc_id']}")
        embedding = self.generate_embedding(doc["text"])
        
        # Save it to cache for next time
        if use_cache:
            self.cache_manager.store_embedding(
                doc_id=doc["doc_id"],
                embedding=embedding,
                doc_hash=doc["hash"],
                filename=doc.get("filename"),
                doc_length=doc.get("doc_length")
            )
        
        return embedding
    
    def generate_embeddings_batch(self, documents: List[Dict], 
                                 use_cache: bool = True) -> Dict[str, np.ndarray]:
        embeddings = {}
        docs_to_embed = []
        doc_ids_to_embed = []
        
        # Checking cache for each document, skipping those we already have
        for doc in documents:
            if use_cache:
                cached_embedding = self.cache_manager.get_embedding(
                    doc["doc_id"], doc["hash"]
                )
                if cached_embedding is not None:
                    embeddings[doc["doc_id"]] = cached_embedding
                    continue
            
            # Collecting documents that need new embeddings
            docs_to_embed.append(doc)
            doc_ids_to_embed.append(doc["doc_id"])
        
        # Generating embeddings in batch for all uncached documents
        if docs_to_embed:
            texts = [doc["text"] for doc in docs_to_embed]
            print(f"Generating embeddings for {len(texts)} documents...")
            new_embeddings = self.model.encode(
                texts, 
                convert_to_numpy=True,
                show_progress_bar=True
            )
            
            # Storing each new embedding in cache
            for i, doc in enumerate(docs_to_embed):
                embedding = new_embeddings[i]
                embeddings[doc["doc_id"]] = embedding
                
                if use_cache:
                    self.cache_manager.store_embedding(
                        doc_id=doc["doc_id"],
                        embedding=embedding,
                        doc_hash=doc["hash"],
                        filename=doc.get("filename"),
                        doc_length=doc.get("doc_length")
                    )
        
        return embeddings
    
    def generate_query_embedding(self, query: str) -> np.ndarray:
        return self.generate_embedding(query)

