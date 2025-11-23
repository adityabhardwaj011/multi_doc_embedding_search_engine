import numpy as np
import faiss
import pickle
from typing import List, Dict, Tuple
from pathlib import Path
from .embedder import EmbeddingGenerator
from .preprocessor import DocumentPreprocessor


class VectorSearchEngine:
    
    def __init__(self, embedder: EmbeddingGenerator, 
                 preprocessor: DocumentPreprocessor,
                 index_path: str = "cache/faiss_index.index"):
        self.embedder = embedder
        self.preprocessor = preprocessor
        self.index_path = Path(index_path)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.doc_ids_path = Path(str(index_path).replace('.index', '_doc_ids.pkl'))
        
        self.index = None
        self.doc_ids = []
        self.documents = {}
        self.dimension = None
    
    def build_index(self, documents: List[Dict], embeddings: Dict[str, np.ndarray]):
        if not embeddings:
            raise ValueError("No embeddings provided")
        
        # Figuring out the dimension from the first embedding
        first_embedding = next(iter(embeddings.values()))
        self.dimension = len(first_embedding)
        
        normalized_embeddings = []
        self.doc_ids = []
        
        # Normalizing embeddings so we can use inner product for cosine similarity
        for doc in documents:
            doc_id = doc["doc_id"]
            if doc_id in embeddings:
                embedding = embeddings[doc_id]
                # Normalizing the embedding vector for cosine similarity
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
                normalized_embeddings.append(embedding.astype('float32'))
                self.doc_ids.append(doc_id)
                self.documents[doc_id] = doc
        
        if not normalized_embeddings:
            raise ValueError("No valid embeddings to index")
        
        # Building the FAISS index for fast similarity search
        embeddings_array = np.array(normalized_embeddings)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings_array)
        
        print(f"Built FAISS index with {len(self.doc_ids)} documents")
        self.save_index()
    
    def load_index(self) -> bool:
        if not self.index_path.exists() or not self.doc_ids_path.exists():
            return False
        
        try:
            self.index = faiss.read_index(str(self.index_path))
            self.dimension = self.index.d
            
            with open(self.doc_ids_path, 'rb') as f:
                self.doc_ids = pickle.load(f)
            
            print(f"Loaded FAISS index with dimension {self.dimension} and {len(self.doc_ids)} documents")
            return True
        except Exception as e:
            print(f"Error loading index: {e}")
            return False
    
    def save_index(self):
        if self.index is not None:
            faiss.write_index(self.index, str(self.index_path))
            with open(self.doc_ids_path, 'wb') as f:
                pickle.dump(self.doc_ids, f)
            print(f"Saved FAISS index to {self.index_path} and doc_ids to {self.doc_ids_path}")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        if self.index is None or len(self.doc_ids) == 0:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Converting query to embedding
        query_embedding = self.embedder.generate_query_embedding(query)
        query_embedding = query_embedding.astype('float32')
        
        # Normalizing query embedding to match how documents are stored
        norm = np.linalg.norm(query_embedding)
        if norm > 0:
            query_embedding = query_embedding / norm
        
        # Reshaping for FAISS as it needs 2D array
        query_embedding = query_embedding.reshape(1, -1)
        
        # Searchin  for similar documents
        scores, indices = self.index.search(query_embedding, min(top_k, len(self.doc_ids)))
        
        results = []
        query_words = set(query.lower().split())
        
        # Building result list with explanations
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < 0 or idx >= len(self.doc_ids):
                continue
            
            doc_id = self.doc_ids[idx]
            doc = self.documents.get(doc_id)
            
            if not doc:
                continue
            
            # Generating explanation for why this document matched
            explanation = self._generate_explanation(
                query, query_words, doc, float(score)
            )
            
            # Getting a preview snippet
            preview = doc["text"][:200] + "..." if len(doc["text"]) > 200 else doc["text"]
            
            results.append({
                "doc_id": doc_id,
                "score": float(score),
                "preview": preview,
                "explanation": explanation,
                "filename": doc.get("filename", doc_id),
                "doc_length": doc.get("doc_length", 0)
            })
        
        return results
    
    def _generate_explanation(self, query: str, query_words: set, 
                            doc: Dict, score: float) -> Dict:
        # Finding which keywords from the query appear in the document
        doc_text = doc["text"]
        doc_words = set(doc_text.lower().split())
        
        overlapping_keywords = query_words.intersection(doc_words)
        
        # Calculating what percentage of query words matched
        if len(query_words) > 0:
            overlap_ratio = len(overlapping_keywords) / len(query_words)
        else:
            overlap_ratio = 0.0
        
        # normalizing document lenght
        doc_length = doc.get("doc_length", 0)
        length_normalization = min(1.0, 1000.0 / max(doc_length, 1))
        
        # Creating explanation
        if overlapping_keywords:
            keywords_str = ", ".join(list(overlapping_keywords)[:5])
            explanation_text = (
                f"This document matched because it contains keywords: {keywords_str}. "
                f"Similarity score: {score:.3f}, Keyword overlap: {overlap_ratio:.1%}"
            )
        else:
            explanation_text = (
                f"This document matched based on semantic similarity (score: {score:.3f}). "
                f"No direct keyword overlap, but content is semantically related."
            )
        
        return {
            "why_matched": explanation_text,
            "overlapping_keywords": list(overlapping_keywords),
            "overlap_ratio": overlap_ratio,
            "length_normalization": length_normalization,
            "similarity_score": score
        }

