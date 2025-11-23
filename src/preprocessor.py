import os
import re
import hashlib
from pathlib import Path
from typing import List, Dict
from bs4 import BeautifulSoup


class DocumentPreprocessor:
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def clean_text(self, text: str) -> str:
        # Removing any HTML tags if present
        soup = BeautifulSoup(text, 'html.parser')
        text = soup.get_text()
        # Making everything lowercase for consistency
        text = text.lower()
        # Replacing multiple spaces/newlines with single space
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text
    
    def compute_hash(self, text: str) -> str:
        # Creating a hash of the text to check if document changed
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    def load_document(self, filepath: Path) -> Dict:
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                raw_text = f.read()
            
            # Cleaning the text and computing hash for cache checking
            cleaned_text = self.clean_text(raw_text)
            doc_hash = self.compute_hash(cleaned_text)
            
            return {
                "doc_id": filepath.stem,
                "filename": filepath.name,
                "filepath": str(filepath),
                "text": cleaned_text,
                "doc_length": len(cleaned_text),
                "hash": doc_hash
            }
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
    
    def load_all_documents(self) -> List[Dict]:
        documents = []
        txt_files = list(self.data_dir.glob("*.txt"))
        
        if not txt_files:
            print(f"No .txt files found in {self.data_dir}")
            return documents
        
        print(f"Found {len(txt_files)} text files. Processing...")
        
        for filepath in txt_files:
            doc = self.load_document(filepath)
            if doc:
                documents.append(doc)
        
        print(f"Successfully processed {len(documents)} documents")
        return documents
    
    def get_document_by_id(self, doc_id: str) -> Dict:
        filepath = self.data_dir / f"{doc_id}.txt"
        if filepath.exists():
            return self.load_document(filepath)
        return None

