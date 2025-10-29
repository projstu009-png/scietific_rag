from typing import List, Optional
import requests
import numpy as np
from dataclasses import dataclass

@dataclass
class EmbeddingConfig:
    model: str
    base_url: str
    dimension: int
    batch_size: int
    normalize: bool
    task_type: str

class OllamaEmbedder:
    """Wrapper for Ollama embedding models"""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.base_url = f"{config.base_url}/api/embeddings"
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i:i + self.config.batch_size]
            batch_embeddings = self._embed_batch(batch)
            embeddings.extend(batch_embeddings)
        
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        return self._embed_batch([text])[0]
    
    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Internal method to embed a batch"""
        embeddings = []
        
        for text in texts:
            # Prepend task instruction for nomic-embed
            if self.config.task_type == "search_document":
                prompt = f"search_document: {text}"
            elif self.config.task_type == "search_query":
                prompt = f"search_query: {text}"
            else:
                prompt = text
            
            response = requests.post(
                self.base_url,
                json={
                    "model": self.config.model,
                    "prompt": prompt
                }
            )
            
            if response.status_code == 200:
                embedding = response.json()["embedding"]
                
                # Normalize if required
                if self.config.normalize:
                    embedding = self._normalize(embedding)
                
                embeddings.append(embedding)
            else:
                raise Exception(f"Embedding failed: {response.text}")
        
        return embeddings
    
    @staticmethod
    def _normalize(embedding: List[float]) -> List[float]:
        """L2 normalize embedding"""
        norm = np.linalg.norm(embedding)
        return (np.array(embedding) / norm).tolist()
