from typing import List, Dict, Optional, Any
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, SearchRequest,
    HnswConfigDiff, ScalarQuantization,
    ScalarQuantizationConfig, ScalarType,  # Add these imports
    QuantizationSearchParams
)
import uuid

class QdrantVectorStore:
    """Qdrant vector store with advanced features"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.collection_name = config["collection_name"]
        
        # Initialize client
        if config["mode"] == "server":
            self.client = QdrantClient(
                host=config["host"],
                port=config["port"]
            )
        else:
            # Local mode for development
            self.client = QdrantClient(path=config.get("path", "./qdrant_data"))
        
        self._create_collection_if_not_exists()
    
    def _create_collection_if_not_exists(self):
        """Create collection with optimized configuration"""
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)
        
        if not exists:
            distance_map = {
                "cosine": Distance.COSINE,
                "dot": Distance.DOT,
                "euclidean": Distance.EUCLID
            }
            
            hnsw_config = HnswConfigDiff(
                m=self.config["hnsw_config"]["m"],
                ef_construct=self.config["hnsw_config"]["ef_construct"]
            )
            
            # Fix: Correct quantization configuration structure
            quantization_config = None
            if self.config["quantization"]["enabled"]:
                quant_type = self.config["quantization"].get("type", "scalar")
                
                if quant_type == "scalar":
                    quantization_config = ScalarQuantization(
                        scalar=ScalarQuantizationConfig(
                            type=ScalarType.INT8,
                            quantile=0.99,
                            always_ram=True
                        )
                    )
            
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.config.get("vector_size", 768),
                    distance=distance_map[self.config["distance_metric"]],
                    hnsw_config=hnsw_config
                ),
                quantization_config=quantization_config,
                on_disk_payload=self.config.get("on_disk_payload", True)
            )

    
    def add_documents(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """Add documents to the vector store"""
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(texts))]
        
        points = []
        for i, (text, embedding, metadata) in enumerate(zip(texts, embeddings, metadatas)):
            payload = {
                "text": text,
                **metadata
            }
            
            points.append(
                PointStruct(
                    id=ids[i],
                    vector=embedding,
                    payload=payload
                )
            )
        
        # Upload in batches
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch
            )
        
        return ids
    
    def similarity_search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filter_conditions: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Perform similarity search"""
        
        # Fix: Correct quantization search params structure
        search_params = None
        if self.config.get("quantization", {}).get("enabled", False):
            from qdrant_client.models import SearchParams, QuantizationSearchParams
            search_params = SearchParams(
                quantization=QuantizationSearchParams(
                    ignore=False,
                    rescore=True
                )
            )
        
        # Build filter
        query_filter = None
        if filter_conditions:
            conditions = []
            for key, value in filter_conditions.items():
                conditions.append(
                    FieldCondition(key=key, match={"value": value})
                )
            query_filter = Filter(must=conditions)
        
        try:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k,
                query_filter=query_filter,
                search_params=search_params,
                score_threshold=score_threshold
            )
            
            return [
                {
                    "id": result.id,
                    "score": result.score,
                    "text": result.payload.get("text", ""),
                    "metadata": {k: v for k, v in result.payload.items() if k != "text"}
                }
                for result in results
            ]
        except Exception as e:
            print(f"Search error: {e}")
            # Retry without quantization params
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k,
                query_filter=query_filter,
                score_threshold=score_threshold
            )
            
            return [
                {
                    "id": result.id,
                    "score": result.score,
                    "text": result.payload.get("text", ""),
                    "metadata": {k: v for k, v in result.payload.items() if k != "text"}
                }
                for result in results
            ]

    
    def hybrid_search(
        self,
        query_embedding: List[float],
        query_text: str,
        top_k: int = 10,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3
    ) -> List[Dict[str, Any]]:
        """Hybrid search combining dense and sparse retrieval"""
        # Dense search
        dense_results = self.similarity_search(query_embedding, top_k=top_k * 2)
        
        # Sparse search (BM25-like using text matching)
        # Note: For production, consider using Qdrant's sparse vectors
        sparse_results = self._text_search(query_text, top_k=top_k * 2)
        
        # Combine results using reciprocal rank fusion
        combined = self._reciprocal_rank_fusion(
            [dense_results, sparse_results],
            weights=[dense_weight, sparse_weight]
        )
        
        return combined[:top_k]
    
    def _text_search(self, query_text: str, top_k: int) -> List[Dict[str, Any]]:
        """Simple text-based search using payload index"""
        # This is a simplified version - implement proper BM25 for production
        results = self.client.scroll(
            collection_name=self.collection_name,
            limit=top_k,
            with_payload=True,
            with_vectors=False
        )[0]
        
        # Score by text overlap (simplified)
        query_words = set(query_text.lower().split())
        scored_results = []
        
        for result in results:
            text = result.payload.get("text", "")
            text_words = set(text.lower().split())
            overlap = len(query_words & text_words) / len(query_words) if query_words else 0
            
            scored_results.append({
                "id": result.id,
                "score": overlap,
                "text": text,
                "metadata": {k: v for k, v in result.payload.items() if k != "text"}
            })
        
        return sorted(scored_results, key=lambda x: x["score"], reverse=True)[:top_k]
    
    @staticmethod
    def _reciprocal_rank_fusion(
        result_lists: List[List[Dict[str, Any]]],
        weights: Optional[List[float]] = None,
        k: int = 60
    ) -> List[Dict[str, Any]]:
        """Reciprocal Rank Fusion for combining multiple result lists"""
        if weights is None:
            weights = [1.0] * len(result_lists)
        
        # Calculate RRF scores
        rrf_scores = {}
        
        for results, weight in zip(result_lists, weights):
            for rank, result in enumerate(results, 1):
                doc_id = result["id"]
                score = weight / (k + rank)
                
                if doc_id not in rrf_scores:
                    rrf_scores[doc_id] = {
                        "score": 0,
                        "result": result
                    }
                rrf_scores[doc_id]["score"] += score
        
        # Sort by RRF score
        sorted_results = sorted(
            rrf_scores.items(),
            key=lambda x: x[1]["score"],
            reverse=True
        )
        
        return [
            {**item[1]["result"], "score": item[1]["score"]}
            for item in sorted_results
        ]
