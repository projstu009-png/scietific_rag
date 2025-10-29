# src/retrieval/hybrid_retriever.py
from typing import List, Dict, Any
import numpy as np
from typing import Optional
import re
from rank_bm25 import BM25Okapi, BM25Plus
import nltk
from nltk.corpus import stopwords

class ScientificBM25:
    """Domain-specific BM25 for physics papers"""
    
    def __init__(self, k1=1.5, b=0.75, delta=0.5):
        # BM25Plus works better for scientific text
        self.k1 = k1
        self.b = b
        self.delta = delta
        
        # Don't remove scientific stopwords!
        # Keep: "not", "no", "between", "above", "below", etc.
        self.stopwords = set(['the', 'a', 'an', 'and', 'or', 'but'])
    
    def preprocess_scientific(self, text: str) -> List[str]:
        """Scientific-aware tokenization with bigram and Greek preservation"""
        # Lowercase and normalize text
        norm_text = text.lower()
        
        # Replace equations with generic token
        norm_text = re.sub(r'\$[^$]+\$|\\\[.*?\\\]', ' __EQUATION__ ', norm_text)
        
        bigram_phrases = ["dark matter", "quantum field", "scalar field", "carbon capture", "heavy water"]  # Extend as needed
        
        # Protect bigrams: replace with underscore-connected token
        for phrase in bigram_phrases:
            norm_text = norm_text.replace(phrase, phrase.replace(" ", "_"))
        
        # Greek letters: preserve and match unicode and LaTeX forms
        greek_patterns = [
            r"\balpha\b|\\alpha|α", r"\bbeta\b|\\beta|β", r"\bgamma\b|\\gamma|γ",
            r"\bdelta\b|\\delta|δ", r"\bepsilon\b|\\epsilon|ε",
            r"\bmu\b|\\mu|μ", r"\bnu\b|\\nu|ν", r"\btau\b|\\tau|τ",
            # Extend as needed
        ]
        # Protect Greek words (all forms) with special token
        for pattern in greek_patterns:
            norm_text = re.sub(pattern, lambda m: "_" + m.group(0) + "_", norm_text)
        
        # Tokenize: keep words, Greek, numbers, bigram tokens, and equations
        pattern = r'__EQUATION__|[a-zA-Z_]+(?:_[a-zA-Z_]+)?|\d+\.?\d*|[μντβαγδε]'
        tokens = re.findall(pattern, norm_text)
        
        # Remove stopwords as before
        tokens = [t for t in tokens if t not in self.stopwords]
        
        return tokens


class SectionAwareRetriever:
    """Retrieval with paper section awareness"""
    
    def __init__(self, vector_store, embedder, config: Dict):
        self.vector_store = vector_store
        self.embedder = embedder
        self.section_weights = config['scientific']['advanced_retrieval']['section_weights']
    
    def retrieve_with_section_boost(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Dict]:
        """Boost scores based on section type"""
        # Get raw results
        query_embedding = self.embedder.embed_query(query)
        results = self.vector_store.similarity_search(
            query_embedding,
            top_k=top_k * 3  # Over-retrieve for reranking
        )
        
        # Apply section weights
        for result in results:
            section = result['metadata'].get('section', 'unknown').lower()
            base_score = result['score']
            
            # Apply weight
            weight = self.section_weights.get(section, 1.0)
            result['weighted_score'] = base_score * weight
            result['section'] = section
        
        # Re-sort by weighted score
        results = sorted(
            results,
            key=lambda x: x['weighted_score'],
            reverse=True
        )[:top_k]
        
        return results


class HybridRetriever:
    """Advanced hybrid retrieval combining dense and sparse methods"""
    
    def __init__(self, vector_store, embedder, config: Dict[str, Any]):
        self.vector_store = vector_store
        self.embedder = embedder
        self.config = config
        
        # Initialize BM25
        self.bm25 = None
        self.scientific_bm25 = ScientificBM25()
        self.documents = []
        self.tokenized_docs = []
    
    def index_documents(self, documents: List[Dict[str, Any]]):
        """Index documents for both dense and sparse retrieval"""
        self.documents = documents
        
        # Tokenize for BM25
        print(f"Tokenizing {len(documents)} documents for BM25...")
        self.tokenized_docs = []
        
        for doc in documents:
            try:
                tokens = self.scientific_bm25.preprocess_scientific(doc['text'])
                self.tokenized_docs.append(tokens)
            except Exception as e:
                print(f"Warning: Failed to tokenize document: {e}")
                # Fallback: simple split
                self.tokenized_docs.append(doc['text'].lower().split())
        
        # Initialize BM25
        if self.tokenized_docs:
            self.bm25 = BM25Plus(self.tokenized_docs)
            print(f"✓ BM25 index created with {len(self.tokenized_docs)} documents")
        else:
            print("⚠ Warning: No documents tokenized for BM25")
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        filters: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """Hybrid retrieval with RRF fusion"""
        
        results = []
        
        # Dense retrieval
        try:
            query_embedding = self.embedder.embed_query(query)
            dense_results = self.vector_store.similarity_search(
                query_embedding,
                top_k=top_k * 2,
                filter_conditions=filters
            )
            results.append(('dense', dense_results))
        except Exception as e:
            print(f"Dense retrieval failed: {e}")
            dense_results = []
        
        # Sparse retrieval (BM25)
        sparse_results = []
        if self.bm25 and self.tokenized_docs:
            try:
                tokenized_query = self.scientific_bm25.preprocess_scientific(query)
                bm25_scores = self.bm25.get_scores(tokenized_query)
                
                # Get top-k from BM25
                top_indices = np.argsort(bm25_scores)[::-1][:top_k * 2]
                
                for idx in top_indices:
                    if idx < len(self.documents):
                        doc = self.documents[idx].copy()
                        doc['score'] = float(bm25_scores[idx])
                        sparse_results.append(doc)
                
                results.append(('sparse', sparse_results))
            except Exception as e:
                print(f"BM25 retrieval failed: {e}")
        
        # If we have results, fuse them
        if results:
            result_lists = [r[1] for r in results]
            weights = []
            
            if dense_results:
                weights.append(dense_weight)
            if sparse_results:
                weights.append(sparse_weight)
            
            fused = self._reciprocal_rank_fusion(result_lists, weights)
            return fused[:top_k]
        
        # Fallback to just dense results
        return dense_results[:top_k] if dense_results else []
    
    def _reciprocal_rank_fusion(
        self,
        results_lists: List[List[Dict]],
        weights: List[float],
        k: int = 60
    ) -> List[Dict]:
        """RRF with configurable k parameter"""
        scores = {}
        
        for results, weight in zip(results_lists, weights):
            for rank, result in enumerate(results, 1):
                # Use chunk_id or id as unique identifier
                doc_id = result.get('chunk_id', result.get('id', str(rank)))
                score = weight / (k + rank)
                
                if doc_id not in scores:
                    scores[doc_id] = {'score': 0, 'doc': result}
                scores[doc_id]['score'] += score
        
        # Sort by fused score
        sorted_results = sorted(
            scores.values(),
            key=lambda x: x['score'],
            reverse=True
        )
        
        return [item['doc'] for item in sorted_results]
