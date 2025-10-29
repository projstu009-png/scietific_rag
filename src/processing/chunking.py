# src/processing/chunking.py - ADD QUALITY FILTERING

from typing import List, Dict, Any
from dataclasses import dataclass
import re

@dataclass
class Chunk:
    text: str
    metadata: Dict[str, Any]
    chunk_index: int
    quality_score: float = 0.0  # NEW

class AdvancedChunker:
    """Advanced chunking with quality filtering"""
    
    def __init__(self, config: Dict[str, Any]):
        self.chunk_size = config.get("chunk_size", 512)
        self.chunk_overlap = config.get("chunk_overlap", 100)
        
        raw_separators = config.get("separators", ["\n\n", "\n", ". ", " "])
        self.separators = [sep for sep in raw_separators if sep]
        
        if not self.separators:
            self.separators = ["\n\n", "\n", ". ", " "]
        
        self.min_chunk_length = config.get("min_chunk_length", 50)
        self.max_chunk_length = config.get("max_chunk_length", 1000)
        
        # NEW: Quality thresholds
        self.min_quality_score = config.get("min_quality_score", 0.5)
    
    def chunk_documents(self, documents: List[Any]) -> List[Chunk]:
        """Chunk documents with quality filtering"""
        all_chunks = []
        
        for doc in documents:
            chunks = self._chunk_text(doc.text, doc.metadata)
            
            # Score each chunk
            for chunk in chunks:
                chunk.quality_score = self._compute_quality_score(chunk.text)
            
            # Filter low-quality chunks
            quality_chunks = [
                c for c in chunks 
                if c.quality_score >= self.min_quality_score
            ]
            
            all_chunks.extend(quality_chunks)
            
            if len(quality_chunks) < len(chunks):
                print(f"⚠️ Filtered {len(chunks) - len(quality_chunks)} low-quality chunks")
        
        return all_chunks
    
    def _compute_quality_score(self, text: str) -> float:
        """Compute chunk quality score (0-1)"""
        
        if not text or len(text) < 10:
            return 0.0
        
        score = 1.0
        
        # Penalty 1: Too many pipe characters (table fragments)
        pipe_ratio = text.count('|') / len(text)
        if pipe_ratio > 0.1:  # More than 10% pipes
            score *= (1 - pipe_ratio)
        
        # Penalty 2: Too many "None" values
        none_count = text.count('None')
        if none_count > 5:
            score *= max(0.1, 1 - (none_count / 50))
        
        # Penalty 3: Too repetitive characters
        if len(set(text)) < len(text) * 0.1:  # Less than 10% unique chars
            score *= 0.3
        
        # Penalty 4: Not enough actual words
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text)
        word_ratio = len(words) / max(len(text.split()), 1)
        if word_ratio < 0.3:  # Less than 30% real words
            score *= word_ratio
        
        # Bonus 1: Contains sentences (has periods)
        if text.count('.') >= 2:
            score *= 1.1
        
        # Bonus 2: Has varied punctuation
        punctuation = set(c for c in text if c in '.,;:!?')
        if len(punctuation) >= 3:
            score *= 1.05
        
        # Bonus 3: Contains scientific keywords
        scientific_keywords = [
            'equation', 'theory', 'experiment', 'result', 'analysis',
            'data', 'method', 'conclusion', 'abstract', 'research',
            'study', 'observation', 'hypothesis', 'model', 'system'
        ]
        
        keyword_count = sum(1 for kw in scientific_keywords if kw in text.lower())
        if keyword_count > 0:
            score *= (1 + keyword_count * 0.05)
        
        return min(score, 1.0)
    
    def _chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """Chunk text using hierarchical splitting"""
        chunks = []
        
        for separator in self.separators:
            if not separator:
                continue
            
            splits = text.split(separator)
            
            current_chunk = ""
            chunk_index = 0
            
            for split in splits:
                if not split.strip():
                    continue
                
                potential_chunk = current_chunk + separator + split if current_chunk else split
                
                if len(potential_chunk) <= self.max_chunk_length:
                    current_chunk = potential_chunk
                else:
                    if len(current_chunk) >= self.min_chunk_length:
                        chunks.append(Chunk(
                            text=current_chunk,
                            metadata={**metadata, "chunk_method": f"split_by_{repr(separator)}"},
                            chunk_index=chunk_index
                        ))
                        chunk_index += 1
                    
                    if self.chunk_overlap > 0:
                        overlap_text = current_chunk[-self.chunk_overlap:]
                        current_chunk = overlap_text + separator + split
                    else:
                        current_chunk = split
            
            if len(current_chunk) >= self.min_chunk_length:
                chunks.append(Chunk(
                    text=current_chunk,
                    metadata={**metadata, "chunk_method": f"split_by_{repr(separator)}"},
                    chunk_index=chunk_index
                ))
            
            if chunks and all(self.min_chunk_length <= len(c.text) <= self.max_chunk_length for c in chunks):
                return chunks
            
            chunks = []
        
        if not chunks:
            chunks.append(Chunk(
                text=text,
                metadata={**metadata, "chunk_method": "no_split"},
                chunk_index=0
            ))
        
        return chunks


# src/rag_system.py - UPDATE ingest_papers

class ScientificRAGSystem:
    """Main RAG system with quality filtering"""
    
    def ingest_papers(self, pdf_paths: List[str]):
        """Ingest with quality filtering"""
        all_chunks = []
        
        for pdf_path in pdf_paths:
            print(f"Processing: {pdf_path}")
            
            documents = self.pdf_processor.process_pdf(pdf_path)
            
            # PRE-FILTER: Remove documents with too many table fragments
            quality_docs = []
            for doc in documents:
                pipe_ratio = doc.text.count('|') / len(doc.text) if doc.text else 0
                if pipe_ratio < 0.3:  # Less than 30% pipes
                    quality_docs.append(doc)
                else:
                    print(f"⚠️ Skipping table-heavy page {doc.page_number}")
            
            chunks = self.chunker.chunk_documents(quality_docs)
            
            # Extract metadata
            for chunk in chunks:
                equations = self.equation_processor.extract_latex_equations(chunk.text)
                chunk.metadata['equations'] = [eq['raw'] for eq in equations]
                
                citations = self.citation_processor.extract_citations(chunk.text)
                chunk.metadata['citations'] = citations
                chunk.metadata['quality_score'] = chunk.quality_score
            
            all_chunks.extend(chunks)
        
        print(f"✓ Total quality chunks: {len(all_chunks)}")
        
        if not all_chunks:
            print("⚠️ WARNING: No quality chunks extracted!")
            return
        
        # Log quality distribution
        scores = [c.quality_score for c in all_chunks]
        print(f"Quality scores - Min: {min(scores):.2f}, Max: {max(scores):.2f}, Avg: {sum(scores)/len(scores):.2f}")
        
        # Generate embeddings
        print("Generating embeddings...")
        texts = [chunk.text for chunk in all_chunks]
        embeddings = self.embedder.embed_documents(texts)
        
        # Add to vector store
        print("Adding to vector store...")
        metadatas = [chunk.metadata for chunk in all_chunks]
        self.vectorstore.add_documents(texts, embeddings, metadatas)
        
        # Index for BM25
        print("Indexing for hybrid retrieval...")
        doc_dicts = [
            {'text': chunk.text, **chunk.metadata}
            for chunk in all_chunks
        ]
        self.retriever.index_documents(doc_dicts)
        
        print("✓ Ingestion complete!")