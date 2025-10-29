# src/rag_system.py
import yaml
from pathlib import Path
from typing import Dict, Any, List

from src.embeddings.ollama_embedder import OllamaEmbedder, EmbeddingConfig
from src.vectorstore.qdrant_store import QdrantVectorStore
from src.processing.pdf_processor import PDFProcessor
from src.processing.chunking import AdvancedChunker
from src.processing.citation_processor import CitationProcessor
from src.processing.equations_processor import EquationProcessor
from src.retrieval.hybrid_retriever import HybridRetriever, SectionAwareRetriever
from src.llm.openai_compatible import OpenAICompatibleLLM
from src.agents.rag_agent import ScientificRAGAgent


class ScientificRAGSystem:
    """Main RAG system orchestrator"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize components
        self._init_embedder()
        self._init_vectorstore()
        self._init_processors()
        self._init_llm()
        self._init_retriever()
        self._init_agent()
        
        print("Scientific RAG System initialized successfully!")
    
    def _init_embedder(self):
        """Initialize embedding model"""
        emb_config = self.config['embeddings']
        self.embedder = OllamaEmbedder(
            EmbeddingConfig(
                model=emb_config['model'],
                base_url=emb_config['base_url'],
                dimension=emb_config['dimension'],
                batch_size=emb_config['batch_size'],
                normalize=emb_config['normalize'],
                task_type=emb_config['task_type']
            )
        )
    
    def _init_vectorstore(self):
        """Initialize vector store"""
        vs_config = self.config['vectorstore']
        vs_config['vector_size'] = self.config['embeddings']['dimension']
        self.vectorstore = QdrantVectorStore(vs_config)
    
    def _init_processors(self):
        """Initialize document processors"""
        self.pdf_processor = PDFProcessor(self.config['processing'])
        self.chunker = AdvancedChunker(self.config['processing'])
        self.citation_processor = CitationProcessor()
        self.equation_processor = EquationProcessor()
    
    def _init_llm(self):
        """Initialize LLM"""
        self.llm = OpenAICompatibleLLM(self.config['llm'])
    
    def _init_retriever(self):
        """Initialize retriever"""
        self.retriever = HybridRetriever(
            self.vectorstore,
            self.embedder,
            self.config
        )
        
        # Also init section-aware retriever
        self.section_retriever = SectionAwareRetriever(
            self.vectorstore,
            self.embedder,
            self.config
        )
    
    def _init_agent(self):
        """Initialize agent"""
        self.agent = ScientificRAGAgent(
            self.retriever,
            self.llm,
            self.config['agent']
        )
    
    def ingest_papers(self, pdf_paths: List[str]):
        """Ingest multiple papers"""
        all_chunks = []
        
        for pdf_path in pdf_paths:
            print(f"Processing: {pdf_path}")
            
            # Process PDF
            documents = self.pdf_processor.process_pdf(pdf_path)
            
            # Chunk documents
            chunks = self.chunker.chunk_documents(documents)
            
            # Extract equations and citations for each chunk
            for chunk in chunks:
                # Extract equations
                equations = self.equation_processor.extract_latex_equations(
                    chunk.text
                )
                chunk.metadata['equations'] = [eq['raw'] for eq in equations]
                
                # Extract citations
                citations = self.citation_processor.extract_citations(chunk.text)
                chunk.metadata['citations'] = citations
            
            all_chunks.extend(chunks)
        
        print(f"Total chunks: {len(all_chunks)}")
        
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
        
        print("Ingestion complete!")
    
    def query(self, question: str) -> Dict[str, Any]:
        """Query the system"""
        return self.agent.query(question)
    
    def query_simple(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """Simple query without agent (faster)"""
        # Retrieve
        results = self.retriever.retrieve(question, top_k=top_k)
        
        # Generate answer
        contexts = [r['text'] for r in results]
        answer = self.llm.generate_with_context(question, contexts)
        
        return {
            'answer': answer,
            'sources': results
        }