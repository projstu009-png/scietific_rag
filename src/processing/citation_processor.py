# src/processing/citation_processor.py
import re
from typing import List, Dict, Set
# src/processing/citation_processor.py - ADD at top:
try:
    from scholarly import scholarly
    SCHOLARLY_AVAILABLE = True
except ImportError:
    SCHOLARLY_AVAILABLE = False
    print("Warning: scholarly library not available. Citation lookup disabled.")

class CitationProcessor:
    """Extract and verify scientific citations"""
    
    def extract_citations(self, text: str) -> Dict[str, List[str]]:
        """Extract all citation types"""
        citations = {
            'numeric': self._extract_numeric(text),  # [1], [2]
            'author_year': self._extract_author_year(text),  # (Smith et al., 2020)
            'arxiv': self._extract_arxiv(text),  # arXiv:2104.12345
            'doi': self._extract_doi(text)  # doi:10.1234/xyz
        }
        return citations
    
    def _extract_numeric(self, text: str) -> List[str]:
        return re.findall(r'\[(\d+(?:,\s*\d+)*)\]', text)
    
    def _extract_author_year(self, text: str) -> List[str]:
        patterns = [
            r'\(([A-Z][a-z]+(?:\s+et\s+al\.?)?[\s,]+\d{4}[a-z]?)\)',
            r'([A-Z][a-z]+(?:\s+and\s+[A-Z][a-z]+)?[\s,]+\d{4}[a-z]?)'
        ]
        citations = []
        for pattern in patterns:
            citations.extend(re.findall(pattern, text))
        return citations
    
    def _extract_arxiv(self, text: str) -> List[str]:
        return re.findall(r'arXiv:(\d{4}\.\d{4,5})', text)
    
    def _extract_doi(self, text: str) -> List[str]:
        return re.findall(r'doi:\s*(10\.\d{4,9}/[-._;()/:\w]+)', text, re.IGNORECASE)
    
    def verify_citation_exists(self, citation: str, references: List[str]) -> bool:
        """Verify citation appears in reference list"""
        # Implement fuzzy matching against reference list
        pass
    
    def build_citation_graph(self, papers: List[Dict]) -> Dict:
        """Build citation network for graph-based retrieval"""
        graph = {}
        for paper in papers:
            paper_id = paper['paper_id']
            citations = paper.get('citations', [])
            graph[paper_id] = {
                'cites': citations,
                'cited_by': []
            }
        
        # Build reverse edges
        for paper_id, data in graph.items():
            for cited in data['cites']:
                if cited in graph:
                    graph[cited]['cited_by'].append(paper_id)
        
        return graph