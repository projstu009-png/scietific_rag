import pymupdf  # PyMuPDF
import pymupdf4llm
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import re

@dataclass
class Document:
    text: str
    metadata: Dict[str, Any]
    page_number: Optional[int] = None

class PDFProcessor:
    """Advanced PDF processor for scientific papers"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.preserve_equations = config.get("preserve_equations", True)
        self.extract_citations = config.get("extract_citations", True)
        self.extract_tables = config.get("extract_tables", True)
    
    def process_pdf(self, pdf_path: str) -> List[Document]:
        """Process PDF and extract structured content"""
        doc = pymupdf.open(pdf_path)
        documents = []
        
        # Extract metadata
        metadata = self._extract_metadata(doc)
        
        # Process each page
        for page_num, page in enumerate(doc):
            # Extract text with formatting
            text = self._extract_text_with_formatting(page)
            
            # Extract equations if enabled
            if self.preserve_equations:
                equations = self._extract_equations(page)
                text = self._integrate_equations(text, equations)
            
            # Extract tables if enabled
            if self.extract_tables:
                tables = self._extract_tables(page)
                text = self._integrate_tables(text, tables)
            
            # Extract citations if enabled
            citations = []
            if self.extract_citations:
                citations = self._extract_citations(text)
            
            page_metadata = {
                **metadata,
                "page_number": page_num + 1,
                "citations": citations
            }
            
            documents.append(Document(
                text=text,
                metadata=page_metadata,
                page_number=page_num + 1
            ))
        
        doc.close()
        return documents
    
    def process_pdf_with_pymupdf4llm(self, pdf_path: str) -> str:
        """Use PyMuPDF4LLM for LLM-optimized extraction"""
        md_text = pymupdf4llm.to_markdown(pdf_path)
        return md_text
    
    def _extract_metadata(self, doc: pymupdf.Document) -> Dict[str, Any]:
        """Extract PDF metadata"""
        metadata = doc.metadata
        
        return {
            "title": metadata.get("title", ""),
            "author": metadata.get("author", ""),
            "subject": metadata.get("subject", ""),
            "keywords": metadata.get("keywords", ""),
            "num_pages": len(doc),
            "format": metadata.get("format", "")
        }
    
    def _extract_text_with_formatting(self, page: pymupdf.Page) -> str:
        """Extract text while preserving formatting"""
        # Use dict mode for structured extraction
        blocks = page.get_text("dict")["blocks"]
        
        text_content = []
        for block in blocks:
            if block["type"] == 0:  # Text block
                for line in block["lines"]:
                    line_text = ""
                    for span in line["spans"]:
                        line_text += span["text"]
                    text_content.append(line_text)
        
        return "\n".join(text_content)
    
    def _extract_equations(self, page: pymupdf.Page) -> List[str]:
        """Extract mathematical equations (simplified)"""
        # This is a simplified version
        # For production, consider using specialized LaTeX extraction tools
        text = page.get_text()
        
        # Detect inline math (simplified pattern)
        inline_math = re.findall(r'\$(.+?)\$', text)
        
        # Detect display math (simplified pattern)
        display_math = re.findall(r'\$\$(.+?)\$\$', text, re.DOTALL)
        
        return inline_math + display_math
    
    def _integrate_equations(self, text: str, equations: List[str]) -> str:
        """Integrate equations into text"""
        if equations:
            equations_str = "\n\nEquations:\n" + "\n".join(f"- {eq}" for eq in equations)
            text += equations_str
        return text
    
    def _extract_tables(self, page: pymupdf.Page) -> List[Dict[str, Any]]:
        """Extract tables from page"""
        tables = page.find_tables()
        
        extracted_tables = []
        for table in tables:
            table_data = table.extract()
            extracted_tables.append({
                "data": table_data,
                "bbox": table.bbox
            })
        
        return extracted_tables
    
    def _integrate_tables(self, text: str, tables: List[Dict[str, Any]]) -> str:
        """Integrate tables into text"""
        if tables:
            tables_str = "\n\nTables:\n"
            for i, table in enumerate(tables, 1):
                tables_str += f"\nTable {i}:\n"
                for row in table["data"]:
                    tables_str += " | ".join(str(cell) for cell in row) + "\n"
            text += tables_str
        return text
    
    def _extract_citations(self, text: str) -> List[str]:
        """Extract citations from text"""
        # Pattern for common citation formats
        patterns = [
            r'\[(\d+)\]',  # [1], [2]
            r'\(([A-Z][a-z]+\s+et\s+al\.,?\s+\d{4})\)',  # (Author et al., 2020)
            r'\(([A-Z][a-z]+\s+and\s+[A-Z][a-z]+,?\s+\d{4})\)'  # (Author and Author, 2020)
        ]
        
        citations = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            citations.extend(matches)
        
        return list(set(citations))
