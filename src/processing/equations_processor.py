# src/processing/equation_processor.py
from sympy.parsing.latex import parse_latex
from sympy import simplify, latex
import re
from typing import List, Dict, Any
import hashlib

class EquationProcessor:
    """Advanced equation extraction and normalization"""
    
        
    def __init__(self):
        # Use both ANTLR and Lark backends
        self.backends = ['antlr', 'lark']

    def extract_latex_equations(self, text: str) -> List[Dict[str, Any]]:
        """Extract LaTeX with better patterns"""
        patterns = {
            'display': [
                r'\\begin{equation}(.*?)\\end{equation}',
                r'\\begin{align}(.*?)\\end{align}',
                r'\\begin{gather}(.*?)\\end{gather}',
                r'\\\[(.*?)\\\]',
                r'\$\$(.*?)\$\$'
            ],
            'inline': [
                r'\\\((.*?)\\\)',
                r'\$([^\$]+)\$'
            ]
        }
        
        equations = []
        for eq_type, pattern_list in patterns.items():
            for pattern in pattern_list:
                for match in re.finditer(pattern, text, re.DOTALL):
                    eq = match.group(1).strip()
                    normalized = self.normalize_equation(eq)
                    equations.append({
                        'raw': eq,
                        'normalized': normalized,
                        'type': eq_type,
                        'position': match.start(),
                        'hash': hashlib.md5(normalized.encode()).hexdigest()
                    })
        return equations

        
    def normalize_equation(self, latex_str: str) -> str:
        """Try multiple parsing strategies"""
        # Try default parser first
        try:
            expr = parse_latex(latex_str)
            normalized = simplify(expr)
            return latex(normalized)
        except Exception as e:
            # Try with transformers for complex expressions
            try:
                from sympy.parsing.latex.lark import parse_latex as lark_parse
                expr = lark_parse(latex_str)
                normalized = simplify(expr)
                return latex(normalized)
            except:
                pass
        
        # Critical fallback: string-based normalization
        return self._fallback_normalize(latex_str)
    
    def _fallback_normalize(self, latex_str: str) -> str:
        """Robust string-based normalization for unparseable LaTeX"""
        # Remove formatting commands
        clean = re.sub(r'\\(left|right|big|Big|bigg|Bigg)', '', latex_str)
        # Normalize spacing
        clean = re.sub(r'\\,|\\;|\\!|\\:', '', clean)
        # Standardize fractions
        clean = re.sub(r'\\dfrac', r'\\frac', clean)
        # Remove unnecessary braces
        clean = re.sub(r'\{([^{}])\}', r'\1', clean)
        return clean.strip()
    
    def compute_equation_similarity(self, eq1: str, eq2: str) -> float:
        """Semantic equation similarity"""
        try:
            expr1 = parse_latex(eq1)
            expr2 = parse_latex(eq2)
            # Check symbolic equivalence
            return 1.0 if simplify(expr1 - expr2) == 0 else 0.0
        except:
            # Fallback to string similarity
            from difflib import SequenceMatcher
            return SequenceMatcher(None, eq1, eq2).ratio()