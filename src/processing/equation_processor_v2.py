# src/processing/equation_processor_v2.py
import re
import hashlib
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class Equation:
    raw: str
    normalized: str
    type: str  # 'inline' or 'display'
    position: int
    hash: str
    variables: List[str]
    operators: List[str]

class RobustEquationProcessor:
    """Production-grade equation extraction without fragile parsing"""
    
    def __init__(self):
        # Common LaTeX environments
        self.display_envs = [
            r'\\begin{equation}(.*?)\\end{equation}',
            r'\\begin{equation\*}(.*?)\\end{equation\*}',
            r'\\begin{align}(.*?)\\end{align}',
            r'\\begin{gather}(.*?)\\end{gather}',
            r'\\\[(.*?)\\\]',
            r'\$\$(.*?)\$\$'
        ]
        
        self.inline_patterns = [
            r'\\\((.*?)\\\)',
            r'\$([^\$\n]{1,200})\$'  # Limit inline length
        ]
        
        # Physics-specific symbols
        self.physics_symbols = {
            'hbar', 'psi', 'phi', 'theta', 'alpha', 'beta', 'gamma',
            'partial', 'nabla', 'int', 'sum', 'prod'
        }
    
    def extract_equations(self, text: str) -> List[Equation]:
        """Extract all equations with robust fallback"""
        equations = []
        seen_hashes = set()
        
        # Extract display equations
        for pattern in self.display_envs:
            for match in re.finditer(pattern, text, re.DOTALL):
                eq = self._process_match(match, 'display', seen_hashes)
                if eq:
                    equations.append(eq)
        
        # Extract inline equations
        for pattern in self.inline_patterns:
            for match in re.finditer(pattern, text):
                eq = self._process_match(match, 'inline', seen_hashes)
                if eq:
                    equations.append(eq)
        
        return equations
    
    def _process_match(
        self, 
        match: re.Match, 
        eq_type: str, 
        seen_hashes: set
    ) -> Equation | None:
        """Process a single equation match"""
        raw = match.group(1).strip()
        
        # Skip if empty or too short
        if not raw or len(raw) < 2:
            return None
        
        # Normalize
        normalized = self.normalize_equation(raw)
        eq_hash = hashlib.md5(normalized.encode()).hexdigest()
        
        # Skip duplicates
        if eq_hash in seen_hashes:
            return None
        seen_hashes.add(eq_hash)
        
        # Extract features
        variables = self._extract_variables(normalized)
        operators = self._extract_operators(normalized)
        
        return Equation(
            raw=raw,
            normalized=normalized,
            type=eq_type,
            position=match.start(),
            hash=eq_hash,
            variables=variables,
            operators=operators
        )
    
    def normalize_equation(self, latex_str: str) -> str:
        """Robust string-based normalization - NO SymPy parsing"""
        
        # Remove comments
        clean = re.sub(r'%.*$', '', latex_str, flags=re.MULTILINE)
        
        # Remove size/spacing commands
        clean = re.sub(r'\\(?:left|right|big|Big|bigg|Bigg)', '', clean)
        clean = re.sub(r'\\[,;:!]', '', clean)
        
        # Standardize commands
        clean = re.sub(r'\\dfrac', r'\\frac', clean)
        clean = re.sub(r'\\tfrac', r'\\frac', clean)
        
        # Remove unnecessary braces around single chars
        clean = re.sub(r'\{([a-zA-Z0-9])\}', r'\1', clean)
        
        # Normalize whitespace
        clean = re.sub(r'\s+', ' ', clean)
        
        # Sort terms in commutative operations (heuristic)
        # E.g., "a + b" and "b + a" should be similar
        clean = self._normalize_commutative(clean)
        
        return clean.strip()
    
    def _normalize_commutative(self, expr: str) -> str:
        """Normalize commutative operations (addition, multiplication)"""
        # Simple heuristic: sort terms in sums
        # This is NOT mathematically rigorous but helps with matching
        
        # Split on + and - (preserve signs)
        terms = re.split(r'([+-])', expr)
        
        if len(terms) > 2:
            # Sort terms while preserving operators
            term_pairs = []
            for i in range(0, len(terms), 2):
                term = terms[i]
                op = terms[i+1] if i+1 < len(terms) else ''
                term_pairs.append((term.strip(), op))
            
            # Sort by term content
            term_pairs.sort(key=lambda x: x[0])
            
            # Reconstruct
            return ''.join(f'{term}{op}' for term, op in term_pairs)
        
        return expr
    
    def _extract_variables(self, latex_str: str) -> List[str]:
        """Extract variable names"""
        # Single letters (a-z, A-Z) not part of commands
        variables = set(re.findall(r'(?<!\\)\b([a-zA-Z])\b', latex_str))
        
        # Greek letters
        greek = set(re.findall(r'\\(alpha|beta|gamma|delta|theta|phi|psi|omega)', latex_str))
        
        return sorted(variables | greek)
    
    def _extract_operators(self, latex_str: str) -> List[str]:
        """Extract mathematical operators"""
        operators = set()
        
        # Standard operators
        operators.update(re.findall(r'([+\-*/=<>])', latex_str))
        
        # LaTeX operators
        ops = ['frac', 'int', 'sum', 'prod', 'partial', 'nabla', 'times', 'cdot']
        for op in ops:
            if f'\\{op}' in latex_str:
                operators.add(op)
        
        return sorted(operators)
    
    def compute_similarity(self, eq1: str, eq2: str) -> float:
        """Compute equation similarity without parsing"""
        
        # Normalize both
        norm1 = self.normalize_equation(eq1)
        norm2 = self.normalize_equation(eq2)
        
        # Exact match
        if norm1 == norm2:
            return 1.0
        
        # Feature-based similarity
        vars1 = set(self._extract_variables(norm1))
        vars2 = set(self._extract_variables(norm2))
        
        ops1 = set(self._extract_operators(norm1))
        ops2 = set(self._extract_operators(norm2))
        
        # Jaccard similarity
        var_sim = len(vars1 & vars2) / len(vars1 | vars2) if vars1 | vars2 else 0
        op_sim = len(ops1 & ops2) / len(ops1 | ops2) if ops1 | ops2 else 0
        
        # String similarity (edit distance)
        from difflib import SequenceMatcher
        str_sim = SequenceMatcher(None, norm1, norm2).ratio()
        
        # Weighted combination
        return 0.4 * var_sim + 0.3 * op_sim + 0.3 * str_sim
    
    def is_physics_equation(self, equation: Equation) -> bool:
        """Heuristic to identify physics equations"""
        # Check for physics symbols
        for symbol in self.physics_symbols:
            if symbol in equation.variables or f'\\{symbol}' in equation.raw:
                return True
        
        # Check for common physics patterns
        physics_patterns = [
            r'\\hbar',
            r'\\partial',
            r'\\nabla',
            r'\\hat{H}',  # Hamiltonian
            r'\\vec',
            r'mc\^2'
        ]
        
        return any(re.search(p, equation.raw) for p in physics_patterns)