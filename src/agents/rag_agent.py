# src/agents/rag_agent.py - FIXED VERSION

import re
from typing import Dict, List, Any
from langgraph.graph import StateGraph, END
from typing import TypedDict

class AgentState(TypedDict):
    """State for RAG agent"""
    query: str
    sub_queries: List[str]
    retrieved_chunks: List[Dict]
    answer: str
    confidence: float
    verification_results: Dict
    iterations: int
    max_iterations: int
    retrieval_quality: float  # NEW: Track retrieval quality

class ScientificRAGAgent:
    """Agentic RAG with hallucination prevention"""
    
    def __init__(self, retriever, llm, config: Dict):
        self.retriever = retriever
        self.llm = llm
        self.config = config
        
        # CRITICAL: Thresholds for retrieval quality
        self.min_retrieval_score = config.get("min_retrieval_score", 0.5)
        self.min_chunks_required = config.get("min_chunks_required", 2)
        self.min_chunk_length = config.get("min_chunk_length", 100)
        
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build LangGraph workflow with quality gates"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("decompose_query", self.decompose_query)
        workflow.add_node("retrieve", self.retrieve_documents)
        workflow.add_node("check_retrieval_quality", self.check_retrieval_quality)
        workflow.add_node("rerank", self.rerank_documents)
        workflow.add_node("generate_answer", self.generate_answer)
        workflow.add_node("verify_answer", self.verify_answer)
        workflow.add_node("refine_answer", self.refine_answer)
        workflow.add_node("no_context_response", self.no_context_response)
        
        # Add edges
        workflow.set_entry_point("decompose_query")
        workflow.add_edge("decompose_query", "retrieve")
        workflow.add_edge("retrieve", "check_retrieval_quality")
        
        # CRITICAL: Branch based on retrieval quality
        workflow.add_conditional_edges(
            "check_retrieval_quality",
            self.has_sufficient_context,
            {
                "sufficient": "rerank",
                "insufficient": "no_context_response"
            }
        )
        
        workflow.add_edge("rerank", "generate_answer")
        workflow.add_edge("generate_answer", "verify_answer")
        
        workflow.add_conditional_edges(
            "verify_answer",
            self.should_refine,
            {
                "refine": "refine_answer",
                "end": END
            }
        )
        
        workflow.add_edge("refine_answer", END)
        workflow.add_edge("no_context_response", END)
        
        return workflow.compile()
    
    def check_retrieval_quality(self, state: AgentState) -> AgentState:
        """NEW: Assess if retrieved chunks are actually relevant"""
        
        chunks = state["retrieved_chunks"]
        
        if not chunks:
            state["retrieval_quality"] = 0.0
            return state
        
        # Check 1: Minimum number of chunks
        if len(chunks) < self.min_chunks_required:
            state["retrieval_quality"] = 0.3
            return state
        
        # Check 2: Average retrieval score
        scores = [c.get('score', 0) for c in chunks]
        avg_score = sum(scores) / len(scores) if scores else 0
        
        if avg_score < self.min_retrieval_score:
            state["retrieval_quality"] = 0.4
            return state
        
        # Check 3: Chunk content quality (not just table fragments)
        quality_chunks = 0
        for chunk in chunks[:5]:  # Check top 5
            text = chunk.get('text', '')
            
            # Filter out garbage chunks
            if len(text) < self.min_chunk_length:
                continue
            
            # Check if it's just table fragments (lots of | and None)
            if text.count('|') > len(text) / 10:  # More than 10% pipes
                continue
            
            if text.count('None') > 5:  # Too many "None" values
                continue
            
            # Check if it has actual words
            words = re.findall(r'\b[a-zA-Z]{3,}\b', text)
            if len(words) < 10:  # Less than 10 real words
                continue
            
            quality_chunks += 1
        
        # Quality score based on having good chunks
        state["retrieval_quality"] = min(quality_chunks / self.min_chunks_required, 1.0)
        
        return state
    
    def has_sufficient_context(self, state: AgentState) -> str:
        """Decide if we have enough context to answer"""
        
        # STRICT: Require minimum quality threshold
        if state["retrieval_quality"] < 0.7:
            print(f"⚠️ Low retrieval quality: {state['retrieval_quality']:.2f}")
            return "insufficient"
        
        return "sufficient"
    
    def no_context_response(self, state: AgentState) -> AgentState:
        """NEW: Handle queries with no relevant context"""
        
        # Create honest response
        state["answer"] = (
            f"I cannot find relevant information to answer this question in the available documents. "
            f"The documents I have access to appear to be about scientific topics, "
            f"but your question about '{state['query']}' does not match any content in my knowledge base.\n\n"
            f"Retrieval details:\n"
            f"- Retrieved {len(state['retrieved_chunks'])} chunks\n"
            f"- Quality score: {state['retrieval_quality']:.2f}\n"
            f"- Average relevance: {sum(c.get('score', 0) for c in state['retrieved_chunks'][:5]) / max(len(state['retrieved_chunks'][:5]), 1):.2f}\n\n"
            f"Please ask questions related to the scientific papers in the knowledge base."
        )
        
        state["confidence"] = 0.0
        state["verification_results"] = {
            'has_relevant_context': False,
            'answer_quality': 0.0
        }
        
        return state
    
    def generate_answer(self, state: AgentState) -> AgentState:
        """Generate answer with STRICT grounding instructions"""
        
        if not state["retrieved_chunks"]:
            state["answer"] = "No relevant context found."
            state["confidence"] = 0.0
            return state
        
        # Filter to only quality chunks
        quality_chunks = [
            c for c in state["retrieved_chunks"][:10]
            if len(c.get('text', '')) > self.min_chunk_length
            and c.get('text', '').count('|') < len(c.get('text', '')) / 10
        ]
        
        if not quality_chunks:
            state["answer"] = "Retrieved chunks do not contain sufficient information."
            state["confidence"] = 0.0
            return state
        
        context = "\n\n".join([
            f"[{i+1}] {chunk.get('text', '')}"
            for i, chunk in enumerate(quality_chunks)
        ])
        
        # CRITICAL: Very strict system prompt
        system_prompt = """You are a scientific assistant that ONLY answers based on provided context.

CRITICAL RULES:
1. If the context does NOT contain information to answer the question, you MUST say "The provided context does not contain information about [topic]."
2. NEVER use your general knowledge - ONLY use the context provided
3. If you cite something, it MUST be from the numbered context chunks [1], [2], etc.
4. If the context is irrelevant (e.g., just tables, equations without explanation), say so explicitly
5. Do NOT invent facts, citations, or explanations not present in the context

If you cannot answer from context, respond EXACTLY:
"I cannot answer this question based on the provided scientific documents. The context does not contain relevant information about [topic]."
"""
        
        prompt = f"""Context from scientific documents:
{context}

Question: {state['query']}

Answer (ONLY if context is relevant, otherwise state it's not answerable):"""
        
        try:
            answer = self.llm.generate(prompt, system_prompt=system_prompt)
            
            # DOUBLE-CHECK: Does answer claim no context?
            if any(phrase in answer.lower() for phrase in [
                "cannot answer",
                "does not contain",
                "no information",
                "not mentioned in the context",
                "context does not"
            ]):
                state["confidence"] = 0.0
            
            state["answer"] = answer
            
        except Exception as e:
            print(f"LLM generation error: {e}")
            state["answer"] = f"Error generating answer: {str(e)}"
            state["confidence"] = 0.0
        
        return state
    
    def verify_answer(self, state: AgentState) -> AgentState:
        """Enhanced verification with hallucination detection"""
        
        # If no good context, mark as unverified
        if state.get("retrieval_quality", 0) < 0.7:
            state["verification_results"] = {
                'has_relevant_context': False,
                'has_citations': False,
                'consistent_with_context': False,
                'no_hallucination': False
            }
            state["confidence"] = 0.0
            return state
        
        verification = {
            'has_relevant_context': state.get("retrieval_quality", 0) >= 0.7,
            'has_citations': self._check_citations(state["answer"]),
            'consistent_with_context': self._check_consistency(
                state["answer"],
                state["retrieved_chunks"]
            ),
            'no_hallucination': self._check_hallucination(
                state["answer"],
                state["retrieved_chunks"]
            ),
            'answer_is_refusal': self._is_refusal_answer(state["answer"])
        }
        
        # If it's a refusal (no context), that's actually GOOD
        if verification['answer_is_refusal']:
            state["confidence"] = 1.0  # High confidence in saying "I don't know"
        else:
            state["confidence"] = sum(v for k, v in verification.items() if k != 'answer_is_refusal') / 4
        
        state["verification_results"] = verification
        
        return state
    
    def _is_refusal_answer(self, answer: str) -> bool:
        """Check if answer correctly refuses to answer"""
        refusal_phrases = [
            "cannot answer",
            "cannot find",
            "does not contain",
            "no information",
            "no relevant",
            "not mentioned",
            "context does not"
        ]
        return any(phrase in answer.lower() for phrase in refusal_phrases)
    
    def _check_hallucination(self, answer: str, chunks: List[Dict]) -> bool:
        """Enhanced hallucination check"""
        
        if not chunks:
            return False
        
        # If answer is a refusal, no hallucination
        if self._is_refusal_answer(answer):
            return True
        
        # Extract key claims from answer
        sentences = answer.split('.')
        
        context_text = ' '.join([c.get('text', '')[:500] for c in chunks[:3]])
        
        # Check if claims appear in context
        hallucination_detected = False
        for sentence in sentences[:3]:  # Check first 3 sentences
            sentence = sentence.strip()
            if len(sentence) < 20:
                continue
            
            # Extract nouns/entities
            key_words = set(re.findall(r'\b[A-Z][a-z]+\b', sentence))
            
            if not key_words:
                continue
            
            # Check if ANY key words appear in context
            if not any(word in context_text for word in key_words):
                hallucination_detected = True
                break
        
        return not hallucination_detected
    
    def _check_consistency(self, answer: str, chunks: List[Dict]) -> bool:
        """Check consistency with stricter thresholds"""
        
        if not chunks or self._is_refusal_answer(answer):
            return True
        
        answer_words = set(re.findall(r'\b\w{4,}\b', answer.lower()))
        context_words = set()
        
        for chunk in chunks[:3]:
            text = chunk.get('text', '')
            context_words.update(re.findall(r'\b\w{4,}\b', text.lower()))
        
        if not answer_words:
            return False
        
        # STRICT: Require 50% overlap
        overlap = len(answer_words & context_words) / len(answer_words)
        return overlap > 0.5
    
    def _check_citations(self, answer: str) -> bool:
        """Check for proper citations"""
        if self._is_refusal_answer(answer):
            return True  # Refusal doesn't need citations
        
        return bool(re.search(r'\[\d+\]', answer))
    
    def should_refine(self, state: AgentState) -> str:
        """Decide refinement with convergence check"""
        
        # Don't refine refusal answers
        if self._is_refusal_answer(state["answer"]):
            return "end"
        
        # Don't refine if no context
        if state.get("retrieval_quality", 0) < 0.7:
            return "end"
        
        if state["confidence"] < self.config.get("confidence_threshold", 0.7):
            if state["iterations"] < state["max_iterations"]:
                state["iterations"] += 1
                return "refine"
        
        return "end"
    
    def query(self, question: str) -> Dict[str, Any]:
        """Main query interface with quality tracking"""
        initial_state = AgentState(
            query=question,
            sub_queries=[],
            retrieved_chunks=[],
            answer="",
            confidence=0.0,
            verification_results={},
            iterations=0,
            max_iterations=self.config.get("max_iterations", 2),
            retrieval_quality=0.0
        )
        
        final_state = self.graph.invoke(initial_state)
        
        return {
            "answer": final_state["answer"],
            "confidence": final_state["confidence"],
            "sources": final_state["retrieved_chunks"],
            "verification": final_state["verification_results"],
            "sub_queries": final_state["sub_queries"],
            "retrieval_quality": final_state.get("retrieval_quality", 0.0),
            "answerable": final_state.get("retrieval_quality", 0.0) >= 0.7
        }
    
    # Keep other methods (decompose_query, retrieve_documents, etc.) the same
    def decompose_query(self, state: AgentState) -> AgentState:
        """Decompose complex query into sub-queries"""
        if not self.config.get("use_query_decomposition", False):
            state["sub_queries"] = [state["query"]]
            return state
        
        prompt = f"""Given this scientific question, decompose it into 2-3 specific sub-questions.

Question: {state['query']}

Sub-questions (one per line):"""
        
        response = self.llm.generate(prompt, temperature=0.3)
        sub_queries = [q.strip() for q in response.split('\n') if q.strip()]
        
        state["sub_queries"] = sub_queries if sub_queries else [state["query"]]
        return state
    
    def retrieve_documents(self, state: AgentState) -> AgentState:
        """Retrieve for all sub-queries"""
        all_chunks = []
        
        for query in state["sub_queries"]:
            try:
                chunks = self.retriever.retrieve(query, top_k=10)
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"Retrieval error for query '{query}': {e}")
                continue
        
        # Deduplicate
        seen = set()
        unique_chunks = []
        for chunk in all_chunks:
            chunk_id = chunk.get('id', chunk.get('chunk_id', str(hash(chunk.get('text', '')))))
            if chunk_id not in seen:
                seen.add(chunk_id)
                unique_chunks.append(chunk)
        
        state["retrieved_chunks"] = unique_chunks
        
        if not unique_chunks:
            print(f"⚠️ No chunks retrieved for: {state['sub_queries']}")
        
        return state
    
    def rerank_documents(self, state: AgentState) -> AgentState:
        """Rerank with quality filtering"""
        chunks = state["retrieved_chunks"][:20]
        
        # Filter garbage chunks first
        quality_chunks = [
            c for c in chunks
            if len(c.get('text', '')) > self.min_chunk_length
            and c.get('text', '').count('|') < len(c.get('text', '')) / 10
            and c.get('text', '').count('None') < 5
        ]
        
        if not quality_chunks:
            state["retrieved_chunks"] = []
            return state
        
        if self.config.get("rerank", False):
            reranked = self._llm_rerank(state["query"], quality_chunks)
            state["retrieved_chunks"] = reranked[:self.config.get("top_k", 5)]
        else:
            state["retrieved_chunks"] = quality_chunks[:self.config.get("top_k", 5)]
        
        return state
    
    def _llm_rerank(self, query: str, chunks: List[Dict]) -> List[Dict]:
        """Fast reranking (limit LLM calls)"""
        for chunk in chunks[:5]:  # Only rerank top 5
            prompt = f"""Rate relevance (0-10):

Question: {query}
Text: {chunk['text'][:300]}

Score:"""
            
            try:
                score = float(self.llm.generate(prompt, temperature=0.0).strip())
            except:
                score = chunk.get('score', 0.5)
            
            chunk['rerank_score'] = score
        
        return sorted(chunks, key=lambda x: x.get('rerank_score', 0), reverse=True)
    
    def refine_answer(self, state: AgentState) -> AgentState:
        """Refine with context check"""
        
        if not state.get("retrieved_chunks"):
            return state
        
        issues = [k for k, v in state["verification_results"].items() if not v]
        
        context = state["retrieved_chunks"][0].get('text', '')[:500]
        
        prompt = f"""Previous answer had issues: {', '.join(issues)}

Question: {state['query']}
Previous: {state['answer']}
Context: {context}...

Improved answer (or state if unanswerable):"""
        
        try:
            refined = self.llm.generate(prompt, temperature=0.2)
            state["answer"] = refined
        except Exception as e:
            print(f"Refinement error: {e}")
        
        return state