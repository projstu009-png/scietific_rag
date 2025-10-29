# src/evaluation/benchmark.py
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from datasets import Dataset
import json
from typing import List, Dict
import time
import numpy as np
from typing import Any 

class RAGBenchmark:
    """Comprehensive benchmarking for RAG system"""
    
    def __init__(self, rag_system, config: Dict):
        self.rag_system = rag_system
        self.config = config
        self.results = []
    
    def load_test_dataset(self, path: str) -> List[Dict]:
        """Load test questions with ground truth"""
        with open(path, 'r') as f:
            return json.load(f)
    
    def run_benchmark(self, test_data: List[Dict]) -> Dict[str, Any]:
        """Run complete benchmark"""
        results = {
            'system_metrics': [],
            'ragas_scores': {},
            'latency_stats': {},
            'per_question_results': []
        }
        
        latencies = []
        
        for item in test_data:
            question = item['question']
            ground_truth = item['ground_truth']
            
            # Measure latency
            start_time = time.time()
            response = self.rag_system.query(question)
            latency = time.time() - start_time
            latencies.append(latency)
            
            # Store result
            result = {
                'question': question,
                'answer': response['answer'],
                'ground_truth': ground_truth,
                'retrieved_contexts': [c['text'] for c in response['sources']],
                'confidence': response['confidence'],
                'latency': latency
            }
            results['per_question_results'].append(result)
        
        # Compute RAGAS metrics
        ragas_dataset = Dataset.from_dict({
            'question': [r['question'] for r in results['per_question_results']],
            'answer': [r['answer'] for r in results['per_question_results']],
            'contexts': [r['retrieved_contexts'] for r in results['per_question_results']],
            'ground_truth': [r['ground_truth'] for r in results['per_question_results']]
        })
        
        ragas_results = evaluate(
            ragas_dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall
            ]
        )
        
        results['ragas_scores'] = ragas_results
        
        # Latency statistics
        results['latency_stats'] = {
            'mean': np.mean(latencies),
            'median': np.median(latencies),
            'p95': np.percentile(latencies, 95),
            'p99': np.percentile(latencies, 99)
        }
        
        return results
    
    def compare_configurations(
        self,
        configs: List[Dict],
        test_data: List[Dict]
    ) -> Dict[str, Any]:
        """Compare different RAG configurations"""
        comparison = {}
        
        for config_name, config in configs:
            print(f"Testing configuration: {config_name}")
            
            # Reinitialize system with new config
            rag = self._init_rag_with_config(config)
            
            # Run benchmark
            results = self.run_benchmark(test_data)
            comparison[config_name] = results
        
        return comparison
    
    def _init_rag_with_config(self, config: Dict):
        """Initialize RAG system with specific config"""
        # Implementation depends on your RAG initialization
        pass