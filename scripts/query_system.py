# scripts/query_system.py - IMPROVED VERSION

import argparse
from pathlib import Path
import sys
import json

sys.path.append(str(Path(__file__).parent.parent))

from src.rag_system import ScientificRAGSystem


def format_source(source: dict, index: int) -> str:
    """Format a single source nicely"""
    score = source.get('score', 0)
    text = source.get('text', '')
    
    # Truncate long text
    display_text = text[:300] + "..." if len(text) > 300 else text
    
    # Clean up whitespace
    display_text = ' '.join(display_text.split())
    
    return f"[{index}] Score: {score:.3f}\n{display_text}\n"


def print_result(result: dict, show_all_sources: bool = False):
    """Print query result with smart source handling"""
    
    print("\n" + "="*80)
    print("ANSWER:")
    print("="*80)
    print(result['answer'])
    
    # Check if query was answerable
    is_answerable = result.get('answerable', True)
    retrieval_quality = result.get('retrieval_quality', 1.0)
    confidence = result.get('confidence', 0.0)
    
    print("\n" + "="*80)
    print("METADATA:")
    print("="*80)
    print(f"Answerable: {'Yes' if is_answerable else 'No'}")
    print(f"Retrieval Quality: {retrieval_quality:.2f}")
    print(f"Confidence: {confidence:.2f}")
    
    # Only show sources if query was answerable
    if is_answerable and retrieval_quality >= 0.7:
        print("\n" + "="*80)
        print("RELEVANT SOURCES:")
        print("="*80)
        
        # Filter quality sources
        quality_sources = [
            s for s in result['sources'][:10]
            if s.get('score', 0) > 0.5 and len(s.get('text', '')) > 100
        ]
        
        if quality_sources:
            num_to_show = len(quality_sources) if show_all_sources else min(3, len(quality_sources))
            for i, source in enumerate(quality_sources[:num_to_show], 1):
                print(format_source(source, i))
        else:
            print("No high-quality sources available.")
    
    elif result['sources']:
        # Show why sources were rejected
        print("\n" + "="*80)
        print("RETRIEVAL DIAGNOSTICS:")
        print("="*80)
        print(f"Retrieved {len(result['sources'])} chunks, but none were relevant:")
        print(f"- Average score: {sum(s.get('score', 0) for s in result['sources'][:5]) / min(len(result['sources']), 5):.3f}")
        print(f"- Retrieval quality: {retrieval_quality:.2f} (threshold: 0.7)")
        
        if show_all_sources:
            print("\n⚠️ Retrieved chunks (for debugging):")
            for i, source in enumerate(result['sources'][:3], 1):
                text_preview = source.get('text', '')[:150]
                print(f"[{i}] Score: {source.get('score', 0):.3f}")
                print(f"Preview: {text_preview}...\n")
    
    # Show verification details if available
    if result.get('verification'):
        print("\n" + "="*80)
        print("VERIFICATION:")
        print("="*80)
        for key, value in result['verification'].items():
            status = "✓" if value else "✗"
            print(f"{status} {key}: {value}")


def main():
    parser = argparse.ArgumentParser(description='Query the RAG system')
    parser.add_argument('--question', required=True, help='Question to ask')
    parser.add_argument('--config', default='config/config.yaml',
                       help='Config file path')
    parser.add_argument('--mode', choices=['agent', 'simple'], default='agent',
                       help='Query mode')
    parser.add_argument('--output', help='Output JSON file (optional)')
    parser.add_argument('--show-all-sources', action='store_true',
                       help='Show all sources even if low quality')
    parser.add_argument('--debug', action='store_true',
                       help='Show debug information')
    
    args = parser.parse_args()
    
    # Initialize system
    print("Initializing RAG system...")
    rag = ScientificRAGSystem(args.config)
    
    # Query
    print(f"\nProcessing query: '{args.question}'")
    
    if args.mode == 'agent':
        result = rag.query(args.question)
    else:
        result = rag.query_simple(args.question)
    
    # Print result
    print_result(result, show_all_sources=args.show_all_sources or args.debug)
    
    # Save if requested
    if args.output:
        # Convert to JSON-serializable format
        output_data = {
            'question': args.question,
            'answer': result['answer'],
            'confidence': result.get('confidence', 0.0),
            'answerable': result.get('answerable', True),
            'retrieval_quality': result.get('retrieval_quality', 1.0),
            'num_sources': len(result.get('sources', [])),
            'verification': result.get('verification', {}),
        }
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\n✓ Result saved to {args.output}")


if __name__ == '__main__':
    main()