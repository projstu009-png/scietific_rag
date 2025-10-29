# scripts/ingest_papers.py
import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.rag_system import ScientificRAGSystem


def main():
    parser = argparse.ArgumentParser(description='Ingest scientific papers')
    parser.add_argument('--papers', nargs='+', required=True, 
                       help='Paths to PDF files')
    parser.add_argument('--config', default='config/config.yaml',
                       help='Config file path')
    
    args = parser.parse_args()
    
    # Initialize system
    rag = ScientificRAGSystem(args.config)
    
    # Ingest papers
    rag.ingest_papers(args.papers)
    
    print("Done!")


if __name__ == '__main__':
    main()