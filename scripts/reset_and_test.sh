#!/bin/bash

echo "Resetting RAG system..."

# Delete collection
curl -X DELETE 'http://localhost:6333/collections/scientific_papers' 2>/dev/null

echo "Re-ingesting papers..."
python3 scripts/ingest_papers.py --papers ~/Downloads/20050613.pdf

echo -e "\nQuerying system..."
python3 scripts/query_system.py --question "What is Xe-135?"
