class CitationGraph:
    """Citation graph for papers"""

    def __init__(self):
        self.graph = {}  # paper_id -> {'cites': [...], 'cited_by': [...]}

    def add_paper(self, paper_id: str, citations: list):
        if paper_id not in self.graph:
            self.graph[paper_id] = {'cites': [], 'cited_by': []}
        self.graph[paper_id]['cites'] = citations
        # Add reverse edges
        for cited in citations:
            if cited not in self.graph:
                self.graph[cited] = {'cites': [], 'cited_by': []}
            self.graph[cited]['cited_by'].append(paper_id)

    def get_neighbors(self, paper_id: str, direction='cites', max_depth=1):
        results = set()
        queue = [(paper_id, 0)]
        while queue:
            node, depth = queue.pop(0)
            if depth < max_depth:
                for neighbor in self.graph.get(node, {}).get(direction, []):
                    if neighbor not in results:
                        results.add(neighbor)
                        queue.append((neighbor, depth + 1))
        return list(results)
