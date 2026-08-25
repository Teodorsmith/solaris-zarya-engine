import enum
import collections

class GraphCycleDetector:
    def __init__(self, graph):
        self.graph = graph
        self.visited = set()
        self.rec_stack = set()

    def is_cyclic_util(self, node):
        if node in self.rec_stack:
            return True
        if node in self.visited:
            return False

        self.visited.add(node)
        self.rec_stack.add(node)

        for neighbor in self.graph[node]:
            if self.is_cyclic_util(neighbor):
                return True

        self.rec_stack.remove(node)
        return False

    def is_cyclic(self):
        for node in self.graph:
            if self.is_cyclic_util(node):
                return True
        return False

def execute(graph=None, **kwargs) -> dict:
    if graph is None:
        return {
            "result": "Graph is required",
            "error": True
        }

    detector = GraphCycleDetector(graph)
    result = detector.is_cyclic()
    return {
        "result": result
    }