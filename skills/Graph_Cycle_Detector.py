import json
from typing import List, Dict


def detect_cycle(graph: Dict[int, List[int]]) -> bool:
    visited = set()
    rec_stack = set()

    def dfs(node: int) -> bool:
        if node in rec_stack:
            return True
        if node in visited:
            return False

        visited.add(node)
        rec_stack.add(node)

        for neighbor in graph[node]:
            if dfs(neighbor):
                return True

        rec_stack.remove(node)
        return False

    for node in graph:
        if dfs(node):
            return True

    return False


def execute(edges: List[List[int]] = [], **kwargs) -> dict:
    graph = {node: [] for node in set(sum(edges, []))}
    for u, v in edges:
        graph[u].append(v)

    return {'result': detect_cycle(graph)}


if __name__ == '__main__':
    print(json.dumps(execute(edges=[[0, 1], [1, 2], [2, 0]])))
    print(json.dumps(execute(edges=[[0, 1], [1, 2], [2, 3]])))
