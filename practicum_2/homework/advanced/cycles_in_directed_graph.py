import os

import networkx as nx

TEST_GRAPH_FILES = [
    "graph_1_wo_cycles.edgelist",
    "graph_2_wo_cycles.edgelist",
    "graph_3_w_cycles.edgelist",
]

def has_cycles(g: nx.DiGraph):

    def dfs(node, visited, stack):
        visited.add(node)
        stack.add(node)
            
        for neighbor in g.neighbors(node):
            if neighbor not in visited:
                if dfs(neighbor, visited, stack):
                    return True
            elif neighbor in stack:
                return True
        
        stack.remove(node)
        return False
        
    visited = set()
    stack = set()
        
    for node in g.nodes():
       if node not in visited:
            if dfs(node, visited, stack):
                return True
        
    return False




if __name__ == "__main__":
    for filename in TEST_GRAPH_FILES:
        # Load the graph
        G = nx.read_edgelist(f'C:/Users/zvnlxn/IT/spbu-fundamentals-of-algorithms/practicum_2/homework/advanced/{filename}', create_using=nx.DiGraph)
        # Output whether it has cycles
        print(f"Graph {filename} has cycles: {has_cycles(G)}")
