import networkx as nx
import numpy as np
from typing import Any


def max_flow(G: nx.DiGraph, s: Any, t: Any) -> int:
    value: int = 0
    
    current_flow = np.inf

    while nx.shortest_path(G, s, t):
        path = nx.shortest_path(G, s, t)
        for u,v in zip(path, path[1:]):
            current_flow = min(current_flow, G[u][v]['weight'])
        for u,v in zip(path, path[1:]):
            G[u][v]['weight'] -= current_flow
            if G[u][v]['weight'] == 0:
                G.remove_edge(u, v)
        value += current_flow
        current_flow = np.inf

    return value

if __name__ == "__main__":
    # Load the graph
    G = nx.read_edgelist("practicum_3/homework/advanced/graph_1.edgelist", create_using=nx.DiGraph)
    val = max_flow(G, s='0', t='5')
    print(f"Maximum flow is {val}. Should be 23")
