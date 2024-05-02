from typing import Any, Union
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def plot_graph(
    G: Union[nx.Graph, nx.DiGraph], highlighted_edges: list[tuple[Any, Any]] = None
) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    pos = nx.spring_layout(G)
    edge_color_list = ["black"] * len(G.edges)
    if highlighted_edges:
        for i, edge in enumerate(G.edges()):
            if edge in highlighted_edges or (edge[1], edge[0]) in highlighted_edges:
                edge_color_list[i] = "red"
    options = dict(
        font_size=12,
        node_size=500,
        node_color="white",
        edgecolors="black",
        edge_color=edge_color_list,
    )
    nx.draw_networkx(G, pos, ax=ax, **options)
    if nx.is_weighted(G):
        labels = {e: G.edges[e]["weight"] for e in G.edges}
        nx.draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=labels)
    plt.show()

def dfs(graph: nx.DiGraph, s, t, visited, current_flow) -> int:
    if s == t:
        return current_flow
        
    visited.add(s)

    for neighbor in graph.neighbors(s):

        print(f's = {s}')
        print(f'neighbor = {neighbor}')
        print(graph[s][neighbor]['weight'])
        

        if graph[s][neighbor]['weight'] == 0:
            continue
        if neighbor in visited:
            continue

        current_flow = min(current_flow, graph[s][neighbor]['weight'])
        print(f"cur flow = {current_flow}")
        print('-' * 32)
        result = dfs(graph, neighbor, t, visited, current_flow)          

        if result > 0:
            graph[s][neighbor]['weight'] -= result
            print(graph[s][neighbor]['weight'])
            return result     
            
    return 0

def max_flow(G: nx.DiGraph, s: Any, t: Any) -> int:
    value: int = 0
    while True:
        flow = dfs(G, s, t, set(), np.inf)
        value += flow
        if flow == 0:
            break
        
    return value


if __name__ == "__main__":
    # Load the graph
    G = nx.read_edgelist("practicum_3/homework/advanced/graph_1.edgelist", create_using=nx.DiGraph)

    plot_graph(G)
    val = max_flow(G, s='0', t='5')
    print(f"Maximum flow is {val}. Should be 23")
