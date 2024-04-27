from typing import Any, Union
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# def dfs(graph: nx.DiGraph, s, t, vis_nodes, vis_edges: list, value):

#     vis_nodes.add(s)

#     print(f'vis_nodes={vis_nodes}')

#     for neighbor in graph.neighbors(s):

#         if graph[s][neighbor]['weight'] == 0:
#             continue

#         if neighbor in vis_nodes:
#             continue

#         vis_edges.append((s, neighbor))
#         print(f'vis edges={vis_edges}')
#         current_path_flow = min(graph[i][j]['weight'] for i, j in vis_edges)
#         print(f'current path flow={current_path_flow}')
        
#         if neighbor not in vis_nodes and neighbor != t:
#             print('HUY')
#             dfs(graph, neighbor, t, vis_nodes, vis_edges, value)

#         if neighbor == t:
#             print('huy')

#             for i, j in vis_edges:
#                 graph[i][j]['weight'] -= current_path_flow
#                 print(f'graph[{i}][{j}][weight] = {graph[i][j]['weight']}')

#             value += current_path_flow

#             print(f'value={value}')
#             vis_edges.pop()
#             vis_nodes.pop()
        
#     return value
                

# def max_flow(G: nx.DiGraph, s: Any, t: Any) -> int:
#     value: int = 0
#     visited = set()
#     vis_edges = list()

#     value = dfs(G, s, t, visited, vis_edges, value)
    
#     return value

# graph[neighbor][s]['weight'] += result

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

def dfs(graph, s, t, visited, vis_edges, current_flow) -> int:
    if s == t:
        return current_flow
    
    
    visited.add(s)

    for neighbor in graph.neighbors(s):
        print(graph.neighbors(s), 'huy')
        if graph[s][neighbor]['weight'] == 0:
            vis_edges.add((s, neighbor))
            continue
        if neighbor in visited:
            continue

        current_flow = min(current_flow, graph[s][neighbor]['weight'])
        result = dfs(graph, neighbor, t, visited, vis_edges, current_flow)          

        if result > 0:
            graph[s][neighbor]['weight'] -= result
            print(graph[s][neighbor]['weight'])
            return result         
    
    return 0

def max_flow(G: nx.DiGraph, s: Any, t: Any) -> int:
    value: int = 0
    while True:
        flow = dfs(G, s, t, set(), set(), np.inf)
        if flow == 0:
            break
        value += flow

    return value


if __name__ == "__main__":
    # Load the graph
    G = nx.read_edgelist("practicum_3/homework/advanced/graph_1.edgelist", create_using=nx.DiGraph)

    plot_graph(G)
    val = max_flow(G, s='0', t='5')
    print(f"Maximum flow is {val}. Should be 23")
