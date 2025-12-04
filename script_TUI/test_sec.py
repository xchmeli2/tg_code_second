import properties
import re
from graph import vystupni_okoli_uzlu
from collections import deque
from graph import Graph
from properties import is_directed
from graph import naslednici_uzlu
from edge import Edge
from collections import defaultdict, deque
from itertools import combinations
from matrix_operations import save_matrix_to_file


def natural_sort_key(name):
    return (len(name), [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', name)])
"""

def number_of_spanning_trees(graph):
    num_nodes = len(graph.nodes)
    
    required_edges = num_nodes - 1

    if len(graph.edges) < required_edges:
        return 0

    spanning_trees_count = 0

    for edge_subset in combinations(graph.edges, required_edges):
        subgraph = Graph()
        subgraph.nodes = graph.nodes.copy()
        subgraph.edges = list(edge_subset)

        if is_tree(subgraph):
            spanning_trees_count += 1

    return spanning_trees_count

def is_tree(graph):

    # Strom musí být spojený a nemá cykly
    return is_connected(graph) and len(graph.edges) == len(graph.nodes) - 1

def is_connected(graph):

    if not graph.nodes:
        return True

    visited = set()

    def dfs(node):
        if node in visited:
            return
        visited.add(node)
        for neighbor in graph.get_neighbors(node):
            dfs(neighbor)

    # Start DFS od libovolného uzlu
    start_node = next(iter(graph.nodes))
    dfs(start_node)

    return len(visited) == len(graph.nodes)

"""

def laplacian_matrix(graph):
    num_nodes = len(graph.nodes)
    L = [[0] * num_nodes for _ in range(num_nodes)]
    
    # Mapování uzlů na indexy 0..N
    node_index = {node: i for i, node in enumerate(graph.nodes)}

    # Místo get_neighbors projdeme "natvrdo" seznam všech hran
    processed_edges = set()
    for edge in graph.edges:
        # Zde předpokládám, že hrana má atributy 'source' a 'target' 
        # (nebo 'uzel1', 'uzel2', či indexy - upravte dle vaší třídy Edge)
        u_node = edge.node1 
        v_node = edge.node2
        
        if u_node == v_node:
            continue

        # Deduplikace pro neorientovaný graf
        pair = tuple(sorted((u_node, v_node)))
        if pair in processed_edges:
            continue
        processed_edges.add(pair)
        
        i = node_index[u_node]
        j = node_index[v_node]

        # Klíčová změna: Započítáme to oběma smery (ignorujeme šipky)
        
        # 1. Zvýšíme stupně na diagonále oběma
        L[i][i] += 1
        L[j][j] += 1
        
        # 2. Nastavíme -1 na obou symetrických pozicích
        L[i][j] = -1
        L[j][i] = -1

    return L

def determinant(matrix):
    n = len(matrix)
    A = [row[:] for row in matrix]
    det = 1

    for i in range(n):
        max_row = i
        for k in range(i + 1, n):
            if abs(A[k][i]) > abs(A[max_row][i]):
                max_row = k

        if abs(A[max_row][i]) < 1e-10:
            return 0

        if max_row != i:
            A[i], A[max_row] = A[max_row], A[i]
            det *= -1 

        det *= A[i][i]
        pivot = A[i][i]
        for j in range(i, n):
            A[i][j] /= pivot

        for k in range(i + 1, n):
            factor = A[k][i]
            for j in range(i, n):
                A[k][j] -= factor * A[i][j]

    return round(det)

def number_of_spanning_trees(graph):
    if len(graph.edges) < len(graph.nodes) - 1:
        return 0

    L = laplacian_matrix(graph)

    L_minor = [row[:-1] for row in L[:-1]]

    print("Výpočet determinantu minoru Laplaciánu...")
    det = determinant(L_minor)

    print("Hodnota determinantu:", det)

    return det


def minimum_spanning_tree(graph):
    parent = {}
    rank = {}

    def find(node):
        if parent[node] != node:
            parent[node] = find(parent[node])
        return parent[node]

    def union(node1, node2):
        root1 = find(node1)
        root2 = find(node2)

        if root1 != root2:
            if rank[root1] > rank[root2]:
                parent[root2] = root1
            elif rank[root1] < rank[root2]:
                parent[root1] = root2
            else:
                parent[root2] = root1
                rank[root1] += 1

    for node in graph.nodes:
        parent[node] = node
        rank[node] = 0

    sorted_edges = sorted(graph.edges, key=lambda edge: edge.weight if edge.weight is not None else float('inf'))

    mst = []  
    total_weight = 0 
    for edge in sorted_edges:
        node1, node2 = edge.node1, edge.node2
        if find(node1) != find(node2):
            union(node1, node2)
            mst.append(edge)
            total_weight += edge.weight if edge.weight is not None else 0

    edge_names = [edge.name for edge in mst if edge.name is not None]
    edge_names = [edge.name for edge in mst if edge.name is not None]
    print("Hrany v minimální kostře:", edge_names)

    return mst, total_weight


def maximum_spanning_tree(graph):
    parent = {}
    rank = {}

    def find(node):
        if parent[node] != node:
            parent[node] = find(parent[node])
        return parent[node]

    def union(node1, node2):
        root1 = find(node1)
        root2 = find(node2)

        if root1 != root2:
            if rank[root1] > rank[root2]:
                parent[root2] = root1
            elif rank[root1] < rank[root2]:
                parent[root1] = root2
            else:
                parent[root2] = root1
                rank[root1] += 1

    for node in graph.nodes:
        parent[node] = node
        rank[node] = 0

    sorted_edges = sorted(graph.edges, key=lambda edge: edge.weight if edge.weight is not None else -float('inf'), reverse=True)

    mst = []
    total_weight = 0
    for edge in sorted_edges:
        node1, node2 = edge.node1, edge.node2
        if find(node1) != find(node2):
            union(node1, node2)
            mst.append(edge)
            total_weight += edge.weight if edge.weight is not None else 0

    edge_names = [edge.name for edge in mst if edge.name is not None]
    edge_names = [edge.name for edge in mst if edge.name is not None]
    print("Hrany v maximální kostře:", edge_names)

    return mst, total_weight


def shortest_path(graph, start, end):
    distances = {node: float('inf') for node in graph.nodes}
    distances[start] = 0
    previous_nodes = {node: None for node in graph.nodes}
    
    unvisited = list(graph.nodes)
    
    while unvisited:
        current_node = min(unvisited, key=lambda node: distances[node])
        
        if distances[current_node] == float('inf'):
            break
        
        unvisited.remove(current_node)
        
        for edge in vystupni_okoli_uzlu(graph, current_node):
            neighbor = edge.node2
            weight = edge.weight if edge.weight is not None else 1
            distance = distances[current_node] + weight
            
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous_nodes[neighbor] = current_node
    
    path = []
    current = end
    while current is not None:
        path.insert(0, current)
        current = previous_nodes[current]
    
    return path, distances[end] if distances[end] != float('inf') else None

def longest_path_with_cycles(graph, start, end, verbose=False):
    def dfs(node, visited, current_length, path, max_path, max_length):
        visited.add(node)
        path.append(node)

        if node == end:
            if current_length > max_length[0]:
                max_length[0] = current_length
                max_path[:] = path[:]
        else:
            for edge in vystupni_okoli_uzlu(graph, node):
                neighbor = edge.node2
                weight = edge.weight if edge.weight is not None else 1
                if neighbor not in visited:
                    dfs(neighbor, visited, current_length + weight, path, max_path, max_length)

        path.pop()
        visited.remove(node)

    visited = set()
    max_path = []
    max_length = [-float('inf')]
    dfs(start, visited, 0, [], max_path, max_length)

    if max_length[0] == -float('inf'):
        return None, None

    edges = [edge.name for edge in graph.edges if edge.node1 in max_path and edge.node2 in max_path and max_path.index(edge.node1) < max_path.index(edge.node2)]
    edges = [edge.name for edge in graph.edges if edge.node1 in max_path and edge.node2 in max_path and max_path.index(edge.node1) < max_path.index(edge.node2)]
    print("Hrany v nejdelší cestě s cykly:", edges)

    return max_path, max_length[0]

def safest_path(graph, start, end):
    distances = {node: float('inf') for node in graph.nodes}
    distances[start] = 0
    previous_nodes = {node: None for node in graph.nodes}
    unvisited = list(graph.nodes)

    while unvisited:
        current_node = min(unvisited, key=lambda node: distances[node])

        if distances[current_node] == float('inf'):
            break

        unvisited.remove(current_node)

        for edge in vystupni_okoli_uzlu(graph, current_node):
            neighbor = edge.node2
            risk = edge.risk if hasattr(edge, 'risk') and edge.risk is not None else 1
            total_risk = distances[current_node] + risk

            if total_risk < distances[neighbor]:
                distances[neighbor] = total_risk
                previous_nodes[neighbor] = current_node

    path = []
    current = end
    while current is not None:
        path.insert(0, current)
        current = previous_nodes[current]

    edges = [edge.name for edge in graph.edges if edge.node1 in path and edge.node2 in path and path.index(edge.node1) < path.index(edge.node2)]
    edges = [edge.name for edge in graph.edges if edge.node1 in path and edge.node2 in path and path.index(edge.node1) < path.index(edge.node2)]
    print("Hrany v nejbezpečnější cestě:", edges)

    return path, distances[end] if distances[end] != float('inf') else None


def widest_path(graph, start, end):
    capacities = {node: -float('inf') for node in graph.nodes}
    capacities[start] = float('inf')
    previous_nodes = {node: None for node in graph.nodes}
    unvisited = set(graph.nodes)

    while unvisited:
        current_node = max(unvisited, key=lambda node: capacities[node])

        if capacities[current_node] == -float('inf'):
            break

        unvisited.remove(current_node)

        for edge in vystupni_okoli_uzlu(graph, current_node):
            neighbor = edge.node2
            capacity = edge.weight if edge.weight is not None else 1
            path_capacity = min(capacities[current_node], capacity)

            if path_capacity > capacities[neighbor]:
                capacities[neighbor] = path_capacity
                previous_nodes[neighbor] = current_node

    path = []
    current = end
    while current is not None:
        path.insert(0, current)
        current = previous_nodes[current]

    edges = [edge.name for edge in graph.edges if edge.node1 in path and edge.node2 in path and path.index(edge.node1) < path.index(edge.node2)]
    edges = [edge.name for edge in graph.edges if edge.node1 in path and edge.node2 in path and path.index(edge.node1) < path.index(edge.node2)]
    print("Hrany v nejširší cestě:", edges)
    return path if capacities[end] != -float('inf') else None, capacities[end]

def bfs(graph, parent_map, source, sink):
    visited = set()
    queue = deque([source])
    visited.add(source)

    while queue:
        current_node = queue.popleft()

        for edge in graph.get_edges():
            if (edge.node1 == current_node and edge.node2 not in visited and 
                    edge.weight > 0):
                queue.append(edge.node2)
                visited.add(edge.node2)
                parent_map[edge.node2] = edge
                if edge.node2 == sink:
                    return True
    return False

def maximal_flow(graph, source, sink):
    parent_map = {}
    max_flow = 0

    while bfs(graph, parent_map, source, sink):
        path_flow = float('Inf')
        s = sink

        while s != source:
            edge = parent_map[s]
            path_flow = min(path_flow, edge.weight)
            s = edge.node1

        v = sink
        while v != source:
            edge = parent_map[v]
            edge.weight -= path_flow
            reverse_edge = next((e for e in graph.get_edges() if e.node1 == v and e.node2 == edge.node1), None)
            if reverse_edge:
                reverse_edge.weight += path_flow
            else:
                reverse_edge = Edge(v, edge.node1, weight=path_flow, name="reverse_" + (edge.name if edge.name else ""))
                graph.add_edge(reverse_edge.node1, reverse_edge.node2, reverse_edge.weight, reverse_edge.name)
            v = edge.node1

        max_flow += path_flow

    return max_flow

def edmonds_karp_full(graph, source, sink, logger=print, export_csv=True):
    """
    Kompletní implementace Edmonds-Karpova algoritmu s vizualizací, statistikami a exportem.
    
    Args:
        graph: Instance grafu.
        source: Počáteční uzel.
        sink: Cílový uzel.
        logger: Funkce pro výpis (print nebo self.log_output).
        export_csv: Zda exportovat výsledky do CSV.
    """
    import csv
    from collections import deque, defaultdict
    
    # Pomocná BFS pro hledání cesty v reziduálním grafu
    def bfs_residual(residual_graph, s, t, parent):
        visited = set()
        queue = deque([s])
        visited.add(s)
        parent[s] = None

        while queue:
            u = queue.popleft()
            for v in residual_graph[u]:
                capacity = residual_graph[u][v]
                if v not in visited and capacity > 0:
                    queue.append(v)
                    visited.add(v)
                    parent[v] = u
                    if v == t:
                        return True
        return False

    # 1. Inicializace reziduálního grafu
    residual_graph = defaultdict(lambda: defaultdict(float))
    original_capacities = defaultdict(lambda: defaultdict(float))
    
    for edge in graph.edges:
        u, v = edge.node1, edge.node2
        w = edge.weight if edge.weight is not None else 1.0
        
        original_capacities[u][v] += w
        residual_graph[u][v] += w
        
        # Zpětná hrana s nulovou kapacitou (pokud neexistuje)
        if v not in residual_graph or u not in residual_graph[v]:
             residual_graph[v][u] += 0.0

    logger(f"Spouštím Edmonds-Karp Max Flow")
    logger(f"Zdroj: {source}, Cíl: {sink}")
    logger("-" * 50)

    parent = {}
    max_flow = 0
    path_count = 0

    while bfs_residual(residual_graph, source, sink, parent):
        path_count += 1
        
        # Hledání úzkého hrdla (bottleneck)
        path_flow = float('Inf')
        s = sink
        path_nodes = [sink]
        
        while s != source:
            path_flow = min(path_flow, residual_graph[parent[s]][s])
            s = parent[s]
            path_nodes.append(s)
        
        path_nodes.reverse()
        path_str = " -> ".join(map(str, path_nodes))
        
        # Aktualizace reziduálních kapacit
        v = sink
        while v != source:
            u = parent[v]
            residual_graph[u][v] -= path_flow
            residual_graph[v][u] += path_flow
            v = parent[v]

        max_flow += path_flow
        
        logger(f"Krok {path_count}: Nalezena cesta {path_str}")
        logger(f"        Přidán tok: {path_flow} | Celkový tok: {max_flow}")

    # Výpočet finálních toků na hranách
    final_flows = []
    for u in original_capacities:
        for v in original_capacities[u]:
            capacity = original_capacities[u][v]
            remaining = residual_graph[u][v]
            flow = capacity - remaining
            if flow > 0:
                final_flows.append({
                    "source": u,
                    "target": v,
                    "flow": flow,
                    "capacity": capacity,
                    "utilization": (flow / capacity * 100) if capacity > 0 else 0
                })

    # Výpis statistik
    logger("-" * 50)
    logger(f"Výpočet dokončen!")
    logger(f"Celkový maximální tok: {max_flow}")
    logger(f"Počet augmentačních cest: {path_count}")
    
    # Export do CSV
    if export_csv:
        filename = "edmonds_karp.csv"
        try:
            with open(filename, 'w', newline='') as csvfile:
                fieldnames = ['source', 'target', 'flow', 'capacity', 'utilization']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for item in final_flows:
                    writer.writerow(item)
            logger(f"Výsledky exportovány do {filename}")
        except IOError as e:
            logger(f"Chyba při zápisu do CSV: {e}")
            
    return max_flow, final_flows

def minimum_cut(graph, source, sink):
    """
    Najde minimální řez v grafu pomocí Ford-Fulkerson/Edmonds-Karp.
    Vrací seznam hran v řezu a celkovou kapacitu.
    """
    # 1. Vytvoříme kopii grafu (residual graph)
    import copy
    residual_graph = copy.deepcopy(graph)
    
    # 2. Spustíme maximální tok
    max_flow_value = maximal_flow(residual_graph, source, sink)
    
    # 3. Najdeme dosažitelné uzly ze source v residual grafu
    visited = set()
    queue = deque([source])
    visited.add(source)
    
    while queue:
        u = queue.popleft()
        for edge in residual_graph.get_edges():
            if edge.node1 == u and edge.node2 not in visited and edge.weight > 0:
                visited.add(edge.node2)
                queue.append(edge.node2)
    
    # 4. Hrany řezu jsou ty, kde node1 je visited a node2 není
    cut_edges = []
    cut_capacity = 0
    
    for edge in graph.get_edges():
        if edge.node1 in visited and edge.node2 not in visited:
            cut_edges.append(edge)
            cut_capacity += edge.weight
    
    print("Hrany minimálního řezu:")
    for edge in cut_edges:
        print("{} -> {} (Kapacita: {})".format(edge.node1, edge.node2, edge.weight))
    print("Celková kapacita řezu: {}".format(cut_capacity))
    
    return cut_edges, cut_capacity

def dfs(graph, start):
    visited = set()
    order = []

    def visit(node):
        visited.add(node)
        order.append(node)

        if is_directed(graph):
            neighbors = [
                edge.node2 for edge in graph.get_edges() if edge.node1 == node
            ]
        else:
            neighbors = [
                edge.node2 for edge in graph.get_edges() if edge.node1 == node
            ] + [
                edge.node1 for edge in graph.get_edges() if edge.node2 == node
            ]

        neighbors = sorted(neighbors)

        for neighbor in neighbors:
            if neighbor not in visited:
                visit(neighbor)

    if start not in graph.nodes:
        print("Startovní uzel '{}' není v grafu.".format(start))
        return []

    visit(start)

    print("Pořadí DFS:", order)
    return order


def bfs_traversal(graph, start):
    visited = set()
    queue = deque([start])
    order = []

    while queue:
        current_node = queue.popleft()
        if current_node not in visited:
            visited.add(current_node)
            order.append(current_node)
            for edge in graph.get_edges():
                if edge.node1 == current_node:
                    if edge.node2 not in visited:
                        queue.append(edge.node2)
                elif edge.node2 == current_node and not is_directed(graph):
                    if edge.node1 not in visited:
                        queue.append(edge.node1)

    print("Pořadí BFS:", order)
    return order

def level_order(graph, start):
    # Level order traversal is essentially BFS
    print("Pořadí Level order:", end=" ") # The print inside bfs_traversal will handle the list
    return bfs_traversal(graph, start)


def preorder(graph, node, visited=None, order=None):
    if visited is None:
        visited = set()
    if order is None:
        order = []
        
    if node not in visited:
        order.append(node)
        visited.add(node)
        for neighbor in naslednici_uzlu(graph, node):
            preorder(graph, neighbor, visited, order)
            
    return order

def postorder(graph, node, visited=None, order=None):
    if visited is None:
        visited = set()
    if order is None:
        order = []
    
    if node not in visited:
        visited.add(node)
        for neighbor in naslednici_uzlu(graph, node):
            postorder(graph, neighbor, visited, order)
        order.append(node)
        
    return order

def inorder(graph, start_node):
    class TreeNode:
        def __init__(self, name):
            self.name = name
            self.children = []

    def build_tree(graph, node, visited):
        if node in visited:
            return None
        visited.add(node)
        root = TreeNode(node)
        neighbors = sorted(naslednici_uzlu(graph, node))
        for neighbor in neighbors:
            if neighbor not in visited:
                child = build_tree(graph, neighbor, visited)
                if child:
                    root.children.append(child)
        return root

    def inorder_traverse(node, result):
        if not node:
            return
        children = node.children
        if children:
            inorder_traverse(children[0], result)
        result.append(node.name)
        if len(children) > 1:
            inorder_traverse(children[1], result)

    visited = set()
    tree = build_tree(graph, start_node, visited)
    result = []
    inorder_traverse(tree, result)
    return result



def natural_sort_key(name):
    return (len(name), [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', name)])
def save_path_to_file(path_nodes, edges, distance, file_name):
    import os
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    file_path = os.path.join(output_dir, file_name)
    
    with open(file_path, "w") as f:
        f.write("Uzly cesty:\n")
        f.write(" -> ".join(path_nodes) + "\n")
        f.write("Hrany:\n")
        for edge in edges:
            f.write("{} -- {}\n".format(edge[0], edge[1]))
        f.write("Celková vzdálenost: {}\n".format(distance))
    print("Výsledky uloženy do", file_path)

# Import Libuš implementace - čistá kopie upraven pro Test-2
from floyd_warshall_clean import floyd_warshall, Metric

def get_shortest_path(graph, start, end, verbose=False):
    # Validace vstupních uzlů
    if start not in graph.nodes or end not in graph.nodes:
        if verbose:
            print(f"❌ Chyba: Uzel '{start}' nebo '{end}' neexistuje v grafu")
            print(f"   Dostupné uzly: {sorted(graph.nodes)}")
        return [], [], None
    
    # Použijeme Libuš implementaci přímo
    result = floyd_warshall(graph, Metric.SHORTEST, verbose=False)
    
    if verbose:
        print(f"\n🔍 DEBUG reconstruct_path({start} -> {end})")
        idx_s = result.nodes.index(start)
        idx_e = result.nodes.index(end)
        print(f"   Distance[{idx_s}][{idx_e}] = {result.distances[idx_s][idx_e]}")
    
    path_nodes = result.reconstruct_path(start, end)
    
    if not path_nodes:
        if verbose:
            print("   ❌ Cesta nebyla nalezena")
        return [], [], None
    
    idx_s = result.nodes.index(start)
    idx_e = result.nodes.index(end)
    distance = result.distances[idx_s][idx_e]
    
    if verbose:
        print(f"   ✅ Nalezena cesta: {' → '.join(path_nodes)}")
        
    edges = []
    for i in range(len(path_nodes) - 1):
        edges.append((path_nodes[i], path_nodes[i+1]))

    save_path_to_file(path_nodes, edges, distance, "shortest_path.txt")
    save_matrix_to_file(result.distances, result.nodes, result.nodes, "shortest_matrix.txt", title="Matice nejkratších cest")

    if verbose:
        print(f"\n📏 Celková délka: {distance}")
        print(f"🛤️  Cesta ({len(path_nodes)} uzlů): {' → '.join(path_nodes)}")
        print(f"📊 Statistiky:")
        print(f"   • Počet uzlů: {len(path_nodes)}")
        print(f"   • Počet hran: {len(edges)}")

    return path_nodes, edges, distance

def get_safest_path(graph, start, end, verbose=False):
    # Použijeme Libuš implementaci přímo
    result = floyd_warshall(graph, Metric.SAFEST, verbose=False)
    
    if verbose:
        print(f"\n🔍 DEBUG reconstruct_path({start} → {end})")
        print(f"   Metrika: SAFEST (log transformace)")
        idx_s = result.nodes.index(start)
        idx_e = result.nodes.index(end)
        print(f"   Σlog(w) = {result.distances[idx_s][idx_e]:.6f}")
    
    path_nodes = result.reconstruct_path(start, end)
    
    if not path_nodes:
        if verbose:
            print("   ❌ Cesta nebyla nalezena")
        return [], [], None
    
    idx_s = result.nodes.index(start)
    idx_e = result.nodes.index(end)
    log_val = result.distances[idx_s][idx_e]
        
    # Převod zpět: exp(log_val)
    import math
    if math.isinf(log_val):
        prob = 0.0
        percentage = 0.0
    else:
        prob = math.exp(log_val)  # Součin původních vah
        # Pokud jsou váhy ve formátu 0-100 (procenta), prob bude velké číslo
        # Pokud jsou váhy ve formátu 0-1 (pravděpodobnost), prob bude malé číslo
        # Musíme rozlišit:
        if prob > 1:
            # Váhy byly v procentech (např. 8, 5, 1 = 8*5*1 = 40)
            percentage = prob  # Už je to v "procentech" jako číslo
        else:
            # Váhy byly pravděpodobnosti (např. 0.08, 0.05, 0.01)
            percentage = prob * 100
    
    if verbose:
        print(f"   ✅ Nalezena cesta: {' → '.join(path_nodes)}")
        print(f"\n📏 Součin vah: {prob:.6f}")
        print(f"   (Interní hodnota Σlog(w): {log_val:.6f})")
        print(f"   Jako procento: {percentage:.2f}%")
        print(f"🛤️  Cesta ({len(path_nodes)} uzlů): {' → '.join(path_nodes)}")
        
    edges = []
    for i in range(len(path_nodes) - 1):
        edges.append((path_nodes[i], path_nodes[i+1]))

    save_path_to_file(path_nodes, edges, percentage, "safest_path.txt")
    # Uložíme matici s převedenými hodnotami pro uživatele
    restored_matrix = [[math.exp(val) if not math.isinf(val) else 0.0 for val in row] for row in result.distances]
    save_matrix_to_file(restored_matrix, result.nodes, result.nodes, "safest_matrix.txt", title="Matice nejbezpečnějších cest")

    return path_nodes, edges, percentage

def get_widest_path(graph, start, end, verbose=False):
    result = floyd_warshall(graph, Metric.WIDEST, verbose=False)
    
    path_nodes = result.reconstruct_path(start, end)
    val = result.distances[result.nodes.index(start)][result.nodes.index(end)]
    
    if not path_nodes:
        return [], [], None
        
    edges = []
    for i in range(len(path_nodes) - 1):
        edges.append((path_nodes[i], path_nodes[i+1]))

    save_path_to_file(path_nodes, edges, val, "widest_path.txt")
    save_matrix_to_file(result.distances, result.nodes, result.nodes, "widest_matrix.txt", title="Matice nejširších cest")

    return path_nodes, edges, val

def get_longest_path(graph, start, end, verbose=False):
    # Libuš používá BFS pro nejdelší JEDNODUCHOU cestu
    result = floyd_warshall(graph, Metric.LONGEST, verbose=False)
    
    path_nodes = result.reconstruct_path(start, end)
    val = result.distances[result.nodes.index(start)][result.nodes.index(end)]
    
    if not path_nodes:
        return [], [], float('-inf')
        
    edges = []
    for i in range(len(path_nodes) - 1):
        edges.append((path_nodes[i], path_nodes[i+1]))

    return path_nodes, edges, val

def get_narrowest_path(graph, start, end, verbose=False):
    result = floyd_warshall(graph, Metric.NARROWEST, verbose=False)
    
    path_nodes = result.reconstruct_path(start, end)
    val = result.distances[result.nodes.index(start)][result.nodes.index(end)]
    
    if not path_nodes:
        return [], [], None
        
    edges = []
    for i in range(len(path_nodes) - 1):
        edges.append((path_nodes[i], path_nodes[i+1]))

    save_path_to_file(path_nodes, edges, val, "narrowest_path.txt")
    save_matrix_to_file(result.distances, result.nodes, result.nodes, "narrowest_matrix.txt", title="Matice nejužších cest")

    return path_nodes, edges, val

def get_most_dangerous_path(graph, start, end, verbose=False):
    """
    Nejnebezpečnější cesta podle MAX SOUČTU vah.
    Používá DFS s limitem pro rychlé hledání.
    POZNÁMKA: Toto NENÍ totéž jako Libuš MOST_DANGEROUS (max součin)!
    """
    # Validace vstupních uzlů
    if start not in graph.nodes or end not in graph.nodes:
        if verbose:
            print(f"❌ Chyba: Uzel '{start}' nebo '{end}' neexistuje v grafu")
            print(f"   Dostupné uzly: {sorted(graph.nodes)}")
        return [], [], None
    
    # Detekce cyklů v grafu pomocí DFS
    def has_cycle_dfs(node, visited, rec_stack):
        visited.add(node)
        rec_stack.add(node)
        
        for edge in vystupni_okoli_uzlu(graph, node):
            neighbor = edge.node2
            if neighbor not in visited:
                if has_cycle_dfs(neighbor, visited, rec_stack):
                    return True
            elif neighbor in rec_stack:
                return True
        
        rec_stack.remove(node)
        return False
    
    # Kontrola cyklů
    has_cycle = False
    visited_global = set()
    for node in graph.nodes:
        if node not in visited_global:
            if has_cycle_dfs(node, visited_global, set()):
                has_cycle = True
                break
    
    # VŽDY zobraz varování o cyklu (ne jen v verbose módu)
    if has_cycle:
        print(f"\n⚠️  VAROVÁNÍ: Graf obsahuje cyklus!")
        print(f"   Nejdelší cesta může být nekonečně dlouhá.")
        print(f"   Hledám nejdelší jednoduchou cestu (bez opakování uzlů)...\n")
    
    if verbose:
        print(f"🔍 Hledám nejdelší cestu {start} → {end} (DFS s limitem)...")
    
    # Použijeme DFS místo BFS (rychlejší pro velké grafy)
    path_nodes, val = longest_path_with_cycles(graph, start, end, verbose=False)
    
    if val is None or val == float('-inf'):
        if verbose:
            print("   ❌ Cesta nebyla nalezena")
        return [], [], None, False
    
    if val == float('inf'):
        if verbose:
            print("   ⚠️ Detekován cyklus - cesta může být nekonečná")
        # Vrátíme cestu i s inf hodnotou
        pass
        
    edges = []
    if path_nodes:
        for i in range(len(path_nodes) - 1):
            edges.append((path_nodes[i], path_nodes[i+1]))
    
    if verbose:
        if path_nodes:
            print(f"   ✅ Nalezena cesta: {' → '.join(path_nodes)}")
            print(f"\n📏 Celková vzdálenost: {val}")
            if has_cycle:
                print(f"   ℹ️  Poznámka: Toto je nejdelší JEDNODUCHÁ cesta (bez cyklů)")
        else:
            print("   ❌ Žádná cesta")

    # Vrátíme rozšířený tuple s informací o cyklu
    return path_nodes, edges, val, has_cycle


# BY BORECCZ1 - Nejbezpečnější cesta s produktem (maximální produkt pravděpodobností)
def get_safest_path_by_boreccz1(graph, start, end, verbose=False):
    """
    Nejbezpečnější cesta podle SOUČINU vah (MAX product).
    Používá DFS pro hledání všech jednoduchých cest a vybere tu s maximálním součinem.
    """
    import math
    
    if start not in graph.nodes or end not in graph.nodes:
        if verbose:
            print("❌ Počáteční nebo koncový uzel neexistuje v grafu.")
        return [], [], None
    
    # Detekce formátu vah (0-1 vs 0-100)
    all_weights = []
    if hasattr(graph, 'edges'):
        for edge in graph.edges:
            if edge.weight is not None and edge.weight > 0:
                all_weights.append(edge.weight)
    
    # Pokud jsou váhy > 1, normalizujeme je (považujeme za procenta)
    normalize = len(all_weights) > 0 and max(all_weights) > 1
    
    if verbose:
        if normalize:
            print(f"\n🔍 Detekováno: Váhy v procentech (max={max(all_weights):.1f}), normalizuji na 0-1")
        else:
            print(f"\n🔍 Detekováno: Váhy už v rozsahu 0-1")
        print(f"🔍 Hledám nejbezpečnější cestu (max součin) {start} → {end} pomocí DFS...")
    
    # Najdeme všechny jednoduché cesty pomocí DFS
    all_paths = []
    
    def dfs_find_paths(current, target, visited, path, product):
        if current == target:
            all_paths.append((list(path), product))
            return
        
        visited.add(current)
        
        # Projdeme všechny sousedy
        neighbors = []
        if hasattr(graph, 'edges'): # Test-2 Graph object
             for edge in graph.edges:
                if edge.node1 == current:
                    neighbors.append((edge.node2, edge.weight))
        
        for neighbor, weight in neighbors:
            w = weight if weight is not None else 1.0
            
            # Normalizace váhy pokud je potřeba
            if normalize and w > 1:
                w = w / 100.0
            
            if neighbor not in visited and w > 0:
                path.append(neighbor)
                dfs_find_paths(neighbor, target, visited, path, product * w)
                path.pop()
        
        visited.remove(current)
    
    # Spustíme DFS
    try:
        dfs_find_paths(start, end, set(), [start], 1.0)
    except RecursionError:
        if verbose:
            print("❌ Překročena maximální hloubka rekurze.")
        return [], [], None
    
    if not all_paths:
        if verbose:
            print("❌ Cesta nebyla nalezena.")
        return [], [], None
    
    # Najdeme cestu s MAXIMÁLNÍM produktem
    max_path, max_product = max(all_paths, key=lambda x: x[1])
    
    # Vytvoříme seznam hran
    edges = [(max_path[i], max_path[i + 1]) for i in range(len(max_path) - 1)]
    
    # Převedeme výsledek na procenta pro zobrazení
    percentage = max_product * 100
    
    if verbose:
        print(f"   ✅ Nalezena cesta: {' → '.join(max_path)}")
        print(f"\n📏 Pravděpodobnost úspěchu: {percentage:.2f}%")
        print(f"   (Součin normalizovaných vah: {max_product:.6f})")
        print(f"   (Nalezeno {len(all_paths)} jednoduchých cest)")

    return max_path, edges, percentage

def get_most_dangerous_path_by_boreccz1(graph, start, end, verbose=False):
    """
    Nejnebezpečnější cesta podle SOUČINU vah (MIN product).
    Používá DFS pro hledání všech jednoduchých cest bez cyklů (původní implementace).
    """
    import math
    from graph import vystupni_okoli_uzlu
    
    if start not in graph.nodes or end not in graph.nodes:
        if verbose:
            print("❌ Počáteční nebo koncový uzel neexistuje v grafu.")
        return [], [], None
    
    # Detekce formátu vah (0-1 vs 0-100)
    all_weights = []
    if hasattr(graph, 'edges'):
        for edge in graph.edges:
            if edge.weight is not None and edge.weight > 0:
                all_weights.append(edge.weight)
    
    # Pokud jsou váhy > 1, normalizujeme je (považujeme za procenta)
    normalize = len(all_weights) > 0 and max(all_weights) > 1
    
    if verbose:
        if normalize:
            print(f"\n🔍 Detekováno: Váhy v procentech (max={max(all_weights):.1f}), normalizuji na 0-1")
        else:
            print(f"\n🔍 Detekováno: Váhy už v rozsahu 0-1")
        print(f"🔍 Hledám nejnebezpečnější cestu (min součin) {start} → {end} pomocí DFS...")
    
    # Najdeme všechny jednoduché cesty pomocí DFS
    all_paths = []
    
    def dfs_find_paths(current, target, visited, path, product):
        if current == target:
            all_paths.append((list(path), product))
            return
        
        visited.add(current)
        
        # Projdeme všechny sousedy
        # Musíme získat sousedy správně podle struktury grafu
        neighbors = []
        if hasattr(graph, 'edges'): # Test-2 Graph object
             for edge in graph.edges:
                if edge.node1 == current:
                    neighbors.append((edge.node2, edge.weight))
        
        for neighbor, weight in neighbors:
            w = weight if weight is not None else 1.0
            
            # Normalizace váhy pokud je potřeba
            if normalize and w > 1:
                w = w / 100.0
            
            if neighbor not in visited and w > 0:
                path.append(neighbor)
                dfs_find_paths(neighbor, target, visited, path, product * w)
                path.pop()
        
        visited.remove(current)
    
    # Spustíme DFS
    # Limit pro DFS, aby se nezacyklilo na obrovských grafech (i když visited to řeší pro simple paths)
    try:
        dfs_find_paths(start, end, set(), [start], 1.0)
    except RecursionError:
        if verbose:
            print("❌ Překročena maximální hloubka rekurze.")
        return [], [], None
    
    if not all_paths:
        if verbose:
            print("❌ Cesta nebyla nalezena.")
        return [], [], None
    
    # Najdeme cestu s minimálním produktem
    min_path, min_product = min(all_paths, key=lambda x: x[1])
    
    # Vytvoříme seznam hran
    edges = [(min_path[i], min_path[i + 1]) for i in range(len(min_path) - 1)]
    
    # Převedeme výsledek na procenta pro zobrazení
    percentage = min_product * 100
    
    if verbose:
        print(f"   ✅ Nalezena cesta: {' → '.join(min_path)}")
        print(f"\n📏 Pravděpodobnost nebezpečí: {percentage:.2f}%")
        print(f"   (Součin normalizovaných vah: {min_product:.6f})")
        print(f"   (Nalezeno {len(all_paths)} jednoduchých cest)")

    return min_path, edges, percentage

def moore_shortest_path(graph, start, end, verbose=False):
    # Moore's algorithm is BFS for shortest path in unweighted graph
    # We treat all edge weights as 1
    
    if start not in graph.nodes or end not in graph.nodes:
        print("Počáteční nebo koncový uzel neexistuje v grafu.")
        return [], [], None

    queue = deque([start])
    visited = {start}
    predecessor = {start: None}
    
    found = False
    while queue:
        current = queue.popleft()
        if current == end:
            found = True
            break
        
        for edge in vystupni_okoli_uzlu(graph, current):
            neighbor = edge.node2
            if neighbor not in visited:
                visited.add(neighbor)
                predecessor[neighbor] = current
                queue.append(neighbor)
    
    if not found:
        print("Cesta nebyla nalezena.")
        return [], [], None

    # Reconstruct path
    path = []
    curr = end
    while curr is not None:
        path.insert(0, curr)
        curr = predecessor[curr]
    
    # Get edges
    edges = []
    length = 0
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        # Find edge object (take first one found)
        found_edge = None
        for edge in graph.edges:
            if edge.node1 == u and edge.node2 == v:
                found_edge = edge
                break
        if found_edge:
            edges.append(found_edge.name if found_edge.name else f"{u}->{v}")
        else:
            edges.append(f"{u}->{v}")
        length += 1 # In Moore's algorithm, length is number of edges (hops)

    save_path_to_file(path, edges, length, "moore_path.txt")
    print("Moorův algoritmus (BFS) - Nejkratší cesta:", " -> ".join(path))
    print("Délka cesty (počet hran):", length)
    
    return path, edges, length
