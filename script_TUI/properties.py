from collections import deque

def get_adj_list(graph):
    """Vytvoří seznam sousednosti pro efektivní průchody."""
    adj = {node: [] for node in graph.nodes}
    for edge in graph.edges:
        if edge.node1 in adj:
            adj[edge.node1].append(edge.node2)
        # Pro neorientované grafy přidáme i zpětnou hranu, pokud chceme "sousedy"
        # Ale pozor: is_connected a is_bipartite v neorientovaném grafu potřebují obousměrné hrany.
        # V orientovaném grafu is_connected (silná souvislost) je složitější, 
        # ale zde se zřejmě myslí "slabě souvislý" (ignoruje směr) nebo se předpokládá neorientovaný.
        # Původní implementace is_connected dělala BFS ignorující směr (přidávala neighbors i neighbors_reverse).
        # Takže pro is_connected potřebujeme neorientovaný adj list.
    return adj

def get_undirected_adj_list(graph):
    """Vytvoří neorientovaný seznam sousednosti."""
    adj = {node: [] for node in graph.nodes}
    for edge in graph.edges:
        if edge.node1 in adj:
            adj[edge.node1].append(edge.node2)
        if edge.node2 in adj:
            adj[edge.node2].append(edge.node1)
    return adj

def is_weighted(graph):
    return any(edge.weight is not None for edge in graph.edges)

def is_directed(graph):
    # Rychlá kontrola: pokud počet hran != počet unikátních dvojic (bez ohledu na pořadí), může to napovědět.
    # Ale definice je: directed_edges != reverse_edges.
    # Pro velké grafy je set comprehension pomalý.
    # Zkusíme optimalizovat:
    # Pokud najdeme hranu A->B, ale ne B->A, je orientovaný.
    
    edge_set = set()
    for edge in graph.edges:
        edge_set.add((edge.node1, edge.node2))
    
    for u, v in edge_set:
        if (v, u) not in edge_set:
            return True
    return False

def is_connected(graph):
    if not graph.nodes:
        return True
    
    # Použijeme neorientovaný adj list pro "slabou souvislost" (nebo souvislost neorientovaného grafu)
    adj = get_undirected_adj_list(graph)
    
    start_node = next(iter(graph.nodes))
    visited = set()
    queue = deque([start_node])
    visited.add(start_node)
    
    while queue:
        node = queue.popleft()
        for neighbor in adj[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    
    return len(visited) == len(graph.nodes)

def prosty(graph):
    return graph.is_simple

def jednoduchy(graph):
    # Smyčky
    for edge in graph.edges:
        if edge.node1 == edge.node2:
            return False
    return prosty(graph)

def is_finite(graph):
    return True

def is_complete(graph):
    n = len(graph.nodes)
    e = len(graph.edges)
    
    if is_directed(graph):
        # V úplném orientovaném grafu musí být n*(n-1) hran (každý s každým tam i zpět, nebo jen tam? 
        # Definice úplného orientovaného grafu (turnaj) je složitější, ale obvykle se myslí "každý s každým".
        # Pokud "úplný" znamená, že mezi každými dvěma uzly je hrana (v obou směrech), pak E = n*(n-1).
        return e == n * (n - 1)
    else:
        # V neorientovaném grafu (reprezentovaném jako 2x hrany v seznamu nebo 1x?)
        # Třída Graph v file_reader.py pro '-' přidává DVĚ hrany.
        # Takže len(graph.edges) je 2 * počet_neorientovanych_hran.
        # Počet hran v K_n je n*(n-1)/2.
        # Takže v graph.edges by mělo být 2 * (n*(n-1)/2) = n*(n-1).
        return e == n * (n - 1)

def is_regular(graph):
    if not graph.nodes:
        return True
        
    adj = get_undirected_adj_list(graph)
    degrees = [len(neighbors) for neighbors in adj.values()]
    
    if not degrees:
        return True
        
    first_degree = degrees[0]
    return all(deg == first_degree for deg in degrees)

def is_bipartite(graph):
    if not graph.nodes:
        return True
        
    adj = get_undirected_adj_list(graph)
    color_map = {}

    for start_node in graph.nodes:
        if start_node not in color_map:
            color_map[start_node] = 0
            queue = deque([start_node])

            while queue:
                current_node = queue.popleft()
                current_color = color_map[current_node]

                for neighbor in adj[current_node]:
                    if neighbor not in color_map:
                        color_map[neighbor] = 1 - current_color
                        queue.append(neighbor)
                    elif color_map[neighbor] == current_color:
                        return False
    return True

def is_tree(graph):
    # Strom musí být souvislý a počet hran musí být roven počtu uzlů - 1
    if not graph.nodes:
        return True 
    
    # Pozor: Pro neorientovaný graf (kde je každá hrana 2x v graph.edges)
    # je počet "fyzických" hran len(graph.edges) / 2.
    # Takže podmínka je: len(graph.edges) / 2 == len(graph.nodes) - 1
    # Nebo: len(graph.edges) == 2 * (len(graph.nodes) - 1)
    
    # Pokud je graf orientovaný, je to složitější (kořenový strom?), 
    # ale is_tree se obvykle ptá na "je to strom" v teorii grafů (neorientovaný, bez cyklů, souvislý).
    
    # Pro jistotu použijeme obecnou vlastnost: Souvislý a bez cyklů.
    # Ekvivalentně: Souvislý a |E| = |V| - 1.
    
    if not is_connected(graph):
        return False
        
    n = len(graph.nodes)
    e = len(graph.edges)
    
    if is_directed(graph):
        # Pro orientovaný graf (pokud to chápeme jako "underlying graph is tree")
        # tak by počet hran měl být n-1 (pokud jsou jednosměrné).
        # Ale naše is_directed vrací True, pokud directed != reverse.
        return e == n - 1
    else:
        # Neorientovaný (hrany jsou tam 2x)
        return e == 2 * (n - 1)
