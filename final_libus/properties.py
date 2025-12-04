import sys
import re
from collections import defaultdict, deque

class Graph:
    def __init__(self):
        self.nodes = {}  # {node_id: weight}
        self.edges = []  # [(node1, node2, direction, weight, label)]
        self.adj_list = defaultdict(list)  # Pro směrované grafy
        self.adj_list_undirected = defaultdict(set)  # Pro neorientované grafy
        
    def add_node(self, node_id, weight=None):
        """Přidá uzel do grafu"""
        node_id = node_id.rstrip(';')
        if node_id != '*':  # Ignorujeme * oznacujici chybejici uzel v bin. stromu
            self.nodes[node_id] = weight
    
    def add_edge(self, node1, direction, node2, weight=None, label=None):
        """Přidá hranu do grafu, automaticky pojmenuje hranu, pokud nemá label"""
        # Normalizace názvů uzlů
        node1 = node1.rstrip(';')
        node2 = node2.rstrip(';')

        # Kontrola existence uzlů
        if node1 not in self.nodes or node2 not in self.nodes:
            return False

        # Automatické pojmenování hrany, pokud label chybí
        if not label or label.strip() == "":
            label = f"h{node1}{node2}"
        else:
            label = label.rstrip(';')

        # Přidání hrany
        self.edges.append((node1, node2, direction, weight, label))

        # Vytvoření seznamu sousedů pro každý uzel
        if direction == '>':
            self.adj_list[node1].append((node2, weight))
        elif direction == '<':
            self.adj_list[node2].append((node1, weight))
        else:  # direction == '-'
            self.adj_list[node1].append((node2, weight))
            self.adj_list[node2].append((node1, weight))

        # Pro neorientovaný pohled (souvislost)
        self.adj_list_undirected[node1].add(node2)
        self.adj_list_undirected[node2].add(node1)

        return True

def parse_graph_file(filename):
    graph = Graph()
    node_order = []  # uchová pořadí uzlů
    
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            if line.startswith('u '):
                parts = line.split()
                node_id = parts[1]
                node_order.append(node_id)
                if len(parts) > 2:
                    weight = float(parts[2].rstrip(';'))
                else:
                    weight = None
                graph.add_node(node_id, weight)
            
            elif line.startswith('h '):
                parts = line.split()
                node1, direction, node2 = parts[1], parts[2], parts[3]
                weight = None
                label = None
                
                for i in range(4, len(parts)):
                    if parts[i].startswith(':'):
                        label = ' '.join(parts[i:])[1:]
                        break
                    else:
                        try:
                            weight = float(parts[i].rstrip(';'))
                        except ValueError:
                            pass
                
                graph.add_edge(node1, direction, node2, weight, label)
    
    # 🔹 Pokud graf nemá žádné hrany, ale obsahuje '*', vytvoříme binární strom
    if len(graph.edges) == 0 and '*' in node_order:
        for i, node in enumerate(node_order):
            if node == '*':
                continue
            left_i = 2 * i + 1
            right_i = 2 * i + 2
            if left_i < len(node_order) and node_order[left_i] != '*':
                graph.add_edge(node, '-', node_order[left_i])
            if right_i < len(node_order) and node_order[right_i] != '*':
                graph.add_edge(node, '-', node_order[right_i])
    
    return graph

def je_uzlove_ohodnoceny(graph):
    """Zkontroluje, zda je graf uzlově ohodnocený (má alespoň jeden uzel s ohodnocením)"""
    for weight in graph.nodes.values():
        if weight is not None:
            return True
    return False

def je_hranove_ohodnoceny(graph):
    """Zkontroluje, zda je graf hranově ohodnocený (má alespoň jednu hranu s ohodnocením nebo jménem)"""
    for node1, node2, direction, weight, label in graph.edges:
        weight_str = str(weight) if weight is not None else None
        if (weight_str is not None and not weight_str.startswith(':')):
            return True
    return False

def je_orientovany(graph):
    """Zkontroluje, zda je graf orientovaný (obsahuje alespoň jednu orientovanou hranu)"""
    for edge in graph.edges:
        if edge[2] in ['>', '<']:  # direction
            return True
    return False

def je_slabe_souvisly(graph):
    """Zkontroluje, zda je orientovaný graf slabě souvislý (souvislý jako neorientovaný)"""
    if len(graph.nodes) == 0: # prazdny graf
        return True
    
    # BFS z prvního uzlu na neorientované verzi
    start_node = next(iter(graph.nodes)) # startovni uzel
    visited = set()              # Množina navštívených uzlů: {}
    queue = deque([start_node])  # Oboustranna fronta uzlů k prozkoumání: [A]
    visited.add(start_node)      # Označíme start: visited = {A}
    
    while queue:
        node = queue.popleft() # odebere prvni prvek z fronty a ulozi do node
        for neighbor in graph.adj_list_undirected[node]: # seznam vsech sousednich uzlu
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    
    return len(visited) == len(graph.nodes) # kontrola, ze se dostanu do vsech uzlu grafu

def je_silne_souvisly(graph):
    """
    Zkontroluje, zda je orientovaný graf silně souvislý (z každého uzlu do každého).
    
    Silně souvislý = z každého uzlu se lze dostat do každého jiného uzlu po orientovaných hranách.
    
    Algoritmus:
    1. BFS z jednoho uzlu → ověříme, že se z něj dostaneme všude
    2. BFS na transponovaném grafu → ověříme, že se ze všech uzlů dostaneme zpět
    Pokud obě kontroly projdou, graf je silně souvislý.
    """
    if len(graph.nodes) == 0:
        return True
    
    # ========== KONTROLA 1: BFS z prvního uzlu (po normálních šipkách) ==========
    # Ověřujeme: "Dostanu se z uzlu A všude?"
    
    start_node = next(iter(graph.nodes))  # Vybereme libovolný uzel (např. A)
    visited = set()                        # Množina navštívených uzlů
    queue = deque([start_node])            # Fronta uzlů k prozkoumání
    visited.add(start_node)                # Označíme startovní uzel jako navstiveny
    
    # BFS průchod grafem
    while queue:
        node = queue.popleft()  # Vyndáme první uzel z fronty
        
        # Projdeme všechny sousedy (kam vedou šipky z tohoto uzlu)
        for neighbor, _ in graph.adj_list[node]:
            if neighbor not in visited:
                visited.add(neighbor)    # Označíme jako navštívený
                queue.append(neighbor)   # Přidáme do fronty k prozkoumání
    
    # Pokud jsme nenavštívili všechny uzly, graf není silně souvislý
    if len(visited) != len(graph.nodes):
        return False
    
    # ========== KONTROLA 2: BFS na transponovaném grafu (obrácené šipky) ==========
    # Ověřujeme: "Dostane se ze všech uzlů zpět do A?"
    
    # Vytvoříme graf s obráceným směrem všech hran
    # Pokud máme A → B, vytvoříme B → A
    reversed_adj = defaultdict(list)
    for node in graph.adj_list:
        for neighbor, weight in graph.adj_list[node]:
            # Původně: node → neighbor
            # Obrácené: neighbor → node
            reversed_adj[neighbor].append((node, weight))
    
    # BFS na obráceném grafu ze stejného startovního uzlu
    visited = set()
    queue = deque([start_node])
    visited.add(start_node)
    
    while queue:
        node = queue.popleft()
        
        # Procházíme sousedy v obráceném grafu
        for neighbor, _ in reversed_adj[node]: # reversed_adj á stejnou strukturu jako adj_list, jen s obráceným směrem hran
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    
    # Pokud jsme navštívili všechny uzly i v obráceném grafu,
    # znamená to, že ze všech uzlů se lze dostat do startovního uzlu
    # → Graf je silně souvislý
    return len(visited) == len(graph.nodes)

def analyzuj_souvislost(graph):
    """Analyzuje souvislost grafu podle toho, zda je orientovaný"""
    if je_orientovany(graph):
        # Orientovaný graf
        if je_silne_souvisly(graph):
            print("Graf je silně souvislý")
            return True
        elif je_slabe_souvisly(graph):
            print("Graf je slabě souvislý")
            return True
        else:
            print("Graf není souvislý")
            return False
    else:
        # Neorientovaný graf
        if je_slabe_souvisly(graph):
            print("Graf je souvislý")
            return True
        else:
            print("Graf není souvislý")
            return False


def je_prosty(graph):
    """
    Zkontroluje, zda je graf prostý (bez násobných hran).
    
    Prostý graf = MŮŽE mít smyčky, ale NESMÍ mít násobné hrany mezi stejnými uzly.
    """
    edges_set = set()
    
    for edge in graph.edges:
        node1, node2, direction = edge[0], edge[1], edge[2]
        
        if direction == '-':
            # U neorientované hrany pořadí uzlů nezáleží
            # A - B je totéž jako B - A
            edge_tuple = tuple(sorted([node1, node2])) + ('-',) # seřadí uzly abecedně → na pořadí nezáleží
        else:
            # U orientované hrany záleží na směru
            # A > B je jiná hrana než B > A
            edge_tuple = (node1, node2, direction)
        
        # Pokud už tato hrana existuje, máme násobnou hranu
        if edge_tuple in edges_set:
            print(f"Prosty graf: {edge_tuple} already in edges set")
            return False
        edges_set.add(edge_tuple)
    
    return True

def je_jednoduchy(graph):
    """Zkontroluje, zda je graf jednoduchý (prostý a bez smyček)."""
    # Nejprve zkontrolujeme, zda existují smyčky (node1 == node2)
    for edge in graph.edges:
        node1, node2, direction = edge[0], edge[1], edge[2]
        if node1 == node2:
            print(f"Jednoduchy graf: Loop in {node1}")
            return False  # graf obsahuje smyčku -> není jednoduchý

    # Pak zkontrolujeme, že je graf prostý (bez násobných hran)
    return je_prosty(graph)

def je_rovinny(graph):
    """
    Zkontroluje, zda je graf rovinný pomocí Eulerovy formule: v - e + f = 2
    Pro rovinný graf platí: e <= 3v - 6 (pro v >= 3)
    """
    v = len(graph.nodes) # pocet uzlu
    e = len(graph.edges) # pocet hran
    
    if v < 3:
        return True
    
    if je_bipartitni(graph):
        return e <= 2*v - 4
    else:
        return e <= 3*v - 6

def je_konecny(graph):
    """Zkontroluje, zda je graf konečný (má konečný počet uzlů a hran)"""
    # V našem případě jsou všechny grafy ze souboru konečné
    return len(graph.nodes) < float('inf') and len(graph.edges) < float('inf')

def je_uplny(graph):
    """Zkontroluje, zda je graf úplný (každé dva uzly jsou spojeny hranou)"""
    n = len(graph.nodes)
    if n <= 1:
        return True
    
    # Pro úplný graf musí být počet hran roven n(n-1)/2 (neorientovaný) nebo n(n-1) (orientovaný)
    required_edges_undirected = n * (n - 1) // 2
    required_edges_directed = n * (n - 1)
    
    # Spočítáme unikátní hrany
    edges_undirected = set()
    edges_directed = set()
    
    for edge in graph.edges:
        node1, node2, direction = edge[0], edge[1], edge[2]
        if direction == '-':
            edges_undirected.add(tuple(sorted([node1, node2])))
        else:
            if direction == '>':
                edges_directed.add((node1, node2))
            else:  # '<'
                edges_directed.add((node2, node1))
    
    # Zkontrolujeme orientovaný úplný graf
    if len(edges_directed) == required_edges_directed and len(edges_undirected) == 0:
        return True
    
    # Zkontrolujeme neorientovaný úplný graf
    if len(edges_undirected) == required_edges_undirected and len(edges_directed) == 0:
        return True
    
    return False

from collections import defaultdict

def je_regularni(graph):
    """
    Zkontroluje, zda je graf regulární.
    - Neorientovaný graf: všechny uzly mají stejný stupeň.
    - Orientovaný graf: všechny uzly mají stejný vstupní i výstupní stupeň.
    """
    if len(graph.nodes) == 0:
        return True

    # Rozlišujeme orientované a neorientované hrany
    is_directed = any(edge[2] in ['>', '<'] for edge in graph.edges)

    if is_directed:
        in_degrees = defaultdict(int)
        out_degrees = defaultdict(int)

        for node in graph.nodes:
            in_degrees[node] = 0
            out_degrees[node] = 0

        for node1, node2, direction, *_ in graph.edges:
            if direction == '>':
                out_degrees[node1] += 1
                in_degrees[node2] += 1
            elif direction == '<':
                out_degrees[node2] += 1
                in_degrees[node1] += 1
            else:  # neorientovaná hrana
                out_degrees[node1] += 1
                in_degrees[node2] += 1
                out_degrees[node2] += 1
                in_degrees[node1] += 1

        # Kontrola, zda jsou všechny vstupní a výstupní stupně stejné
        return len(set(in_degrees.values())) == 1 and len(set(out_degrees.values())) == 1

    else:
        # Neorientovaný graf
        degrees = defaultdict(int)
        for node in graph.nodes:
            degrees[node] = 0
        for node1, node2, direction, *_ in graph.edges:
            degrees[node1] += 1
            degrees[node2] += 1
        return len(set(degrees.values())) == 1


def je_bipartitni(graph):
    """Zkontroluje, zda je graf bipartitní (pomocí obarvení do 2 barev - BFS)"""
    if len(graph.nodes) == 0:
        return True
    
    color = {}
    
    for start_node in graph.nodes:
        if start_node in color:
            continue
        
        # BFS obarvování
        queue = deque([start_node])
        color[start_node] = 0
        
        while queue:
            node = queue.popleft()
            current_color = color[node]
            
            for neighbor, _ in graph.adj_list[node]:
                if neighbor not in color:
                    color[neighbor] = 1 - current_color
                    queue.append(neighbor)
                elif color[neighbor] == current_color:
                    return False
    
    return True

def analyze_graph(filename):
    """Analyzuje graf ze souboru a vypíše jeho vlastnosti"""
    print(f"Analýza grafu: {filename}")
    print("=" * 50)
    
    graph = parse_graph_file(filename)
    
    print(f"Počet uzlů: {len(graph.nodes)}")
    print(f"Počet hran: {len(graph.edges)}")
    print()
    
    properties = {
        'a) Uzlově ohodnocený': je_uzlove_ohodnoceny(graph),
        'b) Hranově ohodnocený': je_hranove_ohodnoceny(graph),
        'c) Orientovaný': je_orientovany(graph),
        'd) Souvislý': analyzuj_souvislost(graph),
        'e) Prostý': je_prosty(graph),
        'f) Jednoduchý': je_jednoduchy(graph),
        'g) Rovinný': je_rovinny(graph),
        'h) Konečný': je_konecny(graph),
        'i) Úplný': je_uplny(graph),
        'j) Regulární': je_regularni(graph),
        'k) Bipartitní': je_bipartitni(graph)
    }
    
    for prop, value in properties.items():
        status = "ANO" if value else "NE"
        print(f"{prop}: {status}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Použití: python graph_analyzer.py <soubor_grafu.txt>")
        sys.exit(1)
    
    filename = sys.argv[1]
    analyze_graph(filename)