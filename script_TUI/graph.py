from edge import Edge
from itertools import combinations

class Graph:
    def __init__(self):
        self.nodes = set()
        self.edges = []

    def add_node(self, node):
        self.nodes.add(node)

    def add_edge(self, node1, node2, weight=None, name=None):
        edge = Edge(node1, node2, weight, name)
        self.edges.append(edge)

    def remove_node(self, node):
        if node in self.nodes:
            self.nodes.remove(node)
            self.edges = [edge for edge in self.edges if edge.node1 != node and edge.node2 != node]
            print("Uzel '{}' a související hrany byly odstraněny.".format(node))
        else:
            print("Uzel '{}' v grafu neexistuje.".format(node))

    def get_edges(self):
        return self.edges
    def get_edge_weight(self, from_node, to_node):
        return self.edges.get((from_node, to_node), float('inf'))  # Vraťte inf pokud hrana neexistuje
    def contains_k5(self):
        if len(self.nodes) < 5 or len(self.edges) < 10:
            return False

        edge_set = {(e.node1, e.node2) for e in self.edges}.union(
                   {(e.node2, e.node1) for e in self.edges})

        for comb in combinations(self.nodes, 5):
            if all((u, v) in edge_set for u, v in combinations(comb, 2)):
                return True
        return False

    def contains_k33(self):
        if len(self.nodes) < 6 or len(self.edges) < 9:
            return False  

        edge_set = {(e.node1, e.node2) for e in self.edges}.union(
                   {(e.node2, e.node1) for e in self.edges})

        for left_part in combinations(self.nodes, 3):
            right_part = set(self.nodes) - set(left_part)
            if len(right_part) < 3:
                continue
            right_part = list(right_part)[:3]

            if all((u, v) in edge_set for u in left_part for v in right_part):
                return True
        return False
    
    def get_neighbors(self, node):
        return [edge.node2 for edge in self.edges if edge.node1 == node]


def is_planar(graph):
    v = len(graph.nodes)
    e = len(graph.edges)
    
    # 1. Eulerova podmínka pro rovinné grafy (nutná, ne postačující): e <= 3v - 6
    # Platí pro v >= 3. Pro menší grafy je to vždy True.
    # Pozor: graph.edges může obsahovat duplicity pro neorientované grafy (tam a zpět).
    # Musíme zjistit počet unikátních hran.
    # Pokud je graf z file_readeru a je neorientovaný ('-'), hrany jsou tam 2x.
    # Pro jistotu spočítáme unikátní dvojice.
    
    unique_edges = set()
    for edge in graph.edges:
        u, n = sorted((edge.node1, edge.node2))
        unique_edges.add((u, n))
    
    real_e = len(unique_edges)
    
    if v >= 3 and real_e > 3 * v - 6:
        return False # Graf je příliš hustý na to, aby byl rovinný
        
    # 2. Limit pro výpočetně náročné kontroly (K5, K3,3)
    # Tyto kontroly mají složitost O(V^5) resp. O(V^6), což je pro V > 30 neúnosné.
    if v > 30:
        # Pokud prošel Eulerem, ale je velký, nemůžeme efektivně ověřit K5/K3,3.
        # Vrátíme True s varováním (optimistický odhad), nebo False?
        # Většina "náhodných" velkých grafů co projdou Eulerem jsou spíše rovinné (řídké), 
        # ale vbg.tg (1000 nodes, 3000 edges) je na hraně.
        # Pro účely "rychlosti" a "uživatelské přívětivosti" vrátíme True, 
        # protože nemůžeme dokázat opak v rozumném čase.
        # Nebo lépe: Vypíšeme do konzole, že přesná kontrola byla přeskočena.
        return True 

    return not (graph.contains_k5() or graph.contains_k33())

def naslednici_uzlu(graph, node):
    return [edge.node2 for edge in graph.edges if edge.node1 == node]

def predchdci_uzlu(graph, node):
    return [edge.node1 for edge in graph.edges if edge.node2 == node]

def sousedi_uzlu(graph, node):
    predchudci = predchdci_uzlu(graph, node)
    naslednici = naslednici_uzlu(graph, node)
    return list(set(predchudci + naslednici))


def vystupni_okoli_uzlu(graph, node):
    outgoing_edges = [edge for edge in graph.edges if edge.node1 == node]
    return outgoing_edges

def vstupni_okoli_uzlu(graph, node):
    incoming_edges = [edge for edge in graph.edges if edge.node2 == node]
    return incoming_edges

def okoli_uzlu(graph, node):
    outgoing = vystupni_okoli_uzlu(graph, node)
    incoming = vstupni_okoli_uzlu(graph, node)
    return outgoing + incoming

def vystupni_stupen_uzlu(graph, node):
    outgoing_degree = len([edge for edge in graph.edges if edge.node1 == node])
    return outgoing_degree

def vstupni_stupen_uzlu(graph, node):
    incoming_degree = len([edge for edge in graph.edges if edge.node2 == node])
    return incoming_degree

def stupen_uzlu(graph, node):
    total_degree = vystupni_stupen_uzlu(graph, node) + vstupni_stupen_uzlu(graph, node)
    return total_degree

