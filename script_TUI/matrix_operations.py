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

def natural_sort_key(name):
    return (len(name), [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', name)])

import csv

def save_matrix_to_file(matrix, row_labels, col_labels, file_path, title="Matrix"):
    import os
    import unicodedata
    
    # Odstranění diakritiky a vytvoření bezpečného názvu
    safe_title = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode("ascii")
    safe_title = safe_title.replace(" ", "_")
    filename = f"{safe_title}.csv"

    # Cílová složka
    export_dir = os.path.join(os.getcwd(), "csv_export")
    os.makedirs(export_dir, exist_ok=True)
    full_path = os.path.join(export_dir, filename)

    # Převod hodnot na text
    formatted_matrix = [[str(v) for v in row] for row in matrix]
    all_values = [val for row in formatted_matrix for val in row] + [str(l) for l in row_labels] + [str(l) for l in col_labels]
    cell_width = max(len(str(v)) for v in all_values) + 1  # určí šířku sloupce

    with open(full_path, "w", encoding="utf-8") as f:
        # Hlavička
        f.write(" " * cell_width + "".join(f"{str(label):>{cell_width}}" for label in col_labels) + "\n")
        f.write("-" * ((len(col_labels) + 1) * cell_width) + "\n")

        # Každý řádek matice
        for i, label in enumerate(row_labels):
            line = f"{str(label):<{cell_width}}" + "".join(f"{formatted_matrix[i][j]:>{cell_width}}" for j in range(len(col_labels)))
            f.write(line + "\n")

    print(f"💾 Matice '{title}' byla uložena do: {full_path}")

def print_or_save_matrix(matrix, row_labels, col_labels, title="Matrix", file_path=None):
    col_labels = [label if label is not None else "" for label in col_labels]
    row_labels = [label if label is not None else "" for label in row_labels]
    
    print("\n" + title + ":")
    
    # Ask user for action
    while True:
        choice = input("Chcete výsledek (1) Vypsat na obrazovku, (2) Uložit do souboru (TXT a CSV)? [1/2]: ").strip()
        if choice == '1':
            header = "     " + " ".join(["{:>6}".format(name) for name in col_labels])
            print(header)
            for i, row in enumerate(matrix):
                formatted_row = []
                for val in row:
                    if isinstance(val, float):
                        formatted_row.append("{:>6.2f}".format(val))
                    else:
                        formatted_row.append("{:>6}".format(val))
                row_str = "{:>3}  ".format(row_labels[i]) + " ".join(formatted_row)
                print(row_str)
            print("\n")
            break
        elif choice == '2':
            if not file_path:
                # Create a default filename from title if not provided
                safe_title = "".join(c if c.isalnum() else "_" for c in title).lower()
                file_path = f"{safe_title}.txt"
            
            save_matrix_to_file(matrix, row_labels, col_labels, file_path, title)
            break
        else:
            print("Neplatná volba. Zadejte 1 nebo 2.")

"""
def adjacency_matrix(graph):
    sorted_nodes = sorted(graph.nodes, key=natural_sort_key)
    node_index = {node: i for i, node in enumerate(sorted_nodes)}
    
    n = len(sorted_nodes)
    adj_matrix = [[0] * n for _ in range(n)]

    for edge in graph.edges:
        i, j = node_index[edge.node1], node_index[edge.node2]
        adj_matrix[i][j] += 1

    node_names = [getattr(node, 'name', node) for node in sorted_nodes]
    return adj_matrix, node_names
"""

def adjacency_matrix(graph):
    sorted_nodes = sorted(graph.nodes, key=natural_sort_key)
    node_index = {node: i for i, node in enumerate(sorted_nodes)}
    n = len(sorted_nodes)
    
    adj_matrix = [[0] * n for _ in range(n)]
    
    for edge in graph.edges:
        u, v = edge.node1, edge.node2
        i, j = node_index[u], node_index[v]
        
        # Logic from reference:
        # if direction == '>': matrix[i][j] += 1
        # elif direction == '<': matrix[j][i] += 1
        # else: matrix[i][j] += 1; matrix[j][i] += 1
        
        if properties.is_directed(graph):
             adj_matrix[i][j] += 1
        else:
             adj_matrix[i][j] += 1
             adj_matrix[j][i] += 1
            
    node_names = [getattr(node, 'name', node) for node in sorted_nodes]
    return adj_matrix, node_names

def incidence_matrix(graph):
    sorted_nodes = sorted(graph.nodes, key=natural_sort_key)
    node_index = {node: i for i, node in enumerate(sorted_nodes)}

    n = len(sorted_nodes)
    m = len(graph.edges)
    inc_matrix = [[0 for _ in range(m)] for _ in range(n)]

    sorted_edges = sorted(graph.edges, key=lambda edge: natural_sort_key(edge.name) if edge.name is not None else "")

    processed_edges = {}
    edge_counter = 0

    for edge in sorted_edges:
        node1 = edge.node1
        node2 = edge.node2
        i = node_index[node1]
        j = node_index[node2]

        if properties.is_directed(graph):  
            if node1 == node2:  
                inc_matrix[i][edge_counter] = 2
            else:
                inc_matrix[i][edge_counter] = 1 
                inc_matrix[j][edge_counter] = -1  
            edge_counter += 1
        else:
            edge_key = frozenset((node1, node2))
            if edge_key not in processed_edges:
                if node1 == node2:
                    inc_matrix[i][edge_counter] = 2  
                else:
                    inc_matrix[i][edge_counter] = 1
                    inc_matrix[j][edge_counter] = 1
                processed_edges[edge_key] = True
                edge_counter += 1

    inc_matrix = [row[:edge_counter] for row in inc_matrix]

    node_names = [getattr(node, 'name', node) for node in sorted_nodes]
    edge_names = [getattr(edge, 'name', "Edge{}".format(idx)) for idx, edge in enumerate(sorted_edges[:edge_counter])]

    return inc_matrix, node_names, edge_names

def print_incidence_matrix(matrix, node_names, edge_names, graph):
    print_or_save_matrix(matrix, node_names, edge_names, title="Matice incidence", file_path="incidence.txt")

def print_adjacency_matrix(matrix, node_names, graph):
    print_or_save_matrix(matrix, node_names, node_names, title="Matice sousednosti", file_path="adjacency.txt")

def print_incidence_matrix(matrix, node_names, edge_names):
    print("Matice incidence:")
    header = "     " + " ".join(["{:>4}".format(name) for name in edge_names])
    print(header)
    for i, row in enumerate(matrix):
        row_str = "{:>3}  ".format(node_names[i]) + " ".join(["{:>4}".format(val) for val in row])
        print(row_str)
    print("\n")

def print_adjacency_matrix(matrix, node_names):
    print("Matice sousednosti:")
    header = "     " + " ".join(["{:>3}".format(name) for name in node_names])
    print(header)
    for i, row in enumerate(matrix):
        row_str = "{:>3}  ".format(node_names[i]) + " ".join(["{:>3}".format(val) for val in row])
        print(row_str)
    print("\n")

def print_labeled_matrix(matrix, node_names, title="Matrix"):
    print("\n" + title + ":")
    header = "     " + " ".join(["{:>3}".format(name) for name in node_names])
    print(header)
    for i, row in enumerate(matrix):
        row_str = "{:>3}  ".format(node_names[i]) + " ".join(["{:>3}".format(val) for val in row])
        print(row_str)
    print("\n")

def print_power_matrix(graph, power):
    adj_matrix, node_names = adjacency_matrix(graph)
    power_matrix = matrix_power(adj_matrix, power)
def print_power_matrix(graph, power):
    adj_matrix, node_names = adjacency_matrix(graph)
    power_matrix = matrix_power(adj_matrix, power)
    
    file_name = "power_matrix_{}x.txt".format(power)
    title = "Matice sousednosti na {}".format(power)
    print_or_save_matrix(power_matrix, node_names, node_names, title=title, file_path=file_name)

def matrix_multiply(A, B):
    n = len(A)
    result = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            result[i][j] = sum(A[i][k] * B[k][j] for k in range(n))
    return result

def matrix_power(matrix, power):
    n = len(matrix)
    result = [[1 if i == j else 0 for j in range(n)] for i in range(n)]

    for _ in range(power):
        result = matrix_multiply(result, matrix)
    
    return result


def second_matrix_power(graph):
    adj_matrix, _ = adjacency_matrix(graph)
    return matrix_power(adj_matrix, 2), _

def third_matrix_power(graph):
    adj_matrix, _ = adjacency_matrix(graph)
    return matrix_power(adj_matrix, 3), _

def print_matrix(matrix):
    for row in matrix:
        print(" ".join(str(val) for val in row))
    print()



def list_neighbors(graph):
    neighbors = {node: [] for node in graph.nodes}
    
    for edge in graph.edges:
        neighbors[edge.node1].append((edge.node2, edge.weight))
    
    return neighbors

def incident_edges(graph):
    incident_edges = {node: [] for node in graph.nodes}
    
    for edge in graph.edges:
        node1 = edge.node1
        node2 = edge.node2
        name = edge.name
        if name is not None:
            incident_edges[node1].append(name)
            incident_edges[node2].append(name)

    sorted_incident_edges = {node: sorted(edges) for node, edges in sorted(incident_edges.items())}
    return sorted_incident_edges


def floyd_warshall(graph):
    sorted_nodes = sorted(graph.nodes, key=natural_sort_key)
    node_index = {node: i for i, node in enumerate(sorted_nodes)}
    n = len(graph.nodes)
    inf = float('inf')

    length_matrix = [[inf for _ in range(n)] for _ in range(n)]
    predecessor_matrix = [['-' for _ in range(n)] for _ in range(n)]

    for i in range(n):
        length_matrix[i][i] = 0
        predecessor_matrix[i][i] = '0'

    for edge in graph.edges:
        node1 = edge.node1
        node2 = edge.node2
        weight = edge.weight if edge.weight is not None else 1
        
        i, j = node_index[node1], node_index[node2]
        
        # Update length matrix
        # If multiple edges, take the minimum weight
        if length_matrix[i][j] == inf or weight < length_matrix[i][j]:
             length_matrix[i][j] = weight
             predecessor_matrix[i][j] = node1
        
        if not properties.is_directed(graph):
             if length_matrix[j][i] == inf or weight < length_matrix[j][i]:
                 length_matrix[j][i] = weight
                 predecessor_matrix[j][i] = node2

    # Floyd-Warshall Algorithm
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if length_matrix[i][k] + length_matrix[k][j] < length_matrix[i][j]:
                    length_matrix[i][j] = length_matrix[i][k] + length_matrix[k][j]
                    predecessor_matrix[i][j] = predecessor_matrix[k][j]

    return length_matrix, predecessor_matrix

def distance_matrix(graph):
    # Matice délek podle reference (není to Floyd-Warshall, ale přímé délky)
    # - Na hlavní diagonále jsou nuly
    # - Pokud existuje hrana, je tam váha (nebo 1)
    # - Jinak nekonečno
    
    sorted_nodes = sorted(graph.nodes, key=natural_sort_key)
    node_index = {node: i for i, node in enumerate(sorted_nodes)}
    n = len(sorted_nodes)
    inf = float('inf')
    
    matrix = [[inf for _ in range(n)] for _ in range(n)]
    
    for i in range(n):
        matrix[i][i] = 0
        
    for edge in graph.edges:
        u, v = edge.node1, edge.node2
        weight = edge.weight if edge.weight is not None else 1
        i, j = node_index[u], node_index[v]
        
        if properties.is_directed(graph):
            matrix[i][j] = weight
        else:
            matrix[i][j] = weight
            matrix[j][i] = weight
            
    node_names = [getattr(node, 'name', node) for node in sorted_nodes]
    return matrix, node_names

def predecessor_matrix(graph):
    # Matice předchůdců podle reference (přímí předchůdci)
    # - Na hlavní diagonále '0'
    # - Pro hranu A->B: M[A][B] = A
    # - Jinak '-'
    
    sorted_nodes = sorted(graph.nodes, key=natural_sort_key)
    node_index = {node: i for i, node in enumerate(sorted_nodes)}
    n = len(sorted_nodes)
    
    matrix = [['-' for _ in range(n)] for _ in range(n)]
    
    for i in range(n):
        matrix[i][i] = '0'
        
    for edge in graph.edges:
        u, v = edge.node1, edge.node2
        i, j = node_index[u], node_index[v]
        
        # Note: Reference logic:
        # if direction == '>': matrix[i][j] = node1
        # elif direction == '<': matrix[j][i] = node2
        # else: matrix[i][j] = node1; matrix[j][i] = node2
        
        # In our graph structure, edges are stored with direction implicitly if graph is directed?
        # Or does 'properties.is_directed(graph)' imply all edges are directed?
        # Assuming our 'graph.edges' stores edges as (u, v) and we check graph property.
        
        if properties.is_directed(graph):
            matrix[i][j] = getattr(u, 'name', u)
        else:
            matrix[i][j] = getattr(u, 'name', u)
            matrix[j][i] = getattr(v, 'name', v)
            
    node_names = [getattr(node, 'name', node) for node in sorted_nodes]
    return matrix, node_names

def laplacian_matrix(graph):
    adj, node_names = adjacency_matrix(graph)
    n = len(node_names)
    
    # Create Degree matrix D
    D = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        # Degree is sum of non-zero entries in adjacency row
        # Note: In reference, degree = sum(1 for value in A[i] if value != 0)
        # This counts NEIGHBORS, not weighted degree.
        degree = sum(1 for value in adj[i] if value != 0)
        D[i][i] = degree

    # Calculate Laplacian L = D - A
    laplacian = [[D[i][j] - adj[i][j] for j in range(n)] for i in range(n)]
    
    return laplacian, node_names

def sign_matrix(graph):
    # Znamenková matice podle reference:
    # - Na hlavní diagonále jsou nuly (0)
    # - Tam, kde v matici sousednosti je 1 nebo více, je plus (+)
    # - Tam, kde v matici sousednosti je 0, je minus (-)
    
    adj_matrix, node_names = adjacency_matrix(graph)
    n = len(node_names)
    matrix = [['' for _ in range(n)] for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i][j] = '0'
            elif adj_matrix[i][j] >= 1:
                matrix[i][j] = '+'
            else:
                matrix[i][j] = '-'
            
    return matrix, node_names

def list_nodes_and_edges(graph):
    sorted_nodes = sorted(graph.nodes)
    print("Uzly:")
    for node in sorted_nodes:
        print(node)
    
    sorted_edges = sorted(graph.edges, key=lambda edge: (edge.node1, edge.node2))
    print("\nHrany:")
    for edge in sorted_edges:
        if properties.is_directed(graph=graph):
            result = "{} ----> {}".format(edge.node1, edge.node2)
        else:
            result = "{} ---- {}:".format(edge.node1, edge.node2)

        if edge.weight is not None:
            result += " váha: {}, ".format(edge.weight)
        if edge.name is not None:
            result += " název: {} ".format(edge.name)
        print(result)
