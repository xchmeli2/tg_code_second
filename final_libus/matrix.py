"""
Analyzátor grafů - zpracování grafů ze souboru a výpočet různých matic
Použití: python script.py vstupni_soubor.txt
"""

import sys
import re
from collections import defaultdict
import csv
import os
import subprocess
import unicodedata
import sys
import tempfile
from fractions import Fraction

def show_matrix_in_excel(matrix, row_labels, col_labels, title="Matice"):
    """
    Uloží matici do složky csv_export jako CSV/TXT soubor v přehledném, zarovnaném formátu
    """

    import unicodedata

    # Odstranění diakritiky a vytvoření bezpečného názvu
    safe_title = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode("ascii")
    safe_title = safe_title.replace(" ", "_")
    filename = f"{safe_title}.csv"

    # Cílová složka
    export_dir = os.path.join(os.getcwd(), "csv_export")
    os.makedirs(export_dir, exist_ok=True)
    file_path = os.path.join(export_dir, filename)

    # Převod hodnot na text
    formatted_matrix = [[str(v) for v in row] for row in matrix]
    all_values = [val for row in formatted_matrix for val in row] + row_labels + col_labels
    cell_width = max(len(str(v)) for v in all_values) + 1  # určí šířku sloupce

    with open(file_path, "w", encoding="utf-8") as f:
        # Hlavička
        f.write(" " * cell_width + "".join(f"{label:>{cell_width}}" for label in col_labels) + "\n")
        f.write("-" * ((len(col_labels) + 1) * cell_width) + "\n")

        # Každý řádek matice
        for i, label in enumerate(row_labels):
            line = f"{label:<{cell_width}}" + "".join(f"{formatted_matrix[i][j]:>{cell_width}}" for j in range(len(col_labels)))
            f.write(line + "\n")

    print(f"💾 Matice '{title}' byla uložena do: {file_path}")


class Graph:
    """Třída reprezentující graf"""
    
    def __init__(self):
        self.nodes = {}  # {node_id: weight}
        self.edges = []  # [(node1, node2, direction, weight, label)]
        self.node_order = []  # Zachování pořadí uzlů ze souboru
        self.adj_list = defaultdict(list)  # Pro směrované grafy
        self.adj_list_undirected = defaultdict(set)  # Pro neorientované grafy
        
    def add_node(self, node_id, weight=None):
        """Přidá uzel do grafu"""
        node_id = node_id.rstrip(';')
        if node_id != '*':  # Ignorujeme * označující chybějící uzel v bin. stromu
            if node_id not in self.nodes:
                self.nodes[node_id] = weight
                self.node_order.append(node_id)
    
    def add_edge(self, node1, node2, direction, weight=None, label=None):
      """
      Přidá hranu do grafu.
      
      Args:
          node1: První uzel hrany
          node2: Druhý uzel hrany
          direction: Směr hrany ('>', '<', '-')
          weight: Ohodnocení hrany
          label: Označení hrany (pokud None, vygeneruje se h<Node1><Node2>)
      
      Returns:
          True pokud se podařilo přidat hranu, False jinak
      """
      # Normalizace uzlů a labelu
      node1 = node1.strip().rstrip(';')
      node2 = node2.strip().rstrip(';')
      
      if label and label.strip():
          label = label.strip().rstrip(';')
      else:
          label = f"h{node1}{node2}"  # automatické pojmenování hrany

      # Kontrola existence uzlů
      if node1 not in self.nodes or node2 not in self.nodes:
          return False

      # Přidání hrany
      self.edges.append((node1, node2, direction, weight, label))
      print(node1, node2, direction,weight,label)
      
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
    """
    Načte graf ze souboru
    
    Formát souboru:
        u identifikator [ohodnoceni];
        h uzel1 (< | - | >) uzel2 [ohodnoceni] [:označení];
    
    Args:
        filename: Cesta k souboru s grafem
    
    Returns:
        Objekt Graph s načtenými daty
    """
    graph = Graph()
    
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        
        # Přeskočit prázdné řádky a komentáře
        if not line or line.startswith('#'):
            continue
        
        # Parsování uzlu: u identifikator [ohodnoceni];
        if line.startswith('u '):
            parts = line.split()
            node_id = parts[1]
            if len(parts) > 2:
                weight = float(parts[2].rstrip(';'))
            else:
                weight = None
            graph.add_node(node_id, weight)
        
        # Parsování hrany: h uzel1 (< | - | >) uzel2 [ohodnoceni] [:označení];
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
            
            # Ujistit se, že uzly existují (pokud nebyly definovány, vytvoříme je)
            if node1 not in graph.nodes:
                graph.add_node(node1)
            if node2 not in graph.nodes:
                graph.add_node(node2)
            
            graph.add_edge(node1, node2, direction, weight, label)
    
    # 🔹 Pokud graf nemá žádné hrany, ale obsahuje '*', vytvoříme binární strom
    if len(graph.edges) == 0 and '*' in graph.node_order:
        for i, node in enumerate(graph.node_order):
            if node == '*':
                continue
            left_i = 2 * i + 1
            right_i = 2 * i + 2
            if left_i < len(graph.node_order) and graph.node_order[left_i] != '*':
                graph.add_edge(node, '-', graph.node_order[left_i])
            if right_i < len(graph.node_order) and graph.node_order[right_i] != '*':
                graph.add_edge(node, '-', graph.node_order[right_i])
    
    return graph


def print_matrix_with_labels(matrix, row_labels, col_labels, title, format_func=None):
    """Zobrazí matici v Excelu místo tisku do konzole"""
    if format_func:
        formatted_matrix = [[format_func(v) for v in row] for row in matrix]
    else:
        formatted_matrix = matrix
    # !!! Pokud bude potreba matice vypsat, odkomentovat !!!
    show_matrix_in_excel(formatted_matrix, row_labels, col_labels, title)


def print_statistics(title, sum_first_row, sum_first_col, ones_first_row, 
                    ones_first_col, sum_diagonal, zeros_diagonal):
    """
    Vytiskne statistiky pro matici
    
    Args:
        title: Název matice
        sum_first_row: Součet čísel v prvním řádku
        sum_first_col: Součet čísel v prvním sloupci
        ones_first_row: Počet jedniček v prvním řádku
        ones_first_col: Počet jedniček v prvním sloupci
        sum_diagonal: Součet čísel na hlavní diagonále
        zeros_diagonal: Počet nul na hlavní diagonále
    """
    print(f"\n  📊 Statistiky {title}:")
    print(f"  ├─ Součet čísel v prvním řádku: {sum_first_row}")
    print(f"  ├─ Součet čísel v prvním sloupci: {sum_first_col}")
    print(f"  ├─ Počet jedniček v prvním řádku: {ones_first_row}")
    print(f"  ├─ Počet jedniček v prvním sloupci: {ones_first_col}")
    print(f"  ├─ Součet čísel na hlavní diagonále: {sum_diagonal}")
    print(f"  └─ Počet nul na hlavní diagonále: {zeros_diagonal}")


def matice_sousednosti(graph):
    """
    Vytvoří matici sousednosti
    
    Matice sousednosti obsahuje počty hran mezi uzly.
    - Pro neorientovanou hranu A - B: M[A][B] += 1 a M[B][A] += 1
    - Pro orientovanou hranu A > B: M[A][B] += 1
    - Pro orientovanou hranu A < B: M[B][A] += 1
    - Smyčky se počítají jako běžné hrany
    
    POZOR: Pokud jsou mezi dvěma uzly násobné hrany, hodnota není jen 0 nebo 1,
           ale počet těchto hran!
    
    Args:
        graph: Objekt Graph s daty grafu
    
    Returns:
        2D matice sousednosti
    """
    print("\n" + "="*80)
    print("📌 MATICE SOUSEDNOSTI")
    print("="*80)
    
    n = len(graph.node_order)
    matrix = [[0 for _ in range(n)] for _ in range(n)]
    
    # Mapování uzlů na indexy
    node_to_index = {node: i for i, node in enumerate(graph.node_order)}
    
    # Vyplnění matice podle hran
    for node1, node2, direction, _, _ in graph.edges:
        i = node_to_index[node1]
        j = node_to_index[node2]
        
        if direction == '>':  # node1 -> node2
            matrix[i][j] += 1
        elif direction == '<':  # node1 <- node2
            matrix[j][i] += 1
        else:  # direction == '-', neorientovaná hrana
            matrix[i][j] += 1
            matrix[j][i] += 1
    
    # Vytisknutí matice
    print_matrix_with_labels(matrix, graph.node_order, graph.node_order, "matice_sousednosti")
    
    # Výpočet statistik
    sum_first_row = sum(matrix[0])
    sum_first_col = sum(matrix[i][0] for i in range(n))
    ones_first_row = sum(1 for x in matrix[0] if x == 1)
    ones_first_col = sum(1 for i in range(n) if matrix[i][0] == 1)
    sum_diagonal = sum(matrix[i][i] for i in range(n))
    zeros_diagonal = sum(1 for i in range(n) if matrix[i][i] == 0)
    
    print_statistics("matice sousednosti", sum_first_row, sum_first_col, 
                    ones_first_row, ones_first_col, sum_diagonal, zeros_diagonal)
    
    return matrix


def znamenkova_matice(graph, adj_matrix):
    """
    Vytvoří znamenkovou matici a zobrazí statistiky

    Znamenková matice:
    - Na hlavní diagonále jsou nuly (0)
    - Tam, kde v matici sousednosti je 1 nebo více, je plus (+)
    - Tam, kde v matici sousednosti je 0, je minus (-)
    """
    print("\n" + "="*80)
    print("📌 ZNAMENKOVÁ MATICE")
    print("="*80)
    
    n = len(graph.node_order)
    matrix = [['' for _ in range(n)] for _ in range(n)]
    
    # Vyplnění znamenkové matice
    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i][j] = '0'
            elif adj_matrix[i][j] >= 1:
                matrix[i][j] = '+'
            else:
                matrix[i][j] = '-'
    
    # Statistiky
    plus_first_row = sum(1 for x in matrix[0] if x == '+')
    plus_first_col = sum(1 for i in range(n) if matrix[i][0] == '+')
    minus_first_row = sum(1 for x in matrix[0] if x == '-')
    minus_first_col = sum(1 for i in range(n) if matrix[i][0] == '-')
    zero_diagonal = sum(1 for i in range(n) if matrix[i][i] == '0')
    
    print(f"\n  📊 Statistiky znamenkové matice:")
    print(f"  ├─ Počet '+' v prvním řádku: {plus_first_row}")
    print(f"  ├─ Počet '+' v prvním sloupci: {plus_first_col}")
    print(f"  ├─ Počet '-' v prvním řádku: {minus_first_row}")
    print(f"  ├─ Počet '-' v prvním sloupci: {minus_first_col}")
    print(f"  └─ Počet nul na hlavní diagonále: {zero_diagonal}")
    
    # Vytisknutí matice
    print_matrix_with_labels(matrix, graph.node_order, graph.node_order, "Znamenková matice")
    
    return matrix


def multiply_matrices(A, B):
    """
    Vynásobí dvě čtvercové matice
    
    Výpočet: C[i][j] = součet(A[i][k] * B[k][j]) pro všechna k
    
    Args:
        A: První matice
        B: Druhá matice
    
    Returns:
        Výsledná matice C = A * B
    """
    n = len(A)
    C = [[0 for _ in range(n)] for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
    
    return C


def mocniny_matice_sousednosti(graph, adj_matrix):
    """
    Vypočítá 2. a 3. mocninu matice sousednosti
    
    Mocniny matice sousednosti ukazují počet cest dané délky mezi uzly:
    - M² ukazuje počet cest délky 2
    - M³ ukazuje počet cest délky 3
    
    Args:
        graph: Objekt Graph s daty grafu
        adj_matrix: Matice sousednosti
    """
    n = len(graph.node_order)
    
    # ========== DRUHÁ MOCNINA ==========
    print("\n" + "="*80)
    print("📌 DRUHÁ MOCNINA MATICE SOUSEDNOSTI (M²)")
    print("="*80)
    print("   (Ukazuje počet cest délky 2 mezi uzly)")
    
    matrix2 = multiply_matrices(adj_matrix, adj_matrix)
    
    # Vytisknutí matice
    print_matrix_with_labels(matrix2, graph.node_order, graph.node_order, "matice_m2")
    
    # Výpočet statistik
    sum_first_row = sum(matrix2[0])
    sum_first_col = sum(matrix2[i][0] for i in range(n))
    ones_first_row = sum(1 for x in matrix2[0] if x == 1)
    ones_first_col = sum(1 for i in range(n) if matrix2[i][0] == 1)
    sum_diagonal = sum(matrix2[i][i] for i in range(n))
    zeros_diagonal = sum(1 for i in range(n) if matrix2[i][i] == 0)
    
    print_statistics("M²", sum_first_row, sum_first_col, 
                    ones_first_row, ones_first_col, sum_diagonal, zeros_diagonal)
    
    # ========== TŘETÍ MOCNINA ==========
    print("\n" + "="*80)
    print("📌 TŘETÍ MOCNINA MATICE SOUSEDNOSTI (M³)")
    print("="*80)
    print("   (Ukazuje počet cest délky 3 mezi uzly)")
    
    matrix3 = multiply_matrices(matrix2, adj_matrix)
    
    # Vytisknutí matice
    print_matrix_with_labels(matrix3, graph.node_order, graph.node_order, "matice_m3")
    
    # Výpočet statistik
    sum_first_row = sum(matrix3[0])
    sum_first_col = sum(matrix3[i][0] for i in range(n))
    ones_first_row = sum(1 for x in matrix3[0] if x == 1)
    ones_first_col = sum(1 for i in range(n) if matrix3[i][0] == 1)
    sum_diagonal = sum(matrix3[i][i] for i in range(n))
    zeros_diagonal = sum(1 for i in range(n) if matrix3[i][i] == 0)
    
    print_statistics("M³", sum_first_row, sum_first_col, 
                    ones_first_row, ones_first_col, sum_diagonal, zeros_diagonal)


def matice_incidence(graph):
    """
    Vytvoří matici incidence
    
    Matice incidence:
    - Řádky = uzly, Sloupce = hrany
    - Pro orientovanou hranu A -> B: M[A][hrana] = 1, M[B][hrana] = -1
    - Pro neorientovanou hranu A - B: M[A][hrana] = 1, M[B][hrana] = 1
    - Pro smyčku A -> A: M[A][hrana] = 2
    - Jinak: M[uzel][hrana] = 0
    
    Args:
        graph: Objekt Graph s daty grafu
    
    Returns:
        2D matice incidence
    """
    print("\n" + "="*80)
    print("📌 MATICE INCIDENCE")
    print("="*80)
    
    n = len(graph.node_order)
    m = len(graph.edges)
    matrix = [[0 for _ in range(m)] for _ in range(n)]
    
    # Mapování uzlů na indexy
    node_to_index = {node: i for i, node in enumerate(graph.node_order)}
    
    # Popisky hran pro sloupce - použijeme label pokud existuje, jinak vytvoříme z uzlů
    edge_labels = []
    for node1, node2, direction, weight, label in graph.edges:
        if label:
            edge_labels.append(label)
        else:
            # Vytvoříme popisek z uzlů a směru
            if direction == '>':
                edge_labels.append(f"{node1}{node2}")
            elif direction == '<':
                edge_labels.append(f"{node2}{node1}")
            else:
                edge_labels.append(f"{node1}{node2}")
    
    # Vyplnění matice podle hran
    for edge_idx, (node1, node2, direction, _, _) in enumerate(graph.edges):
        i = node_to_index[node1]
        j = node_to_index[node2]
        
        # Smyčka (hrana ze stejného uzlu do sebe)
        if node1 == node2:
            matrix[i][edge_idx] = 2
        elif direction == '>':  # node1 > node2 znamená node1 -> node2 (hrana z node1 do node2)
            matrix[i][edge_idx] = 1   # z node1 (vychází) = +1
            matrix[j][edge_idx] = -1  # do node2 (vchází) = -1
        elif direction == '<':  # node1 < node2 znamená node1 <- node2 (hrana z node2 do node1)
            matrix[j][edge_idx] = 1   # z node2 (vychází) = +1
            matrix[i][edge_idx] = -1  # do node1 (vchází) = -1
        else:  # direction == '-', neorientovaná hrana
            matrix[i][edge_idx] = 1
            matrix[j][edge_idx] = 1
    
    # Vytisknutí matice s užší šířkou sloupců pro hrany
    print_matrix_with_labels(matrix, graph.node_order, edge_labels, 
                           "matice_incidence")
    
    # Výpočet statistik (pokud existují hrany)
    if m > 0 and n > 0:
        sum_first_row = sum(matrix[0])
        sum_first_col = sum(matrix[i][0] for i in range(n))
        ones_first_row = sum(1 for x in matrix[0] if x == 1)
        ones_first_col = sum(1 for i in range(n) if matrix[i][0] == 1)
        
        # Pro obdélníkovou matici nemá smysl hlavní diagonála
        # Vypočítáme diagonálu jen pokud existuje (min(n, m) prvků)
        diag_size = min(n, m)
        sum_diagonal = sum(matrix[i][i] for i in range(diag_size))
        zeros_diagonal = sum(1 for i in range(diag_size) if matrix[i][i] == 0)
        
        print(f"\n  📊 Statistiky matice incidence:")
        print(f"  ├─ Součet čísel v prvním řádku: {sum_first_row}")
        print(f"  ├─ Součet čísel v prvním sloupci: {sum_first_col}")
        print(f"  ├─ Počet jedniček v prvním řádku: {ones_first_row}")
        print(f"  ├─ Počet jedniček v prvním sloupci: {ones_first_col}")
        print(f"  ├─ Součet na (pseudo)diagonále: {sum_diagonal}")
        print(f"  └─ Počet nul na (pseudo)diagonále: {zeros_diagonal}")
    
    return matrix

def matice_delek(graph):
    """
    Vytvoří matici délek
    
    Matice délek:
    - Na hlavní diagonále jsou nuly (vzdálenost uzlu od sebe sama)
    - Pokud existuje hrana mezi uzly, je tam ohodnocení hrany (nebo 1 pokud není ohodnocení)
    - Jinak je tam ∞ (nekonečno)
    
    Args:
        graph: Objekt Graph s daty grafu
    
    Returns:
        2D matice délek
    """
    print("\n" + "="*80)
    print("📌 MATICE DÉLEK")
    print("="*80)
    
    n = len(graph.node_order)
    INF = float('inf')
    matrix = [[INF for _ in range(n)] for _ in range(n)]
    
    # Mapování uzlů na indexy
    node_to_index = {node: i for i, node in enumerate(graph.node_order)}
    
    # Hlavní diagonála = 0 (vzdálenost uzlu od sebe sama)
    for i in range(n):
        matrix[i][i] = 0
    
    # Vyplnění délek podle hran
    for node1, node2, direction, weight, _ in graph.edges:
        i = node_to_index[node1]
        j = node_to_index[node2]
        
        # Pokud hrana nemá ohodnocení, použijeme 1
        length = weight if weight is not None else 1
        
        if direction == '>':  # node1 -> node2
            matrix[i][j] = length
        elif direction == '<':  # node1 <- node2
            matrix[j][i] = length
        else:  # direction == '-', neorientovaná hrana
            matrix[i][j] = length
            matrix[j][i] = length
    
    # Formátovací funkce pro tisk
    def format_value(val):
        if val == INF:
            return '∞'
        elif val == int(val):
            return str(int(val))
        else:
            return f"{val:.1f}"
    
    # Vytisknutí matice
    print_matrix_with_labels(matrix, graph.node_order, graph.node_order, 
                           "matice_delek", format_value)
    
    # Výpočet statistik (ignorujeme nekonečna)
    sum_first_row = sum(x for x in matrix[0] if x != INF)
    sum_first_col = sum(matrix[i][0] for i in range(n) if matrix[i][0] != INF)
    ones_first_row = sum(1 for x in matrix[0] if x == 1)
    ones_first_col = sum(1 for i in range(n) if matrix[i][0] == 1)
    sum_diagonal = sum(matrix[i][i] for i in range(n) if matrix[i][i] != INF)
    zeros_diagonal = sum(1 for i in range(n) if matrix[i][i] == 0)
    
    print_statistics("matice délek", sum_first_row, sum_first_col, 
                    ones_first_row, ones_first_col, sum_diagonal, zeros_diagonal)
    
    return matrix


def matice_predchudcu(graph):
    """
    Vytvoří matici předchůdců
    
    Matice předchůdců:
    - Na hlavní diagonále jsou nuly (0)
    - Pro hranu A -> B: M[A][B] = A (předchůdce uzlu B na hraně AB je A)
    - Pro hranu A - B: M[A][B] = A a M[B][A] = B
    - Jinak: M[i][j] = '-' (žádná hrana)
    
    Args:
        graph: Objekt Graph s daty grafu
    
    Returns:
        2D matice předchůdců
    """
    print("\n" + "="*80)
    print("📌 MATICE PŘEDCHŮDCŮ")
    print("="*80)
    
    n = len(graph.node_order)
    matrix = [['-' for _ in range(n)] for _ in range(n)]
    
    # Mapování uzlů na indexy
    node_to_index = {node: i for i, node in enumerate(graph.node_order)}
    
    # Hlavní diagonála = 0
    for i in range(n):
        matrix[i][i] = '0'
    
    # Vyplnění předchůdců podle hran
    for node1, node2, direction, _, _ in graph.edges:
        i = node_to_index[node1]
        j = node_to_index[node2]
        
        if direction == '>':  # node1 -> node2, předchůdce node2 je node1
            matrix[i][j] = node1
        elif direction == '<':  # node1 <- node2, předchůdce node1 je node2
            matrix[j][i] = node2
        else:  # direction == '-', neorientovaná hrana
            matrix[i][j] = node1
            matrix[j][i] = node2
    
    # Vytisknutí matice
    print_matrix_with_labels(matrix, graph.node_order, graph.node_order, "matice_predchudcu")
    
    # Statistiky pro matici předchůdců (počítáme definované předchůdce)
    defined_first_row = sum(1 for x in matrix[0] if x not in ['-', '0'])
    defined_first_col = sum(1 for i in range(n) if matrix[i][0] not in ['-', '0'])
    
    print(f"\n  📊 Statistiky matice předchůdců:")
    print(f"  ├─ Počet definovaných předchůdců v prvním řádku: {defined_first_row}")
    print(f"  └─ Počet definovaných předchůdců v prvním sloupci: {defined_first_col}")
    #print(matrix[2][2])
    
    #index_C = graph.node_order.index("C")
    #value = matrix[index_C][3]
    #print('Matice ma pro radek C na ctvrtem miste', value)
    
    return matrix


def analyze_graph_matrices(filename):
    """
    Hlavní funkce - analyzuje graf a vytvoří všechny matice
    
    Args:
        filename: Cesta k souboru s grafem
    """
    print("\n" + "🔷"*40)
    print(f"🔷  ANALÝZA GRAFU: {filename}")
    print("🔷"*40)
    
    # Načtení grafu ze souboru
    graph = parse_graph_file(filename)
    
    print(f"\n📊 Základní informace o grafu:")
    print(f"   ├─ Počet uzlů: {len(graph.nodes)}")
    print(f"   ├─ Počet hran: {len(graph.edges)}")
    
    # 1. Matice sousednosti
    adj_matrix = matice_sousednosti(graph)
    
    # 2. Znamenková matice
    znamenkova_matice(graph, adj_matrix)
    
    # 3. Mocniny matice sousednosti (2. a 3.)
    mocniny_matice_sousednosti(graph, adj_matrix)
    
    # 4. Matice incidence
    matice_incidence(graph)
    
    # 5. Matice délek
    matice_delek(graph)
    
    # 6. Matice předchůdců
    matice_predchudcu(graph)
    
    pocet_koster(graph)
    
    # 8️⃣ Minimální kostra pomocí Kruskala
    minimalni_kostra_kruskal(graph)
    
    # 9️⃣ Maximální kostra pomocí Kruskala
    maximalni_kostra_kruskal(graph)
    
    get_matrix_row(adj_matrix, graph, "A")
    get_matrix_column(adj_matrix, graph, "A")
    get_matrix_cell(adj_matrix, graph, "A", "A")
    count_values_greater_than(adj_matrix, graph, "A", 2)
    
    # Závěrečná zpráva
    print("\n" + "🔷"*40)
    print("🔷  ✅ ANALÝZA DOKONČENA")
    print("🔷"*40 + "\n")

def determinant_fraction(matrix):
    """
    Spočítá determinant matice (seznam seznamů) pomocí Gausse
    s přesnou racionální aritmetikou (Fraction) – bez chyb zaokrouhlení.
    """
    n = len(matrix)
    A = [[Fraction(x) for x in row] for row in matrix]
    det = Fraction(1)
    swaps = 0

    for i in range(n):
        # Najdi pivot
        pivot_row = None
        for r in range(i, n):
            if A[r][i] != 0:
                pivot_row = r
                break
        if pivot_row is None:
            return Fraction(0)

        # Prohoď řádky, pokud je třeba
        if pivot_row != i:
            A[i], A[pivot_row] = A[pivot_row], A[i]
            swaps += 1

        pivot = A[i][i]
        for j in range(i+1, n):
            if A[j][i] == 0:
                continue
            factor = A[j][i] / pivot
            for k in range(i, n):
                A[j][k] -= factor * A[i][k]

    for i in range(n):
        det *= A[i][i]
    if swaps % 2 == 1:
        det = -det
    return det

def laplaceova_matice(graph):
    """
    Vytvoří Laplaceovu matici grafu (L = D - A)

    - D = matice stupňů (na diagonále je stupeň uzlu)
    - A = matice sousednosti (počty hran mezi uzly)
    - L = D - A
    """
    print("\n" + "="*80)
    print("📌 LAPLACEOVA MATICE (L = D - A)")
    print("="*80)

    # 1️⃣ Získáme matici sousednosti
    A = matice_sousednosti(graph)
    n = len(graph.node_order)

    # 2️⃣ Vytvoříme matici stupňů D
    D = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        degree = sum(1 for value in A[i] if value != 0)
        D[i][i] = degree

    # 3️⃣ Spočítáme Laplaceovu matici L = D - A
    L = [[D[i][j] - A[i][j] for j in range(n)] for i in range(n)]

    # 4️⃣ Uložíme jako CSV a vypíšeme statistiky
    print_matrix_with_labels(L, graph.node_order, graph.node_order, "Laplaceova matice")

    # 5️⃣ Statistiky
    sum_first_row = sum(L[0])
    sum_first_col = sum(L[i][0] for i in range(n))
    sum_diagonal = sum(L[i][i] for i in range(n))
    zeros_diagonal = sum(1 for i in range(n) if L[i][i] == 0)

    print(f"\n  📊 Statistiky Laplaceovy matice:")
    print(f"  ├─ Součet prvního řádku: {sum_first_row}")
    print(f"  ├─ Součet prvního sloupce: {sum_first_col}")
    print(f"  ├─ Součet diagonály: {sum_diagonal}")
    print(f"  └─ Počet nul na diagonále: {zeros_diagonal}")

    return L

def pocet_koster(graph, remove_row=0, remove_col=0):
    """
    Spočítá počet koster grafu pomocí Kirchhoffovy věty:
    1️⃣ vytvoří Laplaceovu matici
    2️⃣ odstraní z ní 1 řádek a 1 sloupec
    3️⃣ spočítá determinant výsledné matice
    4️⃣ absolutní hodnota determinantu = počet koster
    """
    print("\n" + "="*80)
    print("🌳 POČET KOSTER GRAFU (Kirchhoffova věta)")
    print("="*80)

    # 1️⃣ Získáme Laplaceovu matici
    L = laplaceova_matice(graph)
    n = len(L)
    if n <= 1:
        print("⚠️ Graf má příliš málo uzlů – počet koster = 1.")
        return 1

    # 2️⃣ Odstraníme řádek a sloupec
    reduced = []
    for i in range(n):
        if i == remove_row:
            continue
        row = [L[i][j] for j in range(n) if j != remove_col]
        reduced.append(row)

    # 3️⃣ Spočítáme determinant přesně
    det = determinant_fraction(reduced)
    pocet = abs(int(det))

    # 4️⃣ Výstup
    uzly = graph.node_order if getattr(graph, "node_order", None) else list(graph.nodes.keys())
    odstraneny_uzel = uzly[remove_row] if remove_row < len(uzly) else f"řádek {remove_row}"
    print(f"🧩 Odstraněn řádek/sloupec: {odstraneny_uzel}")
    print(f"📐 Determinant zmenšené matice: {det}")
    print(f"🌲 Počet koster grafu: {pocet}")
    print("="*80)
    return pocet

def minimalni_kostra_kruskal(graph):
    """
    Vytvoří minimální kostru grafu pomocí Kruskalova algoritmu.
    
    - Funguje pro neorientované vážené grafy
    - Vrací seznam hran tvořících minimální kostru
    """
    print("\n" + "="*80)
    print("🌲 MINIMÁLNÍ KOSTRA (Kruskalův algoritmus)")
    print("="*80)

    # --- Pomocné funkce pro Union-Find (disjoint set) ---
    parent = {}
    rank = {}

    def find(node):
        """Najde zástupce množiny (s kompresí cesty)."""
        if parent[node] != node:
            parent[node] = find(parent[node])
        return parent[node]

    def union(node1, node2):
        """Spojí dvě množiny podle ranku."""
        root1 = find(node1)
        root2 = find(node2)
        if root1 == root2:
            return False
        if rank[root1] < rank[root2]:
            parent[root1] = root2
        elif rank[root1] > rank[root2]:
            parent[root2] = root1
        else:
            parent[root2] = root1
            rank[root1] += 1
        return True

    # --- Inicializace ---
    for node in graph.nodes.keys():
        parent[node] = node
        rank[node] = 0

    # --- Načteme neorientované hrany s váhou ---
    edges = []
    for node1, node2, direction, weight, label in graph.edges:
        if direction == '-':  # pouze neorientované hrany
            edges.append((weight, node1, node2, label))

    # --- Seřadíme podle váhy ---
    edges.sort(key=lambda x: x[0])

    # --- Kruskal ---
    mst = []
    total_weight = 0
    for weight, u, v, label in edges:
        if union(u, v):
            mst.append((u, v, weight, label))
            total_weight += weight

    # --- Výstup ---
    print("Hrany minimální kostry:")
    for u, v, w, l in mst:
        print(f"  {u} - {v} | váha: {w} | label: {l}")
    print(f"\nCelková váha kostry: {total_weight}")
    print("="*80 + "\n")

    return mst

def maximalni_kostra_kruskal(graph):
    """
    Vytvoří maximální kostru grafu pomocí Kruskalova algoritmu.

    - Funguje pro neorientované vážené grafy
    - Vrací seznam hran tvořících maximální kostru
    """
    print("\n" + "="*80)
    print("🌲 MAXIMÁLNÍ KOSTRA (Kruskalův algoritmus)")
    print("="*80)

    # --- Pomocné funkce pro Union-Find (disjoint set) ---
    parent = {}
    rank = {}

    def find(node):
        """Najde zástupce množiny (s kompresí cesty)."""
        if parent[node] != node:
            parent[node] = find(parent[node])
        return parent[node]

    def union(node1, node2):
        """Spojí dvě množiny podle ranku."""
        root1 = find(node1)
        root2 = find(node2)
        if root1 == root2:
            return False
        if rank[root1] < rank[root2]:
            parent[root1] = root2
        elif rank[root1] > rank[root2]:
            parent[root2] = root1
        else:
            parent[root2] = root1
            rank[root1] += 1
        return True

    # --- Inicializace ---
    for node in graph.nodes.keys():
        parent[node] = node
        rank[node] = 0

    # --- Načteme neorientované hrany s váhou ---
    edges = []
    for node1, node2, direction, weight, label in graph.edges:
        if direction == '-':  # pouze neorientované hrany
            edges.append((weight, node1, node2))

    # --- Seřadíme podle váhy SESTUPNĚ ---
    edges.sort(key=lambda x: x[0], reverse=True)

    # --- Kruskal pro maximální kostru ---
    mst = []
    total_weight = 0
    for weight, u, v in edges:
        if union(u, v):
            mst.append((u, v, weight))
            total_weight += weight

    # --- Výstup ---
    print("Hrany maximální kostry:")
    for u, v, w in mst:
        print(f"  {u} - {v} | váha: {w}")
    print(f"\nCelková váha maximální kostry: {total_weight}")
    print("="*80 + "\n")

    return mst


def get_matrix_row(matrix, graph, node_name):
    """
    Vypíše řádek matice podle názvu uzlu
    
    Args:
        matrix: Matice (2D list)
        graph: Objekt Graph s daty grafu
        node_name: Název uzlu (např. "A", "B", "v1")
    
    Returns:
        List hodnot v řádku nebo None pokud uzel neexistuje
    """
    if node_name not in graph.nodes:
        print(f"❌ Uzel '{node_name}' neexistuje v grafu!")
        return None
    
    index = graph.node_order.index(node_name)
    row = matrix[index]
    
    print(f"\n📋 Řádek pro uzel '{node_name}' (index {index}):")
    print(f"   {row}")
    
    return row


def get_matrix_column(matrix, graph, node_name):
    """
    Vypíše sloupec matice podle názvu uzlu
    
    Args:
        matrix: Matice (2D list)
        graph: Objekt Graph s daty grafu
        node_name: Název uzlu (např. "A", "B", "v1")
    
    Returns:
        List hodnot ve sloupci nebo None pokud uzel neexistuje
    """
    if node_name not in graph.nodes:
        print(f"❌ Uzel '{node_name}' neexistuje v grafu!")
        return None
    
    index = graph.node_order.index(node_name)
    column = [matrix[i][index] for i in range(len(matrix))]
    
    print(f"\n📋 Sloupec pro uzel '{node_name}' (index {index}):")
    print(f"   {column}")
    
    return column


def get_matrix_cell(matrix, graph, row_node, col_node):
    """
    Vypíše konkrétní buňku matice podle názvů uzlů
    
    Args:
        matrix: Matice (2D list)
        graph: Objekt Graph s daty grafu
        row_node: Název uzlu pro řádek
        col_node: Název uzlu pro sloupec
    
    Returns:
        Hodnota v buňce nebo None pokud některý uzel neexistuje
    """
    if row_node not in graph.nodes:
        print(f"❌ Uzel '{row_node}' neexistuje v grafu!")
        return None
    
    if col_node not in graph.nodes:
        print(f"❌ Uzel '{col_node}' neexistuje v grafu!")
        return None
    
    row_index = graph.node_order.index(row_node)
    col_index = graph.node_order.index(col_node)
    value = matrix[row_index][col_index]
    
    print(f"\n📋 Buňka [{row_node}][{col_node}] (index [{row_index}][{col_index}]):")
    print(f"   Hodnota: {value}")
    
    return value


def count_positive_values(matrix):
    """
    Spočítá celkový počet kladných hodnot v matici
    
    Args:
        matrix: Matice (2D list) s číselnými hodnotami
    
    Returns:
        Počet kladných hodnot (> 0)
    """
    count = 0
    for row in matrix:
        for value in row:
            # Kontrola, zda je hodnota číslo a je kladná
            if isinstance(value, (int, float)) and value > 0:
                count += 1
    return count


def count_values_greater_than(matrix, graph, node_name, threshold):
    """
    Spočítá počet hodnot větších než daný práh v řádku určeného uzlu
    
    Args:
        matrix: Matice (2D list)
        graph: Objekt Graph s daty grafu
        node_name: Název uzlu (řádek matice)
        threshold: Prahová hodnota pro porovnání
    
    Returns:
        Počet hodnot větších než threshold v daném řádku
    """
    if node_name not in graph.nodes:
        print(f"❌ Uzel '{node_name}' neexistuje v grafu!")
        return None
    
    index = graph.node_order.index(node_name)
    row = matrix[index]
    
    count = 0
    for value in row:
        # Kontrola, zda je hodnota číslo a je větší než threshold
        if isinstance(value, (int, float)) and value > threshold:
            count += 1
    
    print(f"\n📊 Statistika pro řádek '{node_name}':")
    print(f"   └─ Počet hodnot > {threshold}: {count}")
    
    return count

def main():
    """Hlavní vstupní bod programu"""
    
    # Kontrola argumentů příkazové řádky
    if len(sys.argv) != 2:
        print("❌ Chyba: Špatný počet argumentů!")
        print("\nPoužití:")
        print("  python script.py <soubor_grafu.txt>")
        print("\nPříklad:")
        print("  python script.py graf.txt")
        sys.exit(1)
    
    filename = sys.argv[1]
    
    # Kontrola existence souboru
    try:
        with open(filename, 'r') as f:
            pass
    except FileNotFoundError:
        print(f"❌ Chyba: Soubor '{filename}' nebyl nalezen!")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Chyba při otevírání souboru: {e}")
        sys.exit(1)
    
    # Spuštění analýzy
    analyze_graph_matrices(filename)


if __name__ == "__main__":
    main()