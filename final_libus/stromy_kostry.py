import sys
from collections import deque
import numpy as np

class UnionFind:
    """Union-Find (Disjoint Set) pro detekci cyklů"""
    def __init__(self, nodes):
        self.parent = {node: node for node in nodes}
        self.rank = {node: 0 for node in nodes}
    
    def find(self, node):
        """Najde kořen množiny (s kompresí cesty)"""
        if self.parent[node] != node:
            self.parent[node] = self.find(self.parent[node])
        return self.parent[node]
    
    def union(self, node1, node2):
        """Spojí dvě množiny, vrací True pokud byly oddělené"""
        root1 = self.find(node1)
        root2 = self.find(node2)
        
        if root1 == root2:
            return False  # Už jsou ve stejné množině (cyklus!)
        
        # Union by rank
        if self.rank[root1] < self.rank[root2]:
            self.parent[root1] = root2
        elif self.rank[root1] > self.rank[root2]:
            self.parent[root2] = root1
        else:
            self.parent[root2] = root1
            self.rank[root1] += 1
        
        return True

class BinaryTree:
    def __init__(self):
        self.nodes = {}  # {node_id: weight}
        self.tree_structure = {}  # {node: (left_child, right_child)}
        self.root = None
    
    def add_node(self, node_id, weight=None):
        """Přidá uzel při načítání ze souboru"""
        node_id = node_id.rstrip(';')
        if node_id != '*':
            self.nodes[node_id] = weight
    
    def build_structure(self):
        """Vytvoří BST strukturu podle vah"""
        if not self.nodes:
            return
        
        # Zjistíme, jestli máme ohodnocené uzly
        has_weighted = any(weight is not None for weight in self.nodes.values())
        
        if has_weighted:
            print("  → Stavím ohodnocený strom podle BST pravidel")
            self._build_bst_structure()
        else:
            print("  → Uzly nemají váhy")
    
    def _build_bst_structure(self):
        """Postaví BST podle vah"""
        self.adjacency = None  # Není to graf
        self.edges = []  # Žádné hrany
        
        self.tree_structure = {}
        self.root = None
        
        # První uzel je kořen
        nodes_list = list(self.nodes.items())
        if not nodes_list:
            return
        
        first_node, first_weight = nodes_list[0]
        self.root = first_node
        self.tree_structure[first_node] = (None, None)
        print(f"    🌳 Kořen: {first_node} (váha {first_weight})")
        
        # Vložíme ostatní uzly
        for node, weight in nodes_list[1:]:
            if weight is not None:
                print(f"    🔹 Vkládám {node} (váha {weight})")
                self._insert_into_bst(node, weight)
    
    def _insert_into_bst(self, node_id, weight):
        """Vloží uzel do BST podle váhy"""
        current = self.root
        
        while True:
            current_weight = self.nodes[current]
            
            if weight < current_weight:
                # Jdeme doleva
                left, right = self.tree_structure[current]
                if left is None:
                    self.tree_structure[current] = (node_id, right)
                    self.tree_structure[node_id] = (None, None)
                    print(f"      → Vloženo jako levý potomek '{current}'")
                    return
                else:
                    current = left
            else:
                # Jdeme doprava
                left, right = self.tree_structure[current]
                if right is None:
                    self.tree_structure[current] = (left, node_id)
                    self.tree_structure[node_id] = (None, None)
                    print(f"      → Vloženo jako pravý potomek '{current}'")
                    return
                else:
                    current = right
    
    def insert_node(self, node_id, weight):
        """Vloží nový uzel do BST"""
        print(f"\n🔹 Vkládám uzel '{node_id}' s váhou {weight}")
        
        # Přidáme do nodes
        self.nodes[node_id] = weight
        
        if not self.root:
            # Prázdný strom
            print(f"  → Strom je prázdný, '{node_id}' se stává kořenem")
            self.root = node_id
            self.tree_structure[node_id] = (None, None)
            export_tree_to_csv(self, 'tree.txt')
            return
        
        # Najdeme místo podle BST pravidel
        current = self.root
        while True:
            current_weight = self.nodes[current]
            print(f"  → Porovnávám s uzlem '{current}' (váha {current_weight})")
            
            if weight < current_weight:
                # Jdeme doleva
                print(f"    {weight} < {current_weight} → jdu doleva")
                left, right = self.tree_structure[current]
                
                if left is None:
                    # Našli jsme místo
                    print(f"    ✓ Vkládám jako levý potomek '{current}'")
                    self.tree_structure[current] = (node_id, right)
                    self.tree_structure[node_id] = (None, None)
                    break
                else:
                    current = left
            else:
                # Jdeme doprava
                print(f"    {weight} >= {current_weight} → jdu doprava")
                left, right = self.tree_structure[current]
                
                if right is None:
                    # Našli jsme místo
                    print(f"    ✓ Vkládám jako pravý potomek '{current}'")
                    self.tree_structure[current] = (left, node_id)
                    self.tree_structure[node_id] = (None, None)
                    break
                else:
                    current = right
        
        export_tree_to_csv(self, 'tree.txt')
        print(f"  ✓ Uzel '{node_id}' byl úspěšně vložen")
    
    
    def _build_from_edges(self, edges):
        """Vytvoří graf z hran - rozlišuje orientované/neorientované"""
        # Vytvoříme adjacency list pro graf
        self.adjacency = {node: [] for node in self.nodes.keys()}
        self.is_directed = False  # Příznak orientovaného grafu
        self.edges = []  # Uložíme hrany pro Kruskalův algoritmus
    
        for node1, node2, weight, direction in edges:
            if node1 in self.adjacency and node2 in self.adjacency:
                # Uložíme hranu (jen jednou pro neorientovaný graf)
                if direction == '-':
                    # Neorientovaná hrana - přidáme ji jen jednou
                    if not any(e[:2] == (node1, node2) or e[:2] == (node2, node1) for e in self.edges):
                        self.edges.append((node1, node2, weight if weight is not None else 1))
                    
                    if node2 not in self.adjacency[node1]:
                        self.adjacency[node1].append(node2)
                    if node1 not in self.adjacency[node2]:
                        self.adjacency[node2].append(node1)
                elif direction in ['>', '<']:
                    self.is_directed = True
                    self.edges.append((node1, node2, weight if weight is not None else 1))
                    
                    # Pro '<' musíme prohodit směr
                    if direction == '<':
                        # A < B znamená B -> A
                        if node1 not in self.adjacency[node2]:
                            self.adjacency[node2].append(node1)
                    else:  # '>'
                        # A > B znamená A -> B
                        if node2 not in self.adjacency[node1]:
                            self.adjacency[node1].append(node2)
    
        # Nastavíme kořen
        nodes_list = list(self.nodes.keys())
        if nodes_list:
            self.root = nodes_list[0]
            print(f"    🌳 Kořen: {self.root}")
        
        # Pro každý uzel vytvoříme tree_structure (pro kompatibilitu)
        for node, neighbors in self.adjacency.items():
            left = neighbors[0] if len(neighbors) > 0 else None
            right = neighbors[1] if len(neighbors) > 1 else None
            self.tree_structure[node] = (left, right)
            
            if neighbors:
                print(f"    {node}: sousedé = {neighbors}")


    def _build_from_level_order(self, nodes_in_order):
        """Vytvoří binární strom z level-order posloupnosti s hvězdičkami"""
        if not nodes_in_order:
            return
        
        self.adjacency = None  # Není to graf
        self.edges = []  # Žádné hrany
        
        # První uzel je kořen
        self.root = nodes_in_order[0]
        self.tree_structure = {}
        
        print(f"    🌳 Kořen: {self.root}")
        
        # Procházíme uzly a přiřazujeme potomky
        # Pro uzel na indexu i:
        #   - levý potomek je na indexu 2*i + 1
        #   - pravý potomek je na indexu 2*i + 2
        
        for i, node in enumerate(nodes_in_order):
            if node == '*':
                continue
            
            left_idx = 2 * i + 1
            right_idx = 2 * i + 2
            
            left = None
            right = None
            
            if left_idx < len(nodes_in_order) and nodes_in_order[left_idx] != '*':
                left = nodes_in_order[left_idx]
            
            if right_idx < len(nodes_in_order) and nodes_in_order[right_idx] != '*':
                right = nodes_in_order[right_idx]
            
            self.tree_structure[node] = (left, right)
            
            if left or right:
                print(f"    {node}: left={left}, right={right}")

    def delete_node(self, node_id):
        """Odstraní uzel ze stromu"""
        if node_id not in self.nodes:
            print(f"⚠ Uzel '{node_id}' neexistuje")
            return False
        
        print(f"\n🗑️ Odstraňuji uzel '{node_id}'")
        
        # Najdeme rodiče
        parent = None
        is_left_child = False
        
        for p, (l, r) in self.tree_structure.items():
            if l == node_id:
                parent = p
                is_left_child = True
                break
            elif r == node_id:
                parent = p
                is_left_child = False
                break
        
        left, right = self.tree_structure.get(node_id, (None, None))
        
        # Sebereme všechny uzly z podstromů
        children_to_reinsert = []
        if left:
            children_to_reinsert.extend(self._collect_subtree(left))
        if right:
            children_to_reinsert.extend(self._collect_subtree(right))
        
        print(f"  → Našel jsem {len(children_to_reinsert)} uzlů k přesunutí")
        
        # Odpojíme uzel od rodiče
        if parent:
            p_left, p_right = self.tree_structure[parent]
            if is_left_child:
                self.tree_structure[parent] = (None, p_right)
            else:
                self.tree_structure[parent] = (p_left, None)
        else:
            # Mažeme kořen
            self.root = None
        
        # Smažeme uzel a jeho podstromy
        nodes_to_remove = [node_id] + [n for n, _ in children_to_reinsert]
        for node in nodes_to_remove:
            if node in self.nodes:
                del self.nodes[node]
            if node in self.tree_structure:
                del self.tree_structure[node]
        
        # Pokud není kořen, znovu vložíme potomky
        if self.root:
            print(f"  → Znovu vkládám {len(children_to_reinsert)} uzlů")
            for child_id, child_weight in children_to_reinsert:
                if child_weight is not None:
                    # Tichý režim - bez debug výpisů
                    self._silent_insert(child_id, child_weight)
        
        export_tree_to_csv(self, 'tree.txt')
        print(f"  ✓ Uzel '{node_id}' byl úspěšně odstraněn")
        return True
    
    def _silent_insert(self, node_id, weight):
        """Vloží uzel bez debug výpisů"""
        self.nodes[node_id] = weight
        
        if not self.root:
            self.root = node_id
            self.tree_structure[node_id] = (None, None)
            return
        
        current = self.root
        while True:
            current_weight = self.nodes[current]
            
            if weight < current_weight:
                left, right = self.tree_structure[current]
                if left is None:
                    self.tree_structure[current] = (node_id, right)
                    self.tree_structure[node_id] = (None, None)
                    return
                else:
                    current = left
            else:
                left, right = self.tree_structure[current]
                if right is None:
                    self.tree_structure[current] = (left, node_id)
                    self.tree_structure[node_id] = (None, None)
                    return
                else:
                    current = right
    
    def _collect_subtree(self, node_id):
        """Rekurzivně sebere všechny uzly z podstromu"""
        result = []
        weight = self.nodes.get(node_id)
        result.append((node_id, weight))
        
        left, right = self.tree_structure.get(node_id, (None, None))
        if left:
            result.extend(self._collect_subtree(left))
        if right:
            result.extend(self._collect_subtree(right))
        
        return result


def export_tree_to_csv(tree, filename):
    """Exportuje strom do textového souboru"""
    if filename.endswith('.csv'):
        filename = filename.replace('.csv', '.txt')
    
    # NOVÉ: Kontrola pro grafy
    if hasattr(tree, 'adjacency') and tree.adjacency:
        print(f"  ⚠ Graf nelze exportovat jako strom")
        return
    
    if not tree.tree_structure or tree.root is None:
        return
    
    # Zjistíme hloubku
    max_depth = 0
    queue = deque([(tree.root, 0)])
    node_depths = {tree.root: 0}
    visited = set([tree.root])
    
    while queue:
        node, depth = queue.popleft()
        max_depth = max(max_depth, depth)
        
        if node not in tree.tree_structure:
            continue
        
        left, right = tree.tree_structure.get(node, (None, None))
        
        if left and left not in visited:
            node_depths[left] = depth + 1
            queue.append((left, depth + 1))
            visited.add(left)
        if right and right not in visited:
            node_depths[right] = depth + 1
            queue.append((right, depth + 1))
            visited.add(right)
    
    # ASCII art
    def build_tree(node, prefix="", is_tail=True):
        if node is None:
            return []
        
        lines = []
        weight = tree.nodes.get(node, '')
        node_str = f"{node}({weight})" if weight not in (None, '') else str(node)
        
        connector = "└── " if is_tail else "├── "
        lines.append(prefix + connector + node_str)
        
        left, right = tree.tree_structure.get(node, (None, None))
        extension = "    " if is_tail else "│   "
        
        if left and right:
            lines.extend(build_tree(left, prefix + extension, False))
            lines.extend(build_tree(right, prefix + extension, True))
        elif left:
            lines.extend(build_tree(left, prefix + extension, True))
        elif right:
            lines.extend(build_tree(right, prefix + extension, True))
        
        return lines
    
    visual = []
    weight = tree.nodes.get(tree.root, '')
    root_str = f"{tree.root}({weight})" if weight not in (None, '') else str(tree.root)
    visual.append(root_str)
    
    left, right = tree.tree_structure.get(tree.root, (None, None))
    if left and right:
        visual.extend(build_tree(left, "", False))
        visual.extend(build_tree(right, "", True))
    elif left:
        visual.extend(build_tree(left, "", True))
    elif right:
        visual.extend(build_tree(right, "", True))
    
    # Tabulka
    node_details = []
    for node in sorted(tree.nodes.keys()):
        weight = tree.nodes.get(node, '')
        depth = node_depths.get(node, '?')
        
        # Najdeme rodiče
        parent = ''
        for p, (l, r) in tree.tree_structure.items():
            if l == node or r == node:
                parent = p
                break
        
        # Pro STROM: použijeme tree_structure
        if not hasattr(tree, 'adjacency') or tree.adjacency is None:
            left, right = tree.tree_structure.get(node, (None, None))
            left_str = left if left else '-'
            right_str = right if right else '-'
        # Pro GRAF: zobrazíme sousedy (ne left/right)
        else:
            neighbors = tree.adjacency.get(node, [])
            left_str = neighbors[0] if len(neighbors) > 0 else '-'
            right_str = neighbors[1] if len(neighbors) > 1 else '-'
            # Pokud má více než 2 sousedy, zobraz to
            if len(neighbors) > 2:
                right_str = f"{right_str},..."
        
        if not left_str or (left_str == '-' and right_str == '-'):
            node_type = '🍃 LIST'
        elif not parent:
            node_type = '🌳 KOŘEN'
        else:
            node_type = '🔸 VNITŘNÍ'
        
        weight_str = str(weight) if weight not in (None, '') else '-'
        parent_str = parent if parent else '-'
        
        node_details.append([node, weight_str, parent_str, left_str, right_str, depth, node_type])
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write('╔════════════════════════════════════════════════════════════════╗\n')
        f.write('║             BINÁRNÍ STROM - VIZUALIZACE                      ║\n')
        f.write('╚════════════════════════════════════════════════════════════════╝\n')
        f.write('\n')
        f.write('📊 ZÁKLADNÍ INFORMACE:\n')
        f.write(f'   🌳 Kořen: {tree.root}\n')
        f.write(f'   📦 Počet uzlů: {len(tree.nodes)}\n')
        f.write(f'   📏 Maximální hloubka: {max_depth}\n')
        f.write('\n')
        f.write('═' * 70 + '\n')
        f.write('\n')
        f.write('🌳 STRUKTURA STROMU:\n')
        f.write('\n')
        for line in visual:
            f.write(line + '\n')
        
        f.write('\n')
        f.write('═' * 70 + '\n')
        f.write('\n')
        f.write('📋 DETAIL UZLŮ:\n')
        f.write('\n')
        f.write(f"{'Uzel':<8} {'Váha':<8} {'Rodič':<10} {'Levý':<8} {'Pravý':<9} {'Hloubka':<10} {'Typ'}\n")
        f.write('─' * 75 + '\n')
        
        for row in node_details:
            uzel, vaha, rodic, levy, pravy, hloubka, typ = row
            f.write(f"{uzel:<8} {vaha:<8} {rodic:<10} {levy:<8} {pravy:<9} {str(hloubka):<10} {typ}\n")
        
        f.write('\n')
        f.write('═' * 70 + '\n')

def parse_binary_tree_file(filename):
    """Parser pro binární strom/graf - zvládne 3 formáty"""
    tree = BinaryTree()
    edges = []
    nodes_in_order = []  # Pro level-order formát
    
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            if not line or line.startswith('#'):
                continue
            
            # Načítání uzlů
            if line.startswith('u '):
                parts = line.split()
                node_id = parts[1].rstrip(';')
                
                # Uložíme pořadí pro level-order
                nodes_in_order.append(node_id)
                
                # Zkusíme načíst váhu
                weight = None
                if len(parts) > 2 and node_id != '*':
                    try:
                        weight = float(parts[2].rstrip(';'))
                    except ValueError:
                        pass
                
                tree.add_node(node_id, weight)
            
            # Načítání hran (pro grafy)
            elif line.startswith('h '):
                # Formát: h A - B 5; nebo h A > B 3; nebo h A < B 2;
                parts = line.replace(';', '').split()
                if len(parts) >= 4:
                    node1 = parts[1]
                    direction = parts[2]  # '-', '>', nebo '<'
                    node2 = parts[3]
                    
                    # Kontrola, jestli je to platný směr
                    if direction in ['-', '>', '<']:
                        weight = None
                        # Váha může být na indexu 4, nebo tam může být :h1
                        if len(parts) > 4:
                            try:
                                # Pokusíme se parsovat jako číslo
                                weight_str = parts[4].split(':')[0]  # Odřízneme :h1 pokud existuje
                                weight = float(weight_str)
                            except ValueError:
                                pass
                        edges.append((node1, node2, weight, direction))
    
    # Rozhodnutí, jak stavět strukturu:
    if edges:
        # FORMÁT 3: Graf s hranami
        print(f"  → Načetl jsem {len(edges)} hran (graf)")
        tree._build_from_edges(edges)
    elif any(weight is not None for weight in tree.nodes.values()):
        # FORMÁT 1: BST s vahami
        print("  → Stavím BST podle vah")
        tree.build_structure()
    elif '*' in nodes_in_order:
        # FORMÁT 2: Level-order s hvězdičkami
        print("  → Stavím strom podle level-order pozic")
        tree._build_from_level_order(nodes_in_order)
    else:
        print("  → Uzly bez struktury (přidejte váhy nebo hrany)")
    
    return tree

def level_order(tree):
    """Level-order průchod"""
    if not tree.root:
        return []
    
    result = []
    queue = deque([tree.root])
    
    while queue:
        node = queue.popleft()
        result.append(node)
        
        left, right = tree.tree_structure.get(node, (None, None))
        if left:
            queue.append(left)
        if right:
            queue.append(right)
    
    return result


def pre_order(tree):
    """Pre-order průchod"""
    if not tree.root:
        return []
    
    result = []
    
    def helper(node):
        if node is None:
            return
        result.append(node)
        left, right = tree.tree_structure.get(node, (None, None))
        if left:
            helper(left)
        if right:
            helper(right)
    
    helper(tree.root)
    return result


def in_order(tree):
    """In-order průchod"""
    if not tree.root:
        return []
    
    result = []
    
    def helper(node):
        if node is None:
            return
        left, right = tree.tree_structure.get(node, (None, None))
        if left:
            helper(left)
        result.append(node)
        if right:
            helper(right)
    
    helper(tree.root)
    return result


def post_order(tree):
    """Post-order průchod"""
    if not tree.root:
        return []
    
    result = []
    
    def helper(node):
        if node is None:
            return
        left, right = tree.tree_structure.get(node, (None, None))
        if left:
            helper(left)
        if right:
            helper(right)
        result.append(node)
    
    helper(tree.root)
    return result


def find_leaves(tree):
    """Najde listové uzly"""
    leaves = []
    for node, (left, right) in tree.tree_structure.items():
        if left is None and right is None:
            leaves.append(node)
    return leaves


def print_tree_info(tree):
    """Vypíše informace o stromu"""
    print("\n" + "=" * 60)
    print("     INFORMACE O STROMU")
    print("=" * 60)
    
    if not tree.root:
        print("\n⚠ Strom je prázdný!")
        return
    
    # NOVÉ: Kontrola pro grafy
    if hasattr(tree, 'adjacency') and tree.adjacency:
        print("\n⚠ Toto je graf, ne strom!")
        print(f"\n📊 Základní informace:")
        print(f"  • Počet uzlů: {len(tree.nodes)}")
        print(f"  • Uzly: {', '.join(sorted(tree.nodes.keys()))}")
        
        print(f"\n🔗 Sousedé:")
        for node in sorted(tree.adjacency.keys()):
            neighbors = tree.adjacency[node]
            print(f"  • {node}: {', '.join(neighbors) if neighbors else '-'}")
        
        print("\n💡 Použijte BFS/DFS pro prohledávání grafu!")
        print("=" * 60)
        return
    
    # Původní kód pro stromy...
    print(f"\n📊 Základní informace:")
    print(f"  • Kořen: {tree.root}")
    print(f"  • Počet uzlů: {len(tree.nodes)}")
    print(f"  • Uzly: {', '.join(sorted(tree.nodes.keys()))}")
    
    if tree.nodes:
        print(f"\n📝 Váhy uzlů:")
        for node, weight in sorted(tree.nodes.items()):
            print(f"  • {node}: {weight if weight is not None else 'bez váhy'}")
    
    level_result = level_order(tree)
    pre_result = pre_order(tree)
    in_result = in_order(tree)
    post_result = post_order(tree)
    leaves = find_leaves(tree)
    
    print(f"\n🔄 Průchody stromem:")
    print(f"  • Level-order: {' → '.join(level_result)}")
    print(f"  • Pre-order:   {' → '.join(pre_result)}")
    print(f"  • In-order:    {' → '.join(in_result)}")
    print(f"  • Post-order:  {' → '.join(post_result)}")
    
    print(f"\n🍃 Listové uzly:")
    print(f"  • {', '.join(sorted(leaves)) if leaves else 'žádné'}")
    print(f"  • Celkem: {len(leaves)}")
    
    print("\n" + "=" * 60)
    
def kruskal_minimum_spanning_tree(tree):
    """Kruskalův algoritmus pro MINIMÁLNÍ kostru"""
    print("\n" + "=" * 60)
    print("     MINIMÁLNÍ KOSTRA - KRUSKALŮV ALGORITMUS")
    print("=" * 60)
    
    # Kontrola, jestli je to binární strom (ne graf)
    if not hasattr(tree, 'adjacency') or tree.adjacency is None:
        print("\n⚠️ Toto je binární strom, ne graf!")
        print("  • Binární strom má vždy právě 1 kostru (on sám).")
        print("  • Pro výpočet kostry grafu použijte graf s hranami.")
        return
    
    if not tree.adjacency:
        print("\n❌ Graf je prázdný!")
        return
    
    if not hasattr(tree, 'edges') or not tree.edges:
        print("\n❌ Graf nemá uložené hrany!")
        return
    
    if getattr(tree, 'is_directed', False):
        print("\n⚠️ Varování: Graf je orientovaný. Kostra se obvykle počítá pro neorientované grafy.")
    
    print(f"\n📊 Informace o grafu:")
    print(f"  • Počet uzlů: {len(tree.nodes)}")
    print(f"  • Počet hran: {len(tree.edges)}")
    
    # Seřadíme hrany podle váhy (VZESTUPNĚ pro minimum)
    sorted_edges = sorted(tree.edges, key=lambda x: x[2])
    
    print(f"\n📋 Hrany seřazené podle váhy (vzestupně):")
    for i, (u, v, w) in enumerate(sorted_edges, 1):
        print(f"  {i}. {u} - {v} : {w}")
    
    # Inicializace Union-Find
    uf = UnionFind(list(tree.nodes.keys()))
    
    mst_edges = []
    total_weight = 0
    
    print(f"\n🔄 PRŮBĚH ALGORITMU:\n")
    
    for step, (u, v, weight) in enumerate(sorted_edges, 1):
        print(f"Krok {step}: Zkoumám hranu {u} - {v} (váha {weight})")
        
        # Zkontrolujeme, jestli by vytvořila cyklus
        if uf.union(u, v):
            mst_edges.append((u, v, weight))
            total_weight += weight
            print(f"  ✅ PŘIJATO - Hrana přidána do kostry")
            print(f"     Aktuální váha kostry: {total_weight}")
        else:
            print(f"  ❌ ZAMÍTNUTO - Vytvoří cyklus")
        
        print()
        
        # Pokud máme n-1 hran, máme hotovou kostru
        if len(mst_edges) == len(tree.nodes) - 1:
            print("🎉 Kostra je kompletní!")
            break
    
    print("=" * 60)
    print("📋 VÝSLEDEK - MINIMÁLNÍ KOSTRA:")
    print("=" * 60)
    print(f"\n✅ Hrany v kostře:")
    for i, (u, v, w) in enumerate(mst_edges, 1):
        print(f"  {i}. {u} - {v} : {w}")
    
    print(f"\n📊 Statistiky:")
    print(f"  • Počet hran v kostře: {len(mst_edges)}")
    print(f"  • Celková váha: {total_weight}")
    print(f"  • Očekávaný počet hran: {len(tree.nodes) - 1}")
    
    if len(mst_edges) < len(tree.nodes) - 1:
        print(f"\n⚠️ VAROVÁNÍ: Graf není souvislý!")
        print(f"   Kostra má {len(mst_edges)} hran, ale potřebujeme {len(tree.nodes) - 1}")
    
    print("=" * 60)
    
    return mst_edges, total_weight


def kruskal_maximum_spanning_tree(tree):
    """Kruskalův algoritmus pro MINIMÁLNÍ kostru"""
    print("\n" + "=" * 60)
    print("     MINIMÁLNÍ KOSTRA - KRUSKALŮV ALGORITMUS")
    print("=" * 60)
    
    # Kontrola, jestli je to binární strom (ne graf)
    if not hasattr(tree, 'adjacency') or tree.adjacency is None:
        print("\n⚠️ Toto je binární strom, ne graf!")
        print("  • Binární strom má vždy právě 1 kostru (on sám).")
        print("  • Pro výpočet kostry grafu použijte graf s hranami.")
        return
    
    if not tree.adjacency:
        print("\n❌ Graf je prázdný!")
        return
    
    if not hasattr(tree, 'edges') or not tree.edges:
        print("\n❌ Graf nemá uložené hrany!")
        return
    
    if getattr(tree, 'is_directed', False):
        print("\n⚠️ Varování: Graf je orientovaný. Kostra se obvykle počítá pro neorientované grafy.")
    
    print(f"\n📊 Informace o grafu:")
    print(f"  • Počet uzlů: {len(tree.nodes)}")
    print(f"  • Počet hran: {len(tree.edges)}")
    
    # Seřadíme hrany podle váhy (SESTUPNĚ pro maximum)
    sorted_edges = sorted(tree.edges, key=lambda x: x[2], reverse=True)
    
    print(f"\n📋 Hrany seřazené podle váhy (sestupně):")
    for i, (u, v, w) in enumerate(sorted_edges, 1):
        print(f"  {i}. {u} - {v} : {w}")
    
    # Inicializace Union-Find
    uf = UnionFind(list(tree.nodes.keys()))
    
    mst_edges = []
    total_weight = 0
    
    print(f"\n🔄 PRŮBĚH ALGORITMU:\n")
    
    for step, (u, v, weight) in enumerate(sorted_edges, 1):
        print(f"Krok {step}: Zkoumám hranu {u} - {v} (váha {weight})")
        
        # Zkontrolujeme, jestli by vytvořila cyklus
        if uf.union(u, v):
            mst_edges.append((u, v, weight))
            total_weight += weight
            print(f"  ✅ PŘIJATO - Hrana přidána do kostry")
            print(f"     Aktuální váha kostry: {total_weight}")
        else:
            print(f"  ❌ ZAMÍTNUTO - Vytvoří cyklus")
        
        print()
        
        # Pokud máme n-1 hran, máme hotovou kostru
        if len(mst_edges) == len(tree.nodes) - 1:
            print("🎉 Kostra je kompletní!")
            break
    
    print("=" * 60)
    print("📋 VÝSLEDEK - MAXIMÁLNÍ KOSTRA:")
    print("=" * 60)
    print(f"\n✅ Hrany v kostře:")
    for i, (u, v, w) in enumerate(mst_edges, 1):
        print(f"  {i}. {u} - {v} : {w}")
    
    print(f"\n📊 Statistiky:")
    print(f"  • Počet hran v kostře: {len(mst_edges)}")
    print(f"  • Celková váha: {total_weight}")
    print(f"  • Očekávaný počet hran: {len(tree.nodes) - 1}")
    
    if len(mst_edges) < len(tree.nodes) - 1:
        print(f"\n⚠️ VAROVÁNÍ: Graf není souvislý!")
        print(f"   Kostra má {len(mst_edges)} hran, ale potřebujeme {len(tree.nodes) - 1}")
    
    print("=" * 60)
    
    return mst_edges, total_weight

def count_spanning_trees(tree):
    """Spočítá počet koster grafu pomocí Kirchhoffovy věty (Matrix-Tree Theorem)"""
    print("\n" + "=" * 60)
    print("     POČET KOSTER - KIRCHHOFFOVA VĚTA")
    print("=" * 60)
    
    # Kontrola pro binární strom
    if not hasattr(tree, 'adjacency') or tree.adjacency is None:
        print("\n⚠️ Toto je binární strom, ne graf!")
        print("\n📋 VÝSLEDEK:")
        print("=" * 60)
        print(f"  🌳 Počet různých koster: 1")
        print(f"  (Binární strom má vždy právě 1 kostru - on sám)")
        print("=" * 60)
        return 1
    
    if not tree.adjacency:
        print("\n❌ Graf je prázdný!")
        return 0
    
    if getattr(tree, 'is_directed', False):
        print("\n⚠️ Varování: Graf je orientovaný. Kirchhoffova věta platí pro neorientované grafy.")
    
    print(f"\n📊 Informace o grafu:")
    print(f"  • Počet uzlů: {len(tree.nodes)}")
    print(f"  • Počet hran: {len(tree.edges) if hasattr(tree, 'edges') else 'N/A'}")
    
    # Vytvoříme Laplacianovu matici
    nodes = sorted(tree.nodes.keys())
    n = len(nodes)
    node_index = {node: i for i, node in enumerate(nodes)}
    
    # Degree matice (stupně uzlů na diagonále)
    degree_matrix = np.zeros((n, n))
    for node in nodes:
        degree = len(tree.adjacency.get(node, []))
        degree_matrix[node_index[node]][node_index[node]] = degree
    
    # Adjacency matice (1 pokud existuje hrana)
    adjacency_matrix = np.zeros((n, n))
    for node, neighbors in tree.adjacency.items():
        for neighbor in neighbors:
            if neighbor in node_index:  # Kontrola, že soused existuje
                adjacency_matrix[node_index[node]][node_index[neighbor]] = 1
    
    # Laplacianova matice = Degree - Adjacency
    laplacian = degree_matrix - adjacency_matrix
    
    print("\n📐 Laplacianova matice:")
    print("     ", "  ".join(f"{node:>4}" for node in nodes))
    for i, node in enumerate(nodes):
        print(f"  {node:>2}", "  ".join(f"{int(laplacian[i][j]):>4}" for j in range(n)))
    
    # Cofactor matice (odstraníme poslední řádek a sloupec)
    cofactor = laplacian[:-1, :-1]
    
    print(f"\n📐 Cofaktor matice (bez posledního řádku a sloupce):")
    print("     ", "  ".join(f"{node:>4}" for node in nodes[:-1]))
    for i, node in enumerate(nodes[:-1]):
        print(f"  {node:>2}", "  ".join(f"{int(cofactor[i][j]):>4}" for j in range(n-1)))
    
    # Determinant cofactor matice = počet koster
    try:
        det = np.linalg.det(cofactor)
        num_spanning_trees = int(round(det))
        
        print("\n" + "=" * 60)
        print("📋 VÝSLEDEK:")
        print("=" * 60)
        print(f"\n🌳 Počet různých koster: {num_spanning_trees}")
        print("=" * 60)
        
        return num_spanning_trees
    except:
        print("\n❌ Chyba při výpočtu determinantu!")
        return None

def bfs_search(tree, start_node):
    """Prohledávání do šířky (BFS) - funguje na stromech i grafech"""
    print("\n" + "=" * 60)
    print(f"     PROHLEDÁVÁNÍ DO ŠÍŘKY (BFS) od uzlu '{start_node}'")
    print("=" * 60)
    
    if start_node not in tree.nodes:
        print(f"\n❌ Uzel '{start_node}' neexistuje!")
        print(f"   Dostupné uzly: {', '.join(sorted(tree.nodes.keys()))}")
        return
    
    # Vytvoříme graf - inicializace pro všechny uzly
    graph = {node: [] for node in tree.nodes.keys()}
    
    # Kontrola, jestli máme adjacency list (graf) nebo tree_structure (strom)
    if hasattr(tree, 'adjacency') and tree.adjacency:
        is_directed = getattr(tree, 'is_directed', False)
        if is_directed:
            print("\n  → Používám načtené hrany (orientovaný graf)")
        else:
            print("\n  → Používám načtené hrany (neorientovaný graf)")
        # Zkopírujeme adjacency list
        for node in tree.nodes.keys():
            if node in tree.adjacency:
                graph[node] = list(tree.adjacency[node])
    else:
        print("\n  → Používám stromovou strukturu")
        # Pro každý uzel vytvoříme obousměrné hrany
        for parent, (left, right) in tree.tree_structure.items():
            if left and left in tree.nodes:
                if left not in graph[parent]:
                    graph[parent].append(left)
                if parent not in graph[left]:
                    graph[left].append(parent)
            if right and right in tree.nodes:
                if right not in graph[parent]:
                    graph[parent].append(right)
                if parent not in graph[right]:
                    graph[right].append(parent)
    
    # DEBUG: Zobrazíme graf
    print("\n🔍 DEBUG - Graf sousedů:")
    for node, neighbors in sorted(graph.items()):
        print(f"  {node}: {neighbors}")
    
    # BFS
    queue = deque([start_node])
    visited = set([start_node])
    in_queue = set([start_node])  # NOVÉ: Sledujeme, co je ve frontě
    removal_order = []
    all_queue_states = [start_node]
    
    print("\n📊 VIZUALIZACE FRONTY:\n")
    step = 0
    
    while queue:
        step += 1
        
        queue_display = ' '.join(queue)
        print(f"Krok {step}:")
        print(f"  Fronta: [ {queue_display} ]")
        
        current = queue.popleft()
        in_queue.remove(current)  # NOVÉ: Odstraníme ze sledování fronty
        print(f"  🔍 Zpracovávám: {current}")
        
        neighbors = []
        for neighbor in graph.get(current, []):
            # OPRAVENO: Kontrolujeme visited i in_queue
            if neighbor not in visited and neighbor not in in_queue and neighbor != current:
                neighbors.append(neighbor)
                visited.add(neighbor)
                queue.append(neighbor)
                in_queue.add(neighbor)  # NOVÉ: Přidáme do sledování fronty
                all_queue_states.append(neighbor)
        
        if neighbors:
            print(f"  ➕ Přidávám do fronty: {', '.join(neighbors)}")
        else:
            print(f"  • Žádní noví sousedé")
        
        removal_order.append(current)
        print(f"  ➜ Odstraňuji z fronty: {current}")
        print()
    
    print("=" * 60)
    print("📋 VÝSLEDEK BFS:")
    print(f"  • Pořadí odstraňování z fronty: {' → '.join(removal_order)}")
    print(f"  • Navštíveno uzlů: {len(removal_order)}/{len(tree.nodes)}")
    
    # Vizualizace celé fronty s úrovněmi
    print("\n📊 CELÁ FRONTA (pořadí přidávání do fronty):")
    
    node_levels = {start_node: 0}
    temp_queue = deque([start_node])
    temp_visited = set([start_node])
    
    while temp_queue:
        node = temp_queue.popleft()
        
        for neighbor in graph.get(node, []):
            if neighbor not in temp_visited and neighbor != node:
                temp_visited.add(neighbor)
                node_levels[neighbor] = node_levels[node] + 1
                temp_queue.append(neighbor)
    
    levels_dict = {}
    for node in all_queue_states:
        level = node_levels.get(node, 0)
        if level not in levels_dict:
            levels_dict[level] = []
        levels_dict[level].append(node)
    
    level_parts = []
    for level in sorted(levels_dict.keys()):
        level_parts.append(' '.join(levels_dict[level]))
    
    print(f"  [ {' | '.join(level_parts)} ]")
    print(f"\n  Legenda: | = oddělení úrovní")
    print("=" * 60)


def dfs_search(tree, start_node):
    """Prohledávání do hloubky (DFS) - funguje na stromech i grafech"""
    print("\n" + "=" * 60)
    print(f"     PROHLEDÁVÁNÍ DO HLOUBKY (DFS) od uzlu '{start_node}'")
    print("=" * 60)
    
    if start_node not in tree.nodes:
        print(f"\n❌ Uzel '{start_node}' neexistuje!")
        print(f"   Dostupné uzly: {', '.join(sorted(tree.nodes.keys()))}")
        return
    
    # Vytvoříme graf - inicializace pro všechny uzly
    graph = {node: [] for node in tree.nodes.keys()}
    
    # Kontrola, jestli máme adjacency list (graf) nebo tree_structure (strom)
    if hasattr(tree, 'adjacency') and tree.adjacency:
        is_directed = getattr(tree, 'is_directed', False)
        if is_directed:
            print("\n  → Používám načtené hrany (orientovaný graf)")
        else:
            print("\n  → Používám načtené hrany (neorientovaný graf)")
        # Zkopírujeme adjacency list
        for node in tree.nodes.keys():
            if node in tree.adjacency:
                graph[node] = list(tree.adjacency[node])
    else:
        print("\n  → Používám stromovou strukturu")
        # Pro každý uzel vytvoříme obousměrné hrany
        for parent, (left, right) in tree.tree_structure.items():
            if left and left in tree.nodes:
                if left not in graph[parent]:
                    graph[parent].append(left)
                if parent not in graph[left]:
                    graph[left].append(parent)
            if right and right in tree.nodes:
                if right not in graph[parent]:
                    graph[parent].append(right)
                if parent not in graph[right]:
                    graph[right].append(parent)
    
    # DEBUG: Zobrazíme graf
    print("\n🔍 DEBUG - Graf sousedů:")
    for node, neighbors in sorted(graph.items()):
        print(f"  {node}: {neighbors}")
    
    # DFS
    stack = [start_node]
    visited = set()
    in_stack = set([start_node])  # NOVÉ: Sledujeme, co je v zásobníku
    removal_order = []
    all_stack_states = [start_node]
    
    print("\n📊 VIZUALIZACE ZÁSOBNÍKU:\n")
    step = 0
    
    while stack:
        step += 1
        
        stack_display = ' '.join(stack)
        print(f"Krok {step}:")
        print(f"  Zásobník: [ {stack_display} ] ← vrchol")
        
        current = stack.pop()
        in_stack.remove(current)  # NOVÉ: Odstraníme ze sledování zásobníku
        
        if current in visited:
            print(f"  ⏭️  {current} již navštíven, přeskakuji")
            print()
            continue
        
        print(f"  🔍 Zpracovávám: {current}")
        visited.add(current)
        removal_order.append(current)
        
        neighbors = []
        for neighbor in reversed(graph.get(current, [])):
            # OPRAVENO: Kontrolujeme visited i in_stack
            if neighbor not in visited and neighbor not in in_stack and neighbor != current:
                neighbors.append(neighbor)
                stack.append(neighbor)
                in_stack.add(neighbor)  # NOVÉ: Přidáme do sledování zásobníku
                if neighbor not in all_stack_states:
                    all_stack_states.append(neighbor)
        
        if neighbors:
            print(f"  ➕ Přidávám na zásobník: {', '.join(reversed(neighbors))}")
        else:
            print(f"  • Žádní noví sousedé")
        
        print(f"  ➜ Odebrán ze zásobníku: {current}")
        print()
    
    print("=" * 60)
    print("📋 VÝSLEDEK DFS:")
    print(f"  • Pořadí zpracování: {' → '.join(removal_order)}")
    print(f"  • Navštíveno uzlů: {len(removal_order)}/{len(tree.nodes)}")
    
    print("\n📊 CELÝ ZÁSOBNÍK (pořadí přidávání):")
    print(f"  [ {' '.join(all_stack_states)} ] ← vrchol (přidáváno zleva doprava)")
    print(f"\n  První uzel ({start_node}) = počáteční, ostatní = pořadí přidávání")
    print("=" * 60)
    
def show_menu():
    """Zobrazí menu"""
    print("\n" + "=" * 60)
    print("     MENU - BINÁRNÍ STROM / GRAF")
    print("=" * 60)
    print("\n📋 Dostupné operace:")
    print("  1. Zobrazit informace o stromu/grafu")
    print("  2. Vložit nový uzel (U <název> <váha>)")
    print("  3. Smazat uzel (D <název>)")
    print("  4. Exportovat strom do souboru")
    print("  5. Načíst nový strom/graf ze souboru")
    print("  6. Prohledávání do šířky - BFS (od uzlu)")
    print("  7. Prohledávání do hloubky - DFS (od uzlu)")
    print("  8. Spočítat počet koster grafu (Kirchhoffova věta)")
    print("  9. Minimální kostra (Kruskalův algoritmus)")
    print(" 10. Maximální kostra (Kruskalův algoritmus)")
    print("  0. Ukončit program")
    print("\n" + "=" * 60)

def interactive_mode(tree):
    """Interaktivní režim"""
    
    while True:
        show_menu()
        
        choice = input("\n👉 Zadejte volbu (nebo příkaz): ").strip()
        
        if not choice:
            continue
        
        if choice == '0' or choice.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Ukončuji program...")
            break
        
        elif choice == '1' or choice.lower() in ['info', 'i']:
            print_tree_info(tree)
        
        elif choice == '2' or choice.upper().startswith('U '):
            if choice == '2':
                cmd = input("\n👉 Zadejte příkaz (U <název> <váha>): ").strip()
            else:
                cmd = choice
            
            parts = cmd.split()
            if len(parts) != 3 or parts[0].upper() != 'U':
                print("\n❌ Chybný formát! Použijte: U <název> <váha>")
                print("   Příklad: U E 25")
                continue
            
            node_name = parts[1]
            try:
                weight = float(parts[2])
            except ValueError:
                print("\n❌ Váha musí být číslo!")
                continue
            
            if node_name in tree.nodes:
                print(f"\n⚠️ Uzel '{node_name}' již existuje!")
                continue
            
            tree.insert_node(node_name, weight)
            print(f"\n✅ Uzel '{node_name}' s váhou {weight} byl vložen!")
        
        elif choice == '3' or choice.upper().startswith('D '):
            if choice == '3':
                cmd = input("\n👉 Zadejte příkaz (D <název>): ").strip()
            else:
                cmd = choice
            
            parts = cmd.split()
            if len(parts) != 2 or parts[0].upper() != 'D':
                print("\n❌ Chybný formát! Použijte: D <název>")
                print("   Příklad: D B")
                continue
            
            node_name = parts[1]
            
            if node_name not in tree.nodes:
                print(f"\n⚠️ Uzel '{node_name}' neexistuje!")
                print(f"   Dostupné: {', '.join(sorted(tree.nodes.keys()))}")
                continue
            
            confirm = input(f"\n⚠️ Smazat uzel '{node_name}'? (ano/ne): ").strip().lower()
            if confirm in ['ano', 'a', 'y', 'yes']:
                tree.delete_node(node_name)
                print(f"\n✅ Uzel '{node_name}' byl smazán!")
            else:
                print("\n❌ Zrušeno.")
        
        elif choice == '4' or choice.lower() in ['export', 'e']:
            filename = input("\n👉 Název souboru (Enter = tree.txt): ").strip()
            if not filename:
                filename = 'tree.txt'
            
            if not filename.endswith('.txt'):
                filename += '.txt'
            
            export_tree_to_csv(tree, filename)
            print(f"\n✅ Exportováno do '{filename}'")
        
        elif choice == '5' or choice.lower() in ['load', 'l']:
            filename = input("\n👉 Název souboru: ").strip()
            
            if not filename:
                print("\n❌ Zadejte název souboru!")
                continue
            
            try:
                new_tree = parse_binary_tree_file(filename)
                tree.nodes = new_tree.nodes
                tree.tree_structure = new_tree.tree_structure
                tree.root = new_tree.root
                print(f"\n✅ Načteno z '{filename}'")
                print_tree_info(tree)
            except FileNotFoundError:
                print(f"\n❌ Soubor '{filename}' nenalezen!")
            except Exception as e:
                print("menu chyba")
                print(f"\n❌ Chyba: {e}")
        
        elif choice == '6' or choice.upper().startswith('BFS '):
            if choice == '6':
                start = input("\n👉 Zadejte počáteční uzel pro BFS: ").strip()
            else:
                parts = choice.split()
                if len(parts) >= 2:
                    start = parts[1]
                else:
                    print("\n❌ Zadejte uzel! Formát: BFS <uzel>")
                    continue
            
            if not start:
                print("\n❌ Musíte zadat počáteční uzel!")
                continue
            
            bfs_search(tree, start)
        
        elif choice == '7' or choice.upper().startswith('DFS '):
            if choice == '7':
                start = input("\n👉 Zadejte počáteční uzel pro DFS: ").strip()
            else:
                parts = choice.split()
                if len(parts) >= 2:
                    start = parts[1]
                else:
                    print("\n❌ Zadejte uzel! Formát: DFS <uzel>")
                    continue
            
            if not start:
                print("\n❌ Musíte zadat počáteční uzel!")
                continue
            
            dfs_search(tree, start)
        elif choice == '8' or choice.lower() in ['kostry', 'spanning']:
            count_spanning_trees(tree)
            
        elif choice == '9' or choice.lower() in ['min', 'minimum']:
            kruskal_minimum_spanning_tree(tree)

        elif choice == '10' or choice.lower() in ['max', 'maximum']:
            kruskal_maximum_spanning_tree(tree)
                
        else:
            print("\n❌ Neznámý příkaz!")
            print("   U E 25  - vloží uzel E s váhou 25")
            print("   D B     - smaže uzel B")
            print("   BFS A   - BFS od uzlu A")
            print("   DFS A   - DFS od uzlu A")
    
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("     BINÁRNÍ VYHLEDÁVACÍ STROM")
    print("=" * 60)
    
    if len(sys.argv) >= 2:
        filename = sys.argv[1]
        print(f"\n📂 Načítám: {filename}")
        
        try:
            tree = parse_binary_tree_file(filename)
            print(f"✓ Načteno!")
            
            # OPRAVA: Export jen pro stromy, ne grafy
            if not hasattr(tree, 'adjacency') or not tree.adjacency:
                export_tree_to_csv(tree, 'tree-original.txt')
            else:
                print("\n  → Graf (nelze exportovat jako strom)")
            
            if len(sys.argv) == 3:
                target_node = sys.argv[2]
                print_tree_info(tree)
                
                if target_node in tree.nodes:
                    print(f"\n🔍 Detail uzlu '{target_node}':")
                    weight = tree.nodes.get(target_node)
                    
                    # Pro graf ukážeme sousedy
                    if hasattr(tree, 'adjacency') and tree.adjacency:
                        neighbors = tree.adjacency.get(target_node, [])
                        print(f"  • Sousedé: {', '.join(neighbors) if neighbors else '-'}")
                    else:
                        left, right = tree.tree_structure.get(target_node, (None, None))
                        print(f"  • Levý: {left if left else '-'}")
                        print(f"  • Pravý: {right if right else '-'}")
                    
                    print(f"  • Váha: {weight if weight is not None else 'bez váhy'}")
                else:
                    print(f"\n⚠️ Uzel '{target_node}' nenalezen!")
            
            interactive_mode(tree)
            
        except FileNotFoundError:
            print(f"❌ Soubor nenalezen!")
            sys.exit(1)
        except Exception as e:
            import traceback
            print("\n" + "=" * 60)
            print("🔴 CHYBA PŘI ZPRACOVÁNÍ:")
            print("=" * 60)
            print(f"Typ chyby: {type(e).__name__}")
            print(f"Zpráva: {e}")
            print("\n📍 Traceback:")
            traceback.print_exc()
            print("=" * 60)
            sys.exit(1)
    else:
        print("\n⚠️ Nebyl zadán soubor.")
        print("Použití: python3 main.py <soubor.txt> [uzel]")
        print("\nPrázdný strom...")
        
        tree = BinaryTree()
        interactive_mode(tree)
    
    print("\n✨ Program ukončen.\n")