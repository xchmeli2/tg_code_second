from graph import Graph
from edge import Edge
from properties import is_tree

def read_graph_from_file(file_path):
    graph = Graph()
    edge_set = set() 
    is_simple = True

    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(';')
            for part in parts:
                if part.startswith('u'):
                    node_id = part[2:].strip().split()[0]
                    graph.add_node(node_id)
                elif part.startswith('h'):
                    edge_parts = part[2:].strip().split()
                    if len(edge_parts) < 3:
                        print("Warning: Insufficient parts for edge: {}".format(part))
                        continue

                    node1 = edge_parts[0]
                    direction = edge_parts[1]
                    node2 = edge_parts[2]
                    weight = None
                    name = None

                    if len(edge_parts) > 3:
                        if edge_parts[3].startswith(':'):
                            name = edge_parts[3][1:]
                        else:
                            # Try to parse as float (supports both integers and decimals)
                            try:
                                # Support both . and , as decimal separator
                                weight_str = edge_parts[3].replace(',', '.')
                                weight = float(weight_str)
                                if len(edge_parts) > 4 and edge_parts[4].startswith(':'):
                                    name = edge_parts[4][1:]
                            except ValueError:
                                print("Warning: Could not parse weight: {}".format(edge_parts[3]))

                    if direction == '>':
                        edge_tuple = (node1, node2)
                    elif direction == '<':
                        edge_tuple = (node2, node1)
                    elif direction == '-':
                        edge_tuple = tuple(sorted([node1, node2]))

                    if edge_tuple in edge_set:
                        print("Duplicate edge detected between", node1, "and", node2)
                        is_simple = False 
                    else:
                        edge_set.add(edge_tuple)

                    if direction == '>':
                        graph.add_edge(node1, node2, weight, name)
                    elif direction == '<':
                        graph.add_edge(node2, node1, weight, name)
                    elif direction == '-':
                        graph.add_edge(node1, node2, weight, name)
                        graph.add_edge(node2, node1, weight, name)

    graph.is_simple = is_simple

    # Automatic Binary Tree Edge Generation
    # If the file contained '*', we assume it's a level-order binary tree definition.
    # We reconstruct edges based on indices: Parent i -> Children 2i+1, 2i+2
    
    # Check if we should trigger this logic. 
    # Condition: We encountered '*' (which we didn't add to graph.nodes) OR we want to support it for any file that looks like it.
    # The user prompt implies '*' marks it as such.
    # We need to re-read the file or track nodes during the first pass. 
    # Let's refactor slightly to track ordered_nodes during the first pass.
    
    return graph

def read_graph_from_file(file_path):
    graph = Graph()
    edge_set = set() 
    is_simple = True
    
    ordered_nodes = [] # To track chronological order for binary tree reconstruction
    has_asterisk = False

    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(';')
            for part in parts:
                part = part.strip()
                if not part: continue
                
                if part.startswith('u'):
                    node_def = part[2:].strip().split()
                    node_id = node_def[0]
                    
                    if node_id == '*':
                        has_asterisk = True
                        ordered_nodes.append(None) # Placeholder
                    else:
                        graph.add_node(node_id)
                        ordered_nodes.append(node_id)
                        
                elif part.startswith('h'):
                    edge_parts = part[2:].strip().split()
                    if len(edge_parts) < 3:
                        print("Warning: Insufficient parts for edge: {}".format(part))
                        continue

                    node1 = edge_parts[0]
                    direction = edge_parts[1]
                    node2 = edge_parts[2]
                    weight = None
                    name = None

                    if len(edge_parts) > 3:
                        if edge_parts[3].startswith(':'):
                            name = edge_parts[3][1:]
                        else:
                            # Try to parse as float (supports both integers and decimals)
                            try:
                                # Support both . and , as decimal separator
                                weight_str = edge_parts[3].replace(',', '.')
                                weight = float(weight_str)
                                if len(edge_parts) > 4 and edge_parts[4].startswith(':'):
                                    name = edge_parts[4][1:]
                            except ValueError:
                                print("Warning: Could not parse weight: {}".format(edge_parts[3]))

                    if direction == '>':
                        edge_tuple = (node1, node2)
                    elif direction == '<':
                        edge_tuple = (node2, node1)
                    elif direction == '-':
                        edge_tuple = tuple(sorted([node1, node2]))

                    if edge_tuple in edge_set:
                        # print("Duplicate edge detected between", node1, "and", node2)
                        # Don't mark as not simple just for duplicates in this context if we want to be lenient,
                        # but standard logic says multigraph.
                        is_simple = False 
                    else:
                        edge_set.add(edge_tuple)

                    if direction == '>':
                        graph.add_edge(node1, node2, weight, name)
                    elif direction == '<':
                        graph.add_edge(node2, node1, weight, name)
                    elif direction == '-':
                        graph.add_edge(node1, node2, weight, name)
                        graph.add_edge(node2, node1, weight, name)

    # If '*' was detected, generate implicit edges
    if has_asterisk:
        print("Detekován formát binárního stromu (s hvězdičkami). Generuji hrany...")
        n = len(ordered_nodes)
        for i in range(n):
            parent = ordered_nodes[i]
            if parent is None:
                continue
            
            # Left child index: 2*i + 1
            left_index = 2 * i + 1
            if left_index < n:
                left_child = ordered_nodes[left_index]
                if left_child is not None:
                    # Add edge Parent -> Left Child
                    # Check if edge already exists to avoid duplication if mixed format
                    if (parent, left_child) not in edge_set:
                        graph.add_edge(parent, left_child, weight=1) # Default weight 1
                        edge_set.add((parent, left_child))

            # Right child index: 2*i + 2
            right_index = 2 * i + 2
            if right_index < n:
                right_child = ordered_nodes[right_index]
                if right_child is not None:
                     # Add edge Parent -> Right Child
                    if (parent, right_child) not in edge_set:
                        graph.add_edge(parent, right_child, weight=1)
                        edge_set.add((parent, right_child))

    graph.is_simple = is_simple
    return graph

def save_graph_to_file(graph, file_path):
    # Check if it's a tree to potentially use the binary tree format
    if is_tree(graph):
        # Attempt to reconstruct binary tree structure
        # 1. Find root (node with in-degree 0)
        nodes = list(graph.nodes)
        if not nodes:
            with open(file_path, 'w') as f:
                pass
            print("Prázdný graf uložen.")
            return

        # Calculate in-degrees
        in_degrees = {node: 0 for node in nodes}
        for edge in graph.edges:
            in_degrees[edge.node2] = in_degrees.get(edge.node2, 0) + 1
        
        roots = [node for node, deg in in_degrees.items() if deg == 0]
        
        if len(roots) == 1:
            root = roots[0]
            # It's a rooted tree. Let's try to assign indices for binary tree.
            # Index map: node -> index
            node_indices = {root: 0}
            queue = [(root, 0)]
            max_index = 0
            
            valid_bst_structure = True
            
            import collections
            # Use a queue for BFS traversal to assign indices
            bfs_queue = collections.deque([(root, 0)])
            
            while bfs_queue:
                u, idx = bfs_queue.popleft()
                max_index = max(max_index, idx)
                
                # Find children by iterating edges in order to preserve structure
                # The order in graph.edges defines Left (1st) vs Right (2nd)
                children = []
                for edge in graph.edges:
                    if edge.node1 == u:
                        children.append(edge.node2)
                
                # children.sort() # REMOVED: User specified order is structural, not alphabetical
                
                if len(children) > 2:
                    valid_bst_structure = False
                    break
                
                if len(children) == 1:
                    child = children[0]
                    # Simplified logic: Single child is always Left
                    child_idx = 2 * idx + 1
                    
                    node_indices[child] = child_idx
                    bfs_queue.append((child, child_idx))
                    
                elif len(children) == 2:
                    # Simplified logic: First child is Left, Second is Right
                    left_child = children[0]
                    right_child = children[1]
                    
                    l_idx = 2 * idx + 1
                    r_idx = 2 * idx + 2
                    
                    node_indices[left_child] = l_idx
                    node_indices[right_child] = r_idx
                    
                    bfs_queue.append((left_child, l_idx))
                    bfs_queue.append((right_child, r_idx))

            if valid_bst_structure:
                print("Rozpoznána struktura binárního stromu. Ukládám ve formátu s hvězdičkami.")
                with open(file_path, 'w') as f:
                    for i in range(max_index + 1):
                        # Find node with this index
                        node = next((n for n, idx in node_indices.items() if idx == i), None)
                        if node is not None:
                            f.write("u {};\n".format(node))
                        else:
                            f.write("u *;\n")
                print("Graf byl uložen do souboru: {}".format(full_path))
                return

    # Fallback to standard format
    with open(full_path, 'w') as f:
        # Write nodes
        for node in sorted(graph.nodes):
            f.write("u {};\n".format(node))
        
        # Write edges
        for edge in graph.edges:
            line = "h {} > {} ".format(edge.node1, edge.node2)
            if edge.weight is not None:
                line += "{} ".format(edge.weight)
            if edge.name is not None:
                line += ":{}".format(edge.name)
            f.write(line.strip() + ";\n")
    

