import argparse
import csv
import sys
import os
from collections import deque, defaultdict

# Add current directory to sys.path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from file_reader import read_graph_from_file
    from graph import Graph
except ImportError:
    print("Error: Could not import 'file_reader' or 'graph'. Make sure you are running this script from the correct directory.")
    sys.exit(1)

try:
    from rich.console import Console
    from rich.table import Table
    from rich import print as rprint
    HAS_RICH = True
    console = Console()
except ImportError:
    HAS_RICH = False
    print("Warning: 'rich' library not found. Output will be plain text.")

def bfs(residual_graph, source, sink, parent):
    visited = set()
    queue = deque([source])
    visited.add(source)
    parent[source] = None

    while queue:
        u = queue.popleft()
        
        for v in residual_graph[u]:
            capacity = residual_graph[u][v]
            if v not in visited and capacity > 0:
                queue.append(v)
                visited.add(v)
                parent[v] = u
                if v == sink:
                    return True
    return False

def edmonds_karp(graph, source, sink, export_csv=True):
    # 1. Initialize Residual Graph
    # residual_graph[u][v] = remaining capacity
    residual_graph = defaultdict(lambda: defaultdict(float))
    original_capacities = defaultdict(lambda: defaultdict(float))
    
    # Load edges into residual graph
    # If graph is undirected, we assume edges are bidirectional with full capacity
    # If directed, we respect direction.
    # file_reader usually handles direction by creating appropriate edges in Graph object.
    
    # We need to handle potential multi-edges or just sum capacities if multiple edges exist between same nodes
    for edge in graph.edges:
        u, v = edge.node1, edge.node2
        w = edge.weight if edge.weight is not None else 1.0
        
        original_capacities[u][v] += w
        residual_graph[u][v] += w
        
        # Ensure reverse edge exists in residual graph with 0 capacity if it doesn't exist in original
        if v not in residual_graph or u not in residual_graph[v]:
             residual_graph[v][u] += 0.0

    if HAS_RICH:
        rprint(f"[bold green]Starting Edmonds-Karp Max Flow[/bold green]")
        rprint(f"Source: [cyan]{source}[/cyan], Sink: [cyan]{sink}[/cyan]")
        rprint(f"Nodes: {len(graph.nodes)}, Edges: {len(graph.edges)}")
        rprint("-" * 50)
    else:
        print(f"Starting Edmonds-Karp Max Flow")
        print(f"Source: {source}, Sink: {sink}")
        print("-" * 50)

    parent = {}
    max_flow = 0
    path_count = 0

    while bfs(residual_graph, source, sink, parent):
        path_count += 1
        
        # Find bottleneck capacity
        path_flow = float('Inf')
        s = sink
        path_nodes = [sink]
        
        while s != source:
            path_flow = min(path_flow, residual_graph[parent[s]][s])
            s = parent[s]
            path_nodes.append(s)
        
        path_nodes.reverse()
        path_str = " -> ".join(path_nodes)
        
        # Update residual capacities
        v = sink
        while v != source:
            u = parent[v]
            residual_graph[u][v] -= path_flow
            residual_graph[v][u] += path_flow
            v = parent[v]

        max_flow += path_flow
        
        if HAS_RICH:
            rprint(f"Step {path_count}: Found path [bold yellow]{path_str}[/bold yellow]")
            rprint(f"        Added flow: [bold magenta]{path_flow}[/bold magenta] | Total flow: [bold green]{max_flow}[/bold green]")
        else:
            print(f"Step {path_count}: Found path {path_str}")
            print(f"        Added flow: {path_flow} | Total flow: {max_flow}")

    # Calculate final flow on edges
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

    # Output Statistics
    if HAS_RICH:
        rprint("-" * 50)
        rprint(f"[bold]Computation Finished![/bold]")
        rprint(f"Total Max Flow: [bold green]{max_flow}[/bold green]")
        rprint(f"Total Augmenting Paths: {path_count}")
        
        table = Table(title="Flow Distribution")
        table.add_column("Source", style="cyan")
        table.add_column("Target", style="cyan")
        table.add_column("Flow", justify="right", style="magenta")
        table.add_column("Capacity", justify="right")
        table.add_column("Utilization", justify="right")
        
        for item in final_flows:
            table.add_row(
                str(item["source"]),
                str(item["target"]),
                f"{item['flow']:.2f}",
                f"{item['capacity']:.2f}",
                f"{item['utilization']:.1f}%"
            )
        console.print(table)
    else:
        print("-" * 50)
        print(f"Total Max Flow: {max_flow}")
        print(f"Total Augmenting Paths: {path_count}")
        print("\nFlow Distribution:")
        print(f"{'Source':<10} {'Target':<10} {'Flow':<10} {'Capacity':<10} {'Utilization':<10}")
        for item in final_flows:
            print(f"{item['source']:<10} {item['target']:<10} {item['flow']:<10.2f} {item['capacity']:<10.2f} {item['utilization']:.1f}%")

    # Export to CSV
    if export_csv:
        filename = "edmonds_karp.csv"
        try:
            with open(filename, 'w', newline='') as csvfile:
                fieldnames = ['source', 'target', 'flow', 'capacity', 'utilization']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for item in final_flows:
                    writer.writerow(item)
            
            if HAS_RICH:
                rprint(f"\n[bold blue]Result exported to {filename}[/bold blue]")
            else:
                print(f"\nResult exported to {filename}")
        except IOError as e:
            print(f"Error writing to CSV: {e}")

def main():
    parser = argparse.ArgumentParser(description="Edmonds-Karp Max Flow Algorithm")
    parser.add_argument("file", help="Path to the graph file (.tg)")
    parser.add_argument("--source", required=True, help="Source node")
    parser.add_argument("--target", required=True, help="Target (sink) node")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.file):
        print(f"Error: File '{args.file}' not found.")
        sys.exit(1)
        
    # Load graph
    try:
        graph = read_graph_from_file(args.file)
    except Exception as e:
        print(f"Error loading graph: {e}")
        sys.exit(1)
        
    # Validate nodes
    if args.source not in graph.nodes:
        print(f"Error: Source node '{args.source}' not found in graph.")
        sys.exit(1)
    if args.target not in graph.nodes:
        print(f"Error: Target node '{args.target}' not found in graph.")
        sys.exit(1)
        
    edmonds_karp(graph, args.source, args.target)

if __name__ == "__main__":
    main()
