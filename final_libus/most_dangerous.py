#!/usr/bin/env python3
import math
from properties import parse_graph_file, Graph
from typing import List, Tuple, Dict, Optional

EdgeRecord = Tuple[str, str, float]

def graph_to_edges(graph: Graph) -> List[EdgeRecord]:
    edges: List[EdgeRecord] = []
    for node1, node2, direction, weight, label in graph.edges:
        w = 1.0 if weight is None else weight
        if direction == ">":
            edges.append((node1, node2, w))
        elif direction == "<":
            edges.append((node2, node1, w))
        else:
            edges.append((node1, node2, w))
            edges.append((node2, node1, w))
    return edges

def max_safest_path(nodes: List[str], edges: List[EdgeRecord], source: str, target: str):
    """
    Najde cestu s maximálním součinem vah (most dangerous)
    pomocí logaritmického prostoru.
    """
    log_edges: List[Tuple[str, str, float]] = []
    print(f"\nTabulka hran (logX):")
    print(f"{'Start':10} {'Cíl':10} {'váha':>10} {'log(váha)':>12}")
    for start, end, weight in edges:
        if weight <= 0:
            raise ValueError(f"Váha musí být kladná: {start}->{end}={weight}")
        log_w = math.log(weight)
        print(f"{start:10} {end:10} {weight:10.6f} {log_w:12.6f}")
        log_edges.append((start, end, log_w))

    # Bellman-Ford pro největší sumu log(w)
    dist: Dict[str, float] = {node: -math.inf for node in nodes}
    dist[source] = 0.0
    pred: Dict[str, Optional[str]] = {node: None for node in nodes}

    V = len(nodes)
    for _ in range(V-1):
        for u, v, log_w in log_edges:
            if dist[u] + log_w > dist[v]:
                dist[v] = dist[u] + log_w
                pred[v] = u

    # Rekonstrukce cesty
    if dist[target] == -math.inf:
        return None, []

    path = []
    cur = target
    while cur is not None:
        path.append(cur)
        cur = pred[cur]
    path.reverse()

    total_product = 1.0
    for i in range(len(path)-1):
        for u, v, w in edges:
            if u == path[i] and v == path[i+1]:
                total_product *= w
                break

    return total_product, path

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Nejnebezpečnější cesta (max součin vah)")
    parser.add_argument("file", help="Cesta k .tg souboru")
    args = parser.parse_args()

    graph = parse_graph_file(args.file)
    edges = graph_to_edges(graph)
    nodes = sorted(graph.nodes.keys())

    print(f"Dostupné uzly: {', '.join(nodes)}")
    source = input("Zadej počáteční uzel: ").strip()
    target = input("Zadej cílový uzel: ").strip()

    if source not in nodes or target not in nodes:
        print("❌ Neplatný uzel!")
        return

    total_product, path = max_safest_path(nodes, edges, source, target)

    if not path:
        print(f"❌ Cesta z {source} do {target} neexistuje.")
        return

    print(f"\n📊 Nejnebezpečnější cesta z {source} do {target}: {' → '.join(path)}")
    print(f"Hodnota metriky (součin vah): {total_product:.6f}")

if __name__ == "__main__":
    main()
