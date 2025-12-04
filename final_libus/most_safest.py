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

def max_product_path(nodes: List[str], edges: List[EdgeRecord], source: str, target: str):
    """
    Najde cestu s minimálním rizikem (nejbezpečnější)
    """
    # Převod vah na riziko a -log(riziko)
    log_edges: List[Tuple[str, str, float]] = []
    print(f"\nTabulka hran (-log(riziko)):")
    print(f"{'Start':10} {'Cíl':10} {'váha':>10} {'-log(riziko)':>15}")
    for start, end, weight in edges:
        if not (0 < weight <= 1):
            raise ValueError(f"Váha musí být mezi 0 a 1: {start}->{end}={weight}")
        risk = 1 - weight
        neg_log_risk = -math.log(risk) if risk > 0 else float('inf')
        print(f"{start:10} {end:10} {weight:10.6f} {neg_log_risk:15.6f}")
        log_edges.append((start, end, neg_log_risk))

    # Bellman-Ford pro nejmenší sumu -log(riziko)
    dist: Dict[str, float] = {node: math.inf for node in nodes}
    dist[source] = 0.0
    pred: Dict[str, Optional[str]] = {node: None for node in nodes}

    V = len(nodes)
    for _ in range(V-1):
        for u, v, w in log_edges:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                pred[v] = u

    # Rekonstrukce cesty
    if dist[target] == math.inf:
        return None, []

    path = []
    cur = target
    while cur is not None:
        path.append(cur)
        cur = pred[cur]
    path.reverse()

    # Součin pravděpodobností bezpečí
    total_prob = 1.0
    for i in range(len(path)-1):
        for u, v, w in edges:
            if u == path[i] and v == path[i+1]:
                total_prob *= w
                break

    return total_prob, path

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Nejbezpečnější cesta (min riziko)")
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

    total_prob, path = max_product_path(nodes, edges, source, target)

    if not path:
        print(f"❌ Cesta z {source} do {target} neexistuje.")
        return

    print(f"\n📊 Nejbezpečnější cesta z {source} do {target}: {' → '.join(path)}")
    print(f"Hodnota metriky (součin pravděpodobností bezpečí): {total_prob:.6f}")

if __name__ == "__main__":
    main()
