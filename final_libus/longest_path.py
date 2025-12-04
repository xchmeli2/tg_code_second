#!/usr/bin/env python3
import argparse
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import heapq

from properties import Graph, parse_graph_file

@dataclass
class EdgeRecord:
    start: str
    end: str
    weight: float

def graph_to_edges(graph: Graph) -> List[EdgeRecord]:
    edges: List[EdgeRecord] = []
    for node1, node2, direction, weight, label in graph.edges:
        w = 1.0 if weight is None else weight
        if direction == ">":
            edges.append(EdgeRecord(node1, node2, w))
        elif direction == "<":
            edges.append(EdgeRecord(node2, node1, w))
        else:
            edges.append(EdgeRecord(node1, node2, w))
            edges.append(EdgeRecord(node2, node1, w))
    return edges

def build_graph_dict(edges: List[EdgeRecord]) -> Dict[str, List[Tuple[str, float]]]:
    graph_dict = defaultdict(list)
    for edge in edges:
        graph_dict[edge.start].append((edge.end, edge.weight))
    return graph_dict

def longest_path_bfs(source: str, target: str, edges: List[EdgeRecord], max_depth: int = 50) -> Tuple[List[str], float]:
    """
    Najde nejdelší cestu mezi dvěma uzly pomocí priority BFS (heuristický A* styl).
    """
    graph = build_graph_dict(edges)
    # PriorityQueue: (-current_length, path, current_node)
    pq = [(-0.0, [source], source)]
    best_length = float('-inf')
    best_path: List[str] = []
    visited_states = {}  # (node, len(path)) -> max_length

    while pq:
        neg_len, path, node = heapq.heappop(pq)
        current_len = -neg_len

        if node == target:
            if current_len > best_length:
                best_length = current_len
                best_path = path
            continue

        if len(path) >= max_depth:
            continue

        for neighbor, weight in graph.get(node, []):
            if neighbor in path:
                continue  # zabráníme cyklům

            new_len = current_len + weight
            state_key = (neighbor, len(path)+1)
            if visited_states.get(state_key, float('-inf')) >= new_len:
                continue  # už lepší cesta k tomuto stavu existuje
            visited_states[state_key] = new_len

            heapq.heappush(pq, (-new_len, path + [neighbor], neighbor))

    return best_path, best_length
        

def main():
    parser = argparse.ArgumentParser(description="Rychlý výpočet nejdelší cesty")
    parser.add_argument("file", help="Cesta k .tg souboru")
    parser.add_argument("--source", help="Počáteční uzel")
    parser.add_argument("--target", help="Cílový uzel")
    parser.add_argument("--max-depth", type=int, default=50, help="Maximální délka cesty")
    args = parser.parse_args()

    graph = parse_graph_file(args.file)
    edges = graph_to_edges(graph)
    nodes = sorted(graph.nodes.keys())

    print(f"Dostupné uzly: {', '.join(nodes[:20])} ...")
    source = args.source or input("Zadej počáteční uzel: ").strip()
    target = args.target or input("Zadej cílový uzel: ").strip()

    if source not in nodes or target not in nodes:
        print("❌ Neplatný uzel!")
        return

    path, length = longest_path_bfs(source, target, edges, max_depth=args.max_depth)

    if not path:
        print(f"❌ Cesta z {source} do {target} neexistuje nebo byla mimo limit {args.max_depth}")
        return

    print(f"\n📊 Nejdelší cesta z {source} do {target}: {' → '.join(path)}")
    print(f"Délka (součet vah): {length:.6f}")

if __name__ == "__main__":
    main()
