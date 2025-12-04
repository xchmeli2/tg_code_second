"""
Interaktivní Floyd-Warshall pro různé metriky na grafech z properties.py.

Po spuštění skriptu:
1. Načte graf ze souboru (.tg) ve formátu jako `properties.py`.
2. Nabídne seznam metrik (nejkratší, nejdelší, nejbezpečnější, ...)
3. Uživatelská volba určí, jaká metrika se použije.
4. Pro LONGEST používá BFS, pro SAFEST Dijkstra, ostatní Floyd-Warshall
5. Po skončení lze vypsat cestu mezi konkrétní dvojicí uzlů.
"""

from __future__ import annotations

import argparse
import math
import statistics
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import csv
import os
import heapq
from collections import deque
from properties import Graph, parse_graph_file


# ---------------------------------------------------------------------------
# Datové struktury a konfigurace metrik
# ---------------------------------------------------------------------------

class Metric(Enum):
    SHORTEST = "nejkratší"
    LONGEST = "nejdelší"
    SAFEST = "nejbezpečnější"
    MOST_DANGEROUS = "nejnebezpečnější"
    WIDEST = "nejširší"
    NARROWEST = "nejužší"

@dataclass
class MetricConfig:
    description: str
    default_distance: float
    diagonal_value: float
    better: Callable[[float, float], bool]
    combine: Callable[[float, float], float]
    transform_weight: Callable[[float], float] = lambda w: w
    accumulator: str = "sum"
    use_special_algorithm: bool = False  # True = použij speciální algoritmus

def metric_configs() -> Dict[Metric, MetricConfig]:
    def additive_safe(a: float, b: float) -> float:
        if math.isinf(a) or math.isinf(b):
            return math.inf
        if math.isinf(-a) or math.isinf(-b):
            return -math.inf
        return a + b

    def bottleneck(a: float, b: float) -> float:
        return min(a, b)

    return {
        Metric.SHORTEST: MetricConfig(
            description="Minimalizace součtu vah (Dijkstra).",
            default_distance=math.inf,
            diagonal_value=0.0,
            better=lambda current, candidate: candidate < current,
            combine=additive_safe,
        ),
        Metric.LONGEST: MetricConfig(
            description="Maximalizace součtu vah (BFS bez opakování uzlů).",
            default_distance=-math.inf,
            diagonal_value=0.0,
            better=lambda current, candidate: candidate > current,
            combine=additive_safe,
            use_special_algorithm=True,
        ),
        Metric.SAFEST: MetricConfig(
            description="Minimalizace rizika (Dijkstra, minimalizace součinu rizik pomocí log transformace).",
            default_distance=math.inf,
            diagonal_value=0.0,
            better=lambda current, candidate: candidate < current,
            combine=additive_safe,
            transform_weight=lambda w: math.log(w) if w > 0 else math.inf,
            use_special_algorithm=True,
        ),
        Metric.MOST_DANGEROUS: MetricConfig(
            description="Maximalizace součinu hodnot (Dijkstra s -log transformací pro max součin).",
            default_distance=math.inf,  # Dijkstra hledá minimum, ale my chceme maximum
            diagonal_value=0.0,
            better=lambda current, candidate: candidate < current,
            combine=additive_safe,
            transform_weight=lambda w: -math.log(w) if w > 0 else math.inf,
            use_special_algorithm=True,  # Použijeme Dijkstra s -log
        ),
        Metric.WIDEST: MetricConfig(
            description="Maximalizace minimální kapacity (Floyd-Warshall bottleneck).",
            default_distance=-math.inf,
            diagonal_value=math.inf,
            better=lambda current, candidate: candidate > current,
            combine=bottleneck,
            accumulator="bottleneck (max-min)",
        ),
        Metric.NARROWEST: MetricConfig(
            description="Minimalizace maximální kapacity (Floyd-Warshall bottleneck).",
            default_distance=math.inf,
            diagonal_value=0.0,
            better=lambda current, candidate: candidate < current,
            combine=bottleneck,
            accumulator="bottleneck (min-max)",
        ),
    }


@dataclass
class EdgeRecord:
    start: str
    end: str
    weight: float
    label: str


def _default_weight(weight: Optional[float]) -> float:
    return 1.0 if weight is None else weight


def graph_to_edges(graph: Graph) -> List[EdgeRecord]:
    edges: List[EdgeRecord] = []
    for node1, node2, direction, weight, label in graph.edges:
        w = _default_weight(weight)
        lbl = label or f"h{node1}{node2}"
        if direction == ">":
            edges.append(EdgeRecord(node1, node2, w, lbl))
        elif direction == "<":
            edges.append(EdgeRecord(node2, node1, w, lbl))
        else:
            edges.append(EdgeRecord(node1, node2, w, lbl))
            edges.append(EdgeRecord(node2, node1, w, lbl))
    return edges


def graph_stats(graph: Graph, edges: Sequence[EdgeRecord]) -> Dict[str, Any]:
    weights = [edge.weight for edge in edges]
    node_count = len(graph.nodes)
    directed_edges = len(edges)
    possible_directed = node_count * (node_count - 1)
    density = directed_edges / possible_directed if possible_directed else 0.0

    weight_stats = {
        "min_vaha": min(weights) if weights else None,
        "max_vaha": max(weights) if weights else None,
        "prumerna_vaha": statistics.fmean(weights) if weights else None,
        "median_vaha": statistics.median(weights) if weights else None,
    }
    return {
        "pocet_uzlu": node_count,
        "pocet_orientovanych_hran": directed_edges,
        "hustota_orientovana": density,
        **weight_stats,
    }


# ---------------------------------------------------------------------------
# Speciální algoritmy pro LONGEST a SAFEST
# ---------------------------------------------------------------------------

def longest_path_bfs(
    adj: Dict[str, List[Tuple[str, float]]],
    source: str,
    target: str,
) -> Tuple[float, List[str]]:
    """BFS pro nejdelší jednoduchou cestu (bez opakování uzlů)"""
    queue = deque([(source, frozenset([source]), [source], 0.0)])
    best_length = -math.inf
    best_path: List[str] = []
    
    while queue:
        current, visited, path, length = queue.popleft()
        
        if current == target:
            if length > best_length:
                best_length = length
                best_path = list(path)
            continue
        
        for neighbor, weight in adj.get(current, []):
            if neighbor not in visited:
                new_visited = visited | {neighbor}
                new_path = path + [neighbor]
                new_length = length + weight
                queue.append((neighbor, new_visited, new_path, new_length))
    
    return best_length, best_path


def dijkstra_single_source(
    adj: Dict[str, List[Tuple[str, float]]],
    source: str,
    nodes: List[str],
) -> Tuple[Dict[str, float], Dict[str, Optional[str]]]:
    """Dijkstrův algoritmus pro jeden zdrojový uzel"""
    dist = {node: math.inf for node in nodes}
    pred = {node: None for node in nodes}
    dist[source] = 0.0
    
    # Min-heap: (distance, node)
    heap = [(0.0, source)]
    visited = set()
    
    while heap:
        d, u = heapq.heappop(heap)
        
        if u in visited:
            continue
        visited.add(u)
        
        # OPRAVA: Tato kontrola byla špatně - smazána
        # if d > dist[u]:
        #     continue
        
        for v, weight in adj.get(u, []):
            new_dist = dist[u] + weight
            if new_dist < dist[v]:
                dist[v] = new_dist
                pred[v] = u  # ← Tady nastavujeme předchůdce
                heapq.heappush(heap, (new_dist, v))
    
    return dist, pred


# ---------------------------------------------------------------------------
# Floyd-Warshall algoritmus
# ---------------------------------------------------------------------------

@dataclass
class FloydSnapshot:
    step: int
    via_node: str
    distance_matrix: List[List[float]]
    predecessor_matrix: List[List[Optional[str]]]


@dataclass
class FloydResult:
    metric: Metric
    nodes: List[str]
    distances: List[List[float]]
    predecessors: List[List[Optional[str]]]
    snapshots: List[FloydSnapshot]
    stats: Dict[str, Any] = field(default_factory=dict)
    paths_cache: Dict[Tuple[str, str], List[str]] = field(default_factory=dict)  # Cache cest pro BFS/Dijkstra

    def distance(self, source: str, target: str) -> float:
        i = self.nodes.index(source)
        j = self.nodes.index(target)
        return self.distances[i][j]

    def reconstruct_path(self, source: str, target: str) -> List[str]:
        print(f"\n🔍 DEBUG reconstruct_path({source} -> {target})")
        print(f"   Cache keys: {list(self.paths_cache.keys())}")
        print(f"   Looking for: {(source, target)}")
        
        # Pokud máme cestu v cache (pro BFS/Dijkstra), použijeme ji
        if (source, target) in self.paths_cache:
            path = self.paths_cache[(source, target)]
            print(f"   ✅ Found in cache: {path}")
            return path
        
        print(f"   ⚠️ Not in cache, trying predecessor matrix...")
        
        # Jinak rekonstruujeme z matice předchůdců (Floyd-Warshall)
        idx = {node: pos for pos, node in enumerate(self.nodes)}
        if source not in idx or target not in idx:
            print(f"   ❌ Node not in index")
            return []
        
        i, j = idx[source], idx[target]
        dist = self.distances[i][j]
        
        print(f"   Distance[{i}][{j}] = {dist}")
        print(f"   Predecessor[{i}][{j}] = {self.predecessors[i][j]}")
        
        # Pokud je vzdálenost nekonečno, cesta neexistuje
        if math.isinf(dist) and dist > 0:
            print(f"   ❌ Distance is +inf")
            return []
        if math.isinf(-dist):
            print(f"   ❌ Distance is -inf")
            return []
        
        # Rekonstrukce cesty zpětně od cíle
        path = []
        current = target
        current_idx = j
        visited = set()
        
        print(f"   Starting reconstruction from {target}...")
        
        while current != source:
            if current in visited:
                print(f"   ❌ Cycle detected at {current}")
                return []
            visited.add(current)
            path.append(current)
            
            predecessor = self.predecessors[i][current_idx]
            print(f"     {current} <- {predecessor}")
            
            if predecessor is None or predecessor == "0":
                print(f"   ❌ No predecessor")
                return []
            
            current = predecessor
            current_idx = idx[current]
            
            if len(visited) > len(self.nodes):
                print(f"   ❌ Too many steps")
                return []
        
        path.append(source)
        result = list(reversed(path))
        print(f"   ✅ Reconstructed: {result}")
        return result


def _initialize_matrices(
    nodes: List[str],
    edges: Sequence[EdgeRecord],
    config: MetricConfig,
) -> Tuple[List[List[float]], List[List[Optional[str]]]]:
    n = len(nodes)
    idx = {node: i for i, node in enumerate(nodes)}
    dist = [[config.default_distance for _ in range(n)] for _ in range(n)]
    pred: List[List[Optional[str]]] = [[None for _ in range(n)] for _ in range(n)]

    for i in range(n):
        dist[i][i] = config.diagonal_value
        pred[i][i] = "0"

    for edge in edges:
        i, j = idx[edge.start], idx[edge.end]
        weight = config.transform_weight(edge.weight)
        if config.better(dist[i][j], weight):
            dist[i][j] = weight
            pred[i][j] = edge.start

    return dist, pred


def _format_matrix(matrix: List[List[float]], nodes: List[str]) -> str:
    def fmt(value: float) -> str:
        if value == math.inf:
            return "∞".rjust(8)
        if value == -math.inf:
            return "-∞".rjust(8)
        return f"{value:8.2f}"

    header = "        " + " ".join(node.rjust(8) for node in nodes)
    rows = [header]
    for node, row in zip(nodes, matrix):
        rows.append(node.rjust(8) + " " + " ".join(fmt(val) for val in row))
    return "\n".join(rows)


def _format_pred(pred: List[List[Optional[str]]], nodes: List[str]) -> str:
    header = "        " + " ".join(node.rjust(8) for node in nodes)
    rows = [header]
    for src, row in zip(nodes, pred):
        rows.append(
            src.rjust(8)
            + " "
            + " ".join((val or "-").rjust(8) for val in row)
        )
    return "\n".join(rows)


def _restore_values_after_floyd(
    matrix: List[List[float]], 
    metric: Metric
) -> List[List[float]]:
    """Převede hodnoty zpět z transformovaného prostoru"""
    restored = []
    
    if metric == Metric.SAFEST:
        # Pro SAFEST: NECHÁME v log prostoru!
        # Hodnoty jsou součet log(w) = log(součin w)
        # To JE naše metrika - nepřevádíme!
        restored = [row[:] for row in matrix]
    
    elif metric == Metric.MOST_DANGEROUS:
        # Pro MOST_DANGEROUS: log(x) -> exp(x)
        for row in matrix:
            new_row = []
            for value in row:
                if value == math.inf:
                    new_row.append(math.inf)
                elif value == -math.inf:
                    new_row.append(0.0)
                elif value == 0.0:
                    new_row.append(1.0)
                else:
                    new_row.append(math.exp(value))
            restored.append(new_row)
    
    else:
        # Pro ostatní metriky: bez převodu
        restored = [row[:] for row in matrix]
    
    return restored


def compute_with_special_algorithm(
    graph: Graph,
    edges: Sequence[EdgeRecord],
    metric: Metric,
    nodes: List[str],
    verbose: bool = True,
) -> Tuple[List[List[float]], List[List[Optional[str]]], Dict[Tuple[str, str], List[str]]]:
    """Výpočet pro metriky vyžadující speciální algoritmy (LONGEST, SAFEST)"""
    
    print(f"\n{'='*60}")
    print(f"🚀 compute_with_special_algorithm START")
    print(f"   Metrika: {metric}")
    print(f"   Uzly: {nodes}")
    print(f"   Počet hran: {len(edges)}")
    print(f"{'='*60}\n")
    
    n = len(nodes)
    idx = {node: i for i, node in enumerate(nodes)}
    dist = [[math.inf if metric == Metric.SAFEST else -math.inf for _ in range(n)] for _ in range(n)]
    pred: List[List[Optional[str]]] = [[None for _ in range(n)] for _ in range(n)]
    paths_cache: Dict[Tuple[str, str], List[str]] = {}  # Cache pro úplné cesty
    
    # Vytvoření adjacency listu
    adj: Dict[str, List[Tuple[str, float]]] = {node: [] for node in nodes}
    for edge in edges:
        adj[edge.start].append((edge.end, edge.weight))
    
    # Diagonála
    for i in range(n):
        dist[i][i] = 0.0 if metric == Metric.SAFEST else 0.0
        pred[i][i] = "0"
        paths_cache[(nodes[i], nodes[i])] = [nodes[i]]
    
    print(f"\n{'='*60}")
    print(f"Výpočet pomocí {'BFS' if metric == Metric.LONGEST else 'Dijkstra'}")
    print(f"{'='*60}\n")
    
    total_pairs = n * n
    computed = 0
    
    for i, source in enumerate(nodes):
        if verbose:
            print(f"Zpracovávám zdroj: {source} ({i+1}/{n})")
        
        if metric == Metric.LONGEST:
            # BFS pro nejdelší cestu
            for j, target in enumerate(nodes):
                computed += 1
                if i == j:
                    continue
                
                print(i)
                
                length, path = longest_path_bfs(adj, source, target)
                dist[i][j] = length
                
                # Uložíme celou cestu do cache
                if path:
                    paths_cache[(source, target)] = path
                    if len(path) > 1:
                        print(len(path))
                        pred[i][j] = path[-2]  # Předposlední uzel
        
        elif metric == Metric.SAFEST:
            # Dijkstra pro nejbezpečnější cestu
            # Váhy reprezentují RIZIKA (čím větší, tím horší)
            # Chceme najít cestu s MINIMÁLNÍM SOUČINEM rizik
            # Transformace: w -> log(w)
            # Min(log(w1) + log(w2) + ...) = Min(log(w1*w2*...)) = Min(w1*w2*...)
            
            print(f"\n{'='*70}")
            print(f"📊 Zdroj {source}: Transformace všech hran na log")
            print(f"{'='*70}")
            
            adj_transformed: Dict[str, List[Tuple[str, float]]] = {node: [] for node in nodes}
            for edge in edges:
                if edge.weight <= 0:
                    weight = math.inf
                    log_str = "∞"
                else:
                    # log transformace: min součet log = min součin původních vah
                    weight = math.log(edge.weight)
                    log_str = f"{weight:.6f}"
                
                adj_transformed[edge.start].append((edge.end, weight))
                print(f"  {edge.start} → {edge.end}: w={edge.weight:8.4f}  →  log(w)={log_str:>12}")
            
            print(f"\n🔍 Spouštím Dijkstra ze zdroje {source}...")
            dist_dict, pred_dict = dijkstra_single_source(adj_transformed, source, nodes)
            
            print(f"\n📋 Výsledky Dijkstra ze zdroje {source}:")
            print(f"{'Cíl':<8} {'Součet log(w)':<18} {'Součin w':<18} {'Předchůdce':<12}")
            print(f"{'-'*70}")
            for target in nodes:
                dist_log = dist_dict[target]
                if dist_log == math.inf:
                    log_str = "∞"
                    exp_str = "∞"
                elif dist_log == 0.0:
                    log_str = "0.0000"
                    exp_str = "1.0000"
                else:
                    log_str = f"{dist_log:.6f}"
                    exp_str = f"{math.exp(dist_log):.6f}"
                
                pred_str = pred_dict[target] if pred_dict[target] else "-"
                print(f"{target:<8} {log_str:<18} {exp_str:<18} {pred_str:<12}")
            
            print(f"{'='*70}\n")
            
            # Pro každý cílový uzel
            for j, target in enumerate(nodes):
                # Uložíme vzdálenost v transformovaném prostoru
                dist[i][j] = dist_dict[target]
                
                # Rekonstruujeme a uložíme CELOU cestu
                if source == target:
                    paths_cache[(source, target)] = [source]
                    pred[i][j] = "0"
                elif dist_dict[target] < math.inf:
                    # Rekonstrukce celé cesty
                    path = []
                    current = target
                    visited = set()
                    
                    if source == 'A' and target == 'H':
                        print(f"\n🔍 Rekonstrukce cesty A -> H:")
                    
                    while current != source and current is not None:
                        if current in visited:
                            if source == 'A' and target == 'H':
                                print(f"   ❌ Cyklus detekován!")
                            break
                        visited.add(current)
                        path.append(current)
                        
                        next_node = pred_dict.get(current)
                        if source == 'A' and target == 'H':
                            print(f"   {current} <- {next_node}")
                        
                        current = next_node
                        
                        if len(visited) > len(nodes):
                            if source == 'A' and target == 'H':
                                print(f"   ❌ Příliš mnoho kroků!")
                            break
                    
                    # Pokud jsme úspěšně došli ke zdroji
                    if current == source:
                        path.append(source)
                        path.reverse()
                        paths_cache[(source, target)] = path
                        
                        if source == 'A' and target == 'H':
                            print(f"   ✅ Cesta nalezena: {' → '.join(path)}")
                            print(f"\n   📊 Výpočet metriky po cestě:")
                            print(f"   {'Hrana':<12} {'w':<10} {'log(w)':<12} {'Σlog(w)':<12} {'Πw':<12}")
                            print(f"   {'-'*60}")
                            
                            log_sum = 0.0
                            product = 1.0
                            for k in range(len(path) - 1):
                                for edge in edges:
                                    if edge.start == path[k] and edge.end == path[k+1]:
                                        log_val = math.log(edge.weight)
                                        log_sum += log_val
                                        product *= edge.weight
                                        edge_str = f"{path[k]}→{path[k+1]}"
                                        print(f"   {edge_str:<12} {edge.weight:<10.4f} {log_val:<12.6f} {log_sum:<12.6f} {product:<12.6f}")
                                        break
                            
                            print(f"\n   🎯 Finální výsledek:")
                            print(f"      Součin rizik (metrika) = {product:.6f}")
                            print(f"      (Součet log(w) = {log_sum:.6f})  # jen pro kontrolu")
                            print(f"      Ověření: exp(Σlog(w)) = {math.exp(log_sum):.6f}")
                        
                        # Nastavíme předchůdce jako předposlední uzel v cestě
                        if len(path) > 1:
                            pred[i][j] = path[-2]  # Přímý předchůdce cíle
                        else:
                            pred[i][j] = source
                    else:
                        # Cesta neexistuje
                        if source == 'A' and target == 'H':
                            print(f"   ❌ Nedosaženo zdroje! current={current}")
                        pred[i][j] = None
                else:
                    # Nedosažitelné
                    pred[i][j] = None
        
        if verbose and (i + 1) % 5 == 0:
            progress = ((i + 1) * n) / total_pairs * 100
            print(f"  Progres: {progress:.1f}%")
    
    print(f"\n✅ Výpočet dokončen!\n")
    
    # Převod hodnot zpět pro SAFEST
    if metric == Metric.SAFEST:
        # NECHÁME hodnoty v log prostoru - to je naše metrika!
        # Součet log(w) = log(součin w) = minimální "log riziko"
        # NEPŘEVÁDÍME na exp!
        pass
    
    return dist, pred, paths_cache


def floyd_warshall(
    graph: Graph,
    edges: Sequence[EdgeRecord],
    metric: Metric,
    verbose: bool = True,
) -> FloydResult:
    nodes = sorted(graph.nodes.keys())
    config = metric_configs()[metric]
    n = len(nodes)
    snapshots: List[FloydSnapshot] = []
    paths_cache: Dict[Tuple[str, str], List[str]] = {}

    # Pro speciální algoritmy (LONGEST, SAFEST)
    if config.use_special_algorithm:
        dist, pred, paths_cache = compute_with_special_algorithm(graph, edges, metric, nodes, verbose)
        
        # Vytvoříme jen finální snapshot
        final_snapshot = FloydSnapshot(
            step=0,
            via_node="FINAL",
            distance_matrix=deepcopy(dist),
            predecessor_matrix=deepcopy(pred),
        )
        snapshots.append(final_snapshot)
    else:
        # Standardní Floyd-Warshall
        dist, pred = _initialize_matrices(nodes, edges, config)

        initial_snapshot = FloydSnapshot(
            step=0,
            via_node="-",
            distance_matrix=deepcopy(dist),
            predecessor_matrix=deepcopy(pred),
        )
        snapshots.append(initial_snapshot)

        # if verbose:
        #     print("=== Výchozí stav (iterace 0) ===")
        #     print("Matice délek:")
        #     print(_format_matrix(dist, nodes))
        #     print("\nMatice předchůdců:")
        #     print(_format_pred(pred, nodes))
        #     print()

        for k, via in enumerate(nodes):
            for i in range(n):
                for j in range(n):
                    dist_ik = dist[i][k]
                    dist_kj = dist[k][j]
                    
                    # Pro bottleneck metriky
                    if metric in [Metric.WIDEST, Metric.NARROWEST]:
                        if dist_ik == config.default_distance or dist_kj == config.default_distance:
                            continue
                    else:
                        if math.isinf(dist_ik) or math.isinf(dist_kj):
                            continue
                    
                    candidate = config.combine(dist_ik, dist_kj)
                    if config.better(dist[i][j], candidate):
                        dist[i][j] = candidate
                        pred[i][j] = pred[k][j] if pred[k][j] is not None else nodes[k]

            snapshot = FloydSnapshot(
                step=k + 1,
                via_node=via,
                distance_matrix=deepcopy(dist),
                predecessor_matrix=deepcopy(pred),
            )
            snapshots.append(snapshot)

            # if verbose:
            #     print(f"\n=== Iterace {k + 1} (přes uzel {via}) ===")
            #     print("Matice délek:")
            #     print(_format_matrix(dist, nodes))
            #     print("\nMatice předchůdců:")
            #     print(_format_pred(pred, nodes))
            #     print()

        # Převod hodnot zpět podle metriky
        dist = _restore_values_after_floyd(dist, metric)

    stats = graph_stats(graph, edges)
    diagonal_sum = sum(dist[i][i] for i in range(n)) if n > 0 else 0.0
    bunka_11 = dist[0][0] if n > 0 else None
    second_row_sum = sum(dist[1]) if n > 1 else None
    second_col_sum = sum(dist[i][1] for i in range(n)) if n > 1 else None
    stats.update({
        "pocet_iteraci": len(snapshots),
        "popis_metriky": config.description,
        "typ_akumulatoru": config.accumulator,
        "soucet_diagonaly": diagonal_sum,
        "bunka_1_1": bunka_11,
        "soucet_radek_2": second_row_sum,
        "soucet_sloupec_2": second_col_sum,
    })

    return FloydResult(
        metric=metric,
        nodes=nodes,
        distances=dist,
        predecessors=pred,
        snapshots=snapshots,
        stats=stats,
        paths_cache=paths_cache,
    )


# ---------------------------------------------------------------------------
# Export a CLI
# ---------------------------------------------------------------------------

def export_matrices_to_csv(result: FloydResult, base_filename: str) -> None:
    """Exportuje finální matice do CSV"""
    output_dir = "floyd_output"
    os.makedirs(output_dir, exist_ok=True)
    metric_name = result.metric.name.lower()
    
    # Export matice vzdáleností
    distance_file = os.path.join(output_dir, f"{base_filename}_{metric_name}_distances.csv")
    with open(distance_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([''] + result.nodes)
        for i, node in enumerate(result.nodes):
            row = [node]
            for value in result.distances[i]:
                if value == math.inf:
                    row.append('∞')
                elif value == -math.inf:
                    row.append('-∞')
                else:
                    row.append(f"{value:.6f}")
            writer.writerow(row)
    print(f"✅ Matice vzdáleností: {distance_file}")
    
    # Export matice předchůdců
    predecessor_file = os.path.join(output_dir, f"{base_filename}_{metric_name}_predecessors.csv")
    with open(predecessor_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([''] + result.nodes)
        for i, node in enumerate(result.nodes):
            row = [node]
            for pred in result.predecessors[i]:
                row.append(pred if pred else '-')
            writer.writerow(row)
    print(f"✅ Matice předchůdců: {predecessor_file}")


def prompt_metric() -> Metric:
    configs = metric_configs()
    options = list(Metric)
    print("Vyber metrickou variantu:")
    for idx, metric in enumerate(options, start=1):
        print(f"  {idx}. {metric.value} – {configs[metric].description}")

    while True:
        choice = input("Zadej číslo volby: ").strip()
        if not choice.isdigit():
            print("Prosím zadej číslo.")
            continue
        index = int(choice) - 1
        if 0 <= index < len(options):
            return options[index]
        print("Neplatná volba, zkus to znovu.")

def prompt_path_query(result: FloydResult) -> None:
    """Interaktivní dotazování na cesty"""
    nodes = result.nodes
    
    while True:
        print("\n" + "=" * 60)
        print("DOTAZ NA CESTU MEZI UZLY")
        print("=" * 60)
        print(f"Dostupné uzly: {', '.join(nodes)}")
        print("(Pro ukončení zadej prázdný vstup)")
        
        source = input("\n👉 Zadej POČÁTEČNÍ uzel: ").strip()
        if not source:
            print("Ukončuji dotazování.")
            break
        
        if source not in nodes:
            print(f"❌ Uzel '{source}' neexistuje!")
            continue
        
        target = input("👉 Zadej CÍLOVÝ uzel: ").strip()
        if not target:
            print("Ukončuji dotazování.")
            break
        
        if target not in nodes:
            print(f"❌ Uzel '{target}' neexistuje!")
            continue
        
        distance = result.distance(source, target)
        path = result.reconstruct_path(source, target)
        
        print("\n" + "=" * 60)
        print(f"VÝSLEDEK: Cesta z '{source}' do '{target}'")
        print("=" * 60)
        
        if not path:
            if math.isinf(distance) and distance > 0:
                print(f"❌ Cesta neexistuje - uzel '{target}' není dosažitelný")
            elif math.isinf(-distance):
                print(f"❌ Obsahuje negativní cyklus")
            else:
                print(f"❌ Nelze rekonstruovat cestu")
            continue
        
        metric_descriptions = {
            Metric.SHORTEST: ("nejkratší cesta", "Celková délka"),
            Metric.LONGEST: ("nejdelší jednoduchá cesta", "Celková délka"),
            Metric.SAFEST: ("nejbezpečnější cesta (min součin rizik)", "Součin rizik"),
            Metric.MOST_DANGEROUS: ("nejnebezpečnější cesta", "Celková nebezpečnost"),
            Metric.WIDEST: ("nejširší cesta", "Minimální kapacita"),
            Metric.NARROWEST: ("nejužší cesta", "Maximální kapacita"),
        }
        
        metric_desc, value_label = metric_descriptions.get(
            result.metric, (result.metric.value, "Hodnota")
        )
        
        # Pro SAFEST a MOST_DANGEROUS převedeme log hodnotu zpět na součin
        display_value = distance
        if result.metric == Metric.SAFEST and not math.isinf(distance):
            display_value = math.exp(distance)  # exp(Σ log(w)) = Π w
            print(f"\n📏 {value_label}: {display_value:.6f}")
            print(f"   (Interní hodnota Σlog(w): {distance:.6f})")
        elif result.metric == Metric.MOST_DANGEROUS and not math.isinf(distance):
            display_value = math.exp(-distance)  # exp(-Σ(-log(w))) = exp(Σlog(w)) = Π w
            print(f"\n📏 {value_label}: {display_value:.6f}")
            print(f"   (Interní hodnota Σ(-log(w)): {distance:.6f})")
        else:
            print(f"\n📏 {value_label}: {display_value:.6f}")
        
        print(f"   (Metrika: {metric_desc})")
        print(f"\n🛤️  Cesta ({len(path)} uzlů):")
        print(f"   {' → '.join(path)}")
        print(f"\n📊 Statistiky:")
        print(f"   • Počet uzlů: {len(path)}")
        print(f"   • Počet hran: {len(path) - 1}")
        print("=" * 60)
        
def print_formatted_stats(stats: Dict[str, Any]) -> None:
    labels = {
        "pocet_uzlu": "Počet uzlů",
        "pocet_orientovanych_hran": "Počet orientovaných hran",
        "hustota_orientovana": "Hustota (orientovaná)",
        "min_vaha": "Minimální váha hrany",
        "max_vaha": "Maximální váha hrany",
        "prumerna_vaha": "Průměrná váha hrany",
        "median_vaha": "Medián váhy hrany",
        "pocet_iteraci": "Počet iterací",
        "popis_metriky": "Popis metriky",
        "typ_akumulatoru": "Typ akumulace",
    }
    for key, value in stats.items():
        if key in labels:
            print(f"  - {labels[key]}: {value}")


def run_cli(args: argparse.Namespace) -> None:
    graph = parse_graph_file(args.file)
    edges = graph_to_edges(graph)

    if args.metric:
        metric = Metric[args.metric.upper()]
    else:
        metric = prompt_metric()

    result = floyd_warshall(graph, edges, metric, verbose=not args.quiet)
    
    base_filename = os.path.splitext(os.path.basename(args.file))[0]
    print("\n" + "=" * 60)
    print("EXPORT VÝSLEDKŮ")
    print("=" * 60)
    export_matrices_to_csv(result, base_filename)
    
    prompt_path_query(result)
    
    print("\n" + "=" * 60)
    print("STATISTIKY GRAFU")
    print("=" * 60)
    print_formatted_stats(result.stats)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Floyd-Warshall s BFS/Dijkstra pro speciální metriky."
    )
    parser.add_argument("file", help="Cesta k .tg souboru")
    parser.add_argument("--metric", help="Název metriky (shortest, longest, safest, ...)")
    parser.add_argument("--quiet", action="store_true", help="Potlačí výpis matic")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_cli(args)


if __name__ == "__main__":
    main()