"""
Vrstevnicový algoritmus pro nejširší / nejužší cestu mezi dvěma uzly.
Logika je oddělená od Floyd-Warshall skriptu, protože pracuje čistě na principu
postupného rozšiřování fronty (layer-by-layer) ze zadaného startu.

Každá iterace:
1. Vezme všechny uzly, které změnily hodnotu v předchozím kroku.
2. Projde jejich všechny sousedy a případně aktualizuje jejich ohodnocení (hodnota, předchůdce).
3. Uloží snapshot všech uzlů, aby bylo možné průběh zobrazit.

Použití z CLI:
  python3 bottleneck_paths.py graf.tg
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Set
from collections import defaultdict

from properties import Graph, parse_graph_file

@dataclass
class LayerState:
    iteration: int
    values: Dict[str, float]
    predecessors: Dict[str, Optional[str]]


@dataclass
class BottleneckResult:
    mode: str
    source: str
    target: Optional[str]
    values: Dict[str, float]
    predecessors: Dict[str, Optional[str]]
    states: List[LayerState]

    def reconstruct_path(self) -> List[str]:
        if not self.target:
            return []
        
        # Pro widest: hodnota musí být > 0
        # Pro narrowest: hodnota musí být < inf
        value = self.values.get(self.target, 0.0 if self.mode == "widest" else math.inf)
        if self.mode == "widest" and value == 0.0:
            return []
        if self.mode == "narrowest" and math.isinf(value):
            return []
        
        path = []
        current = self.target
        while current is not None:
            path.append(current)
            if current == self.source:
                break
            current = self.predecessors.get(current)
        
        if not path or path[-1] != self.source:
            return []
        
        return list(reversed(path))


def _default_weight(weight: Optional[float]) -> float:
    return 1.0 if weight is None else weight


def load_predecessors(graph: Graph) -> Dict[str, List[Tuple[str, float]]]:
    """
    Vytvoří slovník předchůdců pro každý uzel:
    pred_map[node] = [(parent, weight), ...]
    """
    predecessors: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    
    for node1, node2, direction, weight, _ in graph.edges:
        w = _default_weight(weight)
        
        if direction == '>':
            # node1 -> node2 : node1 je předchůdce node2
            predecessors[node2].append((node1, w))
        elif direction == '<':
            # node2 -> node1
            predecessors[node1].append((node2, w))
        else:
            # neorientovaná hrana: obě směry
            predecessors[node1].append((node2, w))
            predecessors[node2].append((node1, w))
    
    return predecessors


def predecessors_to_successors(
    predecessors: Dict[str, List[Tuple[str, float]]],
    nodes: Sequence[str],
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Převede slovník předchůdců na slovník následníků.
    successors[parent] = [(child, weight), ...]
    """
    successors: Dict[str, List[Tuple[str, float]]] = {node: [] for node in nodes}
    for child, parents in predecessors.items():
        for parent, weight in parents:
            successors.setdefault(parent, []).append((child, weight))
    return successors


def _capture_state(
    nodes: Sequence[str],
    values: Dict[str, float],
    preds: Dict[str, Optional[str]],
    iteration: int,
) -> LayerState:
    return LayerState(
        iteration=iteration,
        values={node: values[node] for node in nodes},
        predecessors={node: preds[node] for node in nodes},
    )


def propagate_widest(graph: Graph, source: str, target: Optional[str]) -> BottleneckResult:
    nodes = sorted(graph.nodes.keys())
    predecessors = load_predecessors(graph)
    successors = predecessors_to_successors(predecessors, nodes)
    
    # Výchozí stav: zdroj má nekonečno, ostatní 0
    values = {node: 0.0 for node in nodes}
    values[source] = math.inf
    preds = {node: None for node in nodes}
    
    states: List[LayerState] = [_capture_state(nodes, values, preds, 0)]
    
    frontier: Set[str] = {source}
    iteration = 0
    
    while frontier:
        iteration += 1
        next_frontier: Set[str] = set()
        
        # Zpracuj všechny uzly z fronty
        for node in frontier:
            node_value = values[node]
            
            # Projdi všechny následníky tohoto uzlu
            for successor, edge_weight in successors.get(node, []):
                # Nová hodnota je minimum z cesty a hrany
                if math.isinf(node_value):
                    candidate = edge_weight
                else:
                    candidate = min(node_value, edge_weight)
                
                # U nejširší cesty chceme maximalizovat hodnotu
                if candidate > values[successor]:
                    values[successor] = candidate
                    preds[successor] = node
                    next_frontier.add(successor)
        
        if not next_frontier:
            break
        
        states.append(_capture_state(nodes, values, preds, iteration))
        frontier = next_frontier
        
        # Pokud máme cíl a dosáhli jsme ho, můžeme skončit
        if target and target in frontier:
            # Ještě jedna iterace pro uložení finálního stavu
            states.append(_capture_state(nodes, values, preds, iteration))
            break
    
    return BottleneckResult(
        mode="widest",
        source=source,
        target=target,
        values=values,
        predecessors=preds,
        states=states,
    )


def propagate_narrowest(graph: Graph, source: str, target: Optional[str]) -> BottleneckResult:
    nodes = sorted(graph.nodes.keys())
    predecessors = load_predecessors(graph)
    successors = predecessors_to_successors(predecessors, nodes)
    
    # Výchozí stav: zdroj má nekonečno, ostatní 0
    values = {node: 0.0 for node in nodes}
    values[source] = math.inf
    preds = {node: None for node in nodes}
    
    states: List[LayerState] = [_capture_state(nodes, values, preds, 0)]
    
    frontier: Set[str] = {source}
    iteration = 0
    
    while frontier:
        iteration += 1
        next_frontier: Set[str] = set()
        
        # Zpracuj všechny uzly z fronty
        for node in frontier:
            node_value = values[node]
            
            # Projdi všechny následníky tohoto uzlu
            for successor, edge_weight in successors.get(node, []):
                # Nová hodnota je minimum z cesty a hrany
                if math.isinf(node_value):
                    candidate = edge_weight
                else:
                    candidate = min(node_value, edge_weight)
                
                # U nejužší cesty chceme také vyšší hodnoty (širší bottleneck)
                if candidate > values[successor]:
                    values[successor] = candidate
                    preds[successor] = node
                    next_frontier.add(successor)
        
        if not next_frontier:
            break
        
        states.append(_capture_state(nodes, values, preds, iteration))
        frontier = next_frontier
        
        # Pokud máme cíl a dosáhli jsme ho, můžeme skončit
        if target and target in frontier:
            states.append(_capture_state(nodes, values, preds, iteration))
            break
    
    return BottleneckResult(
        mode="narrowest",
        source=source,
        target=target,
        values=values,
        predecessors=preds,
        states=states,
    )


def print_states(result: BottleneckResult) -> None:
    for state in result.states:
        label = "Výchozí stav" if state.iteration == 0 else f"Iterace {state.iteration}"
        print(f"{label} (zdroj {result.source})")
        for node in sorted(state.values.keys()):
            value = state.values[node]
            predecessor = state.predecessors[node]
            
            if math.isinf(value):
                value_str = "∞"
            else:
                value_str = str(value)
            
            prev_label = predecessor if predecessor is not None else "-"
            print(f"  ({value_str}, {prev_label}) -> {node}")
        print()


def run_bottleneck(
    graph: Graph,
    mode: str,
    source: str,
    target: Optional[str],
    verbose: bool = True,
) -> BottleneckResult:
    if source not in graph.nodes:
        raise ValueError(f"Start '{source}' není v grafu.")
    if target and target not in graph.nodes:
        raise ValueError(f"Cíl '{target}' není v grafu.")

    if mode == "widest":
        result = propagate_widest(graph, source, target)
    else:
        result = propagate_narrowest(graph, source, target)

    if verbose:
        print(f"\n=== {mode.upper()} průchod: {source} -> {target or '*'} ===")
        print_states(result)

        path = result.reconstruct_path()
        if target:
            if path:
                print(f"Cesta {source} -> {target}: {' -> '.join(path)}")
                print(f"Hodnota metriky: {result.values[target]}")
            else:
                print(f"Cesta {source} -> {target} se nepodařila nalézt.")

    return result


def interactive_mode(graph: Graph) -> None:
    """Interaktivní režim - ptá se uživatele na vstupy."""
    
    # Zobrazení dostupných uzlů
    nodes_list = sorted(graph.nodes.keys())
    print(f"\nDostupné uzly: {', '.join(nodes_list)}")
    
    # Režim
    print("\nVyberte režim:")
    print("  1 - Nejširší cesta (widest)")
    print("  2 - Nejužší cesta (narrowest)")
    while True:
        mode_input = input("Zadejte volbu (1/2): ").strip()
        if mode_input == "1":
            mode = "widest"
            break
        elif mode_input == "2":
            mode = "narrowest"
            break
        else:
            print("Neplatná volba. Zadejte 1 nebo 2.")
    
    # Zdrojový uzel
    print(f"\nDostupné uzly: {', '.join(nodes_list)}")
    while True:
        source = input("Zadejte zdrojový uzel: ").strip()
        if source in graph.nodes:
            break
        print(f"Uzel '{source}' není v grafu. Zkuste to znovu.")
    
    # Cílový uzel (volitelný)
    print(f"\nDostupné uzly: {', '.join(nodes_list)}")
    while True:
        target = input("Zadejte cílový uzel (nebo Enter pro všechny): ").strip()
        if not target:
            target = None
            break
        if target in graph.nodes:
            break
        print(f"Uzel '{target}' není v grafu. Zkuste to znovu.")
    
    # Verbose
    verbose_input = input("\nZobrazit detailní průběh? (ano/ne) [ano]: ").strip().lower()
    verbose = verbose_input != "ne"
    
    # Spuštění algoritmu
    run_bottleneck(graph, mode, source, target, verbose)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Vrstevnicový algoritmus pro nejširší / nejužší cestu."
    )
    parser.add_argument("file", nargs='?', help="Cesta k .tg souboru.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    
    # Pokud není zadán soubor, zeptáme se
    if not args.file:
        print("=== Vrstevnicový algoritmus pro bottleneck cesty ===\n")
        while True:
            file_path = input("Zadejte cestu k .tg souboru: ").strip()
            try:
                graph = parse_graph_file(file_path)
                break
            except FileNotFoundError:
                print(f"Soubor '{file_path}' nebyl nalezen. Zkuste to znovu.")
            except Exception as e:
                print(f"Chyba při načítání souboru: {e}")
                return
    else:
        # Pokud je zadán soubor jako argument, načteme ho
        try:
            graph = parse_graph_file(args.file)
        except FileNotFoundError:
            print(f"Soubor '{args.file}' nebyl nalezen.")
            return
        except Exception as e:
            print(f"Chyba při načítání souboru: {e}")
            return
    
    # Vždy se zeptáme na ostatní parametry interaktivně
    interactive_mode(graph)


if __name__ == "__main__":
    main()