"""
Vrstevnicový algoritmus pro nejširší / nejužší cestu mezi dvěma uzly.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Set

from properties import Graph, parse_graph_file
from nodes import load_predecessors, predecessors_to_successors


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

        value = self.values.get(self.target)
        if value is None:
            return []
        
        # Pro widest: pokud hodnota je -inf, cesta neexistuje
        if self.mode == "widest" and value == -math.inf:
            return []
        # Pro narrowest: pokud hodnota je +inf, cesta neexistuje
        if self.mode == "narrowest" and value == math.inf:
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


def _capture_state(nodes, values, preds, iteration):
    return LayerState(
        iteration=iteration,
        values={node: values[node] for node in nodes},
        predecessors={node: preds[node] for node in nodes},
    )


def propagate_widest(
    graph: Graph, source: str, target: Optional[str]
) -> BottleneckResult:
    """
    Najde cestu s maximálním minimem (nejširší cestu).
    Hledáme cestu, kde je nejužší místo co nejširší.
    """
    nodes = sorted(graph.nodes.keys())

    predecessors = load_predecessors(graph)
    successors = predecessors_to_successors(predecessors, nodes)

    # Inicializace: všechny uzly kromě zdroje mají -inf
    values = {node: -math.inf for node in nodes}
    values[source] = math.inf
    preds = {node: None for node in nodes}
    processed = set()

    states = [_capture_state(nodes, values, preds, 0)]
    iteration = 0

    # KROK 1: Explicitně zpracuj zdrojový uzel
    processed.add(source)
    changed = False
    for succ, w in successors.get(source, []):
        if w > values[succ]:
            values[succ] = w
            preds[succ] = source
            changed = True
    
    if changed:
        iteration += 1
        states.append(_capture_state(nodes, values, preds, iteration))

    # KROK 2: Hlavní cyklus - zpracuj všechny dosažitelné uzly
    while True:
        # Najdi nevyřízený uzel s NEJVYŠŠÍ hodnotou (mimo -inf)
        best_node = None
        best_value = -math.inf
        
        for node in nodes:
            if node not in processed:
                if values[node] > best_value and values[node] > -math.inf:
                    best_node = node
                    best_value = values[node]
        
        # Pokud není co zpracovat, konec
        if best_node is None:
            break
            
        processed.add(best_node)
        
        # Relaxace všech následníků
        changed = False
        for succ, w in successors.get(best_node, []):
            if succ not in processed:  # Pouze nezpracované uzly
                # Candidate je minimum z aktuální cesty a hrany
                candidate = min(best_value, w)
                
                # Aktualizuj pokud jsme našli VYŠŠÍ hodnotu
                if candidate > values[succ]:
                    values[succ] = candidate
                    preds[succ] = best_node
                    changed = True
        
        if changed:
            iteration += 1
            states.append(_capture_state(nodes, values, preds, iteration))
            
        # Bezpečnostní limit - pokud je příliš mnoho iterací, zastav
        if iteration > len(nodes) * 2:
            break

    return BottleneckResult(
        mode="widest",
        source=source,
        target=target,
        values=values,
        predecessors=preds,
        states=states,
    )


def propagate_narrowest(
    graph: Graph, source: str, target: Optional[str]
) -> BottleneckResult:
    """
    Najde cestu s minimálním minimem (nejužší cestu).
    Hledáme cestu, kde je nejužší místo co nejužší.
    """
    nodes = sorted(graph.nodes.keys())

    predecessors = load_predecessors(graph)
    successors = predecessors_to_successors(predecessors, nodes)

    # Inicializace: všechny uzly kromě zdroje mají +inf
    values = {node: math.inf for node in nodes}
    values[source] = math.inf
    preds = {node: None for node in nodes}
    processed = set()

    states = [_capture_state(nodes, values, preds, 0)]
    iteration = 0

    # KROK 1: Explicitně zpracuj zdrojový uzel
    processed.add(source)
    changed = False
    for succ, w in successors.get(source, []):
        if w < values[succ]:
            values[succ] = w
            preds[succ] = source
            changed = True
    
    if changed:
        iteration += 1
        states.append(_capture_state(nodes, values, preds, iteration))

    # KROK 2: Hlavní cyklus - zpracuj všechny dosažitelné uzly
    while True:
        # Najdi nevyřízený uzel s NEJNIŽŠÍ hodnotou (mimo +inf)
        best_node = None
        best_value = math.inf
        
        for node in nodes:
            if node not in processed:
                if values[node] < best_value and values[node] < math.inf:
                    best_node = node
                    best_value = values[node]
        
        # Pokud není co zpracovat, konec
        if best_node is None:
            break
            
        processed.add(best_node)
        
        # Relaxace všech následníků
        changed = False
        for succ, w in successors.get(best_node, []):
            if succ not in processed:  # Pouze nezpracované uzly
                # Candidate je minimum z aktuální cesty a hrany
                candidate = min(best_value, w)
                
                # Aktualizuj pokud jsme našli NIŽŠÍ hodnotu
                if candidate < values[succ]:
                    values[succ] = candidate
                    preds[succ] = best_node
                    changed = True
        
        if changed:
            iteration += 1
            states.append(_capture_state(nodes, values, preds, iteration))
            
        # Bezpečnostní limit - pokud je příliš mnoho iterací, zastav
        if iteration > len(nodes) * 2:
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
                value_str = "∞" if value > 0 else "-∞"
            else:
                value_str = str(value)
            prev_label = predecessor if predecessor is not None else "-"
            print(f"  ({value_str}, {prev_label}) -> {node}")
        print()


def run_bottleneck(graph, mode, source, target, verbose=True):
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
    print(" 1 - Nejširší cesta (widest)")
    print(" 2 - Nejužší cesta (narrowest)")

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
    while True:
        source = input("\nZadejte zdrojový uzel: ").strip()
        if source in graph.nodes:
            break
        print(f"Uzel '{source}' není v grafu. Zkuste to znovu.")

    # Načti předchůdce a následníky
    predecessors = load_predecessors(graph)
    successors_dict = predecessors_to_successors(predecessors, graph.nodes)

    # Výpis následníků zdrojového uzlu
    successors = [nbr for nbr, _ in successors_dict.get(source, [])]
    if successors:
        print(f"\nNásledníci uzlu '{source}': {', '.join(successors)}")
    else:
        print(f"\nUzel '{source}' nemá žádné následníky.")

    # Cílový uzel
    while True:
        target = input("\nZadejte cílový uzel: ").strip()
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
        # Pokud je soubor zadán jako argument
        try:
            graph = parse_graph_file(args.file)
        except FileNotFoundError:
            print(f"Soubor '{args.file}' nebyl nalezen.")
            return
        except Exception as e:
            print(f"Chyba při načítání souboru: {e}")
            return

    # Vždy interaktivní režim na zbytek parametrů
    interactive_mode(graph)


if __name__ == "__main__":
    main()