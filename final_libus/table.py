#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import csv
import os
from collections import defaultdict
from statistics import mean, median, variance, stdev


class Graph:
    def __init__(self):
        self.nodes = []
        self.edges = []  # každý prvek: (node1, node2, direction, weight, label)
        self.node_order = []  # Zachování pořadí uzlů ze souboru
        self.edge_labels = {}

    def add_node(self, node_id, weight=None):
        """Přidá uzel do grafu"""
        node_id = node_id.rstrip(';')
        if node_id != '*' and node_id not in self.nodes:
            self.nodes.append(node_id)
            self.node_order.append(node_id)

    def add_edge(self, node1, node2, direction, weight=None, label=None):
        """Přidá hranu do grafu"""
        node1 = node1.strip().rstrip(';')
        node2 = node2.strip().rstrip(';')

        if label and label.strip():
            label = label.strip().rstrip(';')
        else:
            label = f"h{node1}{node2}"

        if node1 not in self.nodes:
            self.add_node(node1)
        if node2 not in self.nodes:
            self.add_node(node2)

        self.edges.append((node1, node2, direction, weight, label))
        self.edge_labels[(node1, node2, direction)] = label
        return True


def parse_graph_file(filename):
    graph = Graph()
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('u '):
                parts = line.split()
                node_id = parts[1]
                weight = float(parts[2].rstrip(';')) if len(parts) > 2 else None
                graph.add_node(node_id, weight)
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
                graph.add_edge(node1, node2, direction, weight, label)
    return graph


def incident_tables(graph):
    outgoing = {node: set() for node in graph.nodes}
    incoming = {node: set() for node in graph.nodes}
    unified = {}

    for node1, node2, direction, weight, label in graph.edges:
        if not label:
            label = f"h{node1}{node2}"
        if direction == '>':
            outgoing[node1].add(label)
            incoming[node2].add(label)
        elif direction == '<':
            outgoing[node2].add(label)
            incoming[node1].add(label)
        else:
            outgoing[node1].add(label)
            outgoing[node2].add(label)
            incoming[node1].add(label)
            incoming[node2].add(label)

    for node in graph.nodes:
        unified[node] = sorted(outgoing[node] | incoming[node])

    outgoing = {n: sorted(e) for n, e in outgoing.items()}
    incoming = {n: sorted(e) for n, e in incoming.items()}
    return outgoing, incoming, unified


def neighbor_list(graph):
    neighbors = defaultdict(set)
    for node1, node2, direction, weight, label in graph.edges:
        neighbors[node1].add(node2)
        if direction in ['-', '<']:
            neighbors[node2].add(node1)
    return {n: sorted(list(v)) for n, v in neighbors.items()}


def stats_from_table(table):
    counts = [len(v) for v in table.values()]
    if not counts:
        return {"Počet uzlů": 0}
    return {
        "Počet uzlů": len(table),
        "Uzel s nejvíce položkami": max(table, key=lambda n: len(table[n]), default=None),
        "Max počet položek": max(counts),
        "Počet uzlů s 0 položkami": sum(1 for c in counts if c == 0),
        "Počet uzlů s 1 položkou": sum(1 for c in counts if c == 1),
        "Průměr": round(mean(counts), 2),
        "Medián": median(counts),
        "Rozptyl": round(variance(counts), 2) if len(counts) > 1 else 0,
        "Směrodatná odchylka": round(stdev(counts), 2) if len(counts) > 1 else 0,
        "Celkem položek": sum(counts)
    }


def stats_for_nodes_and_edges(graph):
    node_count = len(graph.nodes)
    edge_count = len(graph.edges)
    name_lengths = [len(n) for n in graph.nodes]
    edge_desc_lengths = [len(f"{n1} {d} {n2}") for n1, n2, d, *_ in graph.edges]
    return {
        "Počet uzlů": node_count,
        "Počet hran": edge_count,
        "Poměr hran/uzlů": round(edge_count / node_count, 2) if node_count else 0,
        "Průměrný stupeň uzlu": round(2 * edge_count / node_count, 2) if node_count else 0,
        "Nejkratší název uzlu": min(name_lengths) if name_lengths else 0,
        "Nejdelší název uzlu": max(name_lengths) if name_lengths else 0,
        "Průměrná délka názvu uzlu": round(mean(name_lengths), 2) if name_lengths else 0,
        "Medián délky názvu uzlu": median(name_lengths) if name_lengths else 0,
        "Rozptyl délky názvů uzlů": round(variance(name_lengths), 2) if len(name_lengths) > 1 else 0,
        "Směrodatná odchylka názvů uzlů": round(stdev(name_lengths), 2) if len(name_lengths) > 1 else 0,
        "Průměrná délka popisu hrany": round(mean(edge_desc_lengths), 2) if edge_desc_lengths else 0,
    }


def print_stats(stats, title):
    print(f"\n📊 Statistiky – {title}")
    for k, v in stats.items():
        print(f"  {k}: {v}")


def export_incident_tables(outgoing, incoming, unified, base_dir="."):
    filename = os.path.join(base_dir, "incidentni_hrany.csv")
    os.makedirs(base_dir, exist_ok=True)
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=';')
        for section, data in [
            ("VYSTUPUJÍCÍ HRANY", outgoing),
            ("VSTUPUJÍCÍ HRANY", incoming),
            ("SJEDNOCENÉ HRANY", unified),
        ]:
            writer.writerow([f"--- {section} ---"])
            writer.writerow(["Uzel", "Hrany"])
            for n, e in sorted(data.items()):
                writer.writerow([n, ", ".join(e) if e else "-"])
            writer.writerow([])
        writer.writerow(["--- STATISTIKY ---"])
        for title, data in [
            ("Vystupující hrany", stats_from_table(outgoing)),
            ("Vstupující hrany", stats_from_table(incoming)),
            ("Sjednocené hrany", stats_from_table(unified))
        ]:
            writer.writerow([title])
            for k, v in data.items():
                writer.writerow([k, v])
            writer.writerow([])
    print(f"Soubor uložen: {filename}")


def export_neighbors(neighbors, base_dir="."):
    filename = os.path.join(base_dir, "sousede.csv")
    os.makedirs(base_dir, exist_ok=True)
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(["Uzel", "Sousedi"])
        for n, e in sorted(neighbors.items()):
            writer.writerow([n, ", ".join(e) if e else "-"])
        stats = stats_from_table(neighbors)
        writer.writerow([])
        writer.writerow(["--- STATISTIKY ---"])
        for k, v in stats.items():
            writer.writerow([k, v])
    print(f"Soubor uložen: {filename}")


def export_nodes_and_edges(graph, base_dir="."):
    filename = os.path.join(base_dir, "uzly_a_hrany.csv")
    os.makedirs(base_dir, exist_ok=True)
    stats = stats_for_nodes_and_edges(graph)
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(["--- SEZNAM UZLŮ ---"])
        writer.writerow(["Uzel"])
        for n in sorted(graph.nodes):
            writer.writerow([n])
        writer.writerow([])
        writer.writerow(["--- SEZNAM HRAN ---"])
        writer.writerow(["Hrana", "Spojení"])
        for node1, node2, direction, weight, label in graph.edges:
            writer.writerow([label, f"{node1} {direction} {node2}"])
        writer.writerow([])
        writer.writerow(["--- STATISTIKY ---"])
        for k, v in stats.items():
            writer.writerow([k, v])
    print(f"Soubor uložen: {filename}")


def main():
    if len(sys.argv) != 2:
        print("Použití: python script.py <graf.txt>")
        sys.exit(1)

    filename = sys.argv[1]
    graph = parse_graph_file(filename)
    os.makedirs("csv_export", exist_ok=True)

    outgoing, incoming, unified = incident_tables(graph)
    neighbors = neighbor_list(graph)

    export_incident_tables(outgoing, incoming, unified, base_dir="csv_export")
    export_neighbors(neighbors, base_dir="csv_export")
    export_nodes_and_edges(graph, base_dir="csv_export")

    print_stats(stats_from_table(outgoing), "Vystupující hrany")
    print_stats(stats_from_table(incoming), "Vstupující hrany")
    print_stats(stats_from_table(unified), "Sjednocené hrany")
    print_stats(stats_from_table(neighbors), "Sousedi")
    print_stats(stats_for_nodes_and_edges(graph), "Seznam uzlů a hran")


if __name__ == "__main__":
    main()
