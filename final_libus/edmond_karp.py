from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple
import argparse
import os
import csv
from properties import Graph, parse_graph_file

@dataclass
class EdgeRecord:
  start: str
  end: str
  capacity: float
  flow: float = 0.0

@dataclass
class EKSnapshot:
  step: int
  path: List[str]
  flow_added: float
  residual_matrix: List[List[float]]

@dataclass
class EKResult:
  nodes: List[str]
  edges: List[EdgeRecord]
  max_flow: float
  snapshots: List[EKSnapshot]

def graph_to_edge_matrix(graph: Graph) -> Tuple[List[str], List[List[float]]]:
  nodes = sorted(graph.nodes.keys())
  idx = {node: i for i, node in enumerate(nodes)}
  n = len(nodes)
  cap_matrix = [[0.0]*n for _ in range(n)]
  
  for node1, node2, direction, weight, label in graph.edges:
    w = 1.0 if weight is None else weight
    i, j = idx[node1], idx[node2]
    cap_matrix[i][j] = w
    
  if direction == "-" or direction == "=":
    cap_matrix[j][i] = w
  return nodes, cap_matrix

def bfs(residual: List[List[float]], source_idx: int, sink_idx: int) -> Optional[List[int]]:
  n = len(residual)
  parent = [-1]*n
  visited = [False]*n
  queue = deque([source_idx])
  visited[source_idx] = True

  while queue:
      u = queue.popleft()
      for v in range(n):
          if not visited[v] and residual[u][v] > 0:
              visited[v] = True
              parent[v] = u
              if v == sink_idx:
                  # Sestavení cesty
                  path = []
                  cur = sink_idx
                  while cur != -1:
                      path.append(cur)
                      cur = parent[cur]
                  return list(reversed(path))
              queue.append(v)
  return None

def edmonds_karp(graph: Graph, source: str, sink: str, verbose: bool = True) -> EKResult:
  nodes, cap_matrix = graph_to_edge_matrix(graph)
  idx_map = {node: i for i, node in enumerate(nodes)}
  n = len(nodes)
  residual = deepcopy(cap_matrix)
  snapshots: List[EKSnapshot] = []
  max_flow = 0.0

  source_idx, sink_idx = idx_map[source], idx_map[sink]
  step = 0

  while True:
      path_idx = bfs(residual, source_idx, sink_idx)
      if not path_idx:
          break
      path_flow = min(residual[path_idx[i]][path_idx[i+1]] for i in range(len(path_idx)-1))
      for i in range(len(path_idx)-1):
          u, v = path_idx[i], path_idx[i+1]
          residual[u][v] -= path_flow
          residual[v][u] += path_flow
      max_flow += path_flow
      step += 1
      # Statistiky uzlů: součet residual kapacit do a z uzlu
      node_stats = {nodes[i]: sum(residual[i]) + sum(residual[j][i] for j in range(n)) for i in range(n)}
      snapshots.append(EKSnapshot(
          step=step,
          path=[nodes[i] for i in path_idx],
          flow_added=path_flow,
          residual_matrix=deepcopy(residual)
      ))
      if verbose:
          print(f"Step {step}: augmenting path = {[nodes[i] for i in path_idx]}, flow added = {path_flow}")
          print(f"  Node stats (sum in+out residual): {node_stats}")

  edges: List[EdgeRecord] = []
  for i, u in enumerate(nodes):
      for j, v in enumerate(nodes):
          if cap_matrix[i][j] > 0:
              edges.append(EdgeRecord(u, v, cap_matrix[i][j], cap_matrix[i][j]-residual[i][j]))

  return EKResult(
      nodes=nodes,
      edges=edges,
      max_flow=max_flow,
      snapshots=snapshots
  )

def prompt_ek_query(result: EKResult) -> None:
  print(f"\nMax flow mezi uzly: {result.max_flow:.2f}")
  print("Průběžné augmentační cesty:")
  for snap in result.snapshots:
    print(f"  Iterace {snap.step}: cesta = {' -> '.join(snap.path)}, flow added = {snap.flow_added:.2f}")

def save_ek_result_csv(result: EKResult, filename: str) -> None:
  # Otevřeme soubor pro zápis
  with open(filename, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    # Hlavička CSV
    writer.writerow(["start", "end", "capacity", "flow"])
    # Zápis hran
    for edge in result.edges:
      writer.writerow([edge.start, edge.end, edge.capacity, edge.flow])

def run_ek_cli(args: argparse.Namespace) -> None:
  graph = parse_graph_file(args.file)
  nodes = sorted(graph.nodes.keys())
  source = args.source if args.source else 's'
  sink = args.target if args.target else 't'

  if source not in nodes or sink not in nodes:
      print(f"Chybí požadované uzly 's' nebo 't' v grafu: {', '.join(nodes)}")
      return

  result = edmonds_karp(graph, source, sink, verbose=not args.quiet)
  prompt_ek_query(result)

  # Uložit finální graf do CSV pod pevným názvem
  csv_file = "edmonds_karp.csv"
  save_ek_result_csv(result, csv_file)

def build_ek_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description="Interaktivní Edmonds-Karp algoritmus (max flow).")
  parser.add_argument("file", help="Cesta k .tg souboru.")
  parser.add_argument("--source", help="Zdrojový uzel")
  parser.add_argument("--target", help="Cílový uzel")
  parser.add_argument("--quiet", action="store_true", help="Potlačí průběžný výpis augmentačních cest")
  return parser

def main() -> None:
  parser = build_ek_parser()
  args = parser.parse_args()
  run_ek_cli(args)

if __name__ == "__main__":
    main()

