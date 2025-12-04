# Graph Algorithms Collection

Tato složka obsahuje sadu Python skriptů pro analýzu grafů a spouštění různých grafových algoritmů. Skripty pracují s grafy definovanými v textových souborech (obvykle s příponou `.tg`).

## Požadavky

- Python 3.x
- Standardní knihovny Pythonu (není potřeba instalovat nic navíc, s výjimkou `numpy` pro `stromy_kostry.py`).

## Formát vstupního souboru (.tg)

Skripty očekávají textový soubor definující uzly a hrany:
- **Uzly:** `u <název_uzlu> [váha]`
- **Hrany:** `h <uzel1> <směr> <uzel2> [váha] [popis]`
  - Směr může být: `>` (orientovaná), `<` (orientovaná zpět), `-` (neorientovaná).

Příklad:
```text
u A 10
u B 20
h A > B 5 label_hrany
h B - C 2
```

## Seznam skriptů a použití

### 1. Analýza vlastností grafu (`properties.py`)
Zjišťuje základní vlastnosti grafu (souvislost, orientovanost, rovinnost, atd.).

**Použití:**
```bash
python properties.py soubor_grafu.tg
```

### 2. Tabulky a statistiky (`table.py`)
Generuje incidenční tabulky, seznamy sousedů a statistiky uzlů/hran. Výstupy ukládá do složky `csv_export/`.

**Použití:**
```bash
python table.py soubor_grafu.tg
```

### 3. Stromy a kostry (`stromy_kostry.py`)
Práce s binárními stromy (průchody, vkládání/mazání) a výpočet minimální kostry grafu (Kruskalův algoritmus).

**Použití:**
```bash
python stromy_kostry.py soubor_grafu.tg
```
*Skript se spustí v interaktivním režimu.*

### 4. Bottleneck cesty (`bottleneck_paths_updated.py`)
Hledá nejširší (widest) nebo nejužší (narrowest) cestu mezi dvěma uzly.
- **Widest:** Maximalizuje minimální kapacitu na cestě.
- **Narrowest:** Minimalizuje maximální kapacitu na cestě.

**Použití:**
```bash
python bottleneck_paths_updated.py [soubor_grafu.tg]
```
*Pokud soubor nezadáte, program si ho vyžádá. Následně se interaktivně ptá na zdrojový a cílový uzel.*

### 5. Maximální tok (`edmond_karp.py`)
Vypočítá maximální tok v síti pomocí Edmonds-Karpova algoritmu.

**Použití:**
```bash
python edmond_karp.py soubor_grafu.tg --source S --target T
```
*Pokud nezadáte source/target, použijí se výchozí 's' a 't'.*

### 6. Nejkratší a jiné cesty (`floyd_warshall.py`)
Univerzální nástroj pro hledání cest pomocí různých metrik (Floyd-Warshall, Dijkstra, BFS).
Podporuje metriky: Nejkratší, Nejdelší, Nejbezpečnější, Nejnebezpečnější, Nejširší, Nejužší.

**Použití:**
```bash
python floyd_warshall.py [soubor_grafu.tg]
```
*Skript je plně interaktivní. Po spuštění vyberete metriku a můžete se dotazovat na cesty mezi uzly.*

### 7. Nejdelší cesta (`longest_path.py`)
Hledá nejdelší jednoduchou cestu mezi dvěma uzly (pomocí BFS).

**Použití:**
```bash
python longest_path.py soubor_grafu.tg --source A --target B
```
*Volitelně lze nastavit `--max-depth`.*

### 8. Nejnebezpečnější cesta (`most_dangerous.py`)
Hledá cestu s maximálním součinem vah (např. pro pravděpodobnost selhání).

**Použití:**
```bash
python most_dangerous.py soubor_grafu.tg
```
*Interaktivně se zeptá na start a cíl.*

### 9. Nejbezpečnější cesta (`most_safest.py`)
Hledá cestu s minimálním rizikem (pravděpodobnostně).

**Použití:**
```bash
python most_safest.py soubor_grafu.tg
```

### 10. Maticové operace (`matrix.py`)
Provádí operace s maticemi grafu (sousednosti, incidence, atd.).

**Použití:**
```bash
python matrix.py soubor_grafu.tg
```

---
*Poznámka: Skripty `edmond_karp.py`, `floyd_warshall.py`, `longest_path.py`, `most_dangerous.py`, `most_safest.py` a `bottleneck_paths_updated.py` vyžadují, aby byl soubor `properties.py` ve stejné složce. Skript `bottleneck_paths_updated.py` navíc vyžaduje `nodes.py`. Ostatní skripty (`table.py`, `matrix.py`, `stromy_kostry.py`) jsou samostatné.*
