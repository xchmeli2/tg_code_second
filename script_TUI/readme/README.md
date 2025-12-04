# Dokumentace Grafové Aplikace

Tato aplikace slouží k načítání, analýze a zpracování grafů. Umožňuje provádět různé maticové operace, zjišťovat vlastnosti grafů a hledat cesty. Grafy jsou načítány ze statických souborů.

## Instalace a Spuštění

1.  Ujistěte se, že máte nainstalovaný Python.
2.  Připravte si soubory s definicí grafů ve složce `graphs` (podporované formáty `.txt` nebo `.tg`).
3.  Spusťte aplikaci příkazem:
    ```bash
    python main.py
    ```

## Použití Aplikace

Po spuštění aplikace budete vyzváni k výběru souboru s grafem ze složky `graphs`. Poté se zobrazí hlavní menu s následujícími možnostmi:

### Hlavní Menu

1.  **Graph Properties (Vlastnosti Grafu)**
    *   Zobrazí základní vlastnosti načteného grafu (zda je ohodnocený, orientovaný, souvislý, atd.).
2.  **Enter a node name to get its properties (Vlastnosti Uzlu)**
    *   Po zadání názvu uzlu zobrazí jeho specifické vlastnosti (následníci, předchůdci, stupně, atd.).
3.  **Matrix Operations (Maticové Operace a Algoritmy)**
    *   Otevře podmenu s pokročilými operacemi a algoritmy (viz níže).
4.  **Exit (Ukončit)**
    *   Ukončí aplikaci.

### Menu Maticových Operací a Algoritmů

Zde jsou dostupné funkce, které můžete volat (vybrat z menu). Pokud je název v aplikaci anglicky, je zde uveden český ekvivalent.

| Volba | Název v menu / Funkce | Popis (Česky) |
| :--- | :--- | :--- |
| 1 | **Adjacency Matrix** | **Matice sousednosti**: Zobrazí nebo uloží matici reprezentující sousednost uzlů. |
| 2 | **Incidence Matrix** | **Matice incidence**: Zobrazí nebo uloží matici reprezentující vztahy mezi uzly a hranami. |
| 3 | **Second Power of Adjacency Matrix** | **Druhá mocnina matice sousednosti**: Spočítá A^2 (počet sledů délky 2). |
| 4 | **Third Power of Adjacency Matrix** | **Třetí mocnina matice sousednosti**: Spočítá A^3 (počet sledů délky 3). |
| 5 | **Custom Power of Adjacency Matrix** | **Vlastní mocnina matice sousednosti**: Spočítá A^n pro zadané n. |
| 6 | **Počet koster grafu** | Spočítá celkový počet koster v grafu. |
| 7 | **Minimální kostra grafu** | Najde kostru grafu s minimálním součtem vah hran (např. Kruskalův/Primův algoritmus). |
| 8 | **Maximální kostra grafu** | Najde kostru grafu s maximálním součtem vah hran. |
| 9 | **Nejkratší cesta v grafu** | Najde nejkratší cestu mezi dvěma uzly (podle vah). |
| 10 | **Nejdelší cesta v grafu** | Najde nejdelší cestu mezi dvěma uzly. |
| 11 | **Nejbezpečnější cesta v grafu** | Najde cestu s nejmenším rizikem (nebo největší pravděpodobností úspěchu). |
| 12 | **Nejširší cesta v grafu** | Najde cestu s maximální "šířkou" (úzkým hrdlem/kapacitou). |
| 13 | **Maximalni tah** | **Maximální tok**: Spočítá maximální tok v síti ze zdroje do stoku. |
| 14 | **DFS Traversal** | **Průchod do hloubky**: Projde graf metodou Depth-First Search. |
| 15 | **BFS Traversal** | **Průchod do šířky**: Projde graf metodou Breadth-First Search. |
| 16 | **pre** | **Preorder průchod**: Vypíše uzly v pořadí preorder (pro stromy/DAG). |
| 17 | **post** | **Postorder průchod**: Vypíše uzly v pořadí postorder. |
| 18 | **in** | **Inorder průchod**: Vypíše uzly v pořadí inorder. |

## Programátorské Rozhraní (API)

Pokud chcete používat funkce přímo v kódu (Python), zde je přehled hlavních funkcí s českým popisem.

### Modul `properties.py` (Vlastnosti Grafu)
*   `is_weighted(graph)`: Je graf ohodnocený?
*   `is_directed(graph)`: Je graf orientovaný?
*   `is_connected(graph)`: Je graf souvislý?
*   `jednoduchy(graph)`: Je graf jednoduchý (bez smyček a násobných hran)?
*   `is_finite(graph)`: Je graf konečný?
*   `is_complete(graph)`: Je graf úplný?
*   `is_regular(graph)`: Je graf regulární?
*   `is_bipartite(graph)`: Je graf bipartitní?
*   `prosty(graph)`: Je graf prostý?
*   `is_planar(graph)`: Je graf rovinný?

### Modul `graph.py` (Vlastnosti Uzlů)
*   `naslednici_uzlu(graph, node)`: Vrátí následníky uzlu.
*   `predchdci_uzlu(graph, node)`: Vrátí předchůdce uzlu.
*   `sousedi_uzlu(graph, node)`: Vrátí sousedy uzlu.
*   `stupen_uzlu(graph, node)`: Vrátí stupeň uzlu.

### Modul `matrix_operations.py` (Matice)
*   `adjacency_matrix(graph)`: Vytvoří matici sousednosti.
*   `incidence_matrix(graph)`: Vytvoří matici incidence.
*   `matrix_power(matrix, power)`: Umocní matici.

### Modul `test_sec.py` (Algoritmy)
*   `number_of_spanning_trees(graph)`: Počet koster.
*   `minimum_spanning_tree(graph)`: Minimální kostra.
*   `maximum_spanning_tree(graph)`: Maximální kostra.
*   `get_shortest_path(graph, start, end)`: Nejkratší cesta.
*   `get_longest_path(graph, start, end)`: Nejdelší cesta.
*   `get_safest_path(graph, start, end)`: Nejbezpečnější cesta.
*   `get_widest_path(graph, start, end)`: Nejširší cesta.
*   `maximal_flow(graph, source, sink)`: Maximální tok.
*   `dfs(graph, start)`: DFS průchod.
*   `bfs_traversal(graph, start)`: BFS průchod.
