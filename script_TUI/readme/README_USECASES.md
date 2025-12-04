# Reálné Využití Grafových Algoritmů (Use Cases)

Tento dokument popisuje praktické využití algoritmů a funkcí implementovaných v této aplikaci. Grafové algoritmy jsou klíčové pro řešení mnoha problémů v reálném světě, od navigace a logistiky až po analýzu sociálních sítí a plánování projektů.

## 1. Vlastnosti Grafu a Uzlů

Tyto funkce slouží k základní analýze struktury sítě.

*   **`is_connected` (Je graf souvislý?)**:
    *   **Telekomunikace**: Ověření, zda v síti neexistují izolované ostrovy, které by nemohly komunikovat se zbytkem sítě.
    *   **Sociální sítě**: Analýza komunit – zda je skupina lidí propojená, nebo se skládá z oddělených podskupin.

*   **`is_directed` (Je graf orientovaný?)**:
    *   **Doprava**: Rozlišení mezi jednosměrnými (orientovaný) a obousměrnými (neorientovaný) ulicemi ve městě.
    *   **Web**: Odkazy mezi webovými stránkami jsou orientované (stránka A odkazuje na B, ale ne nutně naopak).

*   **`stupen_uzlu` (Stupeň uzlu)**:
    *   **Sociální sítě**: Identifikace "influencerů" nebo klíčových osobností (uzly s nejvyšším stupněm mají nejvíce kontaktů).
    *   **Epidemiologie**: Identifikace "super-přenašečů" v síti kontaktů při šíření nemoci.

## 2. Kostry Grafu (Spanning Trees)

Kostra grafu propojuje všechny uzly s minimálním počtem hran (bez cyklů).

*   **`minimum_spanning_tree` (Minimální kostra)**:
    *   **Algoritmus**: Kruskalův algoritmus.
    *   **Infrastruktura**: Návrh nejlevnější sítě pro propojení měst (elektrické vedení, vodovod, internetové kabely), kde váha hrany představuje cenu výstavby. Cílem je propojit všechna místa s minimálními celkovými náklady.
    *   **Clusterová analýza**: Využívá se v datech pro shlukování podobných objektů.

*   **`maximum_spanning_tree` (Maximální kostra)**:
    *   **Algoritmus**: Kruskalův algoritmus (modifikovaný pro maximální váhy).
    *   **Spolehlivost sítí**: Pokud váha hrany představuje spolehlivost spojení, maximální kostra najde páteřní síť s nejvyšší celkovou spolehlivostí.

*   **`number_of_spanning_trees` (Počet koster)**:
    *   **Analýza robustnosti**: Čím více koster graf má, tím více existuje alternativních způsobů propojení, což může indikovat vyšší robustnost sítě proti výpadkům.

## 3. Hledání Cest (Pathfinding)

Hledání optimální cesty mezi dvěma body je jedním z nejčastějších problémů.

*   **`get_shortest_path` (Nejkratší cesta)**:
    *   **Algoritmus**: Floyd-Warshallův algoritmus.
    *   **Navigace (GPS)**: Nalezení nejrychlejší nebo nejkratší trasy z bodu A do bodu B na mapě. Váha hrany je vzdálenost nebo čas.
    *   **Počítačové sítě**: Směrování datových paketů (routing) nejkratší cestou k cíli pro minimalizaci zpoždění (latence).

*   **`get_longest_path` (Nejdelší cesta)**:
    *   **Algoritmus**: Floyd-Warshallův algoritmus (modifikovaný).
    *   **Projektové řízení (CPM/PERT)**: Nalezení tzv. "kritické cesty" v harmonogramu projektu. Určuje minimální dobu potřebnou k dokončení projektu – zpoždění jakékoli činnosti na této cestě zpozdí celý projekt.

*   **`get_safest_path` (Nejbezpečnější cesta)**:
    *   **Algoritmus**: Floyd-Warshallův algoritmus (modifikovaný).
    *   **Logistika cenností**: Transport peněz nebo nebezpečného materiálu, kde váha hrany představuje riziko přepadení nebo nehody. Hledáme cestu s minimálním kumulativním rizikem.
    *   **Směrování v nespolehlivých sítích**: Hledání cesty s nejmenší pravděpodobností ztráty paketu.

*   **`get_widest_path` (Nejširší cesta)**:
    *   **Algoritmus**: Floyd-Warshallův algoritmus (modifikovaný).
    *   **Datové přenosy**: Hledání cesty s maximální propustností (bandwidth). "Šířka" cesty je určena jejím nejužším hrdlem (hranou s nejmenší kapacitou). Užitečné pro streamování videa nebo stahování velkých souborů.
    *   **Doprava nadrozměrných nákladů**: Plánování trasy pro vozidlo, které potřebuje určitou minimální šířku silnice nebo výšku podjezdu.

## 4. Toky v Sítích (Network Flows)

*   **`maximal_flow` (Maximální tok)**:
    *   **Algoritmus**: Edmonds-Karpův algoritmus.
    *   **Doprava**: Určení maximální kapacity silniční sítě – kolik aut může projet z města A do města B za hodinu.
    *   **Distribuční sítě**: Maximální množství ropy, vody nebo plynu, které lze přepravit potrubní sítí.
    *   **Logistika**: Maximální počet balíků, které lze doručit distribuční sítí za den.

*   **`minimum_cut` (Minimální řez)**:
    *   **Algoritmus**: Edmonds-Karpův algoritmus (založeno na maximálním toku).
    *   **Identifikace úzkých hrdel**: Nalezení kritických spojení, jejichž přerušení by rozdělilo síť nebo omezilo tok.
    *   **Bezpečnost sítí**: Identifikace nejzranitelnějších míst v síti.


## 5. Průchody Grafem (Traversals)

*   **`bfs_traversal` (Prohledávání do šířky - BFS)**:
    *   **Sociální sítě**: Hledání lidí ve vzdálenosti "k" přátel (přátelé přátel).
    *   **Vysílání (Broadcasting)**: Šíření informace v síti po vrstvách (nejprve sousedům, pak sousedům sousedů).
    *   **Hledání nejbližšího bodu zájmu**: Např. nejbližší nemocnice nebo policejní stanice na mapě.

*   **`dfs_traversal` (Prohledávání do hloubky - DFS)**:
    *   **Řešení bludišť**: Hledání cesty ven z bludiště.
    *   **Analýza závislostí**: Zjištění, v jakém pořadí instalovat softwarové balíčky (topologické uspořádání).

## 6. Maticové Operace

*   **`adjacency_matrix` (Matice sousednosti) & `incidence_matrix` (Matice incidence)**:
    *   **Reprezentace dat**: Standardní formát pro vstup do jiných analytických nástrojů, strojového učení nebo pro uložení struktury grafu do databáze.

*   **`matrix_power` (Mocnění matice)**:
    *   **Analýza dosahu**: $A^k$ (k-tá mocnina matice sousednosti) nám říká, kolik existuje cest délky přesně $k$ mezi libovolnými dvěma uzly. To je užitečné pro analýzu konektivity a vlivu v sociálních sítích.
