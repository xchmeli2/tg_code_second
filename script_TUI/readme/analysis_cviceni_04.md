# Analýza požadavků pro Cvičení 4 (Optimální sledy)

Na základě poskytnutého PDF (ENC-TG • 2024 • 04) a analýzy současného kódu jsem připravil přehled funkcí.

## 1. Optimální cesty v grafu (Úloha 1)

Cílem je najít různé typy "optimálních" cest mezi uzly.

| Typ cesty | Stav | Implementace | Poznámka |
| :--- | :--- | :--- | :--- |
| **Nejkratší cesta** | ✅ Implementováno | `get_shortest_path` (Floyd-Warshall) | Hledá cestu s minimálním součtem vah. |
| **Nejdelší cesta** | ✅ Implementováno | `get_longest_path` (Floyd-Warshall) | Hledá cestu s maximálním součtem vah. **Pozor:** Floyd-Warshall pro nejdelší cestu funguje jen pokud nejsou v grafu kladné cykly (což u hledání nejdelší cesty odpovídá "negativním cyklům" u nejkratší). Pro obecné grafy s cykly je to NP-těžké, ale máme i `longest_path_with_cycles` (DFS), která to řeší hrubou silou. |
| **Nejbezpečnější cesta** | ✅ Implementováno | `get_safest_path` (Floyd-Warshall) | Hledá cestu s minimálním rizikem (multiplikativní nebo aditivní, v kódu je aditivní `min(sum(risk))`). |
| **Nejširší cesta** | ✅ Implementováno | `get_widest_path` (Floyd-Warshall) | Hledá cestu s maximální kapacitou úzkého hrdla (`max(min(weights))`). |

**Doporučení:** Ověřit funkčnost na grafech se zápornými vahami (zmíněno v zadání "2 2 -3"). Floyd-Warshall zvládne záporné hrany, pokud nejsou v záporném cyklu.

## 2. Síťové grafy a CPM (Úlohy 2 a 3)

Cílem je sestrojit síťový graf z tabulky činností, najít kritickou cestu a rezervy.

| Funkce / Pojem | Stav | Poznámka |
| :--- | :--- | :--- |
| **Načtení tabulky činností** | ❌ Chybí | Aplikace umí číst jen grafy (uzly/hrany). Úlohy 2 a 3 zadávají data jako tabulku činností (ID, předchůdci, trvání). Potřebujeme parser pro tento formát. |
| **Sestrojení síťového grafu** | ❌ Chybí | Převod tabulky činností na graf (uzly = události nebo činnosti?). Metoda CPM obvykle používá hranově ohodnocený graf (Activity on Arrow) nebo uzlově ohodnocený (Activity on Node). Zadání "Sestrojte síťový graf" naznačuje potřebu vizualizace nebo reprezentace. |
| **Kritická cesta (CPM)** | ❌ Chybí | Výpočet nejdřívějších/nejpozdějších začátků a konců (ES, EF, LS, LF). |
| **Rezervy (Slack/Float)** | ❌ Chybí | Výpočet celkové a volné rezervy pro každou činnost. |

---

## Doporučený plán implementace

1.  **Modul pro CPM (Critical Path Method)**:
    *   Vytvořit novou třídu/modul pro zpracování projektů.
    *   Funkce pro zadání činností (interaktivně nebo ze souboru - doporučuji CSV/TXT formát pro tabulku).
    *   Implementace výpočtu CPM (Forward pass, Backward pass).
    *   Výpis kritické cesty a rezerv.

2.  **Rozšíření Menu**:
    *   Přidat do hlavního menu sekci "Project Management / CPM".

3.  **Ověření cest**:
    *   Otestovat stávající algoritmy na grafech ze cvičení (včetně záporných vah).
