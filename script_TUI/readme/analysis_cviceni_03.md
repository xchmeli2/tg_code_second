# Analýza požadavků pro Cvičení 3 (Stromy a kostry)

Na základě poskytnutého PDF (ENC-TG • 2024 • 03) a analýzy současného kódu jsem připravil přehled funkcí, které jsou již implementovány, a těch, které je potřeba doplnit.

## 1. Prohledávání a průchody (Traversals)

| Funkce / Pojem | Stav | Poznámka |
| :--- | :--- | :--- |
| **Prohledávání do hloubky (DFS)** | ✅ Implementováno | Funkce `dfs` v `test_sec.py`. |
| **Prohledávání do šířky (BFS)** | ✅ Implementováno | Funkce `bfs_traversal` v `test_sec.py`. |
| **Level-order** | ⚠️ Částečně | Level-order je v podstatě BFS. Máme `bfs_traversal`, což pro stromy odpovídá level-order. Není ale explicitně pojmenováno jako "level-order". |
| **Pre-order** | ✅ Implementováno | Funkce `preorder` v `test_sec.py`. |
| **Post-order** | ✅ Implementováno | Funkce `postorder` v `test_sec.py`. |
| **In-order** | ✅ Implementováno | Funkce `inorder` v `test_sec.py`. |

## 2. Binární vyhledávací strom (BST)

| Funkce / Pojem | Stav | Poznámka |
| :--- | :--- | :--- |
| **Vložení hodnot do BST** | 🟢 Není potřeba | Graf je definován staticky na začátku (ze souboru), interaktivní vkládání není vyžadováno. |
| **Odebrání hodnot z BST** | 🟢 Není potřeba | Graf je definován staticky, odebírání uzlů není vyžadováno. |

## 3. Kostry grafu (Spanning Trees)

| Funkce / Pojem | Stav | Poznámka |
| :--- | :--- | :--- |
| **Počet koster grafu** | ✅ Implementováno | Funkce `number_of_spanning_trees` (využívá Laplaceovu matici a determinant). |
| **Minimální kostra (MST)** | ✅ Implementováno (Kruskal) | Funkce `minimum_spanning_tree` existuje, ale implementuje pouze **Kruskalův algoritmus**. |
| **Jarníkův-Primův algoritmus** | ❌ Chybí | Cvičení vyžaduje explicitně i tento algoritmus (úloha 6a). |
| **Borůvkův-Sollinův algoritmus** | ❌ Chybí | Cvičení vyžaduje explicitně i tento algoritmus (úloha 6c). |
| **Maximální kostra** | ✅ Implementováno | Funkce `maximum_spanning_tree` (upravený Kruskal). |

## 4. Ostatní pojmy

| Funkce / Pojem | Stav | Poznámka |
| :--- | :--- | :--- |
| **Laplaceova matice** | ✅ Implementováno | Funkce `laplacian_matrix` existuje (používá se pro výpočet počtu koster). |

---

## Doporučený plán implementace

Pro splnění požadavků cvičení je potřeba doplnit následující:

1.  **Rozšíření hledání minimální kostry**:
    *   Implementovat **Jarníkův-Primův algoritmus**.
    *   Implementovat **Borůvkův-Sollinův algoritmus**.
    *   Umožnit uživateli v menu vybrat, který algoritmus chce použít.

2.  **Práce s BST (volitelné/dle potřeby)**:
    *   Pokud je cílem pouze vyřešit úlohy "na papíře" pomocí aplikace, je potřeba vytvořit modul pro BST, který umožní vkládat a mazat prvky a vizualizovat strom (nebo vypsat průchody).
    *   *Poznámka: Stávající aplikace je zaměřena na obecné grafy. BST logika by byla samostatný modul nebo rozšíření.*

3.  **Přejmenování/Alias**:
    *   Přidat alias `level_order` pro `bfs_traversal` pro jasnost.
