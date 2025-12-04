# Vzorové slovní úlohy pro aplikaci Teorie Grafů

Tento dokument obsahuje příklady reálných situací, které lze řešit pomocí algoritmů implementovaných v této aplikaci.

## 1. Nejkratší cesta (Dijkstrův / Floyd-Warshallův algoritmus)
**Situace:** Jste logistický manažer a potřebujete přepravit zboží z **Prahy (uzel A)** do **Brna (uzel B)**. Graf představuje silniční síť, kde uzly jsou města a hrany jsou silnice. Váhy hran představují **vzdálenost v kilometrech** nebo **čas jízdy v minutách**.
**Úkol:** Najděte trasu, která je nejkratší nebo nejrychlejší.
**Algoritmus v aplikaci:** `Hledání cest -> Nejkratší cesta`

## 2. Nejbezpečnější cesta (Maximalizace spolehlivosti)
**Situace:** Posíláte tajnou zprávu přes nespolehlivou počítačovou síť. Graf představuje síť serverů. Váha každé hrany (např. 95, 99) představuje **pravděpodobnost v procentech**, že se zpráva mezi uzly neztratí (spolehlivost linky).
**Úkol:** Najděte cestu od odesílatele k příjemci tak, aby celková pravděpodobnost doručení (součin pravděpodobností na cestě) byla co nejvyšší.
**Algoritmus v aplikaci:** `Hledání cest -> Nejbezpečnější (spolehlivost)`

## 3. Nejširší cesta (Úzké hrdlo)
**Situace:** Potřebujete převézt nadměrný náklad (např. turbínu) z továrny na staveniště. Silnice mají různou šířku nebo nosnost mostů. Váha hrany představuje **maximální šířku/nosnost** daného úseku.
**Úkol:** Najděte trasu, kde "nejužší hrdlo" (nejmenší váha na cestě) je co největší, abyste mohli převézt co nejširší náklad.
**Algoritmus v aplikaci:** `Hledání cest -> Nejširší cesta`

## 4. Minimální kostra (MST - Kruskalův / Primův algoritmus)
**Situace:** Budujete optickou síť pro novou čtvrť. Uzly jsou domy a hrany jsou možné trasy pro kabely. Váha hrany je **cena výkopových prací a kabelu** mezi dvěma domy.
**Úkol:** Propojte všechny domy tak, aby byly všechny zapojeny do sítě (přímo nebo nepřímo) a celková cena výstavby byla minimální.
**Algoritmus v aplikaci:** `Kostry -> Minimální kostra`

## 5. Maximální tok (Max Flow)
**Situace:** Řídíte vodovodní síť. Máte zdroj vody (přehrada) a spotřebiště (město). Potrubí mezi uzly má určitou **kapacitu (litry za sekundu)**.
**Úkol:** Zjistěte, jaké maximální množství vody může protékat sítí ze zdroje do města, aniž by prasklo potrubí. Zároveň vás zajímá, které trubky jsou plně vytížené (úzká hrdla).
**Algoritmus v aplikaci:** `Toky a řezy -> Max Tok & Min Řez`

## 6. Kritická cesta (CPM - Nejdelší cesta v DAG)
**Situace:** Plánujete stavbu domu. Uzly jsou milníky (dokončení fáze) a orientované hrany jsou činnosti (např. "postavit zdi"). Váha hrany je **doba trvání činnosti**. Některé činnosti nemohou začít, dokud neskončí předchozí.
**Úkol:** Jaká je minimální doba, za kterou lze celý projekt stihnout? To odpovídá nalezení nejdelší cesty v grafu závislostí, protože zpoždění na této cestě zpozdí celý projekt.
**Algoritmus v aplikaci:** `Hledání cest -> Nejdelší cesta`

## 7. Moorův algoritmus (BFS - Nejkratší cesta v neohodnoceném grafu)
**Situace:** Analyzujete sociální síť. Uzly jsou lidé a hrany představují přátelství.
**Úkol:** Jaký je minimální počet "přeskoků" (přátel přátel), abyste se seznámili s konkrétní osobou? Zde nás nezajímá "síla" přátelství (váha), ale jen počet kroků.
**Algoritmus v aplikaci:** `Hledání cest -> Moorův alg. (BFS)`

## 8. Čínský listonoš (Eulerovský tah)
**Situace:** Pošťák musí roznést poštu ve čtvrti. Musí projít **každou ulici (hranu)** alespoň jednou a vrátit se na začátek.
**Úkol:** Najděte trasu, která projde všechny hrany. Pokud graf obsahuje Eulerovský tah, je to ideální trasa bez opakování ulic.
**Algoritmus v aplikaci:** (Lze ověřit pomocí vlastností grafu - stupně uzlů)
