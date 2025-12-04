"""
Script pro generování 8 různých grafů s 100 uzly a hranami
Kombinace: ohodnocení uzlů/hran × orientovaný/neorientovaný
"""
import random

NUM_NODES = 50  # počet uzlů
MIN_PROB = 0.01  # minimální pravděpodobnost (aby se vyhnulo log(0))
MAX_PROB = 0.99  # maximální pravděpodobnost

def generate_graph1():
    """Graf 1: Uzlově i hranově ohodnocený, ORIENTOVANÝ"""
    with open('graph1.txt', 'w', encoding='utf-8') as f:
        for i in range(NUM_NODES):
            weight = random.randint(-50, 50)
            f.write(f"u N{i} {weight};\n")
        edge_counter = 1
        for i in range(NUM_NODES):
            num_edges = random.randint(5, 10)
            for _ in range(num_edges):
                target = random.randint(0, NUM_NODES - 1)
                if target != i:
                    weight = random.randint(1, 10)
                    direction = random.choice(['>', '<'])
                    f.write(f"h N{i} {direction} N{target} {weight} :h{edge_counter};\n")
                    edge_counter += 1
    print("Graf 1 vygenerován: graph1.txt")

def generate_graph2():
    """Graf 2: Uzlově i hranově ohodnocený, NEORIENTOVANÝ"""
    with open('graph2.txt', 'w', encoding='utf-8') as f:
        for i in range(NUM_NODES):
            weight = random.randint(-50, 50)
            f.write(f"u N{i} {weight};\n")
        edge_counter = 1
        for i in range(NUM_NODES // 2):
            num_edges = random.randint(5, 10)
            targets = random.sample([j for j in range(NUM_NODES) if j != i], min(num_edges, NUM_NODES - 1))
            for target in targets:
                weight = random.randint(1, 100)
                f.write(f"h N{i} - N{target} {weight} :h{edge_counter};\n")
                edge_counter += 1
    print("Graf 2 vygenerován: graph2.txt")

def generate_graph3():
    """Graf 3: Jen uzlově ohodnocený, ORIENTOVANÝ"""
    with open('graph3.txt', 'w', encoding='utf-8') as f:
        for i in range(NUM_NODES):
            weight = random.randint(-50, 50)
            f.write(f"u N{i} {weight};\n")
        edge_counter = 1
        for i in range(NUM_NODES):
            num_edges = random.randint(5, 10)
            for _ in range(num_edges):
                target = random.randint(0, NUM_NODES - 1)
                if target != i:
                    direction = random.choice(['>', '<'])
                    f.write(f"h N{i} {direction} N{target} :h{edge_counter};\n")
                    edge_counter += 1
    print("Graf 3 vygenerován: graph3.txt")

def generate_graph4():
    """Graf 4: Jen uzlově ohodnocený, NEORIENTOVANÝ"""
    with open('graph4.txt', 'w', encoding='utf-8') as f:
        for i in range(NUM_NODES):
            weight = random.randint(-50, 50)
            f.write(f"u N{i} {weight};\n")
        edge_counter = 1
        for i in range(NUM_NODES // 2):
            num_edges = random.randint(5, 10)
            targets = random.sample([j for j in range(NUM_NODES) if j != i], min(num_edges, NUM_NODES - 1))
            for target in targets:
                f.write(f"h N{i} - N{target} :h{edge_counter};\n")
                edge_counter += 1
    print("Graf 4 vygenerován: graph4.txt")

def generate_graph5():
    """Graf 5: Jen hranově ohodnocený, ORIENTOVANÝ"""
    with open('graph5.txt', 'w', encoding='utf-8') as f:
        for i in range(NUM_NODES):
            f.write(f"u N{i};\n")
        edge_counter = 1
        for i in range(NUM_NODES):
            num_edges = random.randint(5, 10)
            for _ in range(num_edges):
                target = random.randint(0, NUM_NODES - 1)
                if target != i:
                    weight = random.randint(1, 100)
                    direction = random.choice(['>', '<'])
                    f.write(f"h N{i} {direction} N{target} {weight} :h{edge_counter};\n")
                    edge_counter += 1
    print("Graf 5 vygenerován: graph5.txt")

def generate_graph6():
    """Graf 6: Jen hranově ohodnocený, NEORIENTOVANÝ"""
    with open('graph6.txt', 'w', encoding='utf-8') as f:
        for i in range(NUM_NODES):
            f.write(f"u N{i};\n")
        edge_counter = 1
        for i in range(NUM_NODES // 2):
            num_edges = random.randint(5, 10)
            targets = random.sample([j for j in range(NUM_NODES) if j != i], min(num_edges, NUM_NODES - 1))
            for target in targets:
                weight = random.randint(1, 100)
                f.write(f"h N{i} - N{target} {weight} :h{edge_counter};\n")
                edge_counter += 1
    print("Graf 6 vygenerován: graph6.txt")

def generate_graph7():
    """Graf 7: Bez ohodnocení, ORIENTOVANÝ"""
    with open('graph7.txt', 'w', encoding='utf-8') as f:
        for i in range(NUM_NODES):
            f.write(f"u N{i};\n")
        edge_counter = 1
        for i in range(NUM_NODES):
            num_edges = random.randint(5, 10)
            for _ in range(num_edges):
                target = random.randint(0, NUM_NODES - 1)
                if target != i:
                    direction = random.choice(['>', '<'])
                    f.write(f"h N{i} {direction} N{target} :h{edge_counter};\n")
                    edge_counter += 1
    print("Graf 7 vygenerován: graph7.txt")

def generate_graph8():
    """Graf 8: Bez ohodnocení, NEORIENTOVANÝ"""
    with open('graph8.txt', 'w', encoding='utf-8') as f:
        for i in range(NUM_NODES):
            f.write(f"u N{i};\n")
        edge_counter = 1
        for i in range(NUM_NODES // 2):
            num_edges = random.randint(5, 10)
            targets = random.sample([j for j in range(NUM_NODES) if j != i], min(num_edges, NUM_NODES - 1))
            for target in targets:
                f.write(f"h N{i} - N{target} :h{edge_counter};\n")
                edge_counter += 1
    print("Graf 8 vygenerován: graph8.txt")

def generate_graph9():
    """Graf 9: Binární strom s 100 uzly a ohodnocením uzlů"""
    with open('graph9.txt', 'w', encoding='utf-8') as f:
        levels = 7  # ≈ 127 pozic, aby max 100 uzlů
        node_counter = 0
        for level in range(levels):
            for i in range(2 ** level):
                if node_counter >= NUM_NODES:
                    break
                # Ohodnocení uzlu mezi -50 a 50
                weight = random.randint(-50, 50)
                f.write(f"u N{node_counter} {weight};\n")
                node_counter += 1
    print("Graf 9 vygenerován: graph9.txt")

def generate_graph11():
    """Graf 11: Orientovaný, hrany ohodnocené pravděpodobnostmi (0,1)"""
    with open('graph11.txt', 'w', encoding='utf-8') as f:
        # Nejprve uzly
        for i in range(NUM_NODES):
            f.write(f"u N{i};\n")
        edge_counter = 1
        for i in range(NUM_NODES):
            num_edges = random.randint(3, 7)  # každé uzlu 3–7 výstupních hran
            for _ in range(num_edges):
                target = random.randint(0, NUM_NODES - 1)
                if target != i:
                    prob = round(random.uniform(MIN_PROB, MAX_PROB), 4)
                    f.write(f"h N{i} > N{target} {prob} :h{edge_counter};\n")
                    edge_counter += 1
    print("Graf 11 vygenerován: graph11.txt")
if __name__ == "__main__":
    print("Generování 9 grafů...")
    print("=" * 60)
    
    generate_graph1()
    generate_graph2()
    generate_graph3()
    generate_graph4()
    generate_graph5()
    generate_graph6()
    generate_graph7()
    generate_graph8()
    generate_graph9()
    generate_graph11()
    
    print("\n" + "=" * 60)
    print("Všechny grafy byly úspěšně vygenerovány!")
