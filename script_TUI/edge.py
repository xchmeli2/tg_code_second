from collections import deque
class Edge:
    def __init__(self, node1, node2, weight=None, name=None):
        self.node1 = node1
        self.node2 = node2
        self.weight = weight
        self.name = name

    def __repr__(self):
        return "{} -> {} (weight: {}, name: {})".format(self.node1, self.node2, self.weight, self.name)
    
        """
class Edge:
    def __init__(self, node1, node2, capacity=None, name=None, weight=None):
        self.node1 = node1
        self.node2 = node2
        self.capacity = capacity  # Atribut pro kapacitu
        self.name = name
        self.flow = 0  # Tok na hraně
        self.weight = weight


    def __repr__(self):
        return "{} -> {} (weight: {}, capacity: {}, flow: {})".format(self.node1, self.node2, self.capacity, self.flow, self.weight)
"""