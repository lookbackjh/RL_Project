class Graph:

    def __init__(self):
        self.root = None
        self.nodes = []
        self.parents = []
        self.children = []
    



class Node:

    def __init__(self, name, value:tuple):
        self.name = name
        self.value = value # denotes selected features