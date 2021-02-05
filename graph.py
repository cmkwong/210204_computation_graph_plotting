from graphviz import Digraph
import re

class Computation_Graph:
    def __init__(self):
        self.graph = Digraph(comment="Computation Graph")
        self.index_identity = 0
        self.operation_count = 0

    def draw(self, parent_key, children_key):
        self.graph.edge(parent_key, children_key)

    def translate(self, raw_string):
        """
        :param raw_string: "<MulBackward0 object at 0x7f8b135b0790>"
        :return: str: operation, address
        """
        words = re.split(' ', raw_string)
        operation = re.sub(r"[<>]", '', words[0])
        address = re.sub(r"[<>]", '', words[-1])
        return operation, address

    def recursive_loop(self, parent):
        parent_key, parent_address = self.translate(str(parent))
        for function in parent.next_functions:
            children_key, children_address = self.translate(str(function[0]))
            if function[0] == None:
                self.graph.edge(parent_key + '\n' + parent_address, children_key + '\n' + children_address)
            else:
                if len(function[0].next_functions) != 0:
                    self.graph.edge(parent_key + '\n' + parent_address, children_key + '\n' + children_address)
                    self.recursive_loop(function[0])
                elif len(function[0].next_functions) == 0:
                    shape = str(function[0].variable.size())
                    self.graph.edge(parent_key + '\n' + parent_address, children_key + '\n' + children_address + '\n' + shape)

    def save(self, file_name, view=True):
        self.graph.render(file_name, view=view)

    def show(self):
        self.graph.view()
