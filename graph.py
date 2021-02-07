from graphviz import Digraph
import re

class Computation_Graph:
    def __init__(self, identity_size):
        self.graph = Digraph(comment="Computation Graph")
        self.identity_size = identity_size
        self.path_count = 0
        self.paths = {}

    def draw(self, parent_key, children_key):
        self.graph.edge(parent_key, children_key)

    def translate(self, raw_string):
        """
        :param raw_string: "<MulBackward0 object at 0x7f8b135b0790>"
        :return: str: operation, address
        """
        data = {}
        words = re.split(' ', raw_string)
        data["key"] = re.sub(r"[<>]", '', words[0])
        data["address"] = re.sub(r"[<>]", '', words[-1])
        return data

    def key_by_shape(self, shape):
        """
        :param shape: tuple: (int,int)
        :return: str: 'K'
        """
        for key, value in self.identity_size.items():
            if shape == value:
                return key

    def none_node(self, parent_info, children_info, path):
        self.graph.edge(parent_info['key'] + '\n' + parent_info['address'], children_info['key'] + '\n' + children_info['address'])
        path += ' > ' + parent_info['key'] + ' > ' + children_info['key']
        self.paths[self.path_count] = path
        self.path_count += 1

    def non_leaf_node(self, parent_info, children_info, function, path):
        self.graph.edge(parent_info['key'] + '\n' + parent_info['address'], children_info['key'] + '\n' + children_info['address'])
        path += ' > ' + parent_info['key']
        self.recursive_loop(function[0], path)

    def leaf_node(self, parent_info, children_info, function, path):
        shape = tuple(function[0].variable.size())
        key = self.key_by_shape(shape)
        self.graph.edge(parent_info['key'] + '\n' + parent_info['address'], children_info['key'] + '\n' + str(shape), label=key)
        path += ' > ' + parent_info['key'] + ' > ' + children_info['key'] + ' > ' + key
        self.paths[self.path_count] = path
        self.path_count += 1

    def recursive_loop(self, parent, path):
        parent_info = self.translate(str(parent))
        for function in parent.next_functions:
            children_info = self.translate(str(function[0]))
            if function[0] == None: # meaning that is gradient cannot passed user-defined variable
                self.none_node(parent_info, children_info, path)
                # self.graph.edge(parent_key + '\n' + parent_address, children_key + '\n' + children_address)
                # path += ' > ' + parent_key
                # self.paths[self.path_count] = path
                # self.path_count += 1
            else:
                if len(function[0].next_functions) != 0: # meaning that is not accumulated node
                    self.non_leaf_node(parent_info, children_info, function, path)
                    # self.graph.edge(parent_key + '\n' + parent_address, children_key + '\n' + children_address)
                    # path += ' > ' + parent_key
                    # self.recursive_loop(function[0], path)
                elif len(function[0].next_functions) == 0: # meaning that is accumulated node
                    self.leaf_node(parent_info, children_info, function, path)
                    # shape = tuple(function[0].variable.size())
                    # key = self.key_by_shape(shape)
                    # self.graph.edge(parent_key + '\n' + parent_address, children_key + '\n' + str(shape), label=key)
                    # path += ' > ' + parent_key + ' > ' + children_key + ' > ' + key
                    # self.paths[self.path_count] = path
                    # self.path_count += 1

    def save(self, file_name, view=True):
        self.graph.render(file_name, view=view)

    def show(self):
        self.graph.view()

