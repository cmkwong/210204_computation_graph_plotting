from graphviz import Digraph
import re

def translate_paths_into_dict_list(raw_paths):
    """
    self.raw_paths = {  0: '> LogSoftmaxBackward > TBackward > MmBackward > ViewBackward > AccumulateGrad > R',
                        1: '> LogSoftmaxBackward > TBackward > MmBackward > SigmoidBackward > AddBackward0 > MmBackward > ViewBackward > AccumulateGrad > K',
                        ...}
    :return: path_dict = {  'R': ['> LogSoftmaxBackward > TBackward > MmBackward > ViewBackward > AccumulateGrad > R', '> MmBackward > ViewBackward > AccumulateGrad > R', ... ],
                            'K': '> LogSoftmaxBackward > TBackward > MmBackward > SigmoidBackward > AddBackward0 > MmBackward > ViewBackward > AccumulateGrad > K', ' > MmBackward > ViewBackward > AccumulateGrad > K', ...],
                            'H': ...
                            ...}
    """
    path_dict = {}
    for _, text_path in raw_paths.items():
        key = text_path.split('>')[-1].strip()
        if key in path_dict:
            path_dict[key].append(text_path)
        else:
            path_dict[key] = []
            path_dict[key].append(text_path)
    return path_dict

def write_txt_path(path_dict):
    """
    :param path_dict: path_dict = { 'R': ['> LogSoftmaxBackward > TBackward > MmBackward > ViewBackward > AccumulateGrad > R', '> MmBackward > ViewBackward > AccumulateGrad > R', ... ],
                                    'K': '> LogSoftmaxBackward > TBackward > MmBackward > SigmoidBackward > AddBackward0 > MmBackward > ViewBackward > AccumulateGrad > K', ' > MmBackward > ViewBackward > AccumulateGrad > K', ...],
                                    'H': ...
                                    ...}
    :return: None
    """
    with open('text_paths.txt', 'w') as f:
        for key, paths in path_dict.items():
            f.write(key + ":\n")
            for i, path in enumerate(paths):
                f.write('\t' + str(i) + ": " + path + '\n')
            f.write('\n')
        f.close()

class Computation_Graph:
    def __init__(self, identity_size):
        self.graph = Digraph(comment="Computation Graph")
        self.identity_size = identity_size
        self.path_count = 0
        self.raw_paths = {}

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
        self.raw_paths[self.path_count] = path
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
        self.raw_paths[self.path_count] = path
        self.path_count += 1

    def recursive_loop(self, parent, path):
        parent_info = self.translate(str(parent))
        for function in parent.next_functions:
            children_info = self.translate(str(function[0]))
            if function[0] == None: # meaning that is gradient cannot passed user-defined variable
                self.none_node(parent_info, children_info, path)
                # self.graph.edge(parent_key + '\n' + parent_address, children_key + '\n' + children_address)
                # path += ' > ' + parent_key
                # self.raw_paths[self.path_count] = path
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
                    # self.raw_paths[self.path_count] = path
                    # self.path_count += 1

    def save(self, file_name, view=True):
        self.graph.render(file_name, view=view)
        path_dict = translate_paths_into_dict_list(self.raw_paths)
        write_txt_path(path_dict)

    def show(self):
        self.graph.view()

