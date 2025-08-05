import networkx as nx


class PrettyPrinter:
    def __init__(self, g: nx.DiGraph):
        self._graph = g
        self._script = ""

    @property
    def get_script(self):
        self._traverse(list(self._graph.nodes)[0])
        return self._script

    def _traverse(self, node):
        node_value = self._graph.nodes[node]['value']
        if node_value is not None:
            try:
                if ';' in node_value or '{' in node_value:
                    self._script += node_value + '\n'
                elif self._script[-1] == '.' or self._script[-1] == ' ' or node_value == '.':
                    self._script += node_value
                else:
                    self._script += ' ' + node_value
            except IndexError:
                self._script += node_value
        else:
            try:
                for child in self._graph.successors(node):
                    self._traverse(child)
            except Exception as e:
                print(e)
