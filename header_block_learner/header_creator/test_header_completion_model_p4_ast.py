import unittest
import os
import networkx as nx

from graph.graph_builder import GraphBuilder


def _load_p4_ast(path: str) -> GraphBuilder:
    """
    Loads graph from json and returns a graph builder.

    Args:
        Full path to the json file.

    Returns:
        GraphBuilder object.
    """
    graph_builder = GraphBuilder()
    graph_builder.load_data(path)
    return graph_builder


def _get_subgraph_by_node_id(graph_builder: GraphBuilder, node_id: int) -> nx.DiGraph:
    """
    Filters the subgraph by node id.

    Args:
        graph_builder: contains the graph, perform extraction.
        node_id: the root of the new graph

    Returns:
        New subgraph
    """
    return graph_builder.extract_subgraph_by_node_id(node_id)


class TestHeaderCompletionModelP4Ast(unittest.TestCase):
    """
    Tests for ensuring that the generated P4 scripts contains the correct header blocks extensions.
    Checking P4 ASTs. Checking that the correct header block subgraph is added to the P4 ast.
    """

    def setUp(self) -> None:
        """
        Sets the path of the validation files.
        """
        self.validation_dir = os.path.join(os.path.dirname(__file__), "validation_out")

    def test_basic_p4(self):
        """
        Test for basic_p4.json file.
        """
        path = os.path.join(self.validation_dir, "basic_p4.json")
        graph_builder = _load_p4_ast(path)
        subgraph = _get_subgraph_by_node_id(graph_builder, 3274)
        self.assertEqual(subgraph.nodes[3274]['class_'], 'HeaderTypeDeclarationContext')
        self.assertEqual(subgraph.nodes[3282]['class_'], 'StructFieldListContext')
        self.assertEqual(subgraph.nodes[3283]['class_'], 'StructFieldListContext')
        self.assertEqual(subgraph.nodes[3284]['class_'], 'StructFieldListContext')

        self.assertEqual(subgraph.nodes[3285]['class_'], 'StructFieldListContext')
        self.assertEqual(subgraph.nodes[4865]['class_'], 'StructFieldContext')
        self.assertEqual(subgraph.nodes[4866]['class_'], 'TypeRefContext')
        self.assertEqual(subgraph.nodes[4867]['class_'], 'TypeNameContext')
        self.assertEqual(subgraph.nodes[4868]['class_'], 'PrefixedTypeContext')
        self.assertEqual(subgraph.nodes[4869]['class_'], 'Type_or_idContext')
        self.assertEqual(subgraph.nodes[4870]['class_'], 'TerminalNodeImpl')
        self.assertEqual(subgraph.nodes[4870]['value'], 'macAddr_t')
        self.assertEqual(subgraph.nodes[4871]['class_'], 'NameContext')
        self.assertEqual(subgraph.nodes[4872]['class_'], 'NonTypeNameContext')
        self.assertEqual(subgraph.nodes[4873]['class_'], 'Type_or_idContext')
        self.assertEqual(subgraph.nodes[4874]['class_'], 'TerminalNodeImpl')
        self.assertEqual(subgraph.nodes[4874]['value'], 'destinationAddress')

        self.assertEqual(subgraph.nodes[4876]['class_'], 'StructFieldListContext')
        self.assertEqual(subgraph.nodes[4877]['class_'], 'StructFieldContext')
        self.assertEqual(subgraph.nodes[4878]['class_'], 'TypeRefContext')
        self.assertEqual(subgraph.nodes[4879]['class_'], 'TypeNameContext')
        self.assertEqual(subgraph.nodes[4880]['class_'], 'PrefixedTypeContext')
        self.assertEqual(subgraph.nodes[4881]['class_'], 'Type_or_idContext')
        self.assertEqual(subgraph.nodes[4882]['class_'], 'TerminalNodeImpl')
        self.assertEqual(subgraph.nodes[4882]['value'], 'macAddr_t')
        self.assertEqual(subgraph.nodes[4883]['class_'], 'NameContext')
        self.assertEqual(subgraph.nodes[4884]['class_'], 'NonTypeNameContext')
        self.assertEqual(subgraph.nodes[4885]['class_'], 'Type_or_idContext')
        self.assertEqual(subgraph.nodes[4886]['class_'], 'TerminalNodeImpl')
        self.assertEqual(subgraph.nodes[4886]['value'], 'sourceAddress')

    def test_basic_p4_2(self):
        """
        Test for basic_p4_2.json file.
        """
        path = os.path.join(self.validation_dir, "basic_p4_2.json")
        graph_builder = _load_p4_ast(path)
        subgraph = _get_subgraph_by_node_id(graph_builder, 3274)
        self.assertEqual(subgraph.nodes[3274]['class_'], 'HeaderTypeDeclarationContext')
        self.assertEqual(subgraph.nodes[3282]['class_'], 'StructFieldListContext')
        self.assertEqual(subgraph.nodes[3283]['class_'], 'StructFieldListContext')
        self.assertEqual(subgraph.nodes[3284]['class_'], 'StructFieldListContext')

        self.assertEqual(subgraph.nodes[3285]['class_'], 'StructFieldListContext')
        self.assertEqual(subgraph.nodes[5427]['class_'], 'StructFieldContext')
        self.assertEqual(subgraph.nodes[5428]['class_'], 'TypeRefContext')
        self.assertEqual(subgraph.nodes[5429]['class_'], 'TypeNameContext')
        self.assertEqual(subgraph.nodes[5430]['class_'], 'PrefixedTypeContext')
        self.assertEqual(subgraph.nodes[5431]['class_'], 'Type_or_idContext')
        self.assertEqual(subgraph.nodes[5432]['class_'], 'TerminalNodeImpl')
        self.assertEqual(subgraph.nodes[5432]['value'], 'macAddr_t')
        self.assertEqual(subgraph.nodes[5433]['class_'], 'NameContext')
        self.assertEqual(subgraph.nodes[5434]['class_'], 'NonTypeNameContext')
        self.assertEqual(subgraph.nodes[5435]['class_'], 'Type_or_idContext')
        self.assertEqual(subgraph.nodes[5436]['class_'], 'TerminalNodeImpl')
        self.assertEqual(subgraph.nodes[5436]['value'], 'destinationAddress')

        self.assertEqual(subgraph.nodes[5438]['class_'], 'StructFieldListContext')
        self.assertEqual(subgraph.nodes[5439]['class_'], 'StructFieldContext')
        self.assertEqual(subgraph.nodes[5440]['class_'], 'TypeRefContext')
        self.assertEqual(subgraph.nodes[5441]['class_'], 'TypeNameContext')
        self.assertEqual(subgraph.nodes[5442]['class_'], 'PrefixedTypeContext')
        self.assertEqual(subgraph.nodes[5443]['class_'], 'Type_or_idContext')
        self.assertEqual(subgraph.nodes[5444]['class_'], 'TerminalNodeImpl')
        self.assertEqual(subgraph.nodes[5444]['value'], 'macAddr_t')
        self.assertEqual(subgraph.nodes[5445]['class_'], 'NameContext')
        self.assertEqual(subgraph.nodes[5446]['class_'], 'NonTypeNameContext')
        self.assertEqual(subgraph.nodes[5447]['class_'], 'Type_or_idContext')
        self.assertEqual(subgraph.nodes[5448]['class_'], 'TerminalNodeImpl')
        self.assertEqual(subgraph.nodes[5448]['value'], 'sourceAddress')

    def test_basic_p4_4(self):
        """
        Test for basic_p4_4.json file.
        """
        path = os.path.join(self.validation_dir, "basic_p4_4.json")
        graph_builder = _load_p4_ast(path)
        subgraph = _get_subgraph_by_node_id(graph_builder, 3274)
        self.assertEqual(subgraph.nodes[3274]['class_'], 'HeaderTypeDeclarationContext')
        self.assertEqual(subgraph.nodes[3282]['class_'], 'StructFieldListContext')
        self.assertEqual(subgraph.nodes[3283]['class_'], 'StructFieldListContext')
        self.assertEqual(subgraph.nodes[3284]['class_'], 'StructFieldListContext')

        self.assertEqual(subgraph.nodes[3285]['class_'], 'StructFieldListContext')
        self.assertEqual(subgraph.nodes[6551]['class_'], 'StructFieldContext')
        self.assertEqual(subgraph.nodes[6552]['class_'], 'TypeRefContext')
        self.assertEqual(subgraph.nodes[6553]['class_'], 'TypeNameContext')
        self.assertEqual(subgraph.nodes[6554]['class_'], 'PrefixedTypeContext')
        self.assertEqual(subgraph.nodes[6555]['class_'], 'Type_or_idContext')
        self.assertEqual(subgraph.nodes[6556]['class_'], 'TerminalNodeImpl')
        self.assertEqual(subgraph.nodes[6556]['value'], 'macAddr_t')
        self.assertEqual(subgraph.nodes[6557]['class_'], 'NameContext')
        self.assertEqual(subgraph.nodes[6558]['class_'], 'NonTypeNameContext')
        self.assertEqual(subgraph.nodes[6559]['class_'], 'Type_or_idContext')
        self.assertEqual(subgraph.nodes[6560]['class_'], 'TerminalNodeImpl')
        self.assertEqual(subgraph.nodes[6560]['value'], 'destinationAddress')

        self.assertEqual(subgraph.nodes[6562]['class_'], 'StructFieldListContext')
        self.assertEqual(subgraph.nodes[6563]['class_'], 'StructFieldContext')
        self.assertEqual(subgraph.nodes[6564]['class_'], 'TypeRefContext')
        self.assertEqual(subgraph.nodes[6565]['class_'], 'TypeNameContext')
        self.assertEqual(subgraph.nodes[6566]['class_'], 'PrefixedTypeContext')
        self.assertEqual(subgraph.nodes[6567]['class_'], 'Type_or_idContext')
        self.assertEqual(subgraph.nodes[6568]['class_'], 'TerminalNodeImpl')
        self.assertEqual(subgraph.nodes[6568]['value'], 'macAddr_t')
        self.assertEqual(subgraph.nodes[6569]['class_'], 'NameContext')
        self.assertEqual(subgraph.nodes[6570]['class_'], 'NonTypeNameContext')
        self.assertEqual(subgraph.nodes[6571]['class_'], 'Type_or_idContext')
        self.assertEqual(subgraph.nodes[6572]['class_'], 'TerminalNodeImpl')
        self.assertEqual(subgraph.nodes[6572]['value'], 'sourceAddress')

    def test_basic_p4_8(self):
        """
        Test for basic_p4_8.json file.
        """
        path = os.path.join(self.validation_dir, "basic_p4_8.json")
        graph_builder = _load_p4_ast(path)
        subgraph = _get_subgraph_by_node_id(graph_builder, 3274)
        self.assertEqual(subgraph.nodes[3274]['class_'], 'HeaderTypeDeclarationContext')
        self.assertEqual(subgraph.nodes[3282]['class_'], 'StructFieldListContext')
        self.assertEqual(subgraph.nodes[3283]['class_'], 'StructFieldListContext')
        self.assertEqual(subgraph.nodes[3284]['class_'], 'StructFieldListContext')

        self.assertEqual(subgraph.nodes[3285]['class_'], 'StructFieldListContext')
        self.assertEqual(subgraph.nodes[8811]['class_'], 'StructFieldContext')
        self.assertEqual(subgraph.nodes[8812]['class_'], 'TypeRefContext')
        self.assertEqual(subgraph.nodes[8813]['class_'], 'TypeNameContext')
        self.assertEqual(subgraph.nodes[8814]['class_'], 'PrefixedTypeContext')
        self.assertEqual(subgraph.nodes[8815]['class_'], 'Type_or_idContext')
        self.assertEqual(subgraph.nodes[8816]['class_'], 'TerminalNodeImpl')
        self.assertEqual(subgraph.nodes[8816]['value'], 'macAddr_t')
        self.assertEqual(subgraph.nodes[8817]['class_'], 'NameContext')
        self.assertEqual(subgraph.nodes[8818]['class_'], 'NonTypeNameContext')
        self.assertEqual(subgraph.nodes[8819]['class_'], 'Type_or_idContext')
        self.assertEqual(subgraph.nodes[8820]['class_'], 'TerminalNodeImpl')
        self.assertEqual(subgraph.nodes[8820]['value'], 'sourceAddress')

    def test_basic_p4_16(self):
        """
        Test for basic_p4_16.json file.
        """
        path = os.path.join(self.validation_dir, "basic_p4_16.json")
        graph_builder = _load_p4_ast(path)
        subgraph = _get_subgraph_by_node_id(graph_builder, 3274)
        self.assertEqual(subgraph.nodes[3274]['class_'], 'HeaderTypeDeclarationContext')
        self.assertEqual(subgraph.nodes[3282]['class_'], 'StructFieldListContext')
        self.assertEqual(subgraph.nodes[3283]['class_'], 'StructFieldListContext')
        self.assertEqual(subgraph.nodes[3284]['class_'], 'StructFieldListContext')

        self.assertEqual(subgraph.nodes[3285]['class_'], 'StructFieldListContext')
        self.assertEqual(subgraph.nodes[13295]['class_'], 'StructFieldContext')
        self.assertEqual(subgraph.nodes[13296]['class_'], 'TypeRefContext')
        self.assertEqual(subgraph.nodes[13297]['class_'], 'TypeNameContext')
        self.assertEqual(subgraph.nodes[13298]['class_'], 'PrefixedTypeContext')
        self.assertEqual(subgraph.nodes[13299]['class_'], 'Type_or_idContext')
        self.assertEqual(subgraph.nodes[13300]['class_'], 'TerminalNodeImpl')
        self.assertEqual(subgraph.nodes[13300]['value'], 'macAddr_t')
        self.assertEqual(subgraph.nodes[13301]['class_'], 'NameContext')
        self.assertEqual(subgraph.nodes[13302]['class_'], 'NonTypeNameContext')
        self.assertEqual(subgraph.nodes[13303]['class_'], 'Type_or_idContext')
        self.assertEqual(subgraph.nodes[13304]['class_'], 'TerminalNodeImpl')
        self.assertEqual(subgraph.nodes[13304]['value'], 'destinationAddress')

        self.assertEqual(subgraph.nodes[13306]['class_'], 'StructFieldListContext')
        self.assertEqual(subgraph.nodes[13307]['class_'], 'StructFieldContext')
        self.assertEqual(subgraph.nodes[13308]['class_'], 'TypeRefContext')
        self.assertEqual(subgraph.nodes[13309]['class_'], 'TypeNameContext')
        self.assertEqual(subgraph.nodes[13310]['class_'], 'PrefixedTypeContext')
        self.assertEqual(subgraph.nodes[13311]['class_'], 'Type_or_idContext')
        self.assertEqual(subgraph.nodes[13312]['class_'], 'TerminalNodeImpl')
        self.assertEqual(subgraph.nodes[13312]['value'], 'macAddr_t')
        self.assertEqual(subgraph.nodes[13313]['class_'], 'NameContext')
        self.assertEqual(subgraph.nodes[13314]['class_'], 'NonTypeNameContext')
        self.assertEqual(subgraph.nodes[13315]['class_'], 'Type_or_idContext')
        self.assertEqual(subgraph.nodes[13316]['class_'], 'TerminalNodeImpl')
        self.assertEqual(subgraph.nodes[13316]['value'], 'sourceAddress')


if __name__ == '__main__':
    unittest.main()
