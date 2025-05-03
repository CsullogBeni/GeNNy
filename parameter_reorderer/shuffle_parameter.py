import json
import copy
import os
import glob


class GraphReorderer:
    """
    A class to reorder specific substructures in a graph, reassign node IDs, and export modified versions.
    """

    def __init__(self, input_path: str, output_dir: str):
        self.input_path = input_path
        self.output_dir = output_dir
        self.original_graph = self._load_graph()
        self.to_be_reordered = []

    def _load_graph(self):
        """
        Loads the graph from the specified input JSON file.

        Returns:
            dict: The graph structure containing nodes and edges.
        """
        with open(self.input_path) as f:
            return json.load(f)

    def _reassign_all_node_ids(self, graph):
        """
        Reassigns all node IDs in the graph to a contiguous 0-based sequence and updates all edges accordingly.

        Args:
            graph (dict): The graph to process.

        Returns:
            dict: The graph with updated node IDs.
        """
        node_id_map = {}
        for i, node in enumerate(graph['nodes']):
            node_id_map[node['nodeId']] = i
            node['nodeId'] = i

        for edge in graph['edges']:
            edge['source'] = node_id_map[edge['source']]
            edge['target'] = node_id_map[edge['target']]

        return graph

    def _collect_reorder_targets(self):
        """
        Identifies and stores graph node patterns that match the ControlTypeDeclarationContext
        with nested NonEmptyParameterListContexts for later reordering.
        """
        for node in self.original_graph['nodes']:
            if node['class_'] == 'ControlTypeDeclarationContext':
                control_node_id = node['nodeId']
                for edge in self.original_graph['edges']:
                    if edge['source'] == control_node_id:
                        param_list_id = edge['target']
                        for node2 in self.original_graph['nodes']:
                            if node2['class_'] == 'ParameterListContext' and node2['nodeId'] == param_list_id:
                                for edge2 in self.original_graph['edges']:
                                    if edge2['source'] == param_list_id:
                                        fst_id = edge2['target']
                                        for node3 in self.original_graph['nodes']:
                                            if node3['nodeId'] == fst_id and node3[
                                                'class_'] == 'NonEmptyParameterListContext':
                                                for edge3 in self.original_graph['edges']:
                                                    if edge3['source'] == fst_id:
                                                        snd_id = edge3['target']
                                                        for node4 in self.original_graph['nodes']:
                                                            if node4['nodeId'] == snd_id and node4[
                                                                'class_'] == 'NonEmptyParameterListContext':
                                                                self.to_be_reordered.append((
                                                                    control_node_id,
                                                                    param_list_id,
                                                                    fst_id,
                                                                    snd_id
                                                                ))

    def reorder_and_save_all(self):
        """
        Reorders the target node patterns in the graph, reassigns node IDs,
        and saves each modified version as a separate JSON file in the output directory.
        """
        self._collect_reorder_targets()
        original_edges = copy.deepcopy(self.original_graph['edges'])

        os.makedirs(self.output_dir, exist_ok=True)

        for idx, (ctrl_id, param_id, fst_id, snd_id) in enumerate(self.to_be_reordered):
            for edge in self.original_graph['edges']:
                if edge['source'] == param_id and edge['target'] == fst_id:
                    edge['target'] = snd_id
                elif edge['source'] == fst_id and edge['target'] == snd_id:
                    edge['source'] = snd_id
                    edge['target'] = fst_id

            modified_graph = copy.deepcopy(self.original_graph)
            new_graph = self._reassign_all_node_ids(modified_graph)

            output_path = os.path.join(self.output_dir, f"reordered_graph_{idx}.json")
            with open(output_path, "w") as f:
                json.dump(new_graph, f, indent=2)

            print(f"✅ Saved: {output_path}")
            self.original_graph['edges'] = copy.deepcopy(original_edges)

    def validate_graphs(self):
        """
        Validates that all exported graphs contain unique, sorted node IDs from 0 to N-1.

        Prints validation results per file and an overall summary.
        """
        all_files = sorted(glob.glob(os.path.join(self.output_dir, "*.json")))
        all_passed = True

        for path in all_files:
            with open(path) as f:
                graph = json.load(f)

            node_ids = sorted(node['nodeId'] for node in graph['nodes'])
            expected = list(range(len(node_ids)))

            if node_ids == expected:
                print(f"✅ OK: {os.path.basename(path)}")
            else:
                print(f"❌ ERROR: {os.path.basename(path)} – Invalid node IDs!")
                all_passed = False

        if all_passed:
            print("\n🎉 All graphs are valid.")
        else:
            print("\n⚠️ Some graphs are invalid. Please check manually.")


def main():
    input_path = "data/basic_p4_v2_normalized.json"
    output_dir = "data/reordered"

    reorderer = GraphReorderer(input_path, output_dir)
    reorderer.reorder_and_save_all()
    reorderer.validate_graphs()


if __name__ == "__main__":
    main()
