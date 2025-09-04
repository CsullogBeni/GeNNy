# Graph Neural Network Learner

This repository contains a simple but extendable implementation of **graph neural networks (GNNs)**.  
The goal is to provide a modular structure for experimenting with GNN-based learning, making it easy to define new learners and adapt them to different tasks.

## Project Structure

### 1. `abstract_graph_learner.py`
This file defines an **abstract base class** for graph learners.  
It provides a **template** that enforces a consistent interface for any model built on top of it.

Key points:
- Uses Python’s `abc` (Abstract Base Class) module.
- Specifies **required methods** like:
  - `forward`: how the input graph and features are processed.
  - `predict`: how predictions are generated for downstream tasks.
- Encourages a **clean separation** between abstract definitions and concrete implementations.

This makes it easier to plug in different GNN variants without rewriting boilerplate code.

---

### 2. `gnn.py`
This file contains a **concrete implementation** of a graph neural network learner that inherits from the abstract class.

Main elements:
- **Graph Convolution Layers**: Encodes node and edge information into latent representations.
- **Forward Pass**:
  - Takes node features and graph structure as input.
  - Propagates information through the network layers.
  - Produces embeddings suitable for classification or regression tasks.
- **Prediction**:
  - Outputs task-specific predictions (e.g., node classification).

The implementation follows the standard design pattern of `forward()` for training and `predict()` for inference, which keeps training and evaluation code consistent.

---

### JSON graph format

Each graph is a JSON object with two arrays:

```
{
  "nodes": [
    { "id": 123, "class_": "HeaderTypeDeclarationContext", "value": null },
    { "id": 124, "class_": "TerminalNodeImpl", "value": ";" }
  ],
  "edges": [
    { "source": 123, "target": 124 }
  ]
}
```

- id – unique per node (string or int).
- class_ – required (string).
- value – optional; only used if class_ == "TerminalNodeImpl".

## How to Use

### Requirements
- Python 3.8+
- Recommended libraries:
  - [PyTorch](https://pytorch.org/) (for deep learning)
  - [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) (for graph data handling)

Install dependencies:
```bash
pip install torch torch-geometric
```

### Example Usage

## Extending the Project

To implement a new graph learner:

1. Create a new file (e.g., `my_gnn.py`).
2. Inherit from `AbstractGraphLearner` defined in `abstract_graph_learner.py`.
3. Implement the following methods:
   - `forward()`
   - `predict()`
4. Add any custom layers, loss functions, or training loops as needed.

### Example Skeleton
```python
from abstract_graph_learner import AbstractGraphLearner

class MyCustomGNN(AbstractGraphLearner):
    def __init__(self, ...):
        super().__init__()
        # define your layers here

    def forward(self, x, edge_index):
        # implement forward pass
        return ...

    def predict(self, x, edge_index):
        # implement prediction logic
        return ...
```