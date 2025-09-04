"""
Provides training and saving of the header classifiers.

This module wires together:
  - `HeaderBlockClassifier`: predicts header *block* root/context nodes.
  - `HeaderFieldClassifier`: predicts `StructFieldContext` nodes (fields) inside headers.
"""

import os
from typing import List

from header_block_classifier import HeaderBlockClassifier
from header_field_classifier import HeaderFieldClassifier


def train_and_save_models(data_dir: str, output_dir: str,
                          epochs: int = 20, hidden_dim: int = 64):
    """
        Train and save header classification models.

        This function coordinates the training of two classifiers that work on JSON
        data describing headers:

        1. **HeaderBlockClassifier** – Learns to identify header block structures
           and predict their root/context nodes.
        2. **HeaderFieldClassifier** – Learns to identify field-level details within
           headers, such as `StructFieldContext` nodes.

        Workflow:
            - Collect all `.json` files from the given `data_dir`.
            - Train the HeaderBlockClassifier using the collected data.
            - Save the trained block model to `output_dir`.
            - Train the HeaderFieldClassifier using the same data.
            - Save the trained field model to `output_dir`.

        Args:
            data_dir (str): Path to the directory containing `.json` training files.
            output_dir (str): Path to the directory where trained models
                will be stored. Created if it does not exist.
            epochs (int, optional): Number of training iterations (default: 20).
            hidden_dim (int, optional): Dimension of hidden layers in both
                classifiers (default: 64).
        """
    json_files: List[str] = [
        os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".json")
    ]

    # --- Header Block classifier ---
    block_model = HeaderBlockClassifier(hidden_dim=hidden_dim)
    print("Training: HeaderBlockClassifier")
    block_model.fit(json_files, epochs=epochs)
    os.makedirs(output_dir, exist_ok=True)
    block_model.save_model(output_dir)
    print("HeaderBlockClassifier saved")

    # --- Header Field classifier ---
    field_model = HeaderFieldClassifier(hidden_dim=hidden_dim)
    print("\nTraining: HeaderFieldClassifier")
    field_model.fit(json_files, epochs=epochs)
    field_model.save_model(output_dir)
    print("HeaderFieldClassifier saved")
