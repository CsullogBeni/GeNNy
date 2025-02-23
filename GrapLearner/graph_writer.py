from abc import ABC, abstractmethod


class GraphWriter(ABC):
    """
    Abstract base class for writing graph data to different file formats.
    """
    @abstractmethod
    def write_to_json(self, file_path: str) -> None:
        """
        Abstract method to write graph data to a JSON file.
        This method must be implemented by subclasses.

        Args:
            file_path (str): The path to the JSON file.
        """
        pass

    @abstractmethod
    def write_to_pkl(self, file_path: str) -> None:
        """
        Method to write graph data to a pickle (.pkl) file.
        Subclasses can override this method to provide specific implementations.

        Args:
            file_path (str): The path to the pickle file.
        """
        pass
