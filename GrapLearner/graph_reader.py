from abc import ABC, abstractmethod


class GraphReader(ABC):
    """
    Abstract base class for reading graph data from different file formats.
    """
    @abstractmethod
    def read_from_json(self, file_path: str) -> None:
        """
        Abstract method to read graph data from a JSON file.
        This method must be implemented by subclasses.

        Args:
            file_path (str): The path to the JSON file.
        """
        pass

    @abstractmethod
    def read_from_pkl(self, file_path: str) -> None:
        """
        Method to read graph data from a pickle (.pkl) file.
        Subclasses can override this method to provide specific implementations.

        Args:
            file_path (str): The path to the pickle file.
        """
        pass
