from abc import ABC, abstractmethod


class GraphReader(ABC):
    """
    Abstract base class for reading graph data from different file formats.
    """
    @abstractmethod
    def read_from_json(self):
        """
        Abstract method to read graph data from a JSON file.
        This method must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def read_from_pkl(self):
        """
        Method to read graph data from a pickle (.pkl) file.
        Subclasses can override this method to provide specific implementations.
        """
        pass
