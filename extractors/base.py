from abc import ABC, abstractmethod
from core.models import CommunicationMessage, EmployeeContext, MarkerVector


class IFeatureExtractor(ABC):
    """
    Base interface for feature extraction.
    """
    @abstractmethod
    def compute_markers(self, message: CommunicationMessage, context: EmployeeContext) -> MarkerVector:
        pass