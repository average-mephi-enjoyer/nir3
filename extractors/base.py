from abc import ABC, abstractmethod
from core.models import CommunicationMessage, EmployeeContext, MarkerVector


class IFeatureExtractor(ABC):
    @abstractmethod
    def compute_markers(self, message: CommunicationMessage, context: EmployeeContext) -> MarkerVector:
        pass