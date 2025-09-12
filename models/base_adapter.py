"""
Base model adapter interface.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseModelAdapter(ABC):
    """Base class for model adapters."""
    
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate a response for the given prompt.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response text
        """
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        return {
            "model_name": self.model_name,
            "adapter_type": self.__class__.__name__,
            "parameters": self.kwargs
        }
