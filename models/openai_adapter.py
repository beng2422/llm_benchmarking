"""
OpenAI model adapter.
"""

import os
from typing import Dict, Any, Optional
from openai import OpenAI
from .base_adapter import BaseModelAdapter


class OpenAIAdapter(BaseModelAdapter):
    """Adapter for OpenAI models."""
    
    def __init__(self, model_name: str = "gpt-4o-mini", api_key: Optional[str] = None, **kwargs):
        super().__init__(model_name, **kwargs)
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        self.client = OpenAI(api_key=self.api_key)
        
        # Default generation parameters
        self.default_params = {
            "max_tokens": 50,
            "temperature": 0.0,
            **kwargs
        }
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate a response using OpenAI API.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response text
        """
        # Merge default params with provided kwargs
        params = {**self.default_params, **kwargs}
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                **params
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {str(e)}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the OpenAI model."""
        info = super().get_model_info()
        info.update({
            "provider": "OpenAI",
            "api_key_set": bool(self.api_key),
            "default_params": self.default_params
        })
        return info
