"""
OpenAI model adapter with enhanced error handling and retry mechanisms.
"""

import os
import time
from typing import Dict, Any, Optional
from openai import OpenAI
from openai.types import APIError
from .base_adapter import BaseModelAdapter
from framework.logger import logger
from framework.config_manager import config


class OpenAIAdapter(BaseModelAdapter):
    """Adapter for OpenAI models."""
    
    def __init__(self, model_name: str = "gpt-4o-mini", api_key: Optional[str] = None, **kwargs):
        super().__init__(model_name, **kwargs)
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        self.client = OpenAI(api_key=self.api_key)
        
        # Get model configuration
        model_config = config.get_model_config(model_name)
        self.timeout = model_config.get('timeout', 30)
        self.max_retries = model_config.get('max_retries', 3)
        self.retry_delay = model_config.get('retry_delay', 1.0)
        
        # Default generation parameters
        self.default_params = model_config.get('params', {
            "max_tokens": 50,
            "temperature": 0.0,
            **kwargs
        })
        
        self.logger = logger
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate a response using OpenAI API with retry logic.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response text
        """
        # Merge default params with provided kwargs
        params = {**self.default_params, **kwargs}
        
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                start_time = time.time()
                
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    timeout=self.timeout,
                    **params
                )
                
                duration = time.time() - start_time
                response_text = response.choices[0].message.content.strip()
                
                # Log successful request
                self.logger.log_model_request(
                    self.model_name, len(prompt), len(response_text), duration, success=True
                )
                
                return response_text
                
            except APIError as e:
                last_exception = e
                duration = time.time() - start_time
                
                # Log failed request
                self.logger.log_model_request(
                    self.model_name, len(prompt), 0, duration, success=False
                )
                
                # Check if we should retry
                if attempt < self.max_retries and self._should_retry(e):
                    self.logger.warning(f"API error (attempt {attempt + 1}/{self.max_retries + 1}): {e}")
                    time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                    continue
                else:
                    self.logger.error(f"OpenAI API error after {attempt + 1} attempts: {e}")
                    break
                    
            except Exception as e:
                last_exception = e
                duration = time.time() - start_time
                
                # Log failed request
                self.logger.log_model_request(
                    self.model_name, len(prompt), 0, duration, success=False
                )
                
                self.logger.error(f"Unexpected error: {e}")
                break
        
        # If we get here, all retries failed
        raise RuntimeError(f"OpenAI API error after {self.max_retries + 1} attempts: {str(last_exception)}")
    
    def _should_retry(self, error: APIError) -> bool:
        """Determine if an error should trigger a retry."""
        # Retry on rate limits, server errors, and timeouts
        retryable_errors = [
            "rate_limit_exceeded",
            "server_error", 
            "timeout",
            "internal_error",
            "service_unavailable"
        ]
        
        error_code = getattr(error, 'code', None)
        error_type = getattr(error, 'type', None)
        
        return (error_code in retryable_errors or 
                error_type in retryable_errors or
                "rate limit" in str(error).lower() or
                "timeout" in str(error).lower())
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the OpenAI model."""
        info = super().get_model_info()
        info.update({
            "provider": "OpenAI",
            "api_key_set": bool(self.api_key),
            "default_params": self.default_params
        })
        return info
