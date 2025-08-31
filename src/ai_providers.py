"""
AI Model Providers for Financial Analysis
Supports multiple AI models including Google Gemini and OpenAI
"""

import os
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod
import google.generativeai as genai
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class AIProvider(ABC):
    """Abstract base class for AI providers"""
    
    @abstractmethod
    def generate_response(self, prompt: str) -> Optional[str]:
        """Generate a response from the AI model"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available"""
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Get the name of the model"""
        pass


class GeminiProvider(AIProvider):
    """Google Gemini AI provider"""
    
    def __init__(self, model_name: str = "gemini-1.5-flash"):
        self.model_name = model_name
        self.model = None
        self._initialize()
    
    def _initialize(self):
        """Initialize Gemini model"""
        api_key = os.getenv('GOOGLE_AI_API_KEY')
        
        if api_key:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(self.model_name)
        else:
            # Try service account
            service_account_path = '/home/rajan/CREDENTIALS/rtc-lms-ef961e47471d.json'
            if os.path.exists(service_account_path):
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = service_account_path
                try:
                    import google.auth
                    credentials, project = google.auth.default(
                        scopes=['https://www.googleapis.com/auth/generative-language']
                    )
                    self.model = genai.GenerativeModel(self.model_name)
                except Exception:
                    pass
    
    def generate_response(self, prompt: str) -> Optional[str]:
        """Generate response from Gemini"""
        if not self.model:
            return None
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def is_available(self) -> bool:
        """Check if Gemini is available"""
        return self.model is not None
    
    def get_model_name(self) -> str:
        """Get the model name"""
        return f"Google {self.model_name}"


class OpenAIProvider(AIProvider):
    """OpenAI GPT provider"""
    
    # Model-specific token limits (conservative estimates)
    MODEL_TOKEN_LIMITS = {
        "gpt-4o": 8000,  # Reduced from actual limit for safety
        "gpt-4o-mini": 16000,
        "gpt-4-turbo": 8000,
        "gpt-3.5-turbo": 4000
    }
    
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model_name = model_name
        self.client = None
        self.max_tokens = self.MODEL_TOKEN_LIMITS.get(model_name, 4000)
        self._initialize()
    
    def _initialize(self):
        """Initialize OpenAI client"""
        api_key = os.getenv('OPENAI_API_KEY')
        
        if api_key:
            self.client = OpenAI(api_key=api_key)
    
    def _truncate_prompt(self, prompt: str, max_chars: int = None) -> str:
        """Truncate prompt to fit within token limits"""
        if max_chars is None:
            # Rough estimate: 1 token ≈ 4 characters
            max_chars = self.max_tokens * 3  # Conservative to leave room for response
        
        if len(prompt) <= max_chars:
            return prompt
        
        # Try to find a good truncation point
        truncated = prompt[:max_chars]
        
        # Try to truncate at a paragraph or sentence boundary
        last_para = truncated.rfind('\n\n')
        if last_para > max_chars * 0.7:  # If we can keep at least 70% of content
            truncated = truncated[:last_para]
        else:
            last_sentence = truncated.rfind('. ')
            if last_sentence > max_chars * 0.7:
                truncated = truncated[:last_sentence + 1]
        
        return truncated + "\n\n[Note: Context truncated due to length limits]"
    
    def generate_response(self, prompt: str) -> Optional[str]:
        """Generate response from OpenAI"""
        if not self.client:
            return None
        
        try:
            # Truncate prompt if needed
            truncated_prompt = self._truncate_prompt(prompt)
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful financial analyst assistant. Provide clear, accurate, and insightful analysis based on the financial data provided."},
                    {"role": "user", "content": truncated_prompt}
                ],
                temperature=0.7,
                max_tokens=min(2000, self.max_tokens // 2)  # Reserve half for input
            )
            return response.choices[0].message.content
        except Exception as e:
            # If still too large, try with more aggressive truncation
            if "rate_limit_exceeded" in str(e) or "tokens" in str(e).lower():
                try:
                    very_short_prompt = self._truncate_prompt(prompt, max_chars=self.max_tokens * 2)
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": "You are a financial analyst. Be concise."},
                            {"role": "user", "content": very_short_prompt}
                        ],
                        temperature=0.7,
                        max_tokens=1000
                    )
                    return response.choices[0].message.content
                except Exception as e2:
                    return f"Error: Context too large. Please try a simpler question or use a different model. ({str(e2)})"
            return f"Error generating response: {str(e)}"
    
    def is_available(self) -> bool:
        """Check if OpenAI is available"""
        return self.client is not None
    
    def get_model_name(self) -> str:
        """Get the model name"""
        return f"OpenAI {self.model_name}"


class AIModelManager:
    """Manager for AI models"""
    
    # Available models configuration
    AVAILABLE_MODELS = {
        "Google Gemini 1.5 Flash": lambda: GeminiProvider("gemini-1.5-flash"),
        "Google Gemini 1.5 Pro": lambda: GeminiProvider("gemini-1.5-pro"),
        "OpenAI GPT-4o": lambda: OpenAIProvider("gpt-4o"),
        "OpenAI GPT-4o Mini": lambda: OpenAIProvider("gpt-4o-mini"),
        "OpenAI GPT-4 Turbo": lambda: OpenAIProvider("gpt-4-turbo"),
        "OpenAI GPT-3.5 Turbo": lambda: OpenAIProvider("gpt-3.5-turbo")
    }
    
    @classmethod
    def get_available_models(cls) -> Dict[str, bool]:
        """Get list of available models with their availability status"""
        available = {}
        for model_name, provider_factory in cls.AVAILABLE_MODELS.items():
            try:
                provider = provider_factory()
                available[model_name] = provider.is_available()
            except Exception:
                available[model_name] = False
        return available
    
    @classmethod
    def get_provider(cls, model_name: str) -> Optional[AIProvider]:
        """Get AI provider for the specified model"""
        if model_name in cls.AVAILABLE_MODELS:
            provider = cls.AVAILABLE_MODELS[model_name]()
            if provider.is_available():
                return provider
        return None
    
    @classmethod
    def get_default_model(cls) -> Optional[str]:
        """Get the first available model as default"""
        available_models = cls.get_available_models()
        for model_name, is_available in available_models.items():
            if is_available:
                return model_name
        return None