#!/usr/bin/env python3
"""
Test script to verify AI model providers
"""

import os
import sys
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.ai_providers import AIModelManager

load_dotenv()

def test_ai_models():
    """Test available AI models"""
    print("Testing AI Model Providers")
    print("=" * 50)
    
    # Get available models
    available_models = AIModelManager.get_available_models()
    
    print("\n📊 Model Availability:")
    for model_name, is_available in available_models.items():
        status = "✅ Available" if is_available else "❌ Not Available"
        print(f"  {model_name}: {status}")
    
    # Test each available model
    print("\n🧪 Testing Available Models:")
    test_prompt = "What is 2 + 2?"
    
    for model_name, is_available in available_models.items():
        if is_available:
            print(f"\n  Testing {model_name}...")
            provider = AIModelManager.get_provider(model_name)
            if provider:
                response = provider.generate_response(test_prompt)
                if response and not response.startswith("Error"):
                    print(f"    ✅ {model_name} works!")
                    print(f"    Response: {response[:100]}...")
                else:
                    print(f"    ❌ {model_name} failed: {response}")
            else:
                print(f"    ❌ Could not initialize {model_name}")
    
    # Get default model
    default_model = AIModelManager.get_default_model()
    if default_model:
        print(f"\n🎯 Default Model: {default_model}")
    else:
        print("\n❌ No default model available")
    
    print("\n" + "=" * 50)
    print("Test Complete!")

if __name__ == "__main__":
    test_ai_models()