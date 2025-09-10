"""
Test script for the Semantic Search System
Note: This requires proper API keys and Weaviate setup to run fully.
"""

import os
from semantic_search import SemanticSearchSystem

def test_import():
    """Test that all imports work correctly."""
    try:
        import google.generativeai as genai
        import weaviate
        import weaviate.classes as wvc
        from dotenv import load_dotenv
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_environment_setup():
    """Test environment variable loading."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        gemini_key = os.getenv("GEMINI_API_KEY")
        weaviate_url = os.getenv("WEAVIATE_URL", "http://localhost:8080")
        
        print(f"✓ Environment loaded")
        print(f"  - Gemini API Key: {'Set' if gemini_key else 'Not set'}")
        print(f"  - Weaviate URL: {weaviate_url}")
        return True
    except Exception as e:
        print(f"✗ Environment setup error: {e}")
        return False

def test_class_initialization():
    """Test that the class can be instantiated (without API calls)."""
    try:
        # This will fail if API keys are not set, but we can catch that
        system = SemanticSearchSystem()
        print("✓ SemanticSearchSystem class can be instantiated")
        return True
    except ValueError as e:
        if "GEMINI_API_KEY" in str(e):
            print("✓ Class correctly validates API key requirement")
            return True
        else:
            print(f"✗ Unexpected validation error: {e}")
            return False
    except Exception as e:
        print(f"✗ Initialization error: {e}")
        return False

def main():
    """Run basic tests."""
    print("Testing Semantic Search System...")
    print("=" * 50)
    
    tests = [
        test_import,
        test_environment_setup,
        test_class_initialization
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All basic tests passed!")
        print("\nTo run the full system:")
        print("1. Set up your .env file with API keys")
        print("2. Start Weaviate (docker or cloud)")
        print("3. Run: python semantic_search.py")
    else:
        print("✗ Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main()