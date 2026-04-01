
import os
import dspy
from strategy_ideation_engine.intelligence.utils import setup_lm
from strategy_ideation_engine.config import settings

def test_api_providers():
    providers = {
        "Groq": "groq/llama-3.3-70b-versatile",
        "OpenAI": "openai/gpt-4o-mini",
        "Anthropic": "anthropic/claude-3-haiku-20240307",
        "Gemini": "gemini/gemini-1.5-flash"
    }
    
    results = {}
    
    for name, model in providers.items():
        print(f"\nTesting {name} provider with model: {model}...")
        try:
            lm = setup_lm(model)
            with dspy.settings.context(lm=lm):
                # Simple completion test
                response = lm("Say 'Hello from " + name + "'", max_tokens=10)
                # dspy.LM returns a list of strings or a single string depending on version
                # In recent dspy, it's often a list of completions.
                print(f"✅ {name} Success: {response}")
                results[name] = "Success"
        except Exception as e:
            print(f"❌ {name} Failed: {str(e)}")
            results[name] = f"Failed: {str(e)}"
            
    print("\n" + "="*30)
    print("API Provider Test Summary")
    print("="*30)
    for name, status in results.items():
        print(f"{name:10}: {status}")

if __name__ == "__main__":
    test_api_providers()
