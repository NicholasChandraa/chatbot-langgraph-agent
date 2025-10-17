from typing import Dict

# Pricing per 1M tokens (update regularly)
PRICING = {
    "openai": {
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    },
    "anthropic": {
        "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
        "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.00},
    },
    "gemini": {
        "gemini-2.5-pro": {"input": 1.25, "output": 5.00},
        "gemini-2.5-flash": {"input": 0.075, "output": 0.30},
    },
    "ollama": {
        "default": {"input": 0.0, "output": 0.0}  # Free (self-hosted)
    }
}


def calculate_cost(provider: str, model: str, tokens: Dict[str, int]) -> float:
    """Calculate estimated cost in USD"""
    provider = provider.lower()

    if provider not in PRICING:
        return 0.0
    
    # Find matching model (exact or prefix match)
    pricing = None
    for model_key in PRICING[provider]:
        if model.startswith(model_key):
            pricing = PRICING[provider][model_key]
            break
    
    if not pricing:
        return 0.0
    
    prompt_cost = (tokens.get("prompt_tokens", 0) / 1_000_000) * pricing["input"]
    completion_cost = (tokens.get("completion_tokens", 0) / 1_000_000) * pricing["output"]

    return round(prompt_cost + completion_cost, 6)



