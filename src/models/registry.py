from typing import Callable, Dict, Any

_MODEL_REGISTRY: Dict[str, Callable[..., Any]] = {}

def register_model(name: str):
    """Decorator to register a pipeline/model."""
    def decorator(cls_or_func):
        _MODEL_REGISTRY[name] = cls_or_func
        return cls_or_func
    return decorator

def get_model(name: str) -> Callable[..., Any]:
    """Retrieve a registered model by name."""
    if name not in _MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' not found in registry. Available: {list(_MODEL_REGISTRY.keys())}")
    return _MODEL_REGISTRY[name]
