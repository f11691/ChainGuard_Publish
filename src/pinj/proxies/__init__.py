from .base import Proxy, ProxyResult, Decision
from .none import NoneProxy
from .rule_based import RuleBasedProxy
from .encoding_defense import EncodingDefenseProxy

# Heavy proxies import torch/transformers lazily inside their modules.
def get_proxy(name: str, **kw):
    name = name.lower()
    if name == "none":
        return NoneProxy()
    if name == "rule_based":
        return RuleBasedProxy()
    if name == "encoding_defense":
        return EncodingDefenseProxy()
    if name == "classifier_guard":
        from .classifier_guard import ClassifierGuardProxy
        return ClassifierGuardProxy(**kw)
    if name == "llama_guard":
        from .classifier_guard import LlamaGuardProxy
        return LlamaGuardProxy(**kw)
    if name == "llm_classifier":
        from .llm_classifier import LLMClassifierProxy
        return LLMClassifierProxy(**kw)
    raise ValueError(f"unknown proxy: {name}")

ALL_PROXIES = [
    "none", "rule_based", "classifier_guard", "llama_guard",
    "encoding_defense", "llm_classifier",
]
