"""Bayesian optimization utilities (Frangi parameter search, etc.)."""

# Re-export frangi_gpu so `from model_script.bo import frangi_gpu` works.
# This keeps backward compatibility with older entrypoints that expect
# frangi_gpu to live under the bo package.
from importlib import import_module
from typing import Any

__all__ = ["frangi_gpu"]


def __getattr__(name: str) -> Any:
    if name == "frangi_gpu":
        return import_module("model_script.bo.frangi_gpu")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
