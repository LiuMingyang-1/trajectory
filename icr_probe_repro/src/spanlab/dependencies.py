from typing import Any


def _missing_dependency_message(package_name: str, extra: str = "") -> str:
    hint = "Install lab dependencies with `pip install -r icr_probe_repro/requirements.txt`."
    if extra:
        hint = f"{hint} {extra}"
    return f"Missing dependency `{package_name}`. {hint}"


def require_transformers() -> Any:
    try:
        import transformers
    except ModuleNotFoundError as exc:
        raise RuntimeError(_missing_dependency_message("transformers")) from exc
    return transformers


def require_sklearn() -> Any:
    try:
        import sklearn
    except ModuleNotFoundError as exc:
        raise RuntimeError(_missing_dependency_message("scikit-learn")) from exc
    return sklearn


def require_torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(_missing_dependency_message("torch")) from exc
    return torch


def load_spacy_model(model_name: str) -> Any:
    try:
        import spacy
    except ModuleNotFoundError as exc:
        raise RuntimeError(_missing_dependency_message("spacy")) from exc

    try:
        return spacy.load(model_name)
    except OSError as exc:
        extra = f"Then install the English pipeline with `python -m spacy download {model_name}`."
        raise RuntimeError(_missing_dependency_message(model_name, extra=extra)) from exc


def require_matplotlib() -> Any:
    try:
        import matplotlib
    except ModuleNotFoundError as exc:
        raise RuntimeError(_missing_dependency_message("matplotlib")) from exc
    return matplotlib
