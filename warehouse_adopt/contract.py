"""User-module contract and source-free auto-construction.

The system never edits a model's source. It needs only

(a) a *constructor* for the architecture — either a ``model_factory``
    callable supplied in a user module, or an auto-construction spec
    string (``torchvision:<name>``, ``timm:<name>``, ``hf:<repo>``); and
(b) the checkpoint's ``state_dict``.

The only unsupported case is a bare tensor dump with no obtainable
constructor.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Dict, Optional, Tuple

import torch
from torch import nn

REQUIRED_CALLABLES: tuple[str, ...] = (
    "model_factory",
    "load_base_checkpoint",
    "get_dataloaders",
    "infer_fn",
    "loss_fn",
)
OPTIONAL_CALLABLES: tuple[str, ...] = ("consistency_fn",)


@dataclass
class UserModule:
    """The five-callable contract (plus one optional callable).

    Attributes
    ----------
    model_factory
        ``() -> nn.Module`` — return a fresh, randomly-initialised
        instance of the architecture. Its weights are always
        overwritten by assembly; only the *structure* matters.
    load_base_checkpoint
        ``(model, path) -> None`` — load pretrained weights into a
        ``model_factory()`` instance, in place.
    get_dataloaders
        ``() -> (train_loader, val_loader)``.
    infer_fn
        ``(model, batch) -> Tensor`` — one forward pass; ``batch`` is
        whatever the dataloader yields.
    loss_fn
        ``(out, batch) -> Tensor`` — scalar task loss for ``out =
        infer_fn(model, batch)``.
    consistency_fn
        Optional ``(out_update, out_reference) -> Tensor`` — scalar
        consistency loss between two sub-model outputs. Defaults to
        MSE when omitted (see :mod:`warehouse_adopt.training`).
    """

    model_factory: Callable[[], nn.Module]
    load_base_checkpoint: Callable[[nn.Module, str], None]
    get_dataloaders: Callable[[], Tuple[Any, Any]]
    infer_fn: Callable[[nn.Module, Any], torch.Tensor]
    loss_fn: Callable[[torch.Tensor, Any], torch.Tensor]
    consistency_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None

    @classmethod
    def from_namespace(cls, ns: Any) -> "UserModule":
        """Build a :class:`UserModule` from any object with the five attributes."""
        missing = [n for n in REQUIRED_CALLABLES if not callable(getattr(ns, n, None))]
        if missing:
            raise AttributeError(
                f"User module is missing required callables: {missing}. "
                f"Required: {list(REQUIRED_CALLABLES)}; optional: {list(OPTIONAL_CALLABLES)}."
            )
        return cls(
            model_factory=ns.model_factory,
            load_base_checkpoint=ns.load_base_checkpoint,
            get_dataloaders=ns.get_dataloaders,
            infer_fn=ns.infer_fn,
            loss_fn=ns.loss_fn,
            consistency_fn=getattr(ns, "consistency_fn", None),
        )


def load_user_module(path: str | Path) -> UserModule:
    """Import a user-supplied ``.py`` file and validate the contract.

    Contract: the file defines top-level callables ``model_factory``,
    ``load_base_checkpoint``, ``get_dataloaders``, ``infer_fn``,
    ``loss_fn`` (and optionally ``consistency_fn``).
    """
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"User module not found: {path}")
    module_name = f"_warehouse_adopt_user_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for {path}")
    module: ModuleType = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return UserModule.from_namespace(module)


def _parse_query_kwargs(query: str) -> Dict[str, Any]:
    """Parse ``k=v&k2=v2`` query kwargs; values literal-eval'd when possible."""
    kwargs: Dict[str, Any] = {}
    if not query:
        return kwargs
    for pair in query.split("&"):
        if not pair:
            continue
        key, sep, value = pair.partition("=")
        if not sep:
            raise ValueError(f"Malformed kwarg {pair!r} in model spec query (expected k=v)")
        try:
            kwargs[key] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            kwargs[key] = value
    return kwargs


def resolve_model_factory(spec: str) -> Callable[[], nn.Module]:
    """Resolve an auto-construction spec into a zero-arg model factory.

    Supported specs (optional ``?k=v&k2=v2`` kwargs suffix):

    * ``torchvision:<name>`` — ``torchvision.models.<name>(**kwargs)``
    * ``timm:<name>``        — ``timm.create_model(name, pretrained=False, **kwargs)``
    * ``hf:<repo>``          — ``AutoModel.from_config(AutoConfig.from_pretrained(repo, **kwargs))``
      (constructor-only; never downloads weights)
    """
    provider, sep, rest = spec.partition(":")
    if not sep or not rest:
        raise ValueError(
            f"Invalid model spec {spec!r}; expected 'torchvision:<name>', "
            f"'timm:<name>', or 'hf:<repo>' (optionally '?k=v&...')."
        )
    name, _, query = rest.partition("?")
    kwargs = _parse_query_kwargs(query)

    if provider == "torchvision":
        try:
            import torchvision.models as tvm
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise ImportError("Model spec 'torchvision:*' requires torchvision.") from exc
        if not hasattr(tvm, name):
            raise ValueError(f"torchvision.models has no constructor named {name!r}")
        ctor = getattr(tvm, name)
        return lambda: ctor(**kwargs)
    if provider == "timm":
        try:
            import timm
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Model spec 'timm:*' requires timm.") from exc
        return lambda: timm.create_model(name, pretrained=False, **kwargs)
    if provider == "hf":
        try:
            from transformers import AutoConfig, AutoModel
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Model spec 'hf:*' requires transformers.") from exc
        return lambda: AutoModel.from_config(AutoConfig.from_pretrained(name, **kwargs))
    raise ValueError(f"Unknown model-spec provider {provider!r} in {spec!r}")


def load_checkpoint_state_dict(
    path: str | Path,
    *,
    strip_prefixes: Tuple[str, ...] = ("module.",),
    key: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    """Load a checkpoint file and return a flat ``state_dict``.

    Handles the common wrappers: a raw state_dict, a dict containing a
    state_dict under ``key`` (or auto-detected ``"state_dict"`` /
    ``"model"``), and DataParallel ``module.`` prefixes.
    """
    obj = torch.load(Path(path), map_location="cpu", weights_only=True)
    if key is not None:
        obj = obj[key]
    elif isinstance(obj, dict) and not all(isinstance(v, torch.Tensor) for v in obj.values()):
        for candidate in ("state_dict", "model_state_dict", "model"):
            if candidate in obj and isinstance(obj[candidate], dict):
                obj = obj[candidate]
                break
    if not isinstance(obj, dict) or not obj:
        raise ValueError(f"Checkpoint {path} does not contain a usable state_dict.")
    state = {}
    for k, v in obj.items():
        if not isinstance(v, torch.Tensor):
            continue
        for prefix in strip_prefixes:
            if k.startswith(prefix):
                k = k[len(prefix):]
                break
        state[k] = v
    return state


def to_device(batch: Any, device: str | torch.device) -> Any:
    """Move a (possibly nested) batch onto ``device``; non-tensors pass through."""
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, (list, tuple)):
        return type(batch)(to_device(b, device) for b in batch)
    if isinstance(batch, dict):
        return {k: to_device(v, device) for k, v in batch.items()}
    return batch
