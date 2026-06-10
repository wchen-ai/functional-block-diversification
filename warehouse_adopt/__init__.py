"""warehouse_adopt — source-free block-warehouse core.

Wrap any pretrained checkpoint without touching its source: decompose
the ``state_dict`` into blocks, keep N variants of every block in an
on-disk warehouse with contributor provenance, assemble sub-models via
``load_state_dict`` (offline or online engine), diversify them with a
consistency loss, and run ensembled inference with uncertainty.

Specification: ``warehouse_adopt/agent_specs/00_core_warehouse.md``.
"""

from .contract import (
    UserModule,
    load_checkpoint_state_dict,
    load_user_module,
    resolve_model_factory,
    to_device,
)
from .decompose import (
    BlockSpec,
    decompose_balanced,
    decompose_boundary,
    decompose_manual,
    decompose_top_level,
    parse_decomposition,
    validate_decomposition,
)
from .engine import (
    ENGINE_MODES,
    POOL_DEVICES,
    AssemblyEngine,
    EngineStats,
    OfflineEngine,
    OnlineEngine,
    make_engine,
    state_dict_bytes,
)
from .inference import EnsemblePrediction, infer_with_uncertainty, sample_configurations
from .training import (
    ConsistencyBalance,
    ConsistencyBalancer,
    StepStats,
    default_consistency_fn,
    diversification_step,
    finetune_warehouse,
    recalibrate_bn,
    sample_distinct_configs,
)
from .warehouse import (
    ProvenanceEntry,
    Warehouse,
    initialize_variants,
    tensor_content_hash,
)

__version__ = "0.1.0"

__all__ = [
    "AssemblyEngine",
    "BlockSpec",
    "ConsistencyBalance",
    "ConsistencyBalancer",
    "ENGINE_MODES",
    "EngineStats",
    "EnsemblePrediction",
    "POOL_DEVICES",
    "OfflineEngine",
    "OnlineEngine",
    "ProvenanceEntry",
    "StepStats",
    "UserModule",
    "Warehouse",
    "decompose_balanced",
    "decompose_boundary",
    "decompose_manual",
    "decompose_top_level",
    "default_consistency_fn",
    "diversification_step",
    "finetune_warehouse",
    "infer_with_uncertainty",
    "initialize_variants",
    "load_checkpoint_state_dict",
    "load_user_module",
    "make_engine",
    "parse_decomposition",
    "recalibrate_bn",
    "resolve_model_factory",
    "sample_configurations",
    "sample_distinct_configs",
    "state_dict_bytes",
    "tensor_content_hash",
    "to_device",
    "validate_decomposition",
]
