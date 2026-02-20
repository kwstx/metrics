"""
Algorithm Version Registry
===========================
Central registry that maps logical operation names to versioned evaluation
algorithms. This allows the system to:

  1. Swap algorithms at runtime (A/B, gradual rollout).
  2. Pin an older version for reproducibility audits.
  3. Record the exact version string in every AuditLogEntry so that
     historical results can be re-derived deterministically.

Each version descriptor carries:
  - A semantic version string (e.g. ``"1.0.0"``).
  - A human-readable changelog note.
  - Effective timestamps so versions can be scheduled to activate
    automatically.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


@dataclass(frozen=True)
class AlgorithmVersionDescriptor:
    """Immutable record describing one algorithm version."""
    version: str
    description: str
    effective_from: datetime = field(default_factory=datetime.utcnow)
    deprecated_at: Optional[datetime] = None


# -----------------------------------------------------------------------
#  Default version table – one entry per API operation.
#  New versions are appended; the latest non-deprecated entry wins.
# -----------------------------------------------------------------------

_REGISTRY: Dict[str, List[AlgorithmVersionDescriptor]] = {
    "submit_action": [
        AlgorithmVersionDescriptor(
            version="1.0.0",
            description="Graph-based action ingestion with domain multiplier and uncertainty metadata.",
        ),
    ],
    "retrieve_forecast": [
        AlgorithmVersionDescriptor(
            version="1.0.0",
            description=(
                "Weighted causal traversal with reliability coefficients, "
                "domain multipliers, and temporal decay."
            ),
        ),
    ],
    "run_counterfactual": [
        AlgorithmVersionDescriptor(
            version="1.0.0",
            description=(
                "Marginal cooperative influence via agent-removal simulation "
                "with disabled-node projection comparison."
            ),
        ),
    ],
    "query_synergy_density": [
        AlgorithmVersionDescriptor(
            version="1.0.0",
            description=(
                "Independent-vs-cooperative amplification ratio with "
                "normalised vector comparison."
            ),
        ),
    ],
    "agent_impact_profile": [
        AlgorithmVersionDescriptor(
            version="1.0.0",
            description=(
                "Aggregated multi-dimensional profile: forecasts, calibration "
                "events, stability coefficients, synergy participation, and "
                "cooperative intelligence vectors."
            ),
        ),
    ],
    "trace_provenance": [
        AlgorithmVersionDescriptor(
            version="1.0.0",
            description=(
                "Full causal-path reconstruction with node/edge extraction, "
                "influence-map propagation, and reproducibility verification."
            ),
        ),
    ],
    "submit_outcome": [
        AlgorithmVersionDescriptor(
            version="1.0.0",
            description=(
                "Realized outcome recording plus magnitude/timing/synergy "
                "deviation calibration with reliability coefficient feedback."
            ),
        ),
    ],
}


def get_current_version(operation: str) -> AlgorithmVersionDescriptor:
    """
    Returns the currently-active version descriptor for *operation*.
    Active = latest entry whose ``effective_from`` ≤ now **and** is not
    deprecated.
    """
    versions = _REGISTRY.get(operation)
    if not versions:
        raise KeyError(f"Unknown operation: {operation}")

    now = datetime.utcnow()
    candidates = [
        v for v in versions
        if v.effective_from <= now and v.deprecated_at is None
    ]
    if not candidates:
        raise RuntimeError(
            f"No active algorithm version for operation '{operation}'"
        )
    # Latest effective_from wins.
    return max(candidates, key=lambda v: v.effective_from)


def register_version(
    operation: str,
    version: str,
    description: str,
    effective_from: Optional[datetime] = None,
) -> AlgorithmVersionDescriptor:
    """
    Append a new version for an operation.  If *effective_from* is in the
    future the version will not become active until that time.
    """
    desc = AlgorithmVersionDescriptor(
        version=version,
        description=description,
        effective_from=effective_from or datetime.utcnow(),
    )
    _REGISTRY.setdefault(operation, []).append(desc)
    return desc


def deprecate_version(operation: str, version: str) -> None:
    """Mark a specific version as deprecated (no longer selectable)."""
    versions = _REGISTRY.get(operation, [])
    for i, v in enumerate(versions):
        if v.version == version and v.deprecated_at is None:
            # frozen dataclass – replace in-list
            _REGISTRY[operation][i] = AlgorithmVersionDescriptor(
                version=v.version,
                description=v.description,
                effective_from=v.effective_from,
                deprecated_at=datetime.utcnow(),
            )
            return
    raise KeyError(
        f"Active version '{version}' not found for operation '{operation}'"
    )


def list_versions(operation: str) -> List[AlgorithmVersionDescriptor]:
    """Return every version descriptor (active + deprecated) for an operation."""
    return list(_REGISTRY.get(operation, []))
