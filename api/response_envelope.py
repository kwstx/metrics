"""
Structured Response Envelope
=============================
Every API response is wrapped in a uniform envelope that carries:

  - ``api_version``   – the algorithm version that produced the result.
  - ``operation``     – the logical operation name.
  - ``status``        – ``"ok"`` or ``"error"``.
  - ``data``          – the structured payload (vectors, causal explanations, etc.).
  - ``causal_explanation`` – human-readable narrative (always present, never omitted).
  - ``audit_id``      – the audit-log entry ID so the caller can retrieve the
                         full computation record.
  - ``timestamp``     – ISO-8601 UTC timestamp of response creation.
  - ``metadata``      – additional key/value pairs (optional).
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass
class ApiResponse:
    """Immutable structured envelope returned by every API endpoint."""

    operation: str
    api_version: str
    status: str  # "ok" | "error"
    data: Any
    causal_explanation: str
    audit_id: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    metadata: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Serialisation helper – keeps callers from caring about dataclass
    # internals.
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "operation": self.operation,
            "api_version": self.api_version,
            "status": self.status,
            "data": self.data,
            "causal_explanation": self.causal_explanation,
            "audit_id": self.audit_id,
            "timestamp": self.timestamp,
        }
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload


def success_envelope(
    operation: str,
    api_version: str,
    data: Any,
    causal_explanation: str,
    audit_id: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> ApiResponse:
    """Convenience factory for a successful response."""
    return ApiResponse(
        operation=operation,
        api_version=api_version,
        status="ok",
        data=data,
        causal_explanation=causal_explanation,
        audit_id=audit_id,
        metadata=metadata,
    )


def error_envelope(
    operation: str,
    api_version: str,
    error_message: str,
    audit_id: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> ApiResponse:
    """Convenience factory for an error response."""
    return ApiResponse(
        operation=operation,
        api_version=api_version,
        status="error",
        data=None,
        causal_explanation=error_message,
        audit_id=audit_id,
        metadata=metadata,
    )
