"""
Auditable Computation Log
=========================
Every API call that triggers a computation is logged to an append-only audit
ledger. Each record captures:
  - the API operation name,
  - the algorithm version used,
  - all input parameters (serialised as JSON),
  - the full output payload (serialised as JSON),
  - wall-clock timing,
  - the requesting identity (optional).

The ledger is stored in the same relational database that backs the rest of
the system so that it participates in transactional guarantees.
"""

from datetime import datetime
import json
import uuid
from typing import Any, Dict, Optional

from sqlalchemy import Column, String, Float, DateTime, Text
from sqlalchemy.orm import Session

from models.base import Base


class AuditLogEntry(Base):
    """
    Immutable record of a single API computation.
    """
    __tablename__ = "api_audit_log"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    operation = Column(String, nullable=False)
    algorithm_version = Column(String, nullable=False)
    request_payload = Column(Text, nullable=False)  # JSON
    response_payload = Column(Text, nullable=False)  # JSON
    duration_ms = Column(Float, nullable=False)
    caller_identity = Column(String, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    status = Column(String, nullable=False, default="success")  # success | error
    error_detail = Column(Text, nullable=True)

    def __repr__(self) -> str:
        return (
            f"<AuditLogEntry(op={self.operation}, v={self.algorithm_version}, "
            f"t={self.timestamp}, status={self.status})>"
        )


class AuditLogger:
    """
    Thin helper that writes AuditLogEntry rows.
    """

    def __init__(self, session: Session):
        self.session = session

    def log(
        self,
        operation: str,
        algorithm_version: str,
        request_payload: Any,
        response_payload: Any,
        duration_ms: float,
        caller_identity: Optional[str] = None,
        status: str = "success",
        error_detail: Optional[str] = None,
    ) -> AuditLogEntry:
        entry = AuditLogEntry(
            operation=operation,
            algorithm_version=algorithm_version,
            request_payload=json.dumps(request_payload, default=str),
            response_payload=json.dumps(response_payload, default=str),
            duration_ms=duration_ms,
            caller_identity=caller_identity,
            status=status,
            error_detail=error_detail,
        )
        self.session.add(entry)
        # NOTE: caller is responsible for commit (batch-friendly).
        return entry

    def query_log(
        self,
        operation: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 50,
    ):
        """Retrieve audit entries with optional filters."""
        q = self.session.query(AuditLogEntry)
        if operation:
            q = q.filter(AuditLogEntry.operation == operation)
        if since:
            q = q.filter(AuditLogEntry.timestamp >= since)
        q = q.order_by(AuditLogEntry.timestamp.desc()).limit(limit)
        return q.all()
