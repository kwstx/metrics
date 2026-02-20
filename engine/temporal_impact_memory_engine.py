from __future__ import annotations

from datetime import datetime
import math
from typing import Dict, Optional

from sqlalchemy.orm import Session

from models.impact_projection import TemporalImpactLedgerEntry


class TemporalImpactMemoryEngine:
    """
    Maintains a rolling temporal ledger for chained task sequences.
    """

    def __init__(self, session: Session):
        self.session = session

    def append_contribution(
        self,
        task_sequence_id: str,
        source_node_id: str,
        impact_vector: Dict[str, float],
        projection_id: Optional[str] = None,
        decay_function: str = "exponential",
        decay_rate: float = 0.01,
        decay_floor: float = 0.0,
        timestamp: Optional[datetime] = None,
        entry_metadata: Optional[Dict[str, float]] = None,
    ) -> TemporalImpactLedgerEntry:
        entry = TemporalImpactLedgerEntry(
            task_sequence_id=task_sequence_id,
            source_node_id=source_node_id,
            projection_id=projection_id,
            impact_vector=impact_vector,
            decay_function=decay_function,
            decay_rate=decay_rate,
            decay_floor=decay_floor,
            timestamp=timestamp or datetime.utcnow(),
            entry_metadata=entry_metadata,
        )
        self.session.add(entry)
        self.session.commit()
        return entry

    def get_accumulated_impact(
        self,
        task_sequence_id: str,
        as_of: Optional[datetime] = None,
    ) -> Dict[str, float]:
        now = as_of or datetime.utcnow()
        rows = (
            self.session.query(TemporalImpactLedgerEntry)
            .filter(TemporalImpactLedgerEntry.task_sequence_id == task_sequence_id)
            .all()
        )
        if not rows:
            return {}

        total: Dict[str, float] = {}
        for row in rows:
            elapsed = max((now - row.timestamp).total_seconds(), 0.0)
            decay_weight = self._decay_weight(
                function_name=row.decay_function,
                elapsed_seconds=elapsed,
                decay_rate=row.decay_rate,
                decay_floor=row.decay_floor,
            )
            for metric, value in (row.impact_vector or {}).items():
                total[metric] = total.get(metric, 0.0) + (float(value) * decay_weight)
        return total

    def _decay_weight(
        self,
        function_name: str,
        elapsed_seconds: float,
        decay_rate: float,
        decay_floor: float,
    ) -> float:
        if function_name == "none":
            return 1.0

        if function_name == "linear":
            weight = 1.0 - (decay_rate * elapsed_seconds)
            return max(decay_floor, weight)

        # default: exponential
        weight = math.exp(-decay_rate * elapsed_seconds)
        return max(decay_floor, weight)
