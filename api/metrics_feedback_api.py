"""
Metrics & Feedback API Layer
=============================
Structured facade that exposes every engine capability through a consistent,
audited, version-tracked interface.  **Every public method**:

  1. Resolves the current algorithm version for the operation.
  2. Delegates to the appropriate engine(s).
  3. Constructs a structured response containing impact vectors *and*
     causal explanations (never raw counts).
  4. Writes an immutable AuditLogEntry before returning.

Public endpoints
~~~~~~~~~~~~~~~~
  - ``submit_action``        – ingest an action into the Impact Graph.
  - ``retrieve_forecast``    – generate a probabilistic impact projection.
  - ``run_counterfactual``   – simulate agent removal and compute marginal
                               cooperative influence.
  - ``query_synergy_density``– compute or retrieve synergy amplification for
                               a multi-agent cluster.
  - ``agent_impact_profile`` – aggregated multi-dimensional profile for an
                               agent.
  - ``trace_provenance``     – reconstruct the full causal path behind any
                               metric.
  - ``submit_outcome``       – record a realized outcome and trigger
                               calibration.
"""

from __future__ import annotations

import time
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

# Internal engines --------------------------------------------------------
from engine.graph_manager import GraphManager
from engine.impact_forecast_engine import ImpactForecastEngine
from engine.counterfactual_simulation import CounterfactualSimulation
from engine.synergy_density_engine import SynergyDensityEngine
from engine.predictive_calibration_engine import PredictiveCalibrationEngine
from engine.cooperative_stability_engine import CooperativeStabilityEngine
from engine.cooperative_intelligence_index_engine import CooperativeIntelligenceIndexEngine
from engine.impact_provenance_tracing_engine import ImpactProvenanceTracingEngine
from engine.persistent_synergy_signature_engine import PersistentSynergySignatureEngine
from engine.temporal_impact_memory_engine import TemporalImpactMemoryEngine

# Internal models ---------------------------------------------------------
from models.impact_graph import ImpactNode, ImpactEdge
from models.impact_projection import (
    ImpactProjection,
    AgentReliability,
    CooperativeStabilityMetric,
    CooperativeIntelligenceMetric,
    SynergyDensityMetric,
    CalibrationEvent,
    ImpactOutcome,
    MarginalCooperativeInfluence,
    SynergySignature,
)

# API infrastructure ------------------------------------------------------
from api.audit_log import AuditLogger
from api.algorithm_registry import get_current_version
from api.response_envelope import (
    ApiResponse,
    success_envelope,
    error_envelope,
)


class MetricsFeedbackAPI:
    """
    Unified API surface for the Cooperative Intelligence Metrics system.

    Each public method returns an :class:`ApiResponse` that satisfies:
      - Structured impact vectors (never raw counts).
      - Causal explanation text.
      - Algorithm version tag.
      - Audit-log entry ID for full traceability.
    """

    def __init__(self, session: Session, caller_identity: Optional[str] = None):
        self.session = session
        self.caller_identity = caller_identity

        # Wire up engines
        self._graph = GraphManager(session)
        self._forecast = ImpactForecastEngine(session)
        self._counterfactual = CounterfactualSimulation(session)
        self._synergy = SynergyDensityEngine(session)
        self._calibration = PredictiveCalibrationEngine(session)
        self._stability = CooperativeStabilityEngine(session)
        self._intelligence = CooperativeIntelligenceIndexEngine(session)
        self._provenance = ImpactProvenanceTracingEngine(session)
        self._signatures = PersistentSynergySignatureEngine(session)
        self._temporal = TemporalImpactMemoryEngine(session)
        self._audit = AuditLogger(session)

    # =====================================================================
    #  1.  submit_action
    # =====================================================================
    def submit_action(
        self,
        outcome_type: str,
        domain_context: Dict[str, Any],
        magnitude: float = 0.0,
        uncertainty: Optional[Dict[str, Any]] = None,
        causal_links: Optional[List[Dict[str, Any]]] = None,
    ) -> ApiResponse:
        """
        Ingest an agent action into the Impact Graph.

        *causal_links* is an optional list of edge descriptors::

            [{"target_node_id": "...", "weight": 0.8, "confidence": 1.0, "delay": 0.0}]

        Returns
        -------
        ApiResponse
            ``data`` contains the new node's structured descriptor and any
            created edges.
        """
        op = "submit_action"
        ver = get_current_version(op)
        t0 = time.perf_counter()
        request_payload = {
            "outcome_type": outcome_type,
            "domain_context": domain_context,
            "magnitude": magnitude,
            "uncertainty": uncertainty,
            "causal_links": causal_links,
        }

        try:
            node = self._graph.add_node(
                outcome_type=outcome_type,
                domain_context=domain_context,
                magnitude=magnitude,
                uncertainty=uncertainty,
            )

            edges_created: List[Dict[str, Any]] = []
            if causal_links:
                for link in causal_links:
                    edge = self._graph.add_causal_edge(
                        source_id=node.id,
                        target_id=link["target_node_id"],
                        weight=link.get("weight", 1.0),
                        confidence=link.get("confidence", 1.0),
                        delay=link.get("delay", 0.0),
                    )
                    edges_created.append({
                        "edge_id": edge.id,
                        "source_node_id": node.id,
                        "target_node_id": edge.target_node_id,
                        "causal_weight": edge.causal_weight,
                        "confidence_score": edge.confidence_score,
                        "propagation_delay": edge.propagation_delay,
                    })

            data = {
                "node": {
                    "id": node.id,
                    "outcome_type": node.outcome_type,
                    "domain_context": node.domain_context,
                    "magnitude": node.magnitude,
                    "uncertainty_metadata": node.uncertainty_metadata,
                    "timestamp": node.timestamp.isoformat() if node.timestamp else None,
                },
                "edges": edges_created,
            }

            explanation = (
                f"Action node '{node.id[:8]}…' of type '{outcome_type}' ingested into "
                f"the Impact Graph with magnitude {magnitude}. "
                f"{len(edges_created)} causal edge(s) established, linking this action "
                f"to downstream outcome nodes. Domain context: {domain_context}."
            )

            duration = (time.perf_counter() - t0) * 1000
            audit = self._audit.log(
                operation=op,
                algorithm_version=ver.version,
                request_payload=request_payload,
                response_payload=data,
                duration_ms=duration,
                caller_identity=self.caller_identity,
            )
            self.session.commit()

            return success_envelope(
                operation=op,
                api_version=ver.version,
                data=data,
                causal_explanation=explanation,
                audit_id=audit.id,
            )

        except Exception as exc:
            duration = (time.perf_counter() - t0) * 1000
            audit = self._audit.log(
                operation=op,
                algorithm_version=ver.version,
                request_payload=request_payload,
                response_payload=None,
                duration_ms=duration,
                caller_identity=self.caller_identity,
                status="error",
                error_detail=str(exc),
            )
            self.session.commit()
            return error_envelope(
                operation=op,
                api_version=ver.version,
                error_message=str(exc),
                audit_id=audit.id,
            )

    # =====================================================================
    #  2.  retrieve_forecast
    # =====================================================================
    def retrieve_forecast(
        self,
        action_node_id: str,
        time_horizon: float = 30.0,
        task_sequence_id: Optional[str] = None,
        decay_function: str = "exponential",
        decay_rate: float = 0.01,
        decay_floor: float = 0.0,
    ) -> ApiResponse:
        """
        Generate a probabilistic downstream impact projection for an action.

        Returns
        -------
        ApiResponse
            ``data`` carries the multi-dimensional predicted impact vector,
            uncertainty bounds, confidence interval, time horizon, and
            dependency references.
        """
        op = "retrieve_forecast"
        ver = get_current_version(op)
        t0 = time.perf_counter()
        request_payload = {
            "action_node_id": action_node_id,
            "time_horizon": time_horizon,
            "task_sequence_id": task_sequence_id,
            "decay_function": decay_function,
            "decay_rate": decay_rate,
            "decay_floor": decay_floor,
        }

        try:
            projection = self._forecast.forecast_action(
                action_node_id=action_node_id,
                time_horizon=time_horizon,
                task_sequence_id=task_sequence_id,
                decay_function=decay_function,
                decay_rate=decay_rate,
                decay_floor=decay_floor,
            )

            data = {
                "projection_id": projection.id,
                "source_node_id": projection.source_node_id,
                "predicted_impact_vector": projection.predicted_impact_vector,
                "uncertainty_bounds": projection.uncertainty_bounds,
                "confidence_interval": projection.confidence_interval,
                "time_horizon": projection.time_horizon,
                "dependency_references": projection.dependency_references,
                "timestamp": projection.timestamp.isoformat() if projection.timestamp else None,
            }

            vec_summary = ", ".join(
                f"{k}: {v:.4f}" for k, v in (projection.predicted_impact_vector or {}).items()
            )
            explanation = (
                f"Impact forecast generated for node '{action_node_id[:8]}…' over a "
                f"{time_horizon}s time horizon. Causal traversal propagated influence "
                f"through the graph using agent reliability coefficients and domain "
                f"multipliers. Predicted vector: [{vec_summary}]. Uncertainty bounds "
                f"reflect base node uncertainty scaled by reliability."
            )

            duration = (time.perf_counter() - t0) * 1000
            audit = self._audit.log(
                operation=op,
                algorithm_version=ver.version,
                request_payload=request_payload,
                response_payload=data,
                duration_ms=duration,
                caller_identity=self.caller_identity,
            )
            self.session.commit()

            return success_envelope(
                operation=op,
                api_version=ver.version,
                data=data,
                causal_explanation=explanation,
                audit_id=audit.id,
            )

        except Exception as exc:
            duration = (time.perf_counter() - t0) * 1000
            audit = self._audit.log(
                operation=op,
                algorithm_version=ver.version,
                request_payload=request_payload,
                response_payload=None,
                duration_ms=duration,
                caller_identity=self.caller_identity,
                status="error",
                error_detail=str(exc),
            )
            self.session.commit()
            return error_envelope(
                operation=op,
                api_version=ver.version,
                error_message=str(exc),
                audit_id=audit.id,
            )

    # =====================================================================
    #  3.  run_counterfactual
    # =====================================================================
    def run_counterfactual(
        self,
        source_node_id: str,
        removed_agent_id: str,
        time_horizon: float = 30.0,
    ) -> ApiResponse:
        """
        Simulate agent removal and compute the marginal cooperative influence
        delta (structured vector, not a scalar).

        Returns
        -------
        ApiResponse
            ``data`` contains full-projection vector, counterfactual vector,
            marginal influence vector, total marginal influence, and
            dependency references.
        """
        op = "run_counterfactual"
        ver = get_current_version(op)
        t0 = time.perf_counter()
        request_payload = {
            "source_node_id": source_node_id,
            "removed_agent_id": removed_agent_id,
            "time_horizon": time_horizon,
        }

        try:
            metric = self._counterfactual.simulate_agent_removal(
                source_node_id=source_node_id,
                removed_agent_id=removed_agent_id,
                time_horizon=time_horizon,
            )

            data = {
                "metric_id": metric.id,
                "source_node_id": metric.source_node_id,
                "removed_agent_id": metric.removed_agent_id,
                "full_projection_vector": metric.full_projection_vector,
                "counterfactual_projection_vector": metric.counterfactual_projection_vector,
                "marginal_influence_vector": metric.marginal_influence_vector,
                "total_marginal_influence": metric.total_marginal_influence,
                "time_horizon": metric.time_horizon,
                "dependency_references": metric.dependency_references,
                "timestamp": metric.timestamp.isoformat() if metric.timestamp else None,
            }

            marginal_desc = ", ".join(
                f"{k}: Δ{v:+.4f}" for k, v in (metric.marginal_influence_vector or {}).items()
            )
            explanation = (
                f"Counterfactual simulation: removed agent '{removed_agent_id}' from "
                f"the causal graph rooted at node '{source_node_id[:8]}…'. "
                f"Full projection was compared against a counterfactual (agent-absent) "
                f"projection. Marginal influence vector: [{marginal_desc}]. "
                f"Total marginal influence = {metric.total_marginal_influence:.4f}. "
                f"The delta isolates the cooperative contribution attributable solely "
                f"to the removed agent."
            )

            duration = (time.perf_counter() - t0) * 1000
            audit = self._audit.log(
                operation=op,
                algorithm_version=ver.version,
                request_payload=request_payload,
                response_payload=data,
                duration_ms=duration,
                caller_identity=self.caller_identity,
            )
            self.session.commit()

            return success_envelope(
                operation=op,
                api_version=ver.version,
                data=data,
                causal_explanation=explanation,
                audit_id=audit.id,
            )

        except Exception as exc:
            duration = (time.perf_counter() - t0) * 1000
            audit = self._audit.log(
                operation=op,
                algorithm_version=ver.version,
                request_payload=request_payload,
                response_payload=None,
                duration_ms=duration,
                caller_identity=self.caller_identity,
                status="error",
                error_detail=str(exc),
            )
            self.session.commit()
            return error_envelope(
                operation=op,
                api_version=ver.version,
                error_message=str(exc),
                audit_id=audit.id,
            )

    # =====================================================================
    #  4.  query_synergy_density
    # =====================================================================
    def query_synergy_density(
        self,
        agent_node_ids: List[str],
    ) -> ApiResponse:
        """
        Compute or retrieve the synergy density for a multi-agent cluster.

        Returns
        -------
        ApiResponse
            ``data`` carries independent impact sum vector, cooperative impact
            vector, and the normalised amplification ratio.
        """
        op = "query_synergy_density"
        ver = get_current_version(op)
        t0 = time.perf_counter()
        request_payload = {"agent_node_ids": agent_node_ids}

        try:
            metric = self._synergy.calculate_synergy_density(agent_node_ids)

            data = {
                "metric_id": metric.id,
                "collaboration_structure": metric.collaboration_structure,
                "independent_impact_sum": metric.independent_impact_sum,
                "cooperative_impact": metric.cooperative_impact,
                "synergy_density_ratio": metric.synergy_density_ratio,
                "timestamp": metric.timestamp.isoformat() if metric.timestamp else None,
            }

            ind_total = sum((metric.independent_impact_sum or {}).values())
            coop_total = sum((metric.cooperative_impact or {}).values())
            explanation = (
                f"Synergy density computed for {len(agent_node_ids)}-agent cluster. "
                f"Each agent's projected impact was computed in isolation (Σ independent "
                f"= {ind_total:.4f}), then compared against cooperative projection "
                f"(Σ cooperative = {coop_total:.4f}). Amplification ratio = "
                f"{metric.synergy_density_ratio:.4f}; values > 1.0 indicate super-additive "
                f"cooperation, < 1.0 indicate destructive interference."
            )

            duration = (time.perf_counter() - t0) * 1000
            audit = self._audit.log(
                operation=op,
                algorithm_version=ver.version,
                request_payload=request_payload,
                response_payload=data,
                duration_ms=duration,
                caller_identity=self.caller_identity,
            )
            self.session.commit()

            return success_envelope(
                operation=op,
                api_version=ver.version,
                data=data,
                causal_explanation=explanation,
                audit_id=audit.id,
            )

        except Exception as exc:
            duration = (time.perf_counter() - t0) * 1000
            audit = self._audit.log(
                operation=op,
                algorithm_version=ver.version,
                request_payload=request_payload,
                response_payload=None,
                duration_ms=duration,
                caller_identity=self.caller_identity,
                status="error",
                error_detail=str(exc),
            )
            self.session.commit()
            return error_envelope(
                operation=op,
                api_version=ver.version,
                error_message=str(exc),
                audit_id=audit.id,
            )

    # =====================================================================
    #  5.  agent_impact_profile
    # =====================================================================
    def agent_impact_profile(
        self,
        agent_id: str,
    ) -> ApiResponse:
        """
        Build a comprehensive, multi-dimensional impact profile for an agent.

        The profile aggregates:
          - All impact projections the agent has produced.
          - Calibration events showing predictive accuracy.
          - Reliability coefficient history.
          - Cooperative stability metrics.
          - Synergy density participation.
          - Cooperative intelligence vectors where the agent appears.

        Returns
        -------
        ApiResponse
            ``data`` is a nested structure with one section per metric
            dimension.  All values are structured vectors, never raw counts.
        """
        op = "agent_impact_profile"
        ver = get_current_version(op)
        t0 = time.perf_counter()
        request_payload = {"agent_id": agent_id}

        try:
            # --- Reliability ------------------------------------------------
            reliability_record = (
                self.session.query(AgentReliability)
                .filter(AgentReliability.agent_id == agent_id)
                .first()
            )
            reliability_coeff = reliability_record.reliability_coefficient if reliability_record else 1.0
            reliability_updated = (
                reliability_record.last_updated.isoformat()
                if reliability_record and reliability_record.last_updated
                else None
            )

            # --- Projections ------------------------------------------------
            agent_nodes = (
                self.session.query(ImpactNode)
                .filter(ImpactNode.outcome_type == "ACTION")
                .all()
            )
            agent_node_ids = [
                n.id for n in agent_nodes
                if (n.domain_context or {}).get("agent_id") == agent_id
                or (n.domain_context or {}).get("agent") == agent_id
            ]

            projections = (
                self.session.query(ImpactProjection)
                .filter(ImpactProjection.source_node_id.in_(agent_node_ids))
                .order_by(ImpactProjection.timestamp.desc())
                .all()
            ) if agent_node_ids else []

            projection_summaries = [
                {
                    "projection_id": p.id,
                    "predicted_impact_vector": p.predicted_impact_vector,
                    "time_horizon": p.time_horizon,
                    "timestamp": p.timestamp.isoformat() if p.timestamp else None,
                }
                for p in projections
            ]

            # --- Calibration Events -----------------------------------------
            projection_ids = [p.id for p in projections]
            calibration_events: List[Dict[str, Any]] = []
            if projection_ids:
                outcomes = (
                    self.session.query(ImpactOutcome)
                    .filter(ImpactOutcome.projection_id.in_(projection_ids))
                    .all()
                )
                outcome_ids = [o.id for o in outcomes]
                if outcome_ids:
                    cal_records = (
                        self.session.query(CalibrationEvent)
                        .filter(CalibrationEvent.outcome_id.in_(outcome_ids))
                        .all()
                    )
                    calibration_events = [
                        {
                            "event_id": c.id,
                            "magnitude_deviation": c.magnitude_deviation,
                            "timing_deviation": c.timing_deviation,
                            "synergy_assumption_error": c.synergy_assumption_error,
                            "new_reliability_coefficient": c.new_reliability_coefficient,
                            "reliability_delta": c.reliability_delta,
                            "timestamp": c.timestamp.isoformat() if c.timestamp else None,
                        }
                        for c in cal_records
                    ]

            # --- Stability --------------------------------------------------
            stability_trend = self._stability.get_stability_trend(agent_id, limit=20)
            aggregate_stability = self._stability.compute_aggregate_agent_stability(agent_id)

            # --- Synergy Participation --------------------------------------
            all_synergy = self.session.query(SynergyDensityMetric).all()
            synergy_participations = [
                {
                    "metric_id": sm.id,
                    "collaboration_structure": sm.collaboration_structure,
                    "synergy_density_ratio": sm.synergy_density_ratio,
                    "timestamp": sm.timestamp.isoformat() if sm.timestamp else None,
                }
                for sm in all_synergy
                if any(
                    nid in agent_node_ids
                    for nid in (sm.collaboration_structure or [])
                )
            ]

            # --- Counterfactual Influence -----------------------------------
            marginal_records = (
                self.session.query(MarginalCooperativeInfluence)
                .filter(MarginalCooperativeInfluence.removed_agent_id == agent_id)
                .all()
            )
            marginal_summaries = [
                {
                    "metric_id": m.id,
                    "source_node_id": m.source_node_id,
                    "marginal_influence_vector": m.marginal_influence_vector,
                    "total_marginal_influence": m.total_marginal_influence,
                    "timestamp": m.timestamp.isoformat() if m.timestamp else None,
                }
                for m in marginal_records
            ]

            # --- Assemble profile -------------------------------------------
            data = {
                "agent_id": agent_id,
                "reliability": {
                    "coefficient": reliability_coeff,
                    "last_updated": reliability_updated,
                },
                "projections": {
                    "count": len(projection_summaries),
                    "records": projection_summaries,
                },
                "calibration": {
                    "count": len(calibration_events),
                    "records": calibration_events,
                },
                "stability": {
                    "aggregate_stability_coefficient": aggregate_stability,
                    "trend": stability_trend,
                },
                "synergy_participation": {
                    "count": len(synergy_participations),
                    "records": synergy_participations,
                },
                "counterfactual_influence": {
                    "count": len(marginal_summaries),
                    "records": marginal_summaries,
                },
            }

            explanation = (
                f"Multi-dimensional impact profile for agent '{agent_id}'. "
                f"Reliability coefficient: {reliability_coeff:.4f}. "
                f"{len(projection_summaries)} forecast(s) on record, "
                f"{len(calibration_events)} calibration event(s), "
                f"aggregate stability coefficient: {aggregate_stability:.4f}. "
                f"Synergy participation count: {len(synergy_participations)}. "
                f"Counterfactual removal simulations: {len(marginal_summaries)}. "
                f"All values are structured vectors; no raw counts are exposed."
            )

            duration = (time.perf_counter() - t0) * 1000
            audit = self._audit.log(
                operation=op,
                algorithm_version=ver.version,
                request_payload=request_payload,
                response_payload=data,
                duration_ms=duration,
                caller_identity=self.caller_identity,
            )
            self.session.commit()

            return success_envelope(
                operation=op,
                api_version=ver.version,
                data=data,
                causal_explanation=explanation,
                audit_id=audit.id,
            )

        except Exception as exc:
            duration = (time.perf_counter() - t0) * 1000
            audit = self._audit.log(
                operation=op,
                algorithm_version=ver.version,
                request_payload=request_payload,
                response_payload=None,
                duration_ms=duration,
                caller_identity=self.caller_identity,
                status="error",
                error_detail=str(exc),
            )
            self.session.commit()
            return error_envelope(
                operation=op,
                api_version=ver.version,
                error_message=str(exc),
                audit_id=audit.id,
            )

    # =====================================================================
    #  6.  trace_provenance
    # =====================================================================
    def trace_provenance(
        self,
        metric_type: str,
        metric_id: str,
    ) -> ApiResponse:
        """
        Reconstruct the full causal path behind any metric.

        Returns
        -------
        ApiResponse
            ``data`` contains the complete provenance record: nodes, edges,
            propagation weights, predictive assumptions, synergy multipliers,
            and a reproducibility verification section.
        """
        op = "trace_provenance"
        ver = get_current_version(op)
        t0 = time.perf_counter()
        request_payload = {
            "metric_type": metric_type,
            "metric_id": metric_id,
        }

        try:
            trace = self._provenance.trace(metric_type, metric_id)

            data = trace  # already a dict / JSON-friendly structure

            explanation = (
                f"Provenance trace for {metric_type} '{metric_id[:8]}…'. "
                f"The full causal path has been reconstructed by walking the "
                f"Impact Graph from the source node(s). The trace includes all "
                f"nodes, edges, propagation weights, predictive assumptions, "
                f"synergy multipliers, and a reproducibility verification "
                f"section. Every score produced by this metric can be re-derived "
                f"from the returned subgraph and parameters."
            )

            duration = (time.perf_counter() - t0) * 1000
            audit = self._audit.log(
                operation=op,
                algorithm_version=ver.version,
                request_payload=request_payload,
                response_payload=data,
                duration_ms=duration,
                caller_identity=self.caller_identity,
            )
            self.session.commit()

            return success_envelope(
                operation=op,
                api_version=ver.version,
                data=data,
                causal_explanation=explanation,
                audit_id=audit.id,
            )

        except Exception as exc:
            duration = (time.perf_counter() - t0) * 1000
            audit = self._audit.log(
                operation=op,
                algorithm_version=ver.version,
                request_payload=request_payload,
                response_payload=None,
                duration_ms=duration,
                caller_identity=self.caller_identity,
                status="error",
                error_detail=str(exc),
            )
            self.session.commit()
            return error_envelope(
                operation=op,
                api_version=ver.version,
                error_message=str(exc),
                audit_id=audit.id,
            )

    # =====================================================================
    #  7.  submit_outcome
    # =====================================================================
    def submit_outcome(
        self,
        projection_id: str,
        realized_impact_vector: Dict[str, float],
        realized_timestamp: Optional[datetime] = None,
        run_calibration: bool = True,
    ) -> ApiResponse:
        """
        Record a realized outcome and optionally trigger recalibration.

        When ``run_calibration`` is True the calibration engine computes
        magnitude deviation, timing deviation, and synergy assumption error,
        then updates the agent's reliability coefficient.

        Returns
        -------
        ApiResponse
            ``data`` contains the outcome record and (if calibration ran)
            the calibration event with deviation metrics and reliability
            update details.
        """
        op = "submit_outcome"
        ver = get_current_version(op)
        t0 = time.perf_counter()
        request_payload = {
            "projection_id": projection_id,
            "realized_impact_vector": realized_impact_vector,
            "realized_timestamp": realized_timestamp.isoformat() if realized_timestamp else None,
            "run_calibration": run_calibration,
        }

        try:
            outcome = self._calibration.record_outcome(
                projection_id=projection_id,
                realized_vector=realized_impact_vector,
                realized_timestamp=realized_timestamp,
            )

            calibration_data = None
            if run_calibration:
                cal = self._calibration.run_calibration(outcome.id)
                calibration_data = {
                    "calibration_event_id": cal.id,
                    "magnitude_deviation": cal.magnitude_deviation,
                    "timing_deviation": cal.timing_deviation,
                    "synergy_assumption_error": cal.synergy_assumption_error,
                    "new_reliability_coefficient": cal.new_reliability_coefficient,
                    "reliability_delta": cal.reliability_delta,
                    "timestamp": cal.timestamp.isoformat() if cal.timestamp else None,
                }

            data = {
                "outcome": {
                    "outcome_id": outcome.id,
                    "projection_id": outcome.projection_id,
                    "realized_impact_vector": outcome.realized_impact_vector,
                    "realized_timestamp": outcome.realized_timestamp.isoformat()
                        if outcome.realized_timestamp else None,
                    "timestamp": outcome.timestamp.isoformat() if outcome.timestamp else None,
                },
                "calibration": calibration_data,
            }

            parts = [
                f"Realized outcome recorded for projection '{projection_id[:8]}…'. "
                f"Realized impact vector: {realized_impact_vector}."
            ]
            if calibration_data:
                parts.append(
                    f" Calibration executed: magnitude deviation = "
                    f"{calibration_data['magnitude_deviation']:.4f}, "
                    f"timing deviation = {calibration_data['timing_deviation']:.4f}, "
                    f"synergy assumption error = "
                    f"{calibration_data['synergy_assumption_error']:.4f}. "
                    f"Agent reliability updated by "
                    f"{calibration_data['reliability_delta']:+.4f} to "
                    f"{calibration_data['new_reliability_coefficient']:.4f}."
                )
            explanation = "".join(parts)

            duration = (time.perf_counter() - t0) * 1000
            audit = self._audit.log(
                operation=op,
                algorithm_version=ver.version,
                request_payload=request_payload,
                response_payload=data,
                duration_ms=duration,
                caller_identity=self.caller_identity,
            )
            self.session.commit()

            return success_envelope(
                operation=op,
                api_version=ver.version,
                data=data,
                causal_explanation=explanation,
                audit_id=audit.id,
            )

        except Exception as exc:
            duration = (time.perf_counter() - t0) * 1000
            audit = self._audit.log(
                operation=op,
                algorithm_version=ver.version,
                request_payload=request_payload,
                response_payload=None,
                duration_ms=duration,
                caller_identity=self.caller_identity,
                status="error",
                error_detail=str(exc),
            )
            self.session.commit()
            return error_envelope(
                operation=op,
                api_version=ver.version,
                error_message=str(exc),
                audit_id=audit.id,
            )

    # =====================================================================
    #  Utility: query audit log
    # =====================================================================
    def query_audit_log(
        self,
        operation: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve audit-log entries for transparency and compliance review.
        """
        entries = self._audit.query_log(
            operation=operation, since=since, limit=limit
        )
        return [
            {
                "id": e.id,
                "operation": e.operation,
                "algorithm_version": e.algorithm_version,
                "request_payload": json.loads(e.request_payload) if e.request_payload else None,
                "response_payload": json.loads(e.response_payload) if e.response_payload else None,
                "duration_ms": e.duration_ms,
                "caller_identity": e.caller_identity,
                "status": e.status,
                "error_detail": e.error_detail,
                "timestamp": e.timestamp.isoformat() if e.timestamp else None,
            }
            for e in entries
        ]
