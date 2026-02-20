"""
Impact Provenance Tracing Engine
================================
Allows reconstruction of the full causal path behind any metric.
Walks the Impact Graph and returns all relevant nodes, edges,
propagation weights, predictive assumptions, and synergy multipliers
involved in producing a specific metric value.  Every score or vector
produced by the system is reproducible and explainable through this
trace logic.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
from sqlalchemy.orm import Session

from models.impact_graph import ImpactNode, ImpactEdge
from models.impact_projection import (
    AgentReliability,
    CalibrationEvent,
    CooperativeIntelligenceMetric,
    CooperativeStabilityMetric,
    DomainMultiplier,
    ImpactOutcome,
    ImpactProjection,
    MarginalCooperativeInfluence,
    SynergyDensityMetric,
    SynergySignature,
    TemporalImpactLedgerEntry,
)
from .graph_manager import GraphManager


# ---------------------------------------------------------------------------
# Trace result typing helpers (plain dicts keep things serialisation-friendly)
# ---------------------------------------------------------------------------

_SUPPORTED_METRIC_TYPES = frozenset([
    "ImpactProjection",
    "SynergyDensityMetric",
    "CooperativeStabilityMetric",
    "CooperativeIntelligenceMetric",
    "CalibrationEvent",
    "MarginalCooperativeInfluence",
    "TemporalImpactLedgerEntry",
    "SynergySignature",
])


class ImpactProvenanceTracingEngine:
    """
    Allows reconstruction of the full causal path behind any metric.
    Walks the Impact Graph and returns all relevant nodes, edges,
    propagation weights, predictive assumptions, and synergy multipliers.
    """

    def __init__(self, session: Session):
        self.session = session
        self.graph_manager = GraphManager(session)

    # ------------------------------------------------------------------
    # Public dispatch
    # ------------------------------------------------------------------

    def trace(self, metric_type: str, metric_id: str) -> Dict[str, Any]:
        """
        Generic trace function that dispatches to specific tracing logic.
        Returns a fully self-contained provenance record that can be used
        to reproduce and explain any metric value.
        """
        if metric_type not in _SUPPORTED_METRIC_TYPES:
            raise ValueError(
                f"Unsupported metric type for tracing: {metric_type}. "
                f"Supported: {sorted(_SUPPORTED_METRIC_TYPES)}"
            )

        dispatch = {
            "ImpactProjection":              self.trace_projection,
            "SynergyDensityMetric":          self.trace_synergy,
            "CooperativeStabilityMetric":    self.trace_stability,
            "CooperativeIntelligenceMetric": self.trace_intelligence,
            "CalibrationEvent":              self.trace_calibration,
            "MarginalCooperativeInfluence":  self.trace_marginal_influence,
            "TemporalImpactLedgerEntry":     self.trace_temporal_entry,
            "SynergySignature":              self.trace_synergy_signature,
        }
        return dispatch[metric_type](metric_id)

    # ------------------------------------------------------------------
    # 1. ImpactProjection trace
    # ------------------------------------------------------------------

    def trace_projection(self, projection_id: str) -> Dict[str, Any]:
        """Reconstructs the full causal path for an ImpactProjection."""
        projection = self._fetch_or_raise(ImpactProjection, projection_id, "ImpactProjection")
        source_node = self._fetch_or_raise(ImpactNode, projection.source_node_id, "source ImpactNode")

        agent_id = self._extract_agent_id(source_node)
        domain = source_node.domain_context.get("domain")

        reliability = self._lookup_reliability(agent_id)
        multiplier_value = self._lookup_domain_multiplier(domain)

        # Causal path reconstruction
        G = self.graph_manager._build_nx_graph()
        path_data = self._extract_causal_subgraph(G, source_node.id)

        # Per-node influence propagation breakdown
        influence_map = self._compute_influence_map(G, [source_node.id], reliability, multiplier_value)

        # Synergy context: any SynergyDensityMetric that references this source node
        synergy_multipliers = self._find_synergy_context(source_node.id)

        # Reproducibility: re-derive the vector and compare
        reproduced = self._recompute_projected_vector(G, [source_node.id], reliability, multiplier_value)
        reproducibility = self._build_reproducibility_report(
            projection.predicted_impact_vector, reproduced
        )

        return {
            "metric_id": projection_id,
            "metric_type": "ImpactProjection",
            "source_node_id": source_node.id,
            "path": path_data,
            "influence_propagation": influence_map,
            "predictive_assumptions": {
                "agent_id": agent_id,
                "agent_reliability": reliability,
                "domain": domain,
                "domain_multiplier": multiplier_value,
                "time_horizon": projection.time_horizon,
                "uncertainty_bounds": projection.uncertainty_bounds,
            },
            "synergy_multipliers": synergy_multipliers,
            "dependency_references": projection.dependency_references or [],
            "result_vector": projection.predicted_impact_vector,
            "reproducibility": reproducibility,
        }

    # ------------------------------------------------------------------
    # 2. SynergyDensityMetric trace
    # ------------------------------------------------------------------

    def trace_synergy(self, metric_id: str) -> Dict[str, Any]:
        """Reconstructs the causal path for a SynergyDensityMetric."""
        metric = self._fetch_or_raise(SynergyDensityMetric, metric_id, "SynergyDensityMetric")

        agent_node_ids: List[str] = metric.collaboration_structure
        constituent_traces: List[Dict[str, Any]] = []

        for node_id in agent_node_ids:
            projection = (
                self.session.query(ImpactProjection)
                .filter(ImpactProjection.source_node_id == node_id)
                .order_by(ImpactProjection.timestamp.desc())
                .first()
            )
            if projection:
                constituent_traces.append(self.trace_projection(projection.id))
            else:
                node = self.session.get(ImpactNode, node_id)
                constituent_traces.append({
                    "source_node_id": node_id,
                    "node_type": node.outcome_type if node else "UNKNOWN",
                    "note": "No historical projection found; trace based on current graph state.",
                })

        # Reproducibility: re-derive the ratio
        sum_ind = sum(metric.independent_impact_sum.values()) if metric.independent_impact_sum else 0.0
        sum_coop = sum(metric.cooperative_impact.values()) if metric.cooperative_impact else 0.0
        reproduced_ratio = (sum_coop / sum_ind) if sum_ind != 0 else (1.0 if sum_coop == 0 else float("inf"))
        reproducibility = {
            "reproduced_synergy_density_ratio": reproduced_ratio,
            "stored_synergy_density_ratio": metric.synergy_density_ratio,
            "match": abs(reproduced_ratio - metric.synergy_density_ratio) < 1e-9,
        }

        return {
            "metric_id": metric_id,
            "metric_type": "SynergyDensityMetric",
            "collaboration_structure": agent_node_ids,
            "synergy_multiplier": metric.synergy_density_ratio,
            "independent_sum": metric.independent_impact_sum,
            "cooperative_impact": metric.cooperative_impact,
            "constituent_traces": constituent_traces,
            "reproducibility": reproducibility,
        }

    # ------------------------------------------------------------------
    # 3. CooperativeStabilityMetric trace
    # ------------------------------------------------------------------

    def trace_stability(self, metric_id: str) -> Dict[str, Any]:
        """Reconstructs the tracing for a CooperativeStabilityMetric."""
        metric = self._fetch_or_raise(CooperativeStabilityMetric, metric_id, "CooperativeStabilityMetric")

        components = {
            "negotiation_convergence_time": metric.negotiation_convergence_time,
            "resource_allocation_variance": metric.resource_allocation_variance,
            "conflict_resolution_frequency": metric.conflict_resolution_frequency,
            "team_performance_stability": metric.team_performance_stability,
        }

        # Reproducibility: re-derive the stability coefficient using the
        # same formula used in CooperativeStabilityEngine
        reproduced_coeff = self._recompute_stability_coefficient(
            metric.negotiation_convergence_time,
            metric.resource_allocation_variance,
            metric.conflict_resolution_frequency,
            metric.team_performance_stability,
        )
        reproducibility = {
            "reproduced_stability_coefficient": reproduced_coeff,
            "stored_stability_coefficient": metric.stability_coefficient,
            "match": abs(reproduced_coeff - metric.stability_coefficient) < 1e-9,
        }

        return {
            "metric_id": metric_id,
            "metric_type": "CooperativeStabilityMetric",
            "agent_id": metric.agent_id,
            "predictive_assumptions": {
                "team_composition": metric.team_composition,
            },
            "components": components,
            "derivation": {
                "formula": "1.0 / (1.0 + 0.3*f_conv + 0.3*f_var + 0.3*f_conf + 0.1*f_instab)",
                "normalization_targets": {
                    "T_NORM": 10.0,
                    "V_NORM": 0.2,
                    "C_NORM": 2.0,
                    "S_NORM": 1.0,
                },
            },
            "result_coefficient": metric.stability_coefficient,
            "reproducibility": reproducibility,
        }

    # ------------------------------------------------------------------
    # 4. CooperativeIntelligenceMetric trace
    # ------------------------------------------------------------------

    def trace_intelligence(self, metric_id: str) -> Dict[str, Any]:
        """Reconstructs the tracing for a CooperativeIntelligenceMetric."""
        metric = self._fetch_or_raise(
            CooperativeIntelligenceMetric, metric_id, "CooperativeIntelligenceMetric"
        )

        sub_metrics = {
            "uncertainty_reduction": metric.uncertainty_reduction,
            "dependency_graph_enrichment": metric.dependency_graph_enrichment,
            "predictive_calibration_improvement": metric.predictive_calibration_improvement,
            "cross_role_integration_depth": metric.cross_role_integration_depth,
        }

        # Reproducibility: re-derive the vector from stored sub-metrics
        reproduced_vector = {
            "uncertainty_reduction": (metric.uncertainty_reduction or {}).get("reduction_ratio", 0.0),
            "dependency_graph_enrichment": (metric.dependency_graph_enrichment or {}).get("enrichment_ratio", 0.0),
            "predictive_calibration_improvement": (metric.predictive_calibration_improvement or {}).get("relative_improvement", 0.0),
            "cross_role_integration_depth": (metric.cross_role_integration_depth or {}).get("normalized_depth", 0.0),
        }

        stored_vector = metric.cooperative_intelligence_vector or {}
        match = all(
            abs(reproduced_vector.get(k, 0.0) - stored_vector.get(k, 0.0)) < 1e-9
            for k in set(reproduced_vector) | set(stored_vector)
        )

        return {
            "metric_id": metric_id,
            "metric_type": "CooperativeIntelligenceMetric",
            "collaboration_structure": metric.collaboration_structure,
            "sub_metrics": sub_metrics,
            "result_vector": stored_vector,
            "reproducibility": {
                "reproduced_vector": reproduced_vector,
                "stored_vector": stored_vector,
                "match": match,
            },
        }

    # ------------------------------------------------------------------
    # 5. CalibrationEvent trace
    # ------------------------------------------------------------------

    def trace_calibration(self, event_id: str) -> Dict[str, Any]:
        """
        Reconstructs the tracing for a CalibrationEvent.
        Links back to the outcome, its projection, and the source node.
        """
        event = self._fetch_or_raise(CalibrationEvent, event_id, "CalibrationEvent")

        outcome = self._fetch_or_raise(ImpactOutcome, event.outcome_id, "ImpactOutcome")
        projection = self._fetch_or_raise(ImpactProjection, outcome.projection_id, "ImpactProjection")
        source_node = self.session.get(ImpactNode, projection.source_node_id)

        agent_id = self._extract_agent_id(source_node) if source_node else None

        # Reproducibility: re-derive deviations
        pred_vec = projection.predicted_impact_vector or {}
        real_vec = outcome.realized_impact_vector or {}
        all_keys = set(pred_vec.keys()) | set(real_vec.keys())
        total_error = sum(abs(real_vec.get(k, 0.0) - pred_vec.get(k, 0.0)) for k in all_keys)
        total_pred_mag = sum(abs(pred_vec.get(k, 0.0)) for k in all_keys)
        reproduced_mag_dev = total_error / max(total_pred_mag, 1.0)

        expected_time = projection.timestamp + timedelta(seconds=projection.time_horizon)
        time_diff = abs((outcome.realized_timestamp - expected_time).total_seconds())
        reproduced_timing_dev = time_diff / max(projection.time_horizon, 1.0)

        return {
            "metric_id": event_id,
            "metric_type": "CalibrationEvent",
            "outcome_id": outcome.id,
            "projection_id": projection.id,
            "source_node_id": projection.source_node_id,
            "agent_id": agent_id,
            "predicted_vector": pred_vec,
            "realized_vector": real_vec,
            "deviations": {
                "magnitude_deviation": event.magnitude_deviation,
                "timing_deviation": event.timing_deviation,
                "synergy_assumption_error": event.synergy_assumption_error,
            },
            "reliability_update": {
                "new_reliability_coefficient": event.new_reliability_coefficient,
                "reliability_delta": event.reliability_delta,
            },
            "reproducibility": {
                "reproduced_magnitude_deviation": round(reproduced_mag_dev, 9),
                "stored_magnitude_deviation": event.magnitude_deviation,
                "magnitude_match": abs(reproduced_mag_dev - event.magnitude_deviation) < 1e-6,
                "reproduced_timing_deviation": round(reproduced_timing_dev, 9),
                "stored_timing_deviation": event.timing_deviation,
                "timing_match": abs(reproduced_timing_dev - event.timing_deviation) < 1e-6,
            },
        }

    # ------------------------------------------------------------------
    # 6. MarginalCooperativeInfluence trace
    # ------------------------------------------------------------------

    def trace_marginal_influence(self, metric_id: str) -> Dict[str, Any]:
        """Reconstructs the tracing for a MarginalCooperativeInfluence record."""
        metric = self._fetch_or_raise(
            MarginalCooperativeInfluence, metric_id, "MarginalCooperativeInfluence"
        )

        source_node = self.session.get(ImpactNode, metric.source_node_id)
        agent_id = self._extract_agent_id(source_node) if source_node else None
        domain = (source_node.domain_context.get("domain") if source_node else None)

        reliability = self._lookup_reliability(agent_id)
        multiplier_value = self._lookup_domain_multiplier(domain)

        # Full graph path from the source node
        G = self.graph_manager._build_nx_graph()
        path_data = self._extract_causal_subgraph(G, metric.source_node_id)

        # Nodes belonging to the removed agent
        removed_agent_nodes = self._get_agent_node_ids(metric.removed_agent_id)

        # Reproducibility: re-derive the marginal influence vector
        reproduced_full = self._recompute_projected_vector(
            G, [metric.source_node_id], reliability, multiplier_value
        )
        G_disabled = self.graph_manager._build_nx_graph(disabled_node_ids=removed_agent_nodes)
        reproduced_cf = self._recompute_projected_vector(
            G_disabled, [metric.source_node_id], reliability, multiplier_value
        )
        reproduced_delta = {
            k: reproduced_full.get(k, 0.0) - reproduced_cf.get(k, 0.0)
            for k in set(reproduced_full) | set(reproduced_cf)
            if abs(reproduced_full.get(k, 0.0) - reproduced_cf.get(k, 0.0)) > 1e-12
        }

        return {
            "metric_id": metric_id,
            "metric_type": "MarginalCooperativeInfluence",
            "source_node_id": metric.source_node_id,
            "removed_agent_id": metric.removed_agent_id,
            "removed_agent_node_ids": sorted(removed_agent_nodes),
            "full_projection_vector": metric.full_projection_vector,
            "counterfactual_projection_vector": metric.counterfactual_projection_vector,
            "marginal_influence_vector": metric.marginal_influence_vector,
            "total_marginal_influence": metric.total_marginal_influence,
            "path": path_data,
            "predictive_assumptions": {
                "agent_id": agent_id,
                "agent_reliability": reliability,
                "domain": domain,
                "domain_multiplier": multiplier_value,
                "time_horizon": metric.time_horizon,
            },
            "reproducibility": {
                "reproduced_marginal_vector": reproduced_delta,
                "stored_marginal_vector": metric.marginal_influence_vector,
                "match": self._vectors_match(reproduced_delta, metric.marginal_influence_vector),
            },
        }

    # ------------------------------------------------------------------
    # 7. TemporalImpactLedgerEntry trace
    # ------------------------------------------------------------------

    def trace_temporal_entry(self, entry_id: str) -> Dict[str, Any]:
        """Reconstructs the tracing for a TemporalImpactLedgerEntry."""
        entry = self._fetch_or_raise(
            TemporalImpactLedgerEntry, entry_id, "TemporalImpactLedgerEntry"
        )

        source_node = self.session.get(ImpactNode, entry.source_node_id)
        projection = self.session.get(ImpactProjection, entry.projection_id) if entry.projection_id else None

        # Compute current decayed value for reproducibility
        now = datetime.utcnow()
        elapsed = max((now - entry.timestamp).total_seconds(), 0.0)
        decay_weight = self._compute_decay_weight(
            entry.decay_function, elapsed, entry.decay_rate, entry.decay_floor
        )
        decayed_vector = {
            k: v * decay_weight for k, v in (entry.impact_vector or {}).items()
        }

        return {
            "metric_id": entry_id,
            "metric_type": "TemporalImpactLedgerEntry",
            "task_sequence_id": entry.task_sequence_id,
            "source_node_id": entry.source_node_id,
            "source_node_type": source_node.outcome_type if source_node else "UNKNOWN",
            "projection_id": entry.projection_id,
            "impact_vector": entry.impact_vector,
            "decay_parameters": {
                "function": entry.decay_function,
                "rate": entry.decay_rate,
                "floor": entry.decay_floor,
            },
            "temporal_context": {
                "entry_timestamp": entry.timestamp.isoformat() if entry.timestamp else None,
                "elapsed_seconds": round(elapsed, 3),
                "current_decay_weight": round(decay_weight, 9),
                "current_decayed_vector": decayed_vector,
            },
            "entry_metadata": entry.entry_metadata,
            "linked_projection_trace": (
                self.trace_projection(projection.id) if projection else None
            ),
        }

    # ------------------------------------------------------------------
    # 8. SynergySignature trace
    # ------------------------------------------------------------------

    def trace_synergy_signature(self, signature_id: str) -> Dict[str, Any]:
        """
        Reconstructs the tracing for a SynergySignature.
        Re-derives amplification and stability from historical metrics.
        """
        signature = self._fetch_or_raise(SynergySignature, signature_id, "SynergySignature")

        pattern = tuple(sorted(signature.collaboration_structure))

        # Gather all underlying SynergyDensityMetrics that produced this signature
        all_synergy_metrics = self.session.query(SynergyDensityMetric).all()
        matching_metrics = [
            m for m in all_synergy_metrics
            if tuple(sorted(m.collaboration_structure)) == pattern
        ]

        ratios = [m.synergy_density_ratio for m in matching_metrics]
        frequency = len(ratios)
        mean_ratio = (sum(ratios) / frequency) if frequency else 0.0

        # Reproduce stability score (CV-based)
        if frequency <= 1 or mean_ratio == 0:
            reproduced_stability = 1.0 if frequency == 1 else 0.0
        else:
            variance = sum((r - mean_ratio) ** 2 for r in ratios) / frequency
            std_dev = variance ** 0.5
            cv = std_dev / abs(mean_ratio)
            reproduced_stability = 1.0 / (1.0 + cv)

        return {
            "metric_id": signature_id,
            "metric_type": "SynergySignature",
            "collaboration_structure": signature.collaboration_structure,
            "observation_frequency": signature.observation_frequency,
            "above_baseline_consistency": signature.above_baseline_consistency,
            "mean_synergy_density_ratio": signature.mean_synergy_density_ratio,
            "amplification_magnitude": signature.amplification_magnitude,
            "stability_score": signature.stability_score,
            "baseline_synergy_density": signature.baseline_synergy_density,
            "underlying_metric_ids": [m.id for m in matching_metrics],
            "underlying_ratios": ratios,
            "reproducibility": {
                "reproduced_frequency": frequency,
                "reproduced_mean_ratio": round(mean_ratio, 9),
                "reproduced_stability_score": round(reproduced_stability, 9),
                "frequency_match": frequency == signature.observation_frequency,
                "mean_ratio_match": abs(mean_ratio - signature.mean_synergy_density_ratio) < 1e-6,
                "stability_match": abs(reproduced_stability - signature.stability_score) < 1e-6,
            },
        }

    # ------------------------------------------------------------------
    # Reproducibility verification
    # ------------------------------------------------------------------

    def verify_reproducibility(
        self, metric_type: str, metric_id: str
    ) -> Dict[str, Any]:
        """
        Convenience wrapper: traces a metric and returns only the
        reproducibility section.  Raises ValueError if the metric
        cannot be reproduced exactly.
        """
        trace_result = self.trace(metric_type, metric_id)
        repro = trace_result.get("reproducibility")
        if repro is None:
            return {"status": "no_reproducibility_check", "metric_type": metric_type}

        # Determine overall match
        is_match = repro.get("match", True)
        if isinstance(is_match, bool) and not is_match:
            return {
                "status": "mismatch",
                "metric_type": metric_type,
                "metric_id": metric_id,
                "details": repro,
            }
        return {
            "status": "verified",
            "metric_type": metric_type,
            "metric_id": metric_id,
            "details": repro,
        }

    # ==================================================================
    # Internal helpers
    # ==================================================================

    # --- Data fetching helpers ---

    def _fetch_or_raise(self, model, record_id: str, label: str):
        """Fetch a record by primary key or raise ValueError."""
        record = self.session.query(model).filter(model.id == record_id).first()
        if not record:
            raise ValueError(f"{label} {record_id} not found")
        return record

    @staticmethod
    def _extract_agent_id(node: ImpactNode) -> Optional[str]:
        if node is None:
            return None
        ctx = node.domain_context or {}
        return ctx.get("agent_id") or ctx.get("agent")

    def _lookup_reliability(self, agent_id: Optional[str]) -> float:
        if not agent_id:
            return 1.0
        record = (
            self.session.query(AgentReliability)
            .filter(AgentReliability.agent_id == agent_id)
            .first()
        )
        return record.reliability_coefficient if record else 1.0

    def _lookup_domain_multiplier(self, domain: Optional[str]) -> float:
        if not domain:
            return 1.0
        record = (
            self.session.query(DomainMultiplier)
            .filter(DomainMultiplier.domain_name == domain)
            .first()
        )
        return record.multiplier if record else 1.0

    def _get_agent_node_ids(self, agent_id: str) -> Set[str]:
        """Return all ImpactNode IDs belonging to an agent."""
        nodes = self.session.query(ImpactNode).all()
        return {
            n.id for n in nodes
            if (n.domain_context.get("agent_id") or n.domain_context.get("agent")) == agent_id
        }

    # --- Graph extraction and influence propagation ---

    def _extract_causal_subgraph(
        self, G: nx.DiGraph, source_node_id: str
    ) -> Dict[str, Any]:
        """
        Extracts all nodes and edges reachable from ``source_node_id``
        together with propagation weights, confidence scores, and delays.
        """
        if source_node_id not in G:
            return {"nodes": [], "edges": []}

        descendants = nx.descendants(G, source_node_id)
        relevant_ids = {source_node_id} | descendants

        nodes_data = []
        for nid in sorted(relevant_ids):
            node = self.session.get(ImpactNode, nid)
            if node:
                nodes_data.append({
                    "id": node.id,
                    "type": node.outcome_type,
                    "magnitude": node.magnitude,
                    "context": node.domain_context,
                })

        edges_data = []
        for u, v in G.edges():
            if u in relevant_ids and v in relevant_ids:
                ed = G.get_edge_data(u, v)
                edges_data.append({
                    "source": u,
                    "target": v,
                    "causal_weight": ed["weight"],
                    "confidence": ed["confidence"],
                    "delay": ed["delay"],
                })

        return {"nodes": nodes_data, "edges": edges_data}

    def _compute_influence_map(
        self,
        G: nx.DiGraph,
        start_node_ids: List[str],
        reliability: float,
        multiplier: float,
    ) -> List[Dict[str, Any]]:
        """
        Performs a step-by-step influence propagation through the graph.
        Returns a list of per-node influence entries ordered by propagation
        sequence, showing how influence flows through the causal network.
        """
        valid_starts = [sid for sid in start_node_ids if sid in G]
        if not valid_starts:
            return []

        influence: Dict[str, float] = {sid: 1.0 for sid in valid_starts}
        entries: List[Dict[str, Any]] = []

        is_dag = nx.is_directed_acyclic_graph(G)

        if is_dag:
            all_descendants: Set[str] = set()
            for sid in valid_starts:
                all_descendants.update(nx.descendants(G, sid))
                all_descendants.add(sid)

            for n_id in nx.topological_sort(G):
                if n_id not in all_descendants:
                    continue
                inf = influence.get(n_id, 0.0)
                if inf <= 0:
                    continue

                node = self.session.get(ImpactNode, n_id)
                contribution = inf * (node.magnitude if node else 0.0) * multiplier * reliability

                entries.append({
                    "node_id": n_id,
                    "incoming_influence": round(inf, 12),
                    "node_magnitude": node.magnitude if node else 0.0,
                    "reliability": reliability,
                    "multiplier": multiplier,
                    "contribution": round(contribution, 12),
                    "outgoing_edges": [],
                })

                for succ in G.successors(n_id):
                    ed = G.get_edge_data(n_id, succ)
                    propagated = inf * ed["weight"] * ed["confidence"]
                    influence[succ] = influence.get(succ, 0.0) + propagated
                    entries[-1]["outgoing_edges"].append({
                        "target": succ,
                        "weight": ed["weight"],
                        "confidence": ed["confidence"],
                        "propagated_influence": round(propagated, 12),
                    })
        else:
            # BFS fallback for cyclic graphs
            queue = list(valid_starts)
            processed: Set[str] = set(valid_starts)

            while queue:
                curr_id = queue.pop(0)
                inf = influence.get(curr_id, 0.0)
                node = self.session.get(ImpactNode, curr_id)
                contribution = inf * (node.magnitude if node else 0.0) * multiplier * reliability

                entry: Dict[str, Any] = {
                    "node_id": curr_id,
                    "incoming_influence": round(inf, 12),
                    "node_magnitude": node.magnitude if node else 0.0,
                    "reliability": reliability,
                    "multiplier": multiplier,
                    "contribution": round(contribution, 12),
                    "outgoing_edges": [],
                }

                for succ in G.successors(curr_id):
                    ed = G.get_edge_data(curr_id, succ)
                    propagated = inf * ed["weight"] * ed["confidence"]
                    influence[succ] = influence.get(succ, 0.0) + propagated
                    entry["outgoing_edges"].append({
                        "target": succ,
                        "weight": ed["weight"],
                        "confidence": ed["confidence"],
                        "propagated_influence": round(propagated, 12),
                    })
                    if succ not in processed:
                        queue.append(succ)
                        processed.add(succ)

                entries.append(entry)

        return entries

    def _recompute_projected_vector(
        self,
        G: nx.DiGraph,
        start_node_ids: List[str],
        reliability: float,
        multiplier: float,
    ) -> Dict[str, float]:
        """
        Mirrors the logic in ImpactForecastEngine._compute_projected_vector
        so re-derivation is deterministic.
        """
        valid_starts = [sid for sid in start_node_ids if sid in G]
        if not valid_starts:
            return {}

        influence: Dict[str, float] = {sid: 1.0 for sid in valid_starts}
        vector: Dict[str, float] = {}

        is_dag = nx.is_directed_acyclic_graph(G)

        if is_dag:
            all_desc: Set[str] = set()
            for sid in valid_starts:
                all_desc.update(nx.descendants(G, sid))
                all_desc.add(sid)

            for n_id in nx.topological_sort(G):
                if n_id not in all_desc:
                    continue
                inf = influence.get(n_id, 0.0)
                if inf <= 0:
                    continue

                node = self.session.get(ImpactNode, n_id)
                if node:
                    v_type = node.outcome_type
                    vector[v_type] = vector.get(v_type, 0.0) + (inf * node.magnitude * multiplier * reliability)

                for succ in G.successors(n_id):
                    ed = G.get_edge_data(n_id, succ)
                    added = inf * ed["weight"] * ed["confidence"]
                    influence[succ] = influence.get(succ, 0.0) + added
        else:
            queue = list(valid_starts)
            processed: Set[str] = set(valid_starts)
            while queue:
                curr = queue.pop(0)
                inf = influence[curr]
                node = self.session.get(ImpactNode, curr)
                if node:
                    v_type = node.outcome_type
                    vector[v_type] = vector.get(v_type, 0.0) + (inf * node.magnitude * multiplier * reliability)
                for succ in G.successors(curr):
                    ed = G.get_edge_data(curr, succ)
                    influence[succ] = influence.get(succ, 0.0) + (inf * ed["weight"] * ed["confidence"])
                    if succ not in processed:
                        queue.append(succ)
                        processed.add(succ)

        return vector

    # --- Synergy context discovery ---

    def _find_synergy_context(self, node_id: str) -> List[Dict[str, Any]]:
        """
        Finds any SynergyDensityMetric whose collaboration_structure
        includes ``node_id`` and returns summary info.
        """
        all_metrics = self.session.query(SynergyDensityMetric).all()
        results = []
        for m in all_metrics:
            if node_id in (m.collaboration_structure or []):
                results.append({
                    "synergy_metric_id": m.id,
                    "synergy_density_ratio": m.synergy_density_ratio,
                    "collaboration_structure": m.collaboration_structure,
                })
        return results

    # --- Reproducibility helpers ---

    @staticmethod
    def _build_reproducibility_report(
        stored_vector: Dict[str, float],
        reproduced_vector: Dict[str, float],
    ) -> Dict[str, Any]:
        all_keys = set(stored_vector.keys()) | set(reproduced_vector.keys())
        per_key = {}
        overall_match = True
        for k in all_keys:
            s = stored_vector.get(k, 0.0)
            r = reproduced_vector.get(k, 0.0)
            match = abs(s - r) < 1e-6
            per_key[k] = {"stored": s, "reproduced": r, "match": match}
            if not match:
                overall_match = False

        return {
            "reproduced_vector": reproduced_vector,
            "stored_vector": stored_vector,
            "per_key_comparison": per_key,
            "match": overall_match,
        }

    @staticmethod
    def _vectors_match(
        a: Dict[str, float], b: Dict[str, float], tol: float = 1e-6
    ) -> bool:
        all_keys = set(a.keys()) | set(b.keys())
        return all(abs(a.get(k, 0.0) - b.get(k, 0.0)) < tol for k in all_keys)

    @staticmethod
    def _recompute_stability_coefficient(
        conv_time: float,
        alloc_var: float,
        conflict_freq: float,
        perf_instability: float,
    ) -> float:
        """
        Mirrors CooperativeStabilityEngine._compute_stability_coefficient
        """
        T_NORM, V_NORM, C_NORM, S_NORM = 10.0, 0.2, 2.0, 1.0

        f_conv = min(conv_time / T_NORM, 3.0)
        f_var = min(alloc_var / V_NORM, 3.0)
        f_conf = min(conflict_freq / C_NORM, 3.0)
        f_instab = min(perf_instability / S_NORM, 3.0)

        structural = (0.3 * f_conv) + (0.3 * f_var) + (0.3 * f_conf)
        performance = 0.1 * f_instab

        return round(1.0 / (1.0 + structural + performance), 4)

    @staticmethod
    def _compute_decay_weight(
        function_name: str,
        elapsed_seconds: float,
        decay_rate: float,
        decay_floor: float,
    ) -> float:
        """Mirrors TemporalImpactMemoryEngine._decay_weight."""
        if function_name == "none":
            return 1.0
        if function_name == "linear":
            return max(decay_floor, 1.0 - (decay_rate * elapsed_seconds))
        # default: exponential
        return max(decay_floor, math.exp(-decay_rate * elapsed_seconds))
