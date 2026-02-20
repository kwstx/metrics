from statistics import fmean, pstdev
from typing import Dict, List, Set

import networkx as nx
from sqlalchemy.orm import Session

from engine.graph_manager import GraphManager
from engine.agent_impact_profile_engine import AgentImpactProfileEngine
from models.impact_graph import ImpactNode
from models.impact_projection import CooperativeIntelligenceMetric


class CooperativeIntelligenceIndexEngine:
    """
    Generates a Cooperative Intelligence Vector focused on collective reasoning gains.
    """

    def __init__(self, session: Session):
        self.session = session
        self.graph_manager = GraphManager(session)
        self.profile_engine = AgentImpactProfileEngine(session)

    def generate_cooperative_intelligence_vector(
        self,
        agent_node_ids: List[str],
        baseline_predictions: Dict[str, Dict[str, float]],
        cooperative_predictions: Dict[str, Dict[str, float]],
        calibration_errors_before: Dict[str, float],
        calibration_errors_after: Dict[str, float],
    ) -> CooperativeIntelligenceMetric:
        if not agent_node_ids:
            raise ValueError("agent_node_ids cannot be empty")

        graph = self.graph_manager._build_nx_graph()
        involved_nodes = self._collect_involved_nodes(graph, agent_node_ids)
        subgraph = graph.subgraph(involved_nodes).copy()

        uncertainty_reduction = self._compute_uncertainty_reduction(
            baseline_predictions,
            cooperative_predictions,
        )
        dependency_graph_enrichment = self._compute_dependency_graph_enrichment(subgraph)
        predictive_calibration_improvement = self._compute_predictive_calibration_improvement(
            calibration_errors_before,
            calibration_errors_after,
        )
        cross_role_integration_depth = self._compute_cross_role_integration_depth(subgraph)

        cooperative_intelligence_vector = {
            "uncertainty_reduction": uncertainty_reduction["reduction_ratio"],
            "dependency_graph_enrichment": dependency_graph_enrichment["enrichment_ratio"],
            "predictive_calibration_improvement": predictive_calibration_improvement["relative_improvement"],
            "cross_role_integration_depth": cross_role_integration_depth["normalized_depth"],
        }

        metric = CooperativeIntelligenceMetric(
            collaboration_structure=agent_node_ids,
            uncertainty_reduction=uncertainty_reduction,
            dependency_graph_enrichment=dependency_graph_enrichment,
            predictive_calibration_improvement=predictive_calibration_improvement,
            cross_role_integration_depth=cross_role_integration_depth,
            cooperative_intelligence_vector=cooperative_intelligence_vector,
        )
        self.session.add(metric)
        self.session.commit()

        self.profile_engine.update_cross_role_integration_depth(
            agent_ids=self._resolve_agent_ids(agent_node_ids),
            normalized_depth=cross_role_integration_depth["normalized_depth"],
        )

        return metric

    def _resolve_agent_ids(self, node_ids: List[str]) -> Set[str]:
        agent_ids: Set[str] = set()
        for node_id in node_ids:
            node = self._get_node(node_id)
            if not node:
                continue
            context = node.domain_context or {}
            agent_id = context.get("agent_id") or context.get("agent")
            if agent_id:
                agent_ids.add(agent_id)
        return agent_ids

    def _collect_involved_nodes(self, graph: nx.DiGraph, start_node_ids: List[str]) -> Set[str]:
        involved_nodes: Set[str] = set()
        for node_id in start_node_ids:
            if node_id not in graph:
                continue
            involved_nodes.add(node_id)
            involved_nodes.update(nx.descendants(graph, node_id))
        return involved_nodes

    def _compute_uncertainty_reduction(
        self,
        baseline_predictions: Dict[str, Dict[str, float]],
        cooperative_predictions: Dict[str, Dict[str, float]],
    ) -> Dict[str, float]:
        baseline_dispersion = self._shared_prediction_dispersion(baseline_predictions)
        cooperative_dispersion = self._shared_prediction_dispersion(cooperative_predictions)

        if baseline_dispersion == 0.0:
            reduction_ratio = 0.0
        else:
            reduction_ratio = (baseline_dispersion - cooperative_dispersion) / baseline_dispersion

        return {
            "baseline_dispersion": baseline_dispersion,
            "cooperative_dispersion": cooperative_dispersion,
            "reduction_ratio": reduction_ratio,
        }

    def _shared_prediction_dispersion(self, predictions: Dict[str, Dict[str, float]]) -> float:
        if not predictions:
            return 0.0

        outcome_sets = [set(outcome_map.keys()) for outcome_map in predictions.values() if outcome_map]
        if not outcome_sets:
            return 0.0

        shared_outcomes = set.intersection(*outcome_sets)
        if not shared_outcomes:
            return 0.0

        dispersions = []
        for outcome in shared_outcomes:
            values = [prediction[outcome] for prediction in predictions.values() if outcome in prediction]
            if len(values) < 2:
                continue
            dispersions.append(pstdev(values))

        if not dispersions:
            return 0.0
        return fmean(dispersions)

    def _compute_dependency_graph_enrichment(self, subgraph: nx.DiGraph) -> Dict[str, float]:
        total_nodes = len(subgraph.nodes)
        total_edges = len(subgraph.edges)
        cross_agent_edges = 0

        for source_node_id, target_node_id in subgraph.edges:
            source_agent = self._get_agent_for_node(source_node_id)
            target_agent = self._get_agent_for_node(target_node_id)
            if source_agent and target_agent and source_agent != target_agent:
                cross_agent_edges += 1

        enrichment_ratio = (cross_agent_edges / total_edges) if total_edges else 0.0
        return {
            "involved_nodes": float(total_nodes),
            "involved_edges": float(total_edges),
            "cross_agent_edges": float(cross_agent_edges),
            "enrichment_ratio": enrichment_ratio,
        }

    def _compute_predictive_calibration_improvement(
        self,
        before: Dict[str, float],
        after: Dict[str, float],
    ) -> Dict[str, float]:
        shared_outcomes = set(before.keys()) & set(after.keys())
        if not shared_outcomes:
            return {
                "mean_error_before": 0.0,
                "mean_error_after": 0.0,
                "absolute_improvement": 0.0,
                "relative_improvement": 0.0,
            }

        mean_before = fmean(before[outcome] for outcome in shared_outcomes)
        mean_after = fmean(after[outcome] for outcome in shared_outcomes)
        absolute_improvement = mean_before - mean_after
        relative_improvement = (absolute_improvement / mean_before) if mean_before else 0.0

        return {
            "mean_error_before": mean_before,
            "mean_error_after": mean_after,
            "absolute_improvement": absolute_improvement,
            "relative_improvement": relative_improvement,
        }

    def _compute_cross_role_integration_depth(self, subgraph: nx.DiGraph) -> Dict[str, float]:
        total_edges = len(subgraph.edges)
        role_handoffs = 0
        roles = set()

        for node_id in subgraph.nodes:
            roles.add(self._get_role_for_node(node_id))

        for source_node_id, target_node_id in subgraph.edges:
            if self._get_role_for_node(source_node_id) != self._get_role_for_node(target_node_id):
                role_handoffs += 1

        if total_edges == 0:
            normalized_depth = 0.0
        else:
            normalized_depth = role_handoffs / total_edges

        path_depth = self._longest_path_depth(subgraph)

        return {
            "unique_roles": float(len(roles)),
            "role_handoffs": float(role_handoffs),
            "path_depth": float(path_depth),
            "normalized_depth": normalized_depth,
        }

    def _longest_path_depth(self, subgraph: nx.DiGraph) -> int:
        if len(subgraph.nodes) == 0:
            return 0
        if nx.is_directed_acyclic_graph(subgraph):
            return nx.dag_longest_path_length(subgraph)
        return 0

    def _get_node(self, node_id: str) -> ImpactNode | None:
        return self.session.get(ImpactNode, node_id)

    def _get_agent_for_node(self, node_id: str) -> str:
        node = self._get_node(node_id)
        if not node:
            return ""
        context = node.domain_context or {}
        return context.get("agent_id") or context.get("agent") or ""

    def _get_role_for_node(self, node_id: str) -> str:
        node = self._get_node(node_id)
        if not node:
            return "unknown"
        context = node.domain_context or {}
        return context.get("role") or context.get("agent_role") or context.get("function") or "unknown"
