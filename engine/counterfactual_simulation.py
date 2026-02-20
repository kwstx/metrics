from typing import Dict, List, Set
from sqlalchemy.orm import Session

from engine.impact_forecast_engine import ImpactForecastEngine
from engine.agent_impact_profile_engine import AgentImpactProfileEngine
from models.impact_graph import ImpactNode
from models.impact_projection import MarginalCooperativeInfluence


class CounterfactualSimulation:
    """
    Recomputes projected impact after removing an agent's outgoing action edges.
    Persists marginal deltas as Marginal Cooperative Influence.
    """

    def __init__(self, session: Session):
        self.session = session
        self.forecast_engine = ImpactForecastEngine(session)
        self.profile_engine = AgentImpactProfileEngine(session)

    def simulate_agent_removal(
        self,
        source_node_id: str,
        removed_agent_id: str,
        time_horizon: float = 30.0
    ) -> MarginalCooperativeInfluence:
        source_node = self.session.get(ImpactNode, source_node_id)
        if not source_node:
            raise ValueError(f"Source node {source_node_id} not found")

        agent_id = source_node.domain_context.get("agent_id") or source_node.domain_context.get("agent")
        domain = source_node.domain_context.get("domain")

        reliability = self.forecast_engine._get_agent_reliability(agent_id)
        multiplier = self.forecast_engine._get_domain_multiplier(domain)

        full_projection = self.forecast_engine._compute_projected_vector(
            [source_node_id],
            reliability,
            multiplier
        )

        disabled_sources = self._get_agent_action_node_ids(removed_agent_id)
        counterfactual_projection = self.forecast_engine._compute_projected_vector(
            [source_node_id],
            reliability,
            multiplier,
            disabled_node_ids=disabled_sources
        )

        marginal_vector = self._compute_delta(full_projection, counterfactual_projection)
        total_marginal = self._aggregate_total_marginal(marginal_vector)
        dependencies = self.forecast_engine._get_dependencies(source_node_id)

        metric = MarginalCooperativeInfluence(
            source_node_id=source_node_id,
            removed_agent_id=removed_agent_id,
            full_projection_vector=full_projection,
            counterfactual_projection_vector=counterfactual_projection,
            marginal_influence_vector=marginal_vector,
            total_marginal_influence=total_marginal,
            time_horizon=time_horizon,
            dependency_references=dependencies,
        )
        self.session.add(metric)
        self.session.commit()

        self.profile_engine.update_marginal_cooperative_influence(
            agent_id=removed_agent_id,
            total_marginal_influence=total_marginal,
            marginal_vector=marginal_vector,
        )

        return metric

    def _get_agent_action_node_ids(self, agent_id: str) -> Set[str]:
        nodes = self.session.query(ImpactNode).all()
        return {
            node.id for node in nodes
            if (node.domain_context.get("agent_id") or node.domain_context.get("agent")) == agent_id
        }

    def _compute_delta(
        self,
        full_projection: Dict[str, float],
        counterfactual_projection: Dict[str, float]
    ) -> Dict[str, float]:
        deltas: Dict[str, float] = {}
        keys = set(full_projection.keys()) | set(counterfactual_projection.keys())
        for key in keys:
            diff = full_projection.get(key, 0.0) - counterfactual_projection.get(key, 0.0)
            if diff != 0.0:
                deltas[key] = diff
        return deltas

    def _aggregate_total_marginal(self, marginal_vector: Dict[str, float]) -> float:
        outcome_deltas = [value for key, value in marginal_vector.items() if "ACTION" not in key]
        if outcome_deltas:
            return sum(outcome_deltas)
        return sum(marginal_vector.values())
