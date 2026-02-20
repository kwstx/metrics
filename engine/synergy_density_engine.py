from typing import List, Dict, Set, Any
from sqlalchemy.orm import Session
from models.impact_projection import SynergyDensityMetric
from models.impact_graph import ImpactNode
from .agent_impact_profile_engine import AgentImpactProfileEngine
from .impact_forecast_engine import ImpactForecastEngine

class SynergyDensityEngine:
    """
    Computes Synergy Density by comparing aggregated independent projections 
    against cooperative projections for multi-agent task clusters.
    """
    def __init__(self, session: Session):
        self.session = session
        self.forecast_engine = ImpactForecastEngine(session)
        self.profile_engine = AgentImpactProfileEngine(session)

    def calculate_synergy_density(self, agent_node_ids: List[str]) -> SynergyDensityMetric:
        """
        1. Compute projected impact for each agent in isolation.
        2. Sum those independent projections.
        3. Compute projected impact when all agents operate together.
        4. Calculate normalized amplification ratio.
        5. Persist as SynergyDensityMetric.
        """
        if not agent_node_ids:
            raise ValueError("agent_node_ids cannot be empty")

        # 1. Independent Projections (In Isolation)
        independent_vectors = []
        for node_id in agent_node_ids:
            # "Isolation" means other agents in the cluster are disabled
            others = set(agent_node_ids) - {node_id}
            vec = self.forecast_engine._compute_projected_vector(
                [node_id], 
                reliability=1.0, 
                multiplier=1.0, 
                disabled_node_ids=others
            )
            independent_vectors.append(vec)
        
        sum_independent = self._sum_vectors(independent_vectors)

        # 2. Cooperative Projection (Together)
        cooperative_vector = self.forecast_engine._compute_projected_vector(
            agent_node_ids, 
            reliability=1.0, 
            multiplier=1.0
        )

        # 3. Calculate Normalized Amplification Ratio
        ratio = self._calculate_amplification_ratio(sum_independent, cooperative_vector)

        # 4. Persist SynergyDensityMetric
        metric = SynergyDensityMetric(
            collaboration_structure=agent_node_ids,
            independent_impact_sum=sum_independent,
            cooperative_impact=cooperative_vector,
            synergy_density_ratio=ratio
        )
        
        self.session.add(metric)
        self.session.commit()

        self.profile_engine.update_synergy_amplification_contribution(
            agent_ids=self._resolve_agent_ids(agent_node_ids),
            synergy_density_ratio=ratio,
        )
        
        return metric

    def _resolve_agent_ids(self, node_ids: List[str]) -> Set[str]:
        agent_ids: Set[str] = set()
        for node_id in node_ids:
            node = self.session.get(ImpactNode, node_id)
            if not node:
                continue
            context = node.domain_context or {}
            agent_id = context.get("agent_id") or context.get("agent")
            if agent_id:
                agent_ids.add(agent_id)
        return agent_ids

    def _sum_vectors(self, vectors: List[Dict[str, float]]) -> Dict[str, float]:
        result = {}
        for v in vectors:
            for k, val in v.items():
                result[k] = result.get(k, 0.0) + val
        return result

    def _calculate_amplification_ratio(self, independent: Dict[str, float], cooperative: Dict[str, float]) -> float:
        """
        Calculates the ratio representing synergy beyond additive expectation.
        Ratio = Total Cooperative Impact / Total Independent Impact
        Uses the sum of magnitudes across all outcome types as a simple scalar for comparison.
        """
        sum_ind = sum(independent.values())
        sum_coop = sum(cooperative.values())
        
        if sum_ind == 0:
            return 1.0 if sum_coop == 0 else float('inf')
        
        return sum_coop / sum_ind
