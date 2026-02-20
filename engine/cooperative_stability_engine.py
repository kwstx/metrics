from typing import Dict, List, Any
from sqlalchemy.orm import Session
from models.impact_projection import CooperativeStabilityMetric
from .agent_impact_profile_engine import AgentImpactProfileEngine
from datetime import datetime

class CooperativeStabilityEngine:
    """
    Constructs Cooperative Stability Metrics that measure how consistently an agent 
    improves collaborative system behavior across varying team compositions.
    
    This engine logs negotiation convergence time, variance in resource allocation efficiency, 
    conflict resolution frequency, and overall team performance stability.
    """
    def __init__(self, session: Session):
        self.session = session
        self.profile_engine = AgentImpactProfileEngine(session)

    def record_stability_metric(
        self,
        agent_id: str,
        negotiation_convergence_time: float,
        resource_allocation_variance: float,
        conflict_resolution_frequency: float,
        team_performance_stability: float,
        team_composition: List[str]
    ) -> CooperativeStabilityMetric:
        """
        Logs negotiation convergence time, variance in resource allocation efficiency, 
        conflict resolution frequency, and overall team performance stability, 
        then computes a StabilityCoefficient reflecting structural cooperation quality.
        """
        stability_coefficient = self._compute_stability_coefficient(
            negotiation_convergence_time,
            resource_allocation_variance,
            conflict_resolution_frequency,
            team_performance_stability
        )

        metric = CooperativeStabilityMetric(
            agent_id=agent_id,
            negotiation_convergence_time=negotiation_convergence_time,
            resource_allocation_variance=resource_allocation_variance,
            conflict_resolution_frequency=conflict_resolution_frequency,
            team_performance_stability=team_performance_stability,
            stability_coefficient=stability_coefficient,
            team_composition=team_composition
        )

        self.session.add(metric)
        self.session.commit()

        self.profile_engine.update_stability_coefficient(
            agent_id=agent_id,
            stability_coefficient=stability_coefficient,
        )

        return metric

    def _compute_stability_coefficient(
        self,
        conv_time: float,
        alloc_var: float,
        conflict_freq: float,
        perf_instability: float
    ) -> float:
        """
        Computes a StabilityCoefficient reflecting structural cooperation quality 
        rather than task productivity.
        
        Formula uses weighted normalized penalty factors to measure structural friction.
        """
        # Normalization constants (target baselines for "good" cooperation)
        T_NORM = 10.0 # Target max convergence time (seconds)
        V_NORM = 0.2  # Target max allocation variance (Gini-like index)
        C_NORM = 2.0  # Target max conflicts per session
        S_NORM = 1.0  # Baseline team output volatility

        # Normalized features (clamped to prevent extreme outliers from zeroing the score)
        # We focus on structural consistency.
        f_conv = min(conv_time / T_NORM, 3.0)
        f_var = min(alloc_var / V_NORM, 3.0)
        f_conf = min(conflict_freq / C_NORM, 3.0)
        f_instab = min(perf_instability / S_NORM, 3.0)

        # Weighting: 90% structural quality (conv, var, conflict), 10% task-linked performance stability.
        # Using 1 / (1 + sum(penalties)) approach.
        structural_penalty = (0.3 * f_conv) + (0.3 * f_var) + (0.3 * f_conf)
        performance_penalty = 0.1 * f_instab
        
        stability_coefficient = 1.0 / (1.0 + structural_penalty + performance_penalty)
        
        return round(float(stability_coefficient), 4)

    def compute_aggregate_agent_stability(self, agent_id: str) -> float:
        """
        Computes the mean stability coefficient for an agent across all historical observations.
        """
        metrics = self.session.query(CooperativeStabilityMetric).filter(
            CooperativeStabilityMetric.agent_id == agent_id
        ).all()
        
        if not metrics:
            return 1.0 # Default to neutral/stable if no data
            
        return sum(m.stability_coefficient for m in metrics) / len(metrics)

    def get_stability_trend(self, agent_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieves the latest stability metrics to analyze trends.
        """
        metrics = self.session.query(CooperativeStabilityMetric).filter(
            CooperativeStabilityMetric.agent_id == agent_id
        ).order_by(CooperativeStabilityMetric.timestamp.desc()).limit(limit).all()
        
        return [
            {
                "timestamp": m.timestamp.isoformat(),
                "coefficient": m.stability_coefficient,
                "composition": m.team_composition
            } for m in metrics
        ]
