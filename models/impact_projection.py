from datetime import datetime
import uuid
from sqlalchemy import Column, String, Float, DateTime, JSON, ForeignKey, Integer
from sqlalchemy.orm import relationship
from .base import Base


class ImpactProjection(Base):
    """
    Represents a probabilistic downstream projection of an agent action.
    """
    __tablename__ = 'impact_projections'

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    source_node_id = Column(String, ForeignKey('impact_nodes.id'), nullable=False)
    
    # Multi-dimensional predicted impact vector (e.g., {"revenue": 100, "user_growth": 50})
    predicted_impact_vector = Column(JSON, nullable=False)
    
    # Uncertainty bounds (e.g., {"min": 80, "max": 120, "confidence": 0.95})
    uncertainty_bounds = Column(JSON, nullable=False)
    
    # Time horizon for the prediction (e.g., in seconds or as a future timestamp)
    time_horizon = Column(Float, nullable=False)
    
    # References to nodes or edges that this projection depends on
    dependency_references = Column(JSON, nullable=True)
    
    # Confidence interval for later recalibration
    confidence_interval = Column(JSON, nullable=False)
    
    timestamp = Column(DateTime, default=datetime.utcnow)

    # Relationship back to the source node
    source_node = relationship("ImpactNode", foreign_keys=[source_node_id])

    def __repr__(self):
        return f"<ImpactProjection(id={self.id[:8]}, source={self.source_node_id[:8]}, horizon={self.time_horizon})>"

class AgentReliability(Base):
    """
    Stores historical reliability coefficients for agents.
    """
    __tablename__ = 'agent_reliability'
    
    agent_id = Column(String, primary_key=True)
    reliability_coefficient = Column(Float, default=1.0)
    last_updated = Column(DateTime, default=datetime.utcnow)


class AgentImpactProfile(Base):
    """
    Stores a structured, multi-dimensional impact profile per agent.
    Intentionally avoids any single scalar rank/leaderboard score.
    """
    __tablename__ = 'agent_impact_profiles'

    agent_id = Column(String, primary_key=True)
    impact_dimensions = Column(JSON, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    last_updated = Column(DateTime, default=datetime.utcnow)

class DomainMultiplier(Base):
    """
    Stores multipliers specific to domains or impact types.
    """
    __tablename__ = 'domain_multipliers'
    
    domain_name = Column(String, primary_key=True)
    multiplier = Column(Float, default=1.0)
    description = Column(String, nullable=True)


class MarginalCooperativeInfluence(Base):
    """
    Stores counterfactual influence deltas for an agent removal simulation.
    """
    __tablename__ = 'marginal_cooperative_influence'

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    source_node_id = Column(String, ForeignKey('impact_nodes.id'), nullable=False)
    removed_agent_id = Column(String, nullable=False)
    full_projection_vector = Column(JSON, nullable=False)
    counterfactual_projection_vector = Column(JSON, nullable=False)
    marginal_influence_vector = Column(JSON, nullable=False)
    total_marginal_influence = Column(Float, nullable=False, default=0.0)
    time_horizon = Column(Float, nullable=False)
    dependency_references = Column(JSON, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)

    source_node = relationship("ImpactNode", foreign_keys=[source_node_id])


class SynergyDensityMetric(Base):
    """
    Stores the results of synergy density computations comparing independent vs cooperative projections.
    """
    __tablename__ = 'synergy_density_metrics'

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # List of agent action node IDs involved in the collaboration
    collaboration_structure = Column(JSON, nullable=False)
    
    # Aggregated sum of independent impact projections
    independent_impact_sum = Column(JSON, nullable=False)
    
    # Impact projection when agents operate together
    cooperative_impact = Column(JSON, nullable=False)
    
    # Normalized amplification ratio: (cooperative_impact / independent_impact_sum)
    synergy_density_ratio = Column(Float, nullable=False)
    
    timestamp = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<SynergyDensityMetric(id={self.id[:8]}, ratio={self.synergy_density_ratio:.4f})>"


class CooperativeIntelligenceMetric(Base):
    """
    Stores structured cooperative intelligence sub-metrics and their multi-dimensional vector form.
    """
    __tablename__ = 'cooperative_intelligence_metrics'

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    collaboration_structure = Column(JSON, nullable=False)

    # Sub-metrics persisted independently.
    uncertainty_reduction = Column(JSON, nullable=False)
    dependency_graph_enrichment = Column(JSON, nullable=False)
    predictive_calibration_improvement = Column(JSON, nullable=False)
    cross_role_integration_depth = Column(JSON, nullable=False)

    # Multi-dimensional vector representation (no scalar collapse).
    cooperative_intelligence_vector = Column(JSON, nullable=False)

    timestamp = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<CooperativeIntelligenceMetric(id={self.id[:8]})>"


class SynergySignature(Base):
    """
    Stores recurring multi-agent collaboration patterns that consistently exceed
    baseline synergy density across longitudinal observations.
    """
    __tablename__ = 'synergy_signatures'

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))

    # Canonicalized agent/node combination participating in the pattern.
    collaboration_structure = Column(JSON, nullable=False)

    # Number of observed historical synergy events for this pattern.
    observation_frequency = Column(Integer, nullable=False)

    # Fraction of observations above global baseline synergy density.
    above_baseline_consistency = Column(Float, nullable=False)

    # Mean synergy density ratio for this pattern.
    mean_synergy_density_ratio = Column(Float, nullable=False)

    # Mean amplification above global baseline.
    amplification_magnitude = Column(Float, nullable=False)

    # Stability of amplification over time (higher means less variance).
    stability_score = Column(Float, nullable=False)

    # Baseline used when this signature was last computed.
    baseline_synergy_density = Column(Float, nullable=False)

    first_observed_at = Column(DateTime, nullable=False)
    last_observed_at = Column(DateTime, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return (
            f"<SynergySignature(id={self.id[:8]}, freq={self.observation_frequency}, "
            f"amp={self.amplification_magnitude:.4f}, stability={self.stability_score:.4f})>"
        )


class ImpactOutcome(Base):
    """
    Stores the realized real-world outcome of an agent's action.
    """
    __tablename__ = 'impact_outcomes'

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    projection_id = Column(String, ForeignKey('impact_projections.id'), nullable=False)
    
    # Realized multi-dimensional impact vector
    realized_impact_vector = Column(JSON, nullable=False)
    
    # When the outcome was realized (for timing deviation)
    realized_timestamp = Column(DateTime, default=datetime.utcnow)
    
    timestamp = Column(DateTime, default=datetime.utcnow)

    # Relationship to projection
    projection = relationship("ImpactProjection")


class CalibrationEvent(Base):
    """
    Persists the results of a predictive calibration comparison.
    """
    __tablename__ = 'calibration_events'

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    outcome_id = Column(String, ForeignKey('impact_outcomes.id'), nullable=False)
    
    magnitude_deviation = Column(Float, nullable=False)
    timing_deviation = Column(Float, nullable=False)
    synergy_assumption_error = Column(Float, nullable=False)
    
    # The reliability coefficient AFTER the update
    new_reliability_coefficient = Column(Float, nullable=False)
    reliability_delta = Column(Float, nullable=False)
    
    timestamp = Column(DateTime, default=datetime.utcnow)

    outcome = relationship("ImpactOutcome")


class TemporalImpactLedgerEntry(Base):
    """
    Stores a rolling impact contribution for a task chain so influence persists over time.
    """
    __tablename__ = 'temporal_impact_ledger_entries'

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    task_sequence_id = Column(String, nullable=False, index=True)
    source_node_id = Column(String, ForeignKey('impact_nodes.id'), nullable=False)
    projection_id = Column(String, ForeignKey('impact_projections.id'), nullable=True)

    # Raw impact contribution at ingestion time.
    impact_vector = Column(JSON, nullable=False)

    # Decay settings used to weight this contribution over time.
    decay_function = Column(String, nullable=False, default="exponential")
    decay_rate = Column(Float, nullable=False, default=0.01)
    decay_floor = Column(Float, nullable=False, default=0.0)

    # Optional chain metadata for debugging and attribution.
    entry_metadata = Column(JSON, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)

    source_node = relationship("ImpactNode", foreign_keys=[source_node_id])
    projection = relationship("ImpactProjection", foreign_keys=[projection_id])


class CooperativeStabilityMetric(Base):
    """
    Measures how consistently an agent improves collaborative system behavior across varying team compositions.
    """
    __tablename__ = 'cooperative_stability_metrics'

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    agent_id = Column(String, nullable=False, index=True)

    # Negotiation convergence time in seconds
    negotiation_convergence_time = Column(Float, nullable=False)

    # Variance in resource allocation efficiency (0.0 means perfect efficiency parity)
    resource_allocation_variance = Column(Float, nullable=False)

    # Frequency of conflict resolution interventions
    conflict_resolution_frequency = Column(Float, nullable=False)

    # Overall team performance stability (standard deviation of team output)
    team_performance_stability = Column(Float, nullable=False)

    # Computed StabilityCoefficient reflecting structural cooperation quality
    stability_coefficient = Column(Float, nullable=False)

    # Team composition used during the measurement (list of agent roles or IDs)
    team_composition = Column(JSON, nullable=False)

    timestamp = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<CooperativeStabilityMetric(agent={self.agent_id[:8]}, stability={self.stability_coefficient:.4f})>"
