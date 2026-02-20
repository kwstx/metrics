from datetime import timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from engine.cooperative_intelligence_index_engine import CooperativeIntelligenceIndexEngine
from engine.cooperative_stability_engine import CooperativeStabilityEngine
from engine.counterfactual_simulation import CounterfactualSimulation
from engine.graph_manager import GraphManager
from engine.impact_forecast_engine import ImpactForecastEngine
from engine.predictive_calibration_engine import PredictiveCalibrationEngine
from engine.synergy_density_engine import SynergyDensityEngine
from models.base import Base
from models.impact_projection import AgentImpactProfile


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


def test_agent_impact_profile_updates_across_dimensions(session):
    manager = GraphManager(session)
    forecast_engine = ImpactForecastEngine(session)
    stability_engine = CooperativeStabilityEngine(session)
    calibration_engine = PredictiveCalibrationEngine(session)
    synergy_engine = SynergyDensityEngine(session)
    intelligence_engine = CooperativeIntelligenceIndexEngine(session)
    counterfactual_engine = CounterfactualSimulation(session)

    node_a = manager.add_node(
        outcome_type="ACTION",
        domain_context={"agent_id": "agent_alpha", "role": "planner", "domain": "code"},
        magnitude=8.0,
    )
    node_b = manager.add_node(
        outcome_type="ACTION",
        domain_context={"agent_id": "agent_beta", "role": "reviewer", "domain": "code"},
        magnitude=6.0,
    )
    node_out = manager.add_node(
        outcome_type="OUTCOME",
        domain_context={"domain": "code", "role": "system"},
        magnitude=10.0,
    )

    manager.add_causal_edge(node_a.id, node_b.id, weight=0.5, confidence=1.0)
    manager.add_causal_edge(node_b.id, node_out.id, weight=1.0, confidence=1.0)

    projection = forecast_engine.forecast_action(
        action_node_id=node_a.id,
        time_horizon=120.0,
        task_sequence_id="task-11",
        decay_rate=0.01,
    )

    stability_engine.record_stability_metric(
        agent_id="agent_alpha",
        negotiation_convergence_time=5.0,
        resource_allocation_variance=0.1,
        conflict_resolution_frequency=0.5,
        team_performance_stability=0.25,
        team_composition=["agent_alpha", "agent_beta"],
    )

    outcome = calibration_engine.record_outcome(
        projection_id=projection.id,
        realized_vector={"ACTION": 10.0, "OUTCOME": 15.0},
        realized_timestamp=projection.timestamp + timedelta(seconds=120),
    )
    calibration_engine.run_calibration(outcome.id)

    synergy_engine.calculate_synergy_density([node_a.id, node_b.id])

    intelligence_engine.generate_cooperative_intelligence_vector(
        agent_node_ids=[node_a.id, node_b.id],
        baseline_predictions={
            "agent_alpha": {"quality": 0.6, "latency": 0.4},
            "agent_beta": {"quality": 0.8, "latency": 0.2},
        },
        cooperative_predictions={
            "agent_alpha": {"quality": 0.7, "latency": 0.3},
            "agent_beta": {"quality": 0.75, "latency": 0.25},
        },
        calibration_errors_before={"quality": 0.2, "latency": 0.12},
        calibration_errors_after={"quality": 0.1, "latency": 0.06},
    )

    counterfactual_engine.simulate_agent_removal(node_a.id, removed_agent_id="agent_beta")

    profile_alpha = session.get(AgentImpactProfile, "agent_alpha")
    profile_beta = session.get(AgentImpactProfile, "agent_beta")

    assert profile_alpha is not None
    assert profile_beta is not None

    expected_dimensions = {
        "marginal_cooperative_influence",
        "synergy_amplification_contribution",
        "predictive_accuracy_index",
        "stability_coefficient",
        "long_term_impact_weight",
        "cross_role_integration_depth",
    }
    assert set(profile_alpha.impact_dimensions.keys()) == expected_dimensions
    assert set(profile_beta.impact_dimensions.keys()) == expected_dimensions

    assert "score" not in profile_alpha.impact_dimensions
    assert "score" not in profile_beta.impact_dimensions

    assert profile_alpha.impact_dimensions["predictive_accuracy_index"]["samples"] >= 1
    assert profile_alpha.impact_dimensions["stability_coefficient"]["samples"] >= 1
    assert profile_alpha.impact_dimensions["long_term_impact_weight"]["samples"] >= 1
    assert profile_alpha.impact_dimensions["cross_role_integration_depth"]["samples"] >= 1
    assert profile_beta.impact_dimensions["marginal_cooperative_influence"]["samples"] >= 1
    assert profile_beta.impact_dimensions["synergy_amplification_contribution"]["samples"] >= 1

