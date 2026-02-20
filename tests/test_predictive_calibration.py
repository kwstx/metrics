import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timedelta

from models.base import Base
from models.impact_graph import ImpactNode, ImpactEdge
from models.impact_projection import (
    ImpactProjection, AgentReliability, SynergyDensityMetric,
    ImpactOutcome, CalibrationEvent
)
from engine.predictive_calibration_engine import PredictiveCalibrationEngine
from engine.impact_forecast_engine import ImpactForecastEngine

@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()

def test_predictive_calibration_full_loop(session):
    # 1. Setup Agents and Nodes
    agent_id = "agent_alpha"
    action_node = ImpactNode(
        id="node_action_1",
        outcome_type="ACTION",
        magnitude=10.0,
        domain_context={"agent_id": agent_id, "domain": "engineering"}
    )
    outcome_node = ImpactNode(
        id="node_outcome_1",
        outcome_type="OUTCOME",
        magnitude=10.0,
        domain_context={"domain": "engineering"}
    )
    edge = ImpactEdge(
        source_node_id=action_node.id,
        target_node_id=outcome_node.id,
        causal_weight=2.0,
        confidence_score=1.0
    )
    
    session.add_all([action_node, outcome_node, edge])
    session.commit()

    # 2. Generate Forecast
    forecast_engine = ImpactForecastEngine(session)
    projection = forecast_engine.forecast_action(action_node.id, time_horizon=60.0)
    
    # Predicted impact: magnitude 10 * weight 2 * reliability 1 = 20
    assert projection.predicted_impact_vector["OUTCOME"] == 20.0

    # 3. Record Real-World Outcome
    calibration_engine = PredictiveCalibrationEngine(session)
    # Realized impact is much higher than predicted (40 instead of 20)
    # and realized 120 seconds after projection started (expected 60)
    realized_vector = {"OUTCOME": 40.0}
    realized_time = projection.timestamp + timedelta(seconds=120)
    
    outcome = calibration_engine.record_outcome(
        projection_id=projection.id,
        realized_vector=realized_vector,
        realized_timestamp=realized_time
    )
    
    # 4. Run Calibration
    calibration = calibration_engine.run_calibration(outcome.id)
    
    # Assertions
    # Magnitude deviation: |40 - 20| / 20 = 1.0
    assert calibration.magnitude_deviation == pytest.approx(1.0)
    # Timing deviation: |120 - 60| / 60 = 1.0
    assert calibration.timing_deviation == pytest.approx(1.0)
    
    # Check Reliability Update
    agent_rel = session.query(AgentReliability).filter(AgentReliability.agent_id == agent_id).first()
    assert agent_rel is not None
    # Penalty was (0.5 * 1.0 + 0.2 * 1.0 + 0.3 * 0.0) = 0.7
    # Update = -0.1 * 0.7 = -0.07
    # New coefficient = 1.0 - 0.07 = 0.93
    assert agent_rel.reliability_coefficient == pytest.approx(0.93)
    assert calibration.new_reliability_coefficient == pytest.approx(0.93)

    # 5. Verify Future Forecast uses updated reliability
    new_projection = forecast_engine.forecast_action(action_node.id, time_horizon=60.0)
    # New prediction: 10 * 2 * 0.93 = 18.6
    assert new_projection.predicted_impact_vector["OUTCOME"] == pytest.approx(18.6)

def test_synergy_calibration(session):
    agent_id = "agent_beta"
    action_node = ImpactNode(
        id="node_synergy_1",
        outcome_type="ACTION",
        magnitude=10.0,
        domain_context={"agent_id": agent_id}
    )
    session.add(action_node)
    
    # Mock a synergy metric
    synergy_metric = SynergyDensityMetric(
        collaboration_structure=[action_node.id],
        independent_impact_sum={"OUTCOME": 10.0},
        cooperative_impact={"OUTCOME": 15.0},
        synergy_density_ratio=1.5
    )
    session.add(synergy_metric)
    
    # Mock a projection
    projection = ImpactProjection(
        source_node_id=action_node.id,
        predicted_impact_vector={"OUTCOME": 15.0}, # Includes synergy
        uncertainty_bounds={},
        time_horizon=30.0,
        confidence_interval={}
    )
    session.add(projection)
    session.commit()
    
    calibration_engine = PredictiveCalibrationEngine(session)
    # Outcome matches cooperative prediction exactly, including timing
    outcome = calibration_engine.record_outcome(
        projection_id=projection.id,
        realized_vector={"OUTCOME": 15.0},
        realized_timestamp=projection.timestamp + timedelta(seconds=30)
    )
    
    calibration = calibration_engine.run_calibration(outcome.id)
    
    # Synergy error should be 0.0
    assert calibration.synergy_assumption_error == 0.0
    # Small boost for accuracy
    agent_rel = session.query(AgentReliability).filter(AgentReliability.agent_id == agent_id).first()
    assert agent_rel.reliability_coefficient > 1.0
