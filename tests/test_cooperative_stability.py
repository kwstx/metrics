import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from engine.cooperative_stability_engine import CooperativeStabilityEngine
from models.base import Base
from models.impact_projection import CooperativeStabilityMetric

@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()

def test_record_stability_metric(session):
    engine = CooperativeStabilityEngine(session)
    agent_id = "agent_beta_9"
    team_comp = ["strategist", "executor", "validator"]
    
    metric = engine.record_stability_metric(
        agent_id=agent_id,
        negotiation_convergence_time=5.5,
        resource_allocation_variance=0.12,
        conflict_resolution_frequency=1.0,
        team_performance_stability=0.4,
        team_composition=team_comp
    )
    
    assert metric.agent_id == agent_id
    assert metric.stability_coefficient > 0.5 
    assert metric.team_composition == team_comp
    
    # Check persistence
    persisted = session.query(CooperativeStabilityMetric).filter_by(agent_id=agent_id).first()
    assert persisted.id == metric.id
    assert persisted.stability_coefficient == metric.stability_coefficient

def test_stability_coefficient_logic(session):
    engine = CooperativeStabilityEngine(session)
    
    # High stability (low penalties)
    # T_NORM=10, V_NORM=0.2, C_NORM=2, S_NORM=1
    # structural_penalty = 0.3 * (1/10) + 0.3 * (0.05/0.2) + 0.3 * (0/2) = 0.03 + 0.075 + 0 = 0.105
    # performance_penalty = 0.1 * (0.2/1) = 0.02
    # total penalty = 0.125
    # score = 1 / 1.125 = 0.8888...
    stable_score = engine._compute_stability_coefficient(1.0, 0.05, 0.0, 0.2)
    
    # Low stability (high penalties) - capped at 3.0
    # penalty = 0.3*3 + 0.3*3 + 0.3*3 + 0.1*3 = 0.9 + 0.9 + 0.9 + 0.3 = 3.0
    # score = 1 / 4.0 = 0.25
    unstable_score = engine._compute_stability_coefficient(50.0, 2.0, 20.0, 10.0)
    
    assert stable_score > unstable_score
    assert unstable_score == pytest.approx(0.25, abs=1e-2)
    assert stable_score > 0.8

def test_aggregate_stability(session):
    engine = CooperativeStabilityEngine(session)
    agent_id = "agent_gamma"
    
    engine.record_stability_metric(agent_id, 2.0, 0.1, 0.0, 0.1, ["role1"])
    engine.record_stability_metric(agent_id, 15.0, 0.5, 5.0, 2.0, ["role1", "role2"])
    
    agg = engine.compute_aggregate_agent_stability(agent_id)
    assert 0.0 < agg < 1.0
    
    trend = engine.get_stability_trend(agent_id)
    assert len(trend) == 2
    assert trend[0]["coefficient"] < trend[1]["coefficient"] # Latest is the second one (desc order by timestamp)
    # wait, latest is index 0 if order_by desc
    # the second record has worse metrics, so lower coefficient.
    # so trend[0] (latest) should be lower than trend[1].
    assert trend[0]["coefficient"] < trend[1]["coefficient"]
