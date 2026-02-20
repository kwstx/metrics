import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from engine.cooperative_intelligence_index_engine import CooperativeIntelligenceIndexEngine
from engine.graph_manager import GraphManager
from models.base import Base


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


def test_generate_cooperative_intelligence_vector(session):
    graph_manager = GraphManager(session)
    ci_engine = CooperativeIntelligenceIndexEngine(session)

    node_a = graph_manager.add_node(
        outcome_type="ACTION",
        domain_context={"agent": "agent_a", "role": "planner"},
        magnitude=1.0,
    )
    node_b = graph_manager.add_node(
        outcome_type="ACTION",
        domain_context={"agent": "agent_b", "role": "implementer"},
        magnitude=1.0,
    )
    node_c = graph_manager.add_node(
        outcome_type="OUTCOME",
        domain_context={"agent": "agent_c", "role": "reviewer"},
        magnitude=2.0,
    )

    graph_manager.add_causal_edge(node_a.id, node_b.id, weight=0.7, confidence=1.0)
    graph_manager.add_causal_edge(node_b.id, node_c.id, weight=0.9, confidence=1.0)

    baseline_predictions = {
        "agent_a": {"quality": 0.60, "latency": 0.40},
        "agent_b": {"quality": 0.80, "latency": 0.20},
    }
    cooperative_predictions = {
        "agent_a": {"quality": 0.68, "latency": 0.32},
        "agent_b": {"quality": 0.72, "latency": 0.28},
    }
    calibration_before = {"quality": 0.18, "latency": 0.12}
    calibration_after = {"quality": 0.09, "latency": 0.06}

    metric = ci_engine.generate_cooperative_intelligence_vector(
        agent_node_ids=[node_a.id, node_b.id],
        baseline_predictions=baseline_predictions,
        cooperative_predictions=cooperative_predictions,
        calibration_errors_before=calibration_before,
        calibration_errors_after=calibration_after,
    )

    assert metric.uncertainty_reduction["reduction_ratio"] > 0.0
    assert metric.dependency_graph_enrichment["cross_agent_edges"] == pytest.approx(2.0)
    assert metric.predictive_calibration_improvement["absolute_improvement"] == pytest.approx(0.075)
    assert metric.cross_role_integration_depth["normalized_depth"] == pytest.approx(1.0)

    vector = metric.cooperative_intelligence_vector
    assert set(vector.keys()) == {
        "uncertainty_reduction",
        "dependency_graph_enrichment",
        "predictive_calibration_improvement",
        "cross_role_integration_depth",
    }
    assert "score" not in vector
