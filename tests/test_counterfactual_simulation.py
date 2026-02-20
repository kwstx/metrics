import pytest

from engine.counterfactual_simulation import CounterfactualSimulation
from engine.graph_manager import GraphManager
from models.impact_graph import init_db


@pytest.fixture
def session():
    Session = init_db("sqlite:///:memory:")
    session = Session()
    yield session
    session.close()


def test_counterfactual_removes_agent_outgoing_edges(session):
    manager = GraphManager(session)
    simulation = CounterfactualSimulation(session)

    source_action = manager.add_node(
        outcome_type="AGENT_ACTION",
        domain_context={"agent": "alpha"},
        magnitude=1.0,
    )
    removed_agent_action = manager.add_node(
        outcome_type="AGENT_ACTION",
        domain_context={"agent": "beta"},
        magnitude=1.0,
    )
    downstream_outcome = manager.add_node(
        outcome_type="SYSTEM_OUTCOME",
        domain_context={"metric": "stability"},
        magnitude=4.0,
    )

    manager.add_causal_edge(source_action.id, removed_agent_action.id, weight=1.0, confidence=1.0)
    manager.add_causal_edge(removed_agent_action.id, downstream_outcome.id, weight=0.5, confidence=1.0)

    metric = simulation.simulate_agent_removal(source_action.id, removed_agent_id="beta")

    assert metric.removed_agent_id == "beta"
    assert metric.full_projection_vector["SYSTEM_OUTCOME"] == pytest.approx(2.0)
    assert metric.counterfactual_projection_vector.get("SYSTEM_OUTCOME", 0.0) == pytest.approx(0.0)
    assert metric.marginal_influence_vector["SYSTEM_OUTCOME"] == pytest.approx(2.0)
    assert metric.total_marginal_influence == pytest.approx(2.0)
