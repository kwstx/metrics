import pytest
from models.impact_graph import init_db, ImpactNode, ImpactEdge
from engine.graph_manager import GraphManager

@pytest.fixture
def session():
    Session = init_db("sqlite:///:memory:")
    session = Session()
    yield session
    session.close()

def test_impact_graph_flow(session):
    manager = GraphManager(session)
    
    # 1. Add nodes
    action_node = manager.add_node(
        outcome_type="AGENT_ACTION",
        domain_context={"agent": "alpha", "task": "refactor"},
        magnitude=1.0,
        uncertainty={"p_success": 0.9}
    )
    
    outcome_node = manager.add_node(
        outcome_type="SYSTEM_STABILITY",
        domain_context={"metric": "uptime"},
        magnitude=0.0, # Resulting outcome
        uncertainty={"precision": "high"}
    )
    
    # 2. Add causal edge
    edge = manager.add_causal_edge(
        source_id=action_node.id,
        target_id=outcome_node.id,
        weight=0.8,
        confidence=0.95,
        delay=3600 # 1 hour propagation delay
    )
    
    # 3. Verify persistence
    saved_node = session.get(ImpactNode, action_node.id)
    assert saved_node.outcome_type == "AGENT_ACTION"
    assert len(saved_node.outgoing_edges) == 1
    
    # 4. Traversal Query
    paths = manager.get_causal_paths(action_node.id, outcome_node.id)
    assert len(paths) == 1
    assert paths[0][0].id == action_node.id
    assert paths[0][1].id == outcome_node.id

    # 5. Cumulative Impact
    # action_node (1.0) -> edge (0.8 weight * 0.95 confidence) -> outcome_node
    # Cumulative impact = 1.0 * 0.8 * 0.95 = 0.76
    impact = manager.compute_cumulative_impact(action_node.id)
    assert impact == pytest.approx(0.76)

if __name__ == "__main__":
    pytest.main([__file__])
