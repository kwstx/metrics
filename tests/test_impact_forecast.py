import pytest
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from models.base import Base
from models.impact_graph import ImpactNode, ImpactEdge
from models.impact_projection import ImpactProjection, AgentReliability, DomainMultiplier
from engine.impact_forecast_engine import ImpactForecastEngine
from engine.graph_manager import GraphManager

@pytest.fixture
def session():
    # Use in-memory SQLite for testing
    engine = create_engine("sqlite:///:memory:")
    # Import all models to ensure they are registered with Base
    from models.impact_graph import ImpactNode
    from models.impact_projection import ImpactProjection
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()

def test_forecast_engine_full_path(session):
    graph_manager = GraphManager(session)
    forecast_engine = ImpactForecastEngine(session)

    # 1. Setup Reliability and Multipliers
    session.add(AgentReliability(agent_id="agent_007", reliability_coefficient=0.8))
    session.add(DomainMultiplier(domain_name="security", multiplier=1.5))
    session.commit()

    # 2. Build Impact Graph
    # Action: Code Security Audit
    action = graph_manager.add_node(
        outcome_type="ACTION",
        domain_context={"agent_id": "agent_007", "domain": "security"},
        magnitude=1.0 # Base action magnitude
    )

    # Mid-level outcome: Vulnerability Patching
    mid_outcome = graph_manager.add_node(
        outcome_type="VULN_FIXED",
        domain_context={"domain": "security"},
        magnitude=2.0
    )

    # Final outcome: System Trust
    final_outcome = graph_manager.add_node(
        outcome_type="SYSTEM_TRUST",
        domain_context={"domain": "business"},
        magnitude=5.0
    )

    # Connect nodes
    graph_manager.add_causal_edge(action.id, mid_outcome.id, weight=0.9, confidence=1.0)
    graph_manager.add_causal_edge(mid_outcome.id, final_outcome.id, weight=0.6, confidence=0.8)

    # 3. Forecast
    projection = forecast_engine.forecast_action(action.id, time_horizon=90.0)

    # 4. Assertions
    assert projection is not None
    assert projection.source_node_id == action.id
    assert projection.time_horizon == 90.0
    
    vector = projection.predicted_impact_vector
    assert "ACTION" in vector
    assert "VULN_FIXED" in vector
    assert "SYSTEM_TRUST" in vector

    # Check weights:
    # Reliability = 0.8, Multiplier = 1.5
    # Combined Factor = 0.8 * 1.5 = 1.2
    
    # ACTION: inf=1.0, mag=1.0 -> 1.0 * 1.0 * 1.2 = 1.2
    assert vector["ACTION"] == pytest.approx(1.2)
    
    # VULN_FIXED: inf=1.0 * 0.9 * 1.0 = 0.9, mag=2.0 -> 0.9 * 2.0 * 1.2 = 2.16
    assert vector["VULN_FIXED"] == pytest.approx(2.16)
    
    # SYSTEM_TRUST: inf=0.9 * 0.6 * 0.8 = 0.432, mag=5.0 -> 0.432 * 5.0 * 1.2 = 2.592
    assert vector["SYSTEM_TRUST"] == pytest.approx(2.592)

    # Check uncertainty
    bounds = projection.uncertainty_bounds
    assert "SYSTEM_TRUST" in bounds
    # error_coeff = (1-0.8) + 0.1 = 0.3
    # min = 2.592 * (1 - 0.3) = 1.8144
    assert bounds["SYSTEM_TRUST"]["min"] == pytest.approx(2.592 * 0.7)

    # Check dependencies
    assert mid_outcome.id in projection.dependency_references
    assert final_outcome.id in projection.dependency_references

if __name__ == "__main__":
    pytest.main([__file__])
