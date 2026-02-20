import sys
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.base import Base
from models.impact_graph import ImpactNode, ImpactEdge
from engine.graph_manager import GraphManager
from engine.synergy_density_engine import SynergyDensityEngine

def test_synergy_calculation():
    # Setup In-memory DB
    engine = create_engine('sqlite:///:memory:')
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    graph_manager = GraphManager(session)
    synergy_engine = SynergyDensityEngine(session)

    # 1. Create Nodes
    node_a = graph_manager.add_node(
        outcome_type="ACTION",
        domain_context={"agent": "agent_a"},
        magnitude=10.0
    )
    node_b = graph_manager.add_node(
        outcome_type="ACTION",
        domain_context={"agent": "agent_b"},
        magnitude=10.0
    )
    node_c = graph_manager.add_node(
        outcome_type="OUTCOME",
        domain_context={},
        magnitude=20.0
    )

    # 2. Create Edges
    graph_manager.add_causal_edge(node_a.id, node_b.id, weight=0.5)
    graph_manager.add_causal_edge(node_b.id, node_c.id, weight=1.0)

    # Manual check
    vec_a = synergy_engine.forecast_engine._compute_projected_vector([node_a.id], 1.0, 1.0, {node_b.id})
    vec_b = synergy_engine.forecast_engine._compute_projected_vector([node_b.id], 1.0, 1.0, {node_a.id})
    print(f"Vector A (Isolation): {vec_a}")
    print(f"Vector B (Isolation): {vec_b}")

    # 3. Calculate Synergy Density
    metric = synergy_engine.calculate_synergy_density([node_a.id, node_b.id])

    print(f"Independent Impact Sum: {metric.independent_impact_sum}")
    print(f"Cooperative Impact: {metric.cooperative_impact}")
    print(f"Synergy Density Ratio: {metric.synergy_density_ratio}")

    # Calculations:
    # A isolated: {ACTION: 10} Sum=10
    # B isolated: {ACTION: 10, OUTCOME: 20} Sum=30
    # Total Indep: Sum=40
    # Cooperative: 
    #   A inf = 1.0 -> Vector += 1.0 * 10 = 10 (ACTION)
    #   B inf = 1.0 (start) + (1.0 * 0.5) = 1.5 -> Vector += 1.5 * 10 = 15 (ACTION)
    #   C inf = 1.5 * 1.0 = 1.5 -> Vector += 1.5 * 20 = 30 (OUTCOME)
    # Total Coop: ACTION: 25, OUTCOME: 30. Sum=55.
    # Ratio: 55 / 40 = 1.375
    
    assert abs(metric.synergy_density_ratio - 1.375) < 0.001
    print("Test Passed!")

if __name__ == "__main__":
    test_synergy_calculation()
