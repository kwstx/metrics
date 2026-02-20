from datetime import datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from engine.graph_manager import GraphManager
from engine.impact_forecast_engine import ImpactForecastEngine
from engine.temporal_impact_memory_engine import TemporalImpactMemoryEngine
from models.base import Base
from models.impact_projection import TemporalImpactLedgerEntry


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


def test_temporal_ledger_accumulates_with_configurable_decay(session):
    manager = GraphManager(session)
    memory = TemporalImpactMemoryEngine(session)
    now = datetime.utcnow()
    node_one = manager.add_node(
        outcome_type="ACTION",
        domain_context={"agent_id": "agent_a"},
        magnitude=1.0,
    )
    node_two = manager.add_node(
        outcome_type="ACTION",
        domain_context={"agent_id": "agent_b"},
        magnitude=1.0,
    )

    memory.append_contribution(
        task_sequence_id="chain_alpha",
        source_node_id=node_one.id,
        projection_id=None,
        impact_vector={"OUTCOME": 10.0},
        decay_function="none",
        decay_rate=0.0,
        timestamp=now - timedelta(seconds=100),
    )
    memory.append_contribution(
        task_sequence_id="chain_alpha",
        source_node_id=node_two.id,
        projection_id=None,
        impact_vector={"OUTCOME": 10.0},
        decay_function="exponential",
        decay_rate=0.01,
        timestamp=now,
    )

    totals = memory.get_accumulated_impact("chain_alpha", as_of=now)

    # no-decay entry contributes full 10, fresh exponential entry contributes ~10
    assert totals["OUTCOME"] == pytest.approx(20.0, rel=1e-3)

    future = now + timedelta(seconds=100)
    decayed_totals = memory.get_accumulated_impact("chain_alpha", as_of=future)

    # First entry remains 10, second decays by exp(-1) ~= 0.3679.
    assert decayed_totals["OUTCOME"] == pytest.approx(13.6788, rel=1e-3)


def test_forecast_chain_returns_temporal_accumulation_instead_of_reset(session):
    manager = GraphManager(session)
    forecast = ImpactForecastEngine(session)

    action_one = manager.add_node(
        outcome_type="ACTION",
        domain_context={"agent_id": "agent_chain", "domain": "ops"},
        magnitude=1.0,
    )
    action_two = manager.add_node(
        outcome_type="ACTION",
        domain_context={"agent_id": "agent_chain", "domain": "ops"},
        magnitude=2.0,
    )

    first_projection = forecast.forecast_action(
        action_one.id,
        time_horizon=30.0,
        task_sequence_id="seq_1",
        decay_function="none",
        decay_rate=0.0,
    )
    second_projection = forecast.forecast_action(
        action_two.id,
        time_horizon=30.0,
        task_sequence_id="seq_1",
        decay_function="none",
        decay_rate=0.0,
    )

    # Chain memory keeps prior contribution instead of resetting per task.
    assert first_projection.predicted_impact_vector["ACTION"] == pytest.approx(1.0)
    assert second_projection.predicted_impact_vector["ACTION"] == pytest.approx(3.0)

    entries = session.query(TemporalImpactLedgerEntry).filter(
        TemporalImpactLedgerEntry.task_sequence_id == "seq_1"
    ).all()
    assert len(entries) == 2
