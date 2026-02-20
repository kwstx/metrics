from datetime import datetime, timedelta
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from models.base import Base
from models.impact_projection import SynergyDensityMetric, SynergySignature
from engine.persistent_synergy_signature_engine import PersistentSynergySignatureEngine


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


def _add_metric(session, collaboration_structure, ratio, at):
    metric = SynergyDensityMetric(
        collaboration_structure=collaboration_structure,
        independent_impact_sum={"OUTCOME": 10.0},
        cooperative_impact={"OUTCOME": 10.0 * ratio},
        synergy_density_ratio=ratio,
        timestamp=at,
    )
    session.add(metric)


def test_detects_recurring_above_baseline_signature(session):
    start = datetime(2026, 2, 1, 12, 0, 0)

    # Persistent high-synergy pattern.
    _add_metric(session, ["agent_b", "agent_a"], 1.60, start + timedelta(days=1))
    _add_metric(session, ["agent_a", "agent_b"], 1.70, start + timedelta(days=2))
    _add_metric(session, ["agent_a", "agent_b"], 1.65, start + timedelta(days=3))

    # Baseline/noisy patterns.
    _add_metric(session, ["agent_c", "agent_d"], 1.00, start + timedelta(days=1))
    _add_metric(session, ["agent_c", "agent_d"], 1.05, start + timedelta(days=2))
    _add_metric(session, ["agent_c", "agent_d"], 0.95, start + timedelta(days=3))
    _add_metric(session, ["agent_e", "agent_f"], 1.20, start + timedelta(days=1))
    _add_metric(session, ["agent_e", "agent_f"], 0.90, start + timedelta(days=2))
    _add_metric(session, ["agent_e", "agent_f"], 1.10, start + timedelta(days=3))

    # Strong solo performance should not become a collective signature.
    _add_metric(session, ["agent_solo"], 2.00, start + timedelta(days=1))
    _add_metric(session, ["agent_solo"], 2.10, start + timedelta(days=2))
    _add_metric(session, ["agent_solo"], 2.20, start + timedelta(days=3))

    session.commit()

    engine = PersistentSynergySignatureEngine(session)
    signatures = engine.detect_persistent_signatures(min_frequency=3, min_consistency=0.7)

    assert len(signatures) == 1
    signature = signatures[0]
    assert signature.collaboration_structure == ["agent_a", "agent_b"]
    assert signature.observation_frequency == 3
    assert signature.amplification_magnitude > 0.0
    assert signature.stability_score > 0.9
    assert signature.above_baseline_consistency == pytest.approx(1.0)
    assert signature.last_observed_at >= signature.first_observed_at


def test_detection_upserts_existing_signature(session):
    start = datetime(2026, 2, 1, 12, 0, 0)

    _add_metric(session, ["agent_a", "agent_b"], 1.5, start + timedelta(days=1))
    _add_metric(session, ["agent_a", "agent_b"], 1.6, start + timedelta(days=2))
    _add_metric(session, ["agent_a", "agent_b"], 1.7, start + timedelta(days=3))

    _add_metric(session, ["agent_x", "agent_y"], 1.0, start + timedelta(days=1))
    _add_metric(session, ["agent_x", "agent_y"], 1.0, start + timedelta(days=2))
    _add_metric(session, ["agent_x", "agent_y"], 1.0, start + timedelta(days=3))
    session.commit()

    engine = PersistentSynergySignatureEngine(session)
    first_run = engine.detect_persistent_signatures(min_frequency=3, min_consistency=0.7)
    assert len(first_run) == 1

    _add_metric(session, ["agent_b", "agent_a"], 1.8, start + timedelta(days=4))
    _add_metric(session, ["agent_x", "agent_y"], 1.0, start + timedelta(days=4))
    session.commit()

    second_run = engine.detect_persistent_signatures(min_frequency=3, min_consistency=0.7)
    assert len(second_run) == 1

    all_signatures = session.query(SynergySignature).all()
    assert len(all_signatures) == 1
    assert all_signatures[0].observation_frequency == 4
