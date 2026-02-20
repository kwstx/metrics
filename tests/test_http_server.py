from datetime import datetime

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from api.http_server import app, get_db
from models.base import Base
from models.impact_graph import ImpactEdge, ImpactNode
from models.impact_projection import DomainMultiplier


def _seed_graph(session):
    alpha = ImpactNode(
        outcome_type="ACTION",
        domain_context={"agent_id": "alpha", "domain": "code"},
        magnitude=10.0,
        uncertainty_metadata={"std_dev": 0.1},
    )
    beta = ImpactNode(
        outcome_type="ACTION",
        domain_context={"agent_id": "beta", "domain": "code"},
        magnitude=8.0,
        uncertainty_metadata={"std_dev": 0.2},
    )
    outcome = ImpactNode(
        outcome_type="OUTCOME",
        domain_context={"domain": "code"},
        magnitude=0.0,
    )
    session.add_all([alpha, beta, outcome])
    session.commit()

    session.add_all(
        [
            ImpactEdge(
                source_node_id=alpha.id,
                target_node_id=outcome.id,
                causal_weight=0.8,
                confidence_score=1.0,
            ),
            ImpactEdge(
                source_node_id=beta.id,
                target_node_id=outcome.id,
                causal_weight=0.6,
                confidence_score=1.0,
            ),
            DomainMultiplier(domain_name="code", multiplier=1.2, description="Code"),
        ]
    )
    session.commit()
    return alpha, beta, outcome


def _client_and_session():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SessionLocal = sessionmaker(bind=engine)
    Base.metadata.create_all(bind=engine)
    session = SessionLocal()

    def override_get_db():
        try:
            yield session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    return client, session


def test_http_end_to_end_flow():
    client, session = _client_and_session()
    alpha, beta, _ = _seed_graph(session)

    action = client.post(
        "/v1/actions",
        json={
            "outcome_type": "ACTION",
            "domain_context": {"agent_id": "gamma", "domain": "code"},
            "magnitude": 6.0,
            "causal_links": [{"target_node_id": alpha.id, "weight": 0.5, "confidence": 0.9}],
            "caller_identity": "http-test",
        },
    )
    action_payload = action.json()
    assert action.status_code == 200
    assert action_payload["status"] == "ok"
    assert isinstance(action_payload["data"]["node"]["id"], str)
    assert action_payload["api_version"]
    assert action_payload["audit_id"]
    new_node_id = action_payload["data"]["node"]["id"]

    forecast = client.post(
        "/v1/forecasts",
        json={"action_node_id": new_node_id, "time_horizon": 30.0, "caller_identity": "http-test"},
    )
    forecast_payload = forecast.json()
    assert forecast.status_code == 200
    assert forecast_payload["status"] == "ok"
    assert isinstance(forecast_payload["data"]["predicted_impact_vector"], dict)
    projection_id = forecast_payload["data"]["projection_id"]

    counterfactual = client.post(
        "/v1/counterfactuals",
        json={"source_node_id": new_node_id, "removed_agent_id": "beta", "caller_identity": "http-test"},
    )
    counterfactual_payload = counterfactual.json()
    assert counterfactual.status_code == 200
    assert counterfactual_payload["status"] == "ok"
    assert "marginal_influence_vector" in counterfactual_payload["data"]

    synergy = client.post(
        "/v1/synergy-density",
        json={"agent_node_ids": [new_node_id, beta.id], "caller_identity": "http-test"},
    )
    synergy_payload = synergy.json()
    assert synergy.status_code == 200
    assert synergy_payload["status"] == "ok"
    assert isinstance(synergy_payload["data"]["cooperative_impact"], dict)

    outcome = client.post(
        "/v1/outcomes",
        json={
            "projection_id": projection_id,
            "realized_impact_vector": {"ACTION": 5.0, "OUTCOME": 4.0},
            "realized_timestamp": datetime.utcnow().isoformat(),
            "run_calibration": True,
            "caller_identity": "http-test",
        },
    )
    outcome_payload = outcome.json()
    assert outcome.status_code == 200
    assert outcome_payload["status"] == "ok"
    assert outcome_payload["data"]["calibration"] is not None

    trace = client.post(
        "/v1/provenance-traces",
        json={"metric_type": "ImpactProjection", "metric_id": projection_id, "caller_identity": "http-test"},
    )
    trace_payload = trace.json()
    assert trace.status_code == 200
    assert trace_payload["status"] == "ok"
    assert "reproducibility" in trace_payload["data"]

    profile = client.post(
        "/v1/agent-impact-profiles",
        json={"agent_id": "gamma", "caller_identity": "http-test"},
    )
    profile_payload = profile.json()
    assert profile.status_code == 200
    assert profile_payload["status"] == "ok"
    assert profile_payload["data"]["agent_id"] == "gamma"

    log = client.post(
        "/v1/audit-log",
        json={"operation": "retrieve_forecast", "limit": 10, "caller_identity": "http-test"},
    )
    log_payload = log.json()
    assert log.status_code == 200
    assert log_payload["status"] == "ok"
    assert any(entry["operation"] == "retrieve_forecast" for entry in log_payload["records"])


def test_http_error_path_is_audited():
    client, _ = _client_and_session()
    bad = client.post(
        "/v1/forecasts",
        json={"action_node_id": "does-not-exist", "caller_identity": "http-test"},
    )
    payload = bad.json()
    assert bad.status_code == 200
    assert payload["status"] == "error"
    assert payload["audit_id"]
