"""
Tests for the Metrics & Feedback API Layer
==========================================
Validates all seven endpoints, the audit log, versioning, structured vector
responses, causal explanations, and error handling.
"""

import json
import pytest
from datetime import datetime, timedelta

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from models.base import Base
from models.impact_graph import ImpactNode, ImpactEdge
from models.impact_projection import (
    ImpactProjection,
    AgentReliability,
    DomainMultiplier,
    SynergyDensityMetric,
    ImpactOutcome,
    CalibrationEvent,
    MarginalCooperativeInfluence,
    CooperativeStabilityMetric,
    CooperativeIntelligenceMetric,
)

from api.audit_log import AuditLogEntry
from api.metrics_feedback_api import MetricsFeedbackAPI
from api.algorithm_registry import (
    get_current_version,
    register_version,
    deprecate_version,
    list_versions,
)
from api.response_envelope import success_envelope, error_envelope


# -----------------------------------------------------------------------
#  Fixtures
# -----------------------------------------------------------------------

@pytest.fixture
def db_session():
    """In-memory SQLite session with all tables created."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


@pytest.fixture
def api(db_session):
    """MetricsFeedbackAPI wired to the test session."""
    return MetricsFeedbackAPI(db_session, caller_identity="test-harness")


@pytest.fixture
def seeded_graph(db_session):
    """
    Builds a minimal graph::

        [agent_alpha ACTION] --(0.8)--> [OUTCOME_A]
        [agent_beta  ACTION] --(0.6)--> [OUTCOME_A]
        [OUTCOME_A]          --(0.5)--> [OUTCOME_B]
    """
    n1 = ImpactNode(
        outcome_type="ACTION",
        domain_context={"agent_id": "alpha", "domain": "code"},
        magnitude=10.0,
        uncertainty_metadata={"std_dev": 0.1, "distribution": "normal"},
    )
    n2 = ImpactNode(
        outcome_type="OUTCOME",
        domain_context={"domain": "code"},
        magnitude=0.0,
    )
    n3 = ImpactNode(
        outcome_type="ACTION",
        domain_context={"agent_id": "beta", "domain": "code"},
        magnitude=8.0,
        uncertainty_metadata={"std_dev": 0.2, "distribution": "normal"},
    )
    n4 = ImpactNode(
        outcome_type="OUTCOME",
        domain_context={"domain": "infra"},
        magnitude=0.0,
    )
    db_session.add_all([n1, n2, n3, n4])
    db_session.commit()

    e1 = ImpactEdge(
        source_node_id=n1.id, target_node_id=n2.id,
        causal_weight=0.8, confidence_score=1.0,
    )
    e2 = ImpactEdge(
        source_node_id=n3.id, target_node_id=n2.id,
        causal_weight=0.6, confidence_score=1.0,
    )
    e3 = ImpactEdge(
        source_node_id=n2.id, target_node_id=n4.id,
        causal_weight=0.5, confidence_score=0.9,
    )
    db_session.add_all([e1, e2, e3])

    # Seed domain multiplier
    dm = DomainMultiplier(domain_name="code", multiplier=1.2, description="Code domain")
    db_session.add(dm)
    db_session.commit()

    return {
        "alpha_node": n1,
        "beta_node": n3,
        "outcome_a": n2,
        "outcome_b": n4,
    }


# -----------------------------------------------------------------------
#  Envelope & Registry unit tests
# -----------------------------------------------------------------------

class TestResponseEnvelope:
    def test_success_envelope_shape(self):
        resp = success_envelope(
            operation="test_op",
            api_version="1.0.0",
            data={"impact_vector": {"CODE": 0.5}},
            causal_explanation="Test explanation.",
            audit_id="aud-1",
        )
        d = resp.to_dict()
        assert d["status"] == "ok"
        assert d["operation"] == "test_op"
        assert "causal_explanation" in d
        assert "audit_id" in d
        assert d["data"]["impact_vector"]["CODE"] == 0.5

    def test_error_envelope_shape(self):
        resp = error_envelope(
            operation="test_op",
            api_version="1.0.0",
            error_message="Something broke",
            audit_id="aud-2",
        )
        d = resp.to_dict()
        assert d["status"] == "error"
        assert d["data"] is None
        assert "broke" in d["causal_explanation"]


class TestAlgorithmRegistry:
    def test_default_versions_exist(self):
        ops = [
            "submit_action", "retrieve_forecast", "run_counterfactual",
            "query_synergy_density", "agent_impact_profile",
            "trace_provenance", "submit_outcome",
        ]
        for op in ops:
            ver = get_current_version(op)
            assert ver.version == "1.0.0"

    def test_register_and_retrieve_new_version(self):
        register_version(
            "submit_action", "2.0.0",
            "Enhanced ingestion with validation.",
        )
        ver = get_current_version("submit_action")
        assert ver.version == "2.0.0"

    def test_deprecate_version(self):
        register_version(
            "submit_action", "3.0.0",
            "Experimental.",
        )
        deprecate_version("submit_action", "3.0.0")
        # Should fall back to 2.0.0 (registered above, still active).
        ver = get_current_version("submit_action")
        assert ver.version != "3.0.0"

    def test_list_versions(self):
        versions = list_versions("submit_action")
        assert len(versions) >= 1

    def test_unknown_operation_raises(self):
        with pytest.raises(KeyError):
            get_current_version("nonexistent_operation")


# -----------------------------------------------------------------------
#  submit_action
# -----------------------------------------------------------------------

class TestSubmitAction:
    def test_basic_action_submission(self, api, db_session):
        resp = api.submit_action(
            outcome_type="ACTION",
            domain_context={"agent_id": "gamma", "domain": "research"},
            magnitude=5.0,
            uncertainty={"std_dev": 0.3},
        )
        assert resp.status == "ok"
        assert resp.data["node"]["outcome_type"] == "ACTION"
        assert resp.data["node"]["magnitude"] == 5.0
        assert resp.causal_explanation  # non-empty
        assert resp.audit_id  # non-empty

        # Verify node in DB
        node = db_session.query(ImpactNode).filter_by(id=resp.data["node"]["id"]).first()
        assert node is not None

    def test_action_with_causal_links(self, api, db_session, seeded_graph):
        target_id = seeded_graph["outcome_a"].id
        resp = api.submit_action(
            outcome_type="ACTION",
            domain_context={"agent_id": "delta", "domain": "code"},
            magnitude=3.0,
            causal_links=[
                {"target_node_id": target_id, "weight": 0.7, "confidence": 0.9},
            ],
        )
        assert resp.status == "ok"
        assert len(resp.data["edges"]) == 1
        assert resp.data["edges"][0]["causal_weight"] == 0.7

    def test_action_audit_logged(self, api, db_session):
        api.submit_action(
            outcome_type="ACTION",
            domain_context={"agent_id": "epsilon"},
            magnitude=1.0,
        )
        entries = db_session.query(AuditLogEntry).filter_by(operation="submit_action").all()
        assert len(entries) >= 1
        # Version may differ from "1.0.0" if registry tests ran first (module-level singleton)
        assert entries[0].algorithm_version is not None
        assert len(entries[0].algorithm_version) > 0
        assert entries[0].status == "success"


# -----------------------------------------------------------------------
#  retrieve_forecast
# -----------------------------------------------------------------------

class TestRetrieveForecast:
    def test_forecast_returns_structured_vector(self, api, seeded_graph):
        resp = api.retrieve_forecast(
            action_node_id=seeded_graph["alpha_node"].id,
            time_horizon=60.0,
        )
        assert resp.status == "ok"
        assert "predicted_impact_vector" in resp.data
        assert isinstance(resp.data["predicted_impact_vector"], dict)
        assert resp.data["time_horizon"] == 60.0
        assert resp.causal_explanation
        assert resp.audit_id

    def test_forecast_includes_uncertainty(self, api, seeded_graph):
        resp = api.retrieve_forecast(
            action_node_id=seeded_graph["alpha_node"].id,
        )
        assert resp.status == "ok"
        assert "uncertainty_bounds" in resp.data

    def test_forecast_unknown_node_returns_error(self, api):
        resp = api.retrieve_forecast(action_node_id="nonexistent-id")
        assert resp.status == "error"
        assert resp.audit_id  # error is also audited


# -----------------------------------------------------------------------
#  run_counterfactual
# -----------------------------------------------------------------------

class TestRunCounterfactual:
    def test_counterfactual_returns_marginal_vector(self, api, seeded_graph):
        resp = api.run_counterfactual(
            source_node_id=seeded_graph["alpha_node"].id,
            removed_agent_id="beta",
        )
        assert resp.status == "ok"
        assert "marginal_influence_vector" in resp.data
        assert "total_marginal_influence" in resp.data
        assert isinstance(resp.data["full_projection_vector"], dict)
        assert resp.causal_explanation
        assert resp.audit_id

    def test_counterfactual_invalid_source_returns_error(self, api):
        resp = api.run_counterfactual(
            source_node_id="bad-id",
            removed_agent_id="beta",
        )
        assert resp.status == "error"


# -----------------------------------------------------------------------
#  query_synergy_density
# -----------------------------------------------------------------------

class TestQuerySynergyDensity:
    def test_synergy_density_computation(self, api, seeded_graph):
        resp = api.query_synergy_density(
            agent_node_ids=[
                seeded_graph["alpha_node"].id,
                seeded_graph["beta_node"].id,
            ],
        )
        assert resp.status == "ok"
        assert "synergy_density_ratio" in resp.data
        assert "independent_impact_sum" in resp.data
        assert "cooperative_impact" in resp.data
        assert isinstance(resp.data["independent_impact_sum"], dict)
        assert resp.causal_explanation
        assert resp.audit_id

    def test_synergy_density_empty_list_returns_error(self, api):
        resp = api.query_synergy_density(agent_node_ids=[])
        assert resp.status == "error"


# -----------------------------------------------------------------------
#  agent_impact_profile
# -----------------------------------------------------------------------

class TestAgentImpactProfile:
    def test_profile_structure(self, api, seeded_graph):
        resp = api.agent_impact_profile(agent_id="alpha")
        assert resp.status == "ok"
        data = resp.data
        assert data["agent_id"] == "alpha"
        assert "reliability" in data
        assert "projections" in data
        assert "calibration" in data
        assert "stability" in data
        assert "synergy_participation" in data
        assert "counterfactual_influence" in data
        assert resp.causal_explanation
        assert resp.audit_id

    def test_profile_unknown_agent_returns_defaults(self, api, seeded_graph):
        resp = api.agent_impact_profile(agent_id="unknown_agent_xyz")
        assert resp.status == "ok"
        assert resp.data["reliability"]["coefficient"] == 1.0

    def test_profile_includes_forecast_data(self, api, seeded_graph):
        # Generate a forecast first
        api.retrieve_forecast(action_node_id=seeded_graph["alpha_node"].id)
        resp = api.agent_impact_profile(agent_id="alpha")
        assert resp.status == "ok"
        assert resp.data["projections"]["count"] >= 1


# -----------------------------------------------------------------------
#  trace_provenance
# -----------------------------------------------------------------------

class TestTraceProvenance:
    def test_trace_projection_provenance(self, api, db_session, seeded_graph):
        # Create a projection first
        forecast_resp = api.retrieve_forecast(
            action_node_id=seeded_graph["alpha_node"].id,
        )
        projection_id = forecast_resp.data["projection_id"]

        resp = api.trace_provenance(
            metric_type="ImpactProjection",
            metric_id=projection_id,
        )
        assert resp.status == "ok"
        assert resp.data is not None
        assert resp.causal_explanation
        assert resp.audit_id

    def test_trace_invalid_type_returns_error(self, api):
        resp = api.trace_provenance(
            metric_type="NonExistentMetricType",
            metric_id="some-id",
        )
        assert resp.status == "error"

    def test_trace_invalid_id_returns_error(self, api):
        resp = api.trace_provenance(
            metric_type="ImpactProjection",
            metric_id="nonexistent-id",
        )
        assert resp.status == "error"


# -----------------------------------------------------------------------
#  submit_outcome
# -----------------------------------------------------------------------

class TestSubmitOutcome:
    def test_outcome_recording_and_calibration(self, api, db_session, seeded_graph):
        # Step 1: create forecast
        forecast_resp = api.retrieve_forecast(
            action_node_id=seeded_graph["alpha_node"].id,
            time_horizon=10.0,
        )
        projection_id = forecast_resp.data["projection_id"]

        # Step 2: submit realized outcome
        resp = api.submit_outcome(
            projection_id=projection_id,
            realized_impact_vector={"OUTCOME": 7.5, "ACTION": 9.0},
            run_calibration=True,
        )
        assert resp.status == "ok"
        assert resp.data["outcome"]["projection_id"] == projection_id
        assert resp.data["calibration"] is not None
        assert "magnitude_deviation" in resp.data["calibration"]
        assert "reliability_delta" in resp.data["calibration"]
        assert resp.causal_explanation
        assert resp.audit_id

    def test_outcome_without_calibration(self, api, db_session, seeded_graph):
        forecast_resp = api.retrieve_forecast(
            action_node_id=seeded_graph["alpha_node"].id,
        )
        projection_id = forecast_resp.data["projection_id"]

        resp = api.submit_outcome(
            projection_id=projection_id,
            realized_impact_vector={"OUTCOME": 5.0},
            run_calibration=False,
        )
        assert resp.status == "ok"
        assert resp.data["calibration"] is None

    def test_outcome_invalid_projection_returns_error(self, api):
        resp = api.submit_outcome(
            projection_id="nonexistent-projection",
            realized_impact_vector={"X": 1.0},
        )
        assert resp.status == "error"


# -----------------------------------------------------------------------
#  Audit log queries
# -----------------------------------------------------------------------

class TestAuditLog:
    def test_audit_log_populated(self, api, db_session, seeded_graph):
        api.submit_action(
            outcome_type="ACTION",
            domain_context={"agent_id": "audit_test"},
            magnitude=1.0,
        )
        api.retrieve_forecast(
            action_node_id=seeded_graph["alpha_node"].id,
        )
        entries = api.query_audit_log()
        assert len(entries) >= 2
        ops = {e["operation"] for e in entries}
        assert "submit_action" in ops
        assert "retrieve_forecast" in ops

    def test_audit_log_filter_by_operation(self, api, db_session, seeded_graph):
        api.submit_action(
            outcome_type="ACTION",
            domain_context={"agent_id": "filter_test"},
            magnitude=1.0,
        )
        entries = api.query_audit_log(operation="submit_action")
        assert all(e["operation"] == "submit_action" for e in entries)

    def test_audit_log_captures_errors(self, api, db_session):
        api.retrieve_forecast(action_node_id="bad-node")
        entries = api.query_audit_log(operation="retrieve_forecast")
        errors = [e for e in entries if e["status"] == "error"]
        assert len(errors) >= 1
        assert errors[0]["error_detail"] is not None


# -----------------------------------------------------------------------
#  End-to-end flow: submit → forecast → outcome → calibrate → profile
# -----------------------------------------------------------------------

class TestEndToEndFlow:
    def test_full_lifecycle(self, api, db_session, seeded_graph):
        # 1. Submit a new action
        action_resp = api.submit_action(
            outcome_type="ACTION",
            domain_context={"agent_id": "alpha", "domain": "code"},
            magnitude=6.0,
            causal_links=[
                {
                    "target_node_id": seeded_graph["outcome_a"].id,
                    "weight": 0.9,
                    "confidence": 0.95,
                },
            ],
        )
        assert action_resp.status == "ok"
        new_node_id = action_resp.data["node"]["id"]

        # 2. Retrieve forecast
        forecast_resp = api.retrieve_forecast(
            action_node_id=new_node_id,
            time_horizon=30.0,
        )
        assert forecast_resp.status == "ok"
        projection_id = forecast_resp.data["projection_id"]

        # 3. Run counterfactual
        cf_resp = api.run_counterfactual(
            source_node_id=new_node_id,
            removed_agent_id="beta",
        )
        assert cf_resp.status == "ok"

        # 4. Query synergy density
        synergy_resp = api.query_synergy_density(
            agent_node_ids=[new_node_id, seeded_graph["beta_node"].id],
        )
        assert synergy_resp.status == "ok"

        # 5. Submit realized outcome
        outcome_resp = api.submit_outcome(
            projection_id=projection_id,
            realized_impact_vector={"OUTCOME": 8.0, "ACTION": 6.0},
            run_calibration=True,
        )
        assert outcome_resp.status == "ok"
        assert outcome_resp.data["calibration"] is not None

        # 6. Trace provenance
        trace_resp = api.trace_provenance(
            metric_type="ImpactProjection",
            metric_id=projection_id,
        )
        assert trace_resp.status == "ok"

        # 7. Agent profile
        profile_resp = api.agent_impact_profile(agent_id="alpha")
        assert profile_resp.status == "ok"
        assert profile_resp.data["projections"]["count"] >= 1

        # 8. Audit log has all 7 entries
        log = api.query_audit_log(limit=100)
        ops = [e["operation"] for e in log]
        assert "submit_action" in ops
        assert "retrieve_forecast" in ops
        assert "run_counterfactual" in ops
        assert "query_synergy_density" in ops
        assert "submit_outcome" in ops
        assert "trace_provenance" in ops
        assert "agent_impact_profile" in ops

        # All responses carry version + audit_id + causal_explanation
        for r in [action_resp, forecast_resp, cf_resp, synergy_resp,
                   outcome_resp, trace_resp, profile_resp]:
            assert r.api_version
            assert r.audit_id
            assert r.causal_explanation

    def test_versioned_algorithm_is_recorded_in_audit(self, api, db_session, seeded_graph):
        api.submit_action(
            outcome_type="ACTION",
            domain_context={"agent_id": "ver_test"},
            magnitude=1.0,
        )
        entries = api.query_audit_log(operation="submit_action")
        assert any(e["algorithm_version"] is not None for e in entries)
