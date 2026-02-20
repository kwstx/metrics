"""
Comprehensive tests for the Impact Provenance Tracing Engine.
Validates that every metric type is traceable, reproducible, and
explainable through the provenance trace logic.
"""

import math
from datetime import datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from models.base import Base
from models.impact_graph import ImpactNode, ImpactEdge
from models.impact_projection import (
    AgentReliability,
    CalibrationEvent,
    CooperativeIntelligenceMetric,
    CooperativeStabilityMetric,
    DomainMultiplier,
    ImpactOutcome,
    ImpactProjection,
    MarginalCooperativeInfluence,
    SynergyDensityMetric,
    SynergySignature,
    TemporalImpactLedgerEntry,
)
from engine.graph_manager import GraphManager
from engine.impact_provenance_tracing_engine import ImpactProvenanceTracingEngine


# ─── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    s = Session()
    yield s
    s.close()


@pytest.fixture
def graph_data(session):
    """
    Creates a rich impact graph with three agents across two domains.

    Graph topology:
        action_a1 ──(0.8)──► outcome_o1 ──(0.6)──► feedback_f1
        action_b1 ──(0.7)──► outcome_o1
        action_c1 ──(0.5)──► outcome_o2
    """
    gm = GraphManager(session)

    # Nodes
    a1 = gm.add_node("ACTION", {"agent_id": "agent_alpha", "domain": "code", "role": "developer"}, magnitude=10.0)
    b1 = gm.add_node("ACTION", {"agent_id": "agent_beta", "domain": "code", "role": "reviewer"}, magnitude=5.0)
    c1 = gm.add_node("ACTION", {"agent_id": "agent_gamma", "domain": "design", "role": "designer"}, magnitude=8.0)
    o1 = gm.add_node("OUTCOME", {"domain": "code"}, magnitude=20.0)
    o2 = gm.add_node("OUTCOME", {"domain": "design"}, magnitude=15.0)
    f1 = gm.add_node("FEEDBACK", {"domain": "code"}, magnitude=3.0)

    # Edges
    gm.add_causal_edge(a1.id, o1.id, weight=0.8, confidence=0.9, delay=1.0)
    gm.add_causal_edge(b1.id, o1.id, weight=0.7, confidence=0.85, delay=2.0)
    gm.add_causal_edge(o1.id, f1.id, weight=0.6, confidence=0.95, delay=0.5)
    gm.add_causal_edge(c1.id, o2.id, weight=0.5, confidence=0.8, delay=1.5)

    # Agent reliability
    rel_alpha = AgentReliability(agent_id="agent_alpha", reliability_coefficient=0.9)
    rel_beta = AgentReliability(agent_id="agent_beta", reliability_coefficient=0.85)
    session.add_all([rel_alpha, rel_beta])

    # Domain multiplier
    dm_code = DomainMultiplier(domain_name="code", multiplier=1.2, description="Code domain")
    dm_design = DomainMultiplier(domain_name="design", multiplier=0.9, description="Design domain")
    session.add_all([dm_code, dm_design])
    session.commit()

    return {
        "a1": a1, "b1": b1, "c1": c1,
        "o1": o1, "o2": o2, "f1": f1,
    }


@pytest.fixture
def tracing_engine(session):
    return ImpactProvenanceTracingEngine(session)


# ─── Test: Unsupported metric type ────────────────────────────────────────


class TestDispatch:
    def test_unsupported_type_raises(self, tracing_engine):
        with pytest.raises(ValueError, match="Unsupported metric type"):
            tracing_engine.trace("UnknownType", "some-id")

    def test_all_supported_types_dispatchable(self, tracing_engine):
        """Each supported type should invoke corresponding method (not crash on dispatch)."""
        from engine.impact_provenance_tracing_engine import _SUPPORTED_METRIC_TYPES
        for mt in _SUPPORTED_METRIC_TYPES:
            # Will raise ValueError for the *record* not found — NOT KeyError
            with pytest.raises(ValueError, match="not found"):
                tracing_engine.trace(mt, "nonexistent-id")


# ─── Test: ImpactProjection trace ─────────────────────────────────────────


class TestTraceProjection:
    def _create_projection(self, session, graph_data) -> ImpactProjection:
        from engine.impact_forecast_engine import ImpactForecastEngine
        fe = ImpactForecastEngine(session)
        return fe.forecast_action(graph_data["a1"].id, time_horizon=30.0)

    def test_trace_structure(self, session, graph_data, tracing_engine):
        proj = self._create_projection(session, graph_data)
        result = tracing_engine.trace("ImpactProjection", proj.id)

        assert result["metric_type"] == "ImpactProjection"
        assert result["metric_id"] == proj.id
        assert result["source_node_id"] == graph_data["a1"].id

        # Path should have nodes and edges
        assert "nodes" in result["path"]
        assert "edges" in result["path"]
        assert len(result["path"]["nodes"]) >= 1

        # Predictive assumptions
        pa = result["predictive_assumptions"]
        assert pa["agent_id"] == "agent_alpha"
        assert pa["agent_reliability"] == 0.9
        assert pa["domain"] == "code"
        assert pa["domain_multiplier"] == 1.2
        assert pa["time_horizon"] == 30.0

    def test_trace_contains_influence_propagation(self, session, graph_data, tracing_engine):
        proj = self._create_projection(session, graph_data)
        result = tracing_engine.trace("ImpactProjection", proj.id)

        ip = result["influence_propagation"]
        assert isinstance(ip, list)
        assert len(ip) >= 1

        # First entry should be the source node
        first = ip[0]
        assert first["node_id"] == graph_data["a1"].id
        assert first["incoming_influence"] == 1.0
        assert first["reliability"] == 0.9
        assert first["multiplier"] == 1.2

    def test_trace_reproducibility_matches(self, session, graph_data, tracing_engine):
        proj = self._create_projection(session, graph_data)
        result = tracing_engine.trace("ImpactProjection", proj.id)

        repro = result["reproducibility"]
        assert repro["match"] is True

    def test_verify_reproducibility_passes(self, session, graph_data, tracing_engine):
        proj = self._create_projection(session, graph_data)
        status = tracing_engine.verify_reproducibility("ImpactProjection", proj.id)
        assert status["status"] == "verified"

    def test_trace_edges_include_weights(self, session, graph_data, tracing_engine):
        proj = self._create_projection(session, graph_data)
        result = tracing_engine.trace("ImpactProjection", proj.id)

        edges = result["path"]["edges"]
        assert len(edges) >= 1
        for edge in edges:
            assert "causal_weight" in edge
            assert "confidence" in edge
            assert "delay" in edge

    def test_not_found_raises(self, tracing_engine):
        with pytest.raises(ValueError, match="not found"):
            tracing_engine.trace_projection("nonexistent-id")


# ─── Test: SynergyDensityMetric trace ─────────────────────────────────────


class TestTraceSynergy:
    def _create_synergy(self, session, graph_data) -> SynergyDensityMetric:
        from engine.synergy_density_engine import SynergyDensityEngine
        sde = SynergyDensityEngine(session)
        return sde.calculate_synergy_density([
            graph_data["a1"].id, graph_data["b1"].id
        ])

    def test_trace_structure(self, session, graph_data, tracing_engine):
        metric = self._create_synergy(session, graph_data)
        result = tracing_engine.trace("SynergyDensityMetric", metric.id)

        assert result["metric_type"] == "SynergyDensityMetric"
        assert result["synergy_multiplier"] == metric.synergy_density_ratio
        assert len(result["collaboration_structure"]) == 2
        assert isinstance(result["constituent_traces"], list)

    def test_reproducibility(self, session, graph_data, tracing_engine):
        metric = self._create_synergy(session, graph_data)
        result = tracing_engine.trace("SynergyDensityMetric", metric.id)

        repro = result["reproducibility"]
        assert repro["match"] is True

    def test_not_found_raises(self, tracing_engine):
        with pytest.raises(ValueError, match="not found"):
            tracing_engine.trace_synergy("nonexistent-id")


# ─── Test: CooperativeStabilityMetric trace ────────────────────────────────


class TestTraceStability:
    def _create_stability(self, session) -> CooperativeStabilityMetric:
        from engine.cooperative_stability_engine import CooperativeStabilityEngine
        cse = CooperativeStabilityEngine(session)
        return cse.record_stability_metric(
            agent_id="agent_alpha",
            negotiation_convergence_time=5.0,
            resource_allocation_variance=0.1,
            conflict_resolution_frequency=1.0,
            team_performance_stability=0.5,
            team_composition=["agent_alpha", "agent_beta"],
        )

    def test_trace_structure(self, session, graph_data, tracing_engine):
        metric = self._create_stability(session)
        result = tracing_engine.trace("CooperativeStabilityMetric", metric.id)

        assert result["metric_type"] == "CooperativeStabilityMetric"
        assert result["agent_id"] == "agent_alpha"
        assert "components" in result
        assert result["components"]["negotiation_convergence_time"] == 5.0

    def test_reproducibility(self, session, graph_data, tracing_engine):
        metric = self._create_stability(session)
        result = tracing_engine.trace("CooperativeStabilityMetric", metric.id)

        repro = result["reproducibility"]
        assert repro["match"] is True
        assert repro["reproduced_stability_coefficient"] == metric.stability_coefficient

    def test_derivation_formula_present(self, session, graph_data, tracing_engine):
        metric = self._create_stability(session)
        result = tracing_engine.trace("CooperativeStabilityMetric", metric.id)

        assert "formula" in result["derivation"]
        assert "normalization_targets" in result["derivation"]

    def test_not_found_raises(self, tracing_engine):
        with pytest.raises(ValueError, match="not found"):
            tracing_engine.trace_stability("nonexistent-id")


# ─── Test: CooperativeIntelligenceMetric trace ─────────────────────────────


class TestTraceIntelligence:
    def _create_intelligence(self, session, graph_data) -> CooperativeIntelligenceMetric:
        from engine.cooperative_intelligence_index_engine import CooperativeIntelligenceIndexEngine
        engine = CooperativeIntelligenceIndexEngine(session)
        return engine.generate_cooperative_intelligence_vector(
            agent_node_ids=[graph_data["a1"].id, graph_data["b1"].id],
            baseline_predictions={
                "agent_alpha": {"code": 10.0, "test": 5.0},
                "agent_beta": {"code": 8.0, "test": 6.0},
            },
            cooperative_predictions={
                "agent_alpha": {"code": 11.0, "test": 5.5},
                "agent_beta": {"code": 9.0, "test": 5.5},
            },
            calibration_errors_before={"code": 0.3, "test": 0.2},
            calibration_errors_after={"code": 0.15, "test": 0.1},
        )

    def test_trace_structure(self, session, graph_data, tracing_engine):
        metric = self._create_intelligence(session, graph_data)
        result = tracing_engine.trace("CooperativeIntelligenceMetric", metric.id)

        assert result["metric_type"] == "CooperativeIntelligenceMetric"
        assert "sub_metrics" in result
        assert "result_vector" in result

    def test_reproducibility(self, session, graph_data, tracing_engine):
        metric = self._create_intelligence(session, graph_data)
        result = tracing_engine.trace("CooperativeIntelligenceMetric", metric.id)

        repro = result["reproducibility"]
        assert repro["match"] is True

    def test_not_found_raises(self, tracing_engine):
        with pytest.raises(ValueError, match="not found"):
            tracing_engine.trace_intelligence("nonexistent-id")


# ─── Test: CalibrationEvent trace ──────────────────────────────────────────


class TestTraceCalibration:
    def _create_calibration(self, session, graph_data):
        from engine.impact_forecast_engine import ImpactForecastEngine
        from engine.predictive_calibration_engine import PredictiveCalibrationEngine

        fe = ImpactForecastEngine(session)
        proj = fe.forecast_action(graph_data["a1"].id, time_horizon=30.0)

        pce = PredictiveCalibrationEngine(session)
        realized = {k: v * 0.9 for k, v in proj.predicted_impact_vector.items()}
        outcome = pce.record_outcome(
            proj.id, realized,
            realized_timestamp=proj.timestamp + timedelta(seconds=32),
        )
        event = pce.run_calibration(outcome.id)
        return proj, outcome, event

    def test_trace_structure(self, session, graph_data, tracing_engine):
        proj, outcome, event = self._create_calibration(session, graph_data)
        result = tracing_engine.trace("CalibrationEvent", event.id)

        assert result["metric_type"] == "CalibrationEvent"
        assert result["outcome_id"] == outcome.id
        assert result["projection_id"] == proj.id
        assert result["agent_id"] == "agent_alpha"

    def test_deviations_present(self, session, graph_data, tracing_engine):
        _, _, event = self._create_calibration(session, graph_data)
        result = tracing_engine.trace("CalibrationEvent", event.id)

        devs = result["deviations"]
        assert "magnitude_deviation" in devs
        assert "timing_deviation" in devs
        assert "synergy_assumption_error" in devs

    def test_reproducibility_magnitude(self, session, graph_data, tracing_engine):
        _, _, event = self._create_calibration(session, graph_data)
        result = tracing_engine.trace("CalibrationEvent", event.id)

        repro = result["reproducibility"]
        assert repro["magnitude_match"] is True

    def test_reproducibility_timing(self, session, graph_data, tracing_engine):
        _, _, event = self._create_calibration(session, graph_data)
        result = tracing_engine.trace("CalibrationEvent", event.id)

        repro = result["reproducibility"]
        assert repro["timing_match"] is True

    def test_not_found_raises(self, tracing_engine):
        with pytest.raises(ValueError, match="not found"):
            tracing_engine.trace_calibration("nonexistent-id")


# ─── Test: MarginalCooperativeInfluence trace ──────────────────────────────


class TestTraceMarginalInfluence:
    def _create_marginal(self, session, graph_data):
        from engine.counterfactual_simulation import CounterfactualSimulation
        cs = CounterfactualSimulation(session)
        return cs.simulate_agent_removal(
            source_node_id=graph_data["a1"].id,
            removed_agent_id="agent_beta",
            time_horizon=30.0,
        )

    def test_trace_structure(self, session, graph_data, tracing_engine):
        metric = self._create_marginal(session, graph_data)
        result = tracing_engine.trace("MarginalCooperativeInfluence", metric.id)

        assert result["metric_type"] == "MarginalCooperativeInfluence"
        assert result["source_node_id"] == graph_data["a1"].id
        assert result["removed_agent_id"] == "agent_beta"
        assert isinstance(result["removed_agent_node_ids"], list)

    def test_path_present(self, session, graph_data, tracing_engine):
        metric = self._create_marginal(session, graph_data)
        result = tracing_engine.trace("MarginalCooperativeInfluence", metric.id)

        assert "nodes" in result["path"]
        assert "edges" in result["path"]

    def test_reproducibility(self, session, graph_data, tracing_engine):
        metric = self._create_marginal(session, graph_data)
        result = tracing_engine.trace("MarginalCooperativeInfluence", metric.id)

        repro = result["reproducibility"]
        assert repro["match"] is True

    def test_not_found_raises(self, tracing_engine):
        with pytest.raises(ValueError, match="not found"):
            tracing_engine.trace_marginal_influence("nonexistent-id")


# ─── Test: TemporalImpactLedgerEntry trace ─────────────────────────────────


class TestTraceTemporalEntry:
    def _create_entry(self, session, graph_data) -> TemporalImpactLedgerEntry:
        from engine.temporal_impact_memory_engine import TemporalImpactMemoryEngine
        tme = TemporalImpactMemoryEngine(session)
        return tme.append_contribution(
            task_sequence_id="seq-001",
            source_node_id=graph_data["a1"].id,
            impact_vector={"code": 15.0, "test": 7.5},
            decay_function="exponential",
            decay_rate=0.01,
            decay_floor=0.1,
            entry_metadata={"time_horizon": 30.0},
        )

    def test_trace_structure(self, session, graph_data, tracing_engine):
        entry = self._create_entry(session, graph_data)
        result = tracing_engine.trace("TemporalImpactLedgerEntry", entry.id)

        assert result["metric_type"] == "TemporalImpactLedgerEntry"
        assert result["task_sequence_id"] == "seq-001"
        assert result["source_node_id"] == graph_data["a1"].id

    def test_decay_parameters(self, session, graph_data, tracing_engine):
        entry = self._create_entry(session, graph_data)
        result = tracing_engine.trace("TemporalImpactLedgerEntry", entry.id)

        dp = result["decay_parameters"]
        assert dp["function"] == "exponential"
        assert dp["rate"] == 0.01
        assert dp["floor"] == 0.1

    def test_temporal_context(self, session, graph_data, tracing_engine):
        entry = self._create_entry(session, graph_data)
        result = tracing_engine.trace("TemporalImpactLedgerEntry", entry.id)

        tc = result["temporal_context"]
        assert tc["current_decay_weight"] > 0.0
        assert tc["current_decay_weight"] <= 1.0
        assert "current_decayed_vector" in tc

    def test_not_found_raises(self, tracing_engine):
        with pytest.raises(ValueError, match="not found"):
            tracing_engine.trace_temporal_entry("nonexistent-id")


# ─── Test: SynergySignature trace ─────────────────────────────────────────


class TestTraceSynergySignature:
    def _create_signature(self, session, graph_data) -> SynergySignature:
        """
        Directly creates SynergyDensityMetric records and a SynergySignature
        to avoid issues where identical graph traversals produce a ratio
        exactly equal to the global baseline.
        """
        agent_ids = sorted([graph_data["a1"].id, graph_data["b1"].id])

        # Create 4 synergy density metrics with varied ratios to ensure
        # at least some are above the global baseline.
        ratios = [1.3, 1.5, 1.4, 1.6]
        for r in ratios:
            m = SynergyDensityMetric(
                collaboration_structure=agent_ids,
                independent_impact_sum={"code": 10.0},
                cooperative_impact={"code": 10.0 * r},
                synergy_density_ratio=r,
            )
            session.add(m)
        session.commit()

        mean_ratio = sum(ratios) / len(ratios)
        baseline = mean_ratio  # All metrics have the same pattern
        amplification = mean_ratio - baseline  # 0 in this case, but see below

        # We need additional "noise" metrics for OTHER patterns so the
        # global baseline is lower than our pattern's mean.
        noise = SynergyDensityMetric(
            collaboration_structure=["fake_a", "fake_b"],
            independent_impact_sum={"x": 5.0},
            cooperative_impact={"x": 4.0},
            synergy_density_ratio=0.8,
        )
        session.add(noise)
        session.commit()

        from engine.persistent_synergy_signature_engine import PersistentSynergySignatureEngine
        psse = PersistentSynergySignatureEngine(session)
        signatures = psse.detect_persistent_signatures(min_frequency=3, min_consistency=0.0)
        assert len(signatures) >= 1, f"Expected at least 1 signature, got {len(signatures)}"
        # Find the one matching our agent_ids pattern
        for s in signatures:
            if sorted(s.collaboration_structure) == agent_ids:
                return s
        return signatures[0]

    def test_trace_structure(self, session, graph_data, tracing_engine):
        sig = self._create_signature(session, graph_data)
        result = tracing_engine.trace("SynergySignature", sig.id)

        assert result["metric_type"] == "SynergySignature"
        assert result["observation_frequency"] >= 3
        assert isinstance(result["underlying_metric_ids"], list)
        assert isinstance(result["underlying_ratios"], list)

    def test_reproducibility_frequency(self, session, graph_data, tracing_engine):
        sig = self._create_signature(session, graph_data)
        result = tracing_engine.trace("SynergySignature", sig.id)

        repro = result["reproducibility"]
        assert repro["frequency_match"] is True

    def test_reproducibility_stability(self, session, graph_data, tracing_engine):
        sig = self._create_signature(session, graph_data)
        result = tracing_engine.trace("SynergySignature", sig.id)

        repro = result["reproducibility"]
        assert repro["stability_match"] is True

    def test_not_found_raises(self, tracing_engine):
        with pytest.raises(ValueError, match="not found"):
            tracing_engine.trace_synergy_signature("nonexistent-id")


# ─── Test: verify_reproducibility convenience method ───────────────────────


class TestVerifyReproducibility:
    def test_verified_projection(self, session, graph_data, tracing_engine):
        from engine.impact_forecast_engine import ImpactForecastEngine
        fe = ImpactForecastEngine(session)
        proj = fe.forecast_action(graph_data["a1"].id, time_horizon=30.0)

        result = tracing_engine.verify_reproducibility("ImpactProjection", proj.id)
        assert result["status"] == "verified"

    def test_verified_stability(self, session, graph_data, tracing_engine):
        from engine.cooperative_stability_engine import CooperativeStabilityEngine
        cse = CooperativeStabilityEngine(session)
        metric = cse.record_stability_metric(
            agent_id="agent_alpha",
            negotiation_convergence_time=5.0,
            resource_allocation_variance=0.1,
            conflict_resolution_frequency=1.0,
            team_performance_stability=0.5,
            team_composition=["agent_alpha", "agent_beta"],
        )

        result = tracing_engine.verify_reproducibility("CooperativeStabilityMetric", metric.id)
        assert result["status"] == "verified"


# ─── Test: Edge cases ─────────────────────────────────────────────────────


class TestEdgeCases:
    def test_projection_no_reliability_no_multiplier(self, session, tracing_engine):
        """Agent without stored reliability or domain multiplier defaults to 1.0."""
        gm = GraphManager(session)
        node = gm.add_node("ACTION", {"agent_id": "nobody", "domain": "unknown_domain"}, magnitude=5.0)

        from engine.impact_forecast_engine import ImpactForecastEngine
        fe = ImpactForecastEngine(session)
        proj = fe.forecast_action(node.id, time_horizon=10.0)

        result = tracing_engine.trace("ImpactProjection", proj.id)
        pa = result["predictive_assumptions"]
        assert pa["agent_reliability"] == 1.0
        assert pa["domain_multiplier"] == 1.0
        assert result["reproducibility"]["match"] is True

    def test_trace_with_isolated_node_no_edges(self, session, tracing_engine):
        """A node with no outgoing edges should still produce a valid trace."""
        gm = GraphManager(session)
        node = gm.add_node("ACTION", {"agent_id": "solo", "domain": "code"}, magnitude=3.0)

        from engine.impact_forecast_engine import ImpactForecastEngine
        fe = ImpactForecastEngine(session)
        proj = fe.forecast_action(node.id, time_horizon=5.0)

        result = tracing_engine.trace("ImpactProjection", proj.id)
        assert len(result["path"]["nodes"]) == 1
        assert len(result["path"]["edges"]) == 0
        assert result["reproducibility"]["match"] is True

    def test_synergy_context_linked_to_projection(self, session, graph_data, tracing_engine):
        """If a synergy metric exists for the projection's node, it shows in the trace."""
        from engine.synergy_density_engine import SynergyDensityEngine
        from engine.impact_forecast_engine import ImpactForecastEngine

        sde = SynergyDensityEngine(session)
        sde.calculate_synergy_density([graph_data["a1"].id, graph_data["b1"].id])

        fe = ImpactForecastEngine(session)
        proj = fe.forecast_action(graph_data["a1"].id, time_horizon=30.0)

        result = tracing_engine.trace("ImpactProjection", proj.id)
        assert len(result["synergy_multipliers"]) >= 1
        assert "synergy_density_ratio" in result["synergy_multipliers"][0]


# ─── Test: Deep causal chain tracing ───────────────────────────────────────


class TestDeepCausalChain:
    def test_multi_hop_trace(self, session, tracing_engine):
        """Verify influence propagation across a 4-node chain."""
        gm = GraphManager(session)

        n1 = gm.add_node("ACTION", {"agent_id": "deep_agent", "domain": "code"}, magnitude=10.0)
        n2 = gm.add_node("OUTCOME", {"domain": "code"}, magnitude=8.0)
        n3 = gm.add_node("OUTCOME", {"domain": "code"}, magnitude=6.0)
        n4 = gm.add_node("FEEDBACK", {"domain": "code"}, magnitude=2.0)

        gm.add_causal_edge(n1.id, n2.id, weight=0.9, confidence=1.0)
        gm.add_causal_edge(n2.id, n3.id, weight=0.7, confidence=0.95)
        gm.add_causal_edge(n3.id, n4.id, weight=0.5, confidence=0.9)

        from engine.impact_forecast_engine import ImpactForecastEngine
        fe = ImpactForecastEngine(session)
        proj = fe.forecast_action(n1.id, time_horizon=60.0)

        result = tracing_engine.trace("ImpactProjection", proj.id)

        # Should contain all 4 nodes
        node_ids = {n["id"] for n in result["path"]["nodes"]}
        assert n1.id in node_ids
        assert n2.id in node_ids
        assert n3.id in node_ids
        assert n4.id in node_ids

        # Should contain 3 edges
        assert len(result["path"]["edges"]) == 3

        # Influence propagation should have 4 entries
        assert len(result["influence_propagation"]) == 4

        # The first node should have influence = 1.0
        assert result["influence_propagation"][0]["incoming_influence"] == 1.0

        # Vector should be fully reproducible
        assert result["reproducibility"]["match"] is True
