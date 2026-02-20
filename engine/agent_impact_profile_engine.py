from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, Iterable

from sqlalchemy.orm import Session

from models.impact_projection import AgentImpactProfile


_DEFAULT_DIMENSIONS = {
    "marginal_cooperative_influence": {"value": 0.0, "samples": 0, "last_vector": {}},
    "synergy_amplification_contribution": {"value": 0.0, "samples": 0},
    "predictive_accuracy_index": {"value": 0.0, "samples": 0},
    "stability_coefficient": {"value": 0.0, "samples": 0},
    "long_term_impact_weight": {"value": 0.0, "samples": 0},
    "cross_role_integration_depth": {"value": 0.0, "samples": 0},
}


class AgentImpactProfileEngine:
    """
    Upserts agent profiles and continuously updates each dimension independently.
    """

    def __init__(self, session: Session):
        self.session = session

    def ensure_profile(self, agent_id: str) -> AgentImpactProfile:
        profile = self.session.get(AgentImpactProfile, agent_id)
        if profile:
            if not profile.impact_dimensions:
                profile.impact_dimensions = deepcopy(_DEFAULT_DIMENSIONS)
                profile.last_updated = datetime.utcnow()
                self.session.commit()
            return profile

        profile = AgentImpactProfile(
            agent_id=agent_id,
            impact_dimensions=deepcopy(_DEFAULT_DIMENSIONS),
        )
        self.session.add(profile)
        self.session.commit()
        return profile

    def update_marginal_cooperative_influence(
        self,
        agent_id: str,
        total_marginal_influence: float,
        marginal_vector: Dict[str, float],
    ) -> AgentImpactProfile:
        profile = self.ensure_profile(agent_id)
        self._update_dimension(profile, "marginal_cooperative_influence", float(total_marginal_influence))
        profile.impact_dimensions["marginal_cooperative_influence"]["last_vector"] = marginal_vector
        return self._persist(profile)

    def update_synergy_amplification_contribution(
        self,
        agent_ids: Iterable[str],
        synergy_density_ratio: float,
    ) -> None:
        for agent_id in set(agent_ids):
            if not agent_id:
                continue
            profile = self.ensure_profile(agent_id)
            self._update_dimension(
                profile,
                "synergy_amplification_contribution",
                float(synergy_density_ratio),
            )
            self._persist(profile)

    def update_predictive_accuracy_index(
        self,
        agent_id: str,
        magnitude_deviation: float,
        timing_deviation: float,
        synergy_assumption_error: float,
        reliability_coefficient: float,
    ) -> AgentImpactProfile:
        penalty = (
            0.5 * min(max(magnitude_deviation, 0.0), 1.0)
            + 0.2 * min(max(timing_deviation, 0.0), 1.0)
            + 0.3 * min(max(synergy_assumption_error, 0.0), 1.0)
        )
        accuracy_from_error = 1.0 - penalty
        normalized_reliability = (max(min(reliability_coefficient, 2.0), 0.1) - 0.1) / 1.9
        predictive_accuracy_index = (0.7 * accuracy_from_error) + (0.3 * normalized_reliability)

        profile = self.ensure_profile(agent_id)
        self._update_dimension(profile, "predictive_accuracy_index", predictive_accuracy_index)
        return self._persist(profile)

    def update_stability_coefficient(self, agent_id: str, stability_coefficient: float) -> AgentImpactProfile:
        profile = self.ensure_profile(agent_id)
        self._update_dimension(profile, "stability_coefficient", float(stability_coefficient))
        return self._persist(profile)

    def update_long_term_impact_weight(
        self,
        agent_id: str,
        impact_vector: Dict[str, float],
        time_horizon: float,
        decay_rate: float = 0.0,
    ) -> AgentImpactProfile:
        total_magnitude = sum(abs(float(v)) for v in (impact_vector or {}).values())
        horizon_multiplier = 1.0 + (min(max(time_horizon, 0.0), 3600.0) / 3600.0)
        decay_adjustment = max(0.0, 1.0 - min(max(decay_rate, 0.0), 1.0))
        long_term_weight = total_magnitude * horizon_multiplier * decay_adjustment

        profile = self.ensure_profile(agent_id)
        self._update_dimension(profile, "long_term_impact_weight", long_term_weight)
        return self._persist(profile)

    def update_cross_role_integration_depth(
        self,
        agent_ids: Iterable[str],
        normalized_depth: float,
    ) -> None:
        for agent_id in set(agent_ids):
            if not agent_id:
                continue
            profile = self.ensure_profile(agent_id)
            self._update_dimension(profile, "cross_role_integration_depth", float(normalized_depth))
            self._persist(profile)

    def _update_dimension(self, profile: AgentImpactProfile, dimension_key: str, observation: float) -> None:
        dimensions = deepcopy(profile.impact_dimensions or _DEFAULT_DIMENSIONS)
        current = dimensions.get(dimension_key, {"value": 0.0, "samples": 0})
        samples = int(current.get("samples", 0)) + 1
        previous_value = float(current.get("value", 0.0))
        updated_value = ((previous_value * (samples - 1)) + observation) / samples

        current["value"] = float(updated_value)
        current["samples"] = samples
        dimensions[dimension_key] = current
        profile.impact_dimensions = dimensions

    def _persist(self, profile: AgentImpactProfile) -> AgentImpactProfile:
        profile.last_updated = datetime.utcnow()
        self.session.commit()
        return profile
