from typing import Dict, List, Tuple
from datetime import datetime
from sqlalchemy.orm import Session
from models.impact_projection import SynergyDensityMetric, SynergySignature


class PersistentSynergySignatureEngine:
    """
    Detects recurring collaboration patterns that consistently produce
    above-baseline synergy density over time and persists them as signatures.
    """

    def __init__(self, session: Session):
        self.session = session

    def detect_persistent_signatures(
        self,
        min_frequency: int = 3,
        min_consistency: float = 0.7,
        baseline_synergy_density: float = 1.0,
    ) -> List[SynergySignature]:
        """
        Mines historical SynergyDensityMetric records and persists signatures
        for recurring multi-agent combinations with reliable amplification.
        """
        metrics = self.session.query(SynergyDensityMetric).order_by(SynergyDensityMetric.timestamp.asc()).all()
        if not metrics:
            return []

        baseline = baseline_synergy_density
        grouped = self._group_by_collaboration_pattern(metrics)
        existing = self._load_existing_signatures()
        persisted: List[SynergySignature] = []

        for pattern, pattern_metrics in grouped.items():
            if len(pattern) < 2:
                # Signatures model collective intelligence, not solo performance.
                continue

            frequency = len(pattern_metrics)
            if frequency < min_frequency:
                continue

            ratios = [m.synergy_density_ratio for m in pattern_metrics]
            mean_ratio = sum(ratios) / frequency

            above_baseline_count = sum(1 for r in ratios if r > baseline)
            consistency = above_baseline_count / frequency

            if min_consistency > 0.0:
                if consistency < min_consistency or mean_ratio <= baseline:
                    continue

            amplification = mean_ratio - baseline
            stability = self._compute_stability_score(ratios)
            first_observed = min(m.timestamp for m in pattern_metrics) or datetime.utcnow()
            last_observed = max(m.timestamp for m in pattern_metrics) or datetime.utcnow()

            signature = existing.get(pattern)
            if signature is None:
                signature = SynergySignature(
                    collaboration_structure=list(pattern),
                    observation_frequency=frequency,
                    above_baseline_consistency=consistency,
                    mean_synergy_density_ratio=mean_ratio,
                    amplification_magnitude=amplification,
                    stability_score=stability,
                    baseline_synergy_density=baseline,
                    first_observed_at=first_observed,
                    last_observed_at=last_observed,
                )
                self.session.add(signature)
            else:
                signature.observation_frequency = frequency
                signature.above_baseline_consistency = consistency
                signature.mean_synergy_density_ratio = mean_ratio
                signature.amplification_magnitude = amplification
                signature.stability_score = stability
                signature.baseline_synergy_density = baseline
                signature.first_observed_at = first_observed
                signature.last_observed_at = last_observed

            persisted.append(signature)

        self.session.commit()
        return persisted

    def _group_by_collaboration_pattern(
        self,
        metrics: List[SynergyDensityMetric]
    ) -> Dict[Tuple[str, ...], List[SynergyDensityMetric]]:
        grouped: Dict[Tuple[str, ...], List[SynergyDensityMetric]] = {}
        for metric in metrics:
            pattern = tuple(sorted(metric.collaboration_structure))
            grouped.setdefault(pattern, []).append(metric)
        return grouped

    def _load_existing_signatures(self) -> Dict[Tuple[str, ...], SynergySignature]:
        signatures = self.session.query(SynergySignature).all()
        return {tuple(sorted(s.collaboration_structure)): s for s in signatures}

    def _compute_stability_score(self, ratios: List[float]) -> float:
        if not ratios:
            return 0.0
        if len(ratios) == 1:
            return 1.0

        mean_ratio = sum(ratios) / len(ratios)
        if mean_ratio == 0:
            return 0.0

        variance = sum((r - mean_ratio) ** 2 for r in ratios) / len(ratios)
        std_dev = variance ** 0.5
        coefficient_of_variation = std_dev / abs(mean_ratio)
        return 1.0 / (1.0 + coefficient_of_variation)
