from typing import Dict, Any, Optional, List
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
import json

from models.impact_projection import (
    ImpactProjection, ImpactOutcome, CalibrationEvent, 
    AgentReliability, SynergyDensityMetric
)
from models.impact_graph import ImpactNode

class PredictiveCalibrationEngine:
    """
    Implements Predictive Calibration Tracking.
    Compares predicted impact vectors with realized real-world outcomes.
    Computes magnitude, timing, and synergy deviations to update agent reliability.
    """
    def __init__(self, session: Session):
        self.session = session

    def record_outcome(
        self, 
        projection_id: str, 
        realized_vector: Dict[str, float], 
        realized_timestamp: Optional[datetime] = None
    ) -> ImpactOutcome:
        """
        Persists a realized outcome for a specific projection once data becomes available.
        """
        projection = self.session.query(ImpactProjection).filter(ImpactProjection.id == projection_id).first()
        if not projection:
            raise ValueError(f"Projection {projection_id} not found")

        outcome = ImpactOutcome(
            projection_id=projection_id,
            realized_impact_vector=realized_vector,
            realized_timestamp=realized_timestamp or datetime.utcnow()
        )
        self.session.add(outcome)
        self.session.commit()
        return outcome

    def run_calibration(self, outcome_id: str) -> CalibrationEvent:
        """
        Computes magnitude deviation, timing deviation, and synergy assumption error.
        Updates each agent’s reliability coefficient accordingly.
        """
        outcome = self.session.query(ImpactOutcome).filter(ImpactOutcome.id == outcome_id).first()
        if not outcome:
            raise ValueError(f"Outcome {outcome_id} not found")

        projection = outcome.projection
        if not projection:
            raise ValueError(f"Projection for outcome {outcome_id} not found")

        # 1. Magnitude Deviation
        # Computed as the normalized L1 distance between predicted and realized vectors
        pred_vec = projection.predicted_impact_vector
        real_vec = outcome.realized_impact_vector
        
        all_keys = set(pred_vec.keys()) | set(real_vec.keys())
        total_error = 0.0
        total_pred_mag = 0.0
        
        for k in all_keys:
            p = pred_vec.get(k, 0.0)
            r = real_vec.get(k, 0.0)
            total_error += abs(r - p)
            total_pred_mag += abs(p)

        magnitude_deviation = total_error / max(total_pred_mag, 1.0)

        # 2. Timing Deviation
        # Difference between realized time and predicted time (start + horizon)
        expected_time = projection.timestamp + timedelta(seconds=projection.time_horizon)
        real_time = outcome.realized_timestamp
        
        time_diff_seconds = abs((real_time - expected_time).total_seconds())
        timing_deviation = time_diff_seconds / max(projection.time_horizon, 1.0)

        # 3. Synergy Assumption Error
        # Check if the node participated in a synergy computation
        synergy_error = 0.0
        # We look for metrics where this node_id is in the collaboration_structure
        # Note: JSON query syntax varies, this is a generic search
        synergy_metrics = self.session.query(SynergyDensityMetric).all()
        relevant_metric = None
        for sm in synergy_metrics:
            if projection.source_node_id in sm.collaboration_structure:
                relevant_metric = sm
                break

        if relevant_metric:
            # Predicted synergy ratio
            pred_synergy = relevant_metric.synergy_density_ratio
            
            # Realized synergy is complex to measure per-agent, so we use
            # the agent's performance relative to the cooperative prediction.
            # If the agent underperformed its cooperative expectation, it's a synergy error.
            real_mag = sum(real_vec.values())
            pred_mag = sum(pred_vec.values())
            
            # If pred_mag already included synergy, then the deviation is the error.
            # We weight this by how 'dense' the synergy was.
            synergy_error = abs((real_mag / max(pred_mag, 1e-9)) - 1.0) * pred_synergy
        else:
            # No synergy assumed -> synergy error is low/zero
            synergy_error = 0.0

        # 4. Update Agent Reliability
        # Feed updates back into the tracking layer
        source_node = self.session.query(ImpactNode).filter(ImpactNode.id == projection.source_node_id).first()
        agent_id = source_node.domain_context.get("agent_id") if source_node else None
        
        reliability_delta = 0.0
        new_coeff = 1.0
        
        if agent_id:
            agent_rel = self.session.query(AgentReliability).filter(AgentReliability.agent_id == agent_id).first()
            if not agent_rel:
                agent_rel = AgentReliability(agent_id=agent_id, reliability_coefficient=1.0)
                self.session.add(agent_rel)
            
            old_coeff = agent_rel.reliability_coefficient
            
            # Penalty logic
            penalty = (0.5 * min(magnitude_deviation, 1.0) + 
                       0.2 * min(timing_deviation, 1.0) + 
                       0.3 * min(synergy_error, 1.0))
            
            # Learning rate for adjustment
            lr = 0.1
            
            if penalty < 0.1:
                # High accuracy reward
                update = 0.05
            else:
                update = -lr * penalty
                
            agent_rel.reliability_coefficient = max(0.1, min(2.0, agent_rel.reliability_coefficient + update))
            agent_rel.last_updated = datetime.utcnow()
            
            new_coeff = agent_rel.reliability_coefficient
            reliability_delta = new_coeff - old_coeff

        # 5. Persist Calibration Event
        calibration = CalibrationEvent(
            outcome_id=outcome.id,
            magnitude_deviation=magnitude_deviation,
            timing_deviation=timing_deviation,
            synergy_assumption_error=synergy_error,
            new_reliability_coefficient=new_coeff,
            reliability_delta=reliability_delta
        )
        self.session.add(calibration)
        self.session.commit()

        return calibration
