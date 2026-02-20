from datetime import datetime
import uuid
from sqlalchemy import Column, String, Float, DateTime, JSON, ForeignKey
from sqlalchemy.orm import relationship
from .base import Base


class ImpactProjection(Base):
    """
    Represents a probabilistic downstream projection of an agent action.
    """
    __tablename__ = 'impact_projections'

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    source_node_id = Column(String, ForeignKey('impact_nodes.id'), nullable=False)
    
    # Multi-dimensional predicted impact vector (e.g., {"revenue": 100, "user_growth": 50})
    predicted_impact_vector = Column(JSON, nullable=False)
    
    # Uncertainty bounds (e.g., {"min": 80, "max": 120, "confidence": 0.95})
    uncertainty_bounds = Column(JSON, nullable=False)
    
    # Time horizon for the prediction (e.g., in seconds or as a future timestamp)
    time_horizon = Column(Float, nullable=False)
    
    # References to nodes or edges that this projection depends on
    dependency_references = Column(JSON, nullable=True)
    
    # Confidence interval for later recalibration
    confidence_interval = Column(JSON, nullable=False)
    
    timestamp = Column(DateTime, default=datetime.utcnow)

    # Relationship back to the source node
    source_node = relationship("ImpactNode", foreign_keys=[source_node_id])

    def __repr__(self):
        return f"<ImpactProjection(id={self.id[:8]}, source={self.source_node_id[:8]}, horizon={self.time_horizon})>"

class AgentReliability(Base):
    """
    Stores historical reliability coefficients for agents.
    """
    __tablename__ = 'agent_reliability'
    
    agent_id = Column(String, primary_key=True)
    reliability_coefficient = Column(Float, default=1.0)
    last_updated = Column(DateTime, default=datetime.utcnow)

class DomainMultiplier(Base):
    """
    Stores multipliers specific to domains or impact types.
    """
    __tablename__ = 'domain_multipliers'
    
    domain_name = Column(String, primary_key=True)
    multiplier = Column(Float, default=1.0)
    description = Column(String, nullable=True)
