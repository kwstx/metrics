from datetime import datetime
import uuid
from sqlalchemy import Column, String, Float, DateTime, JSON, ForeignKey, create_engine
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

Base = declarative_base()

class ImpactNode(Base):
    """
    Represents an agent action or a measurable real-world outcome.
    """
    __tablename__ = 'impact_nodes'

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    outcome_type = Column(String, nullable=False) # e.g., "ACTION", "OUTCOME", "FEEDBACK"
    domain_context = Column(JSON, nullable=False) # e.g., {"domain": "code", "repo": "metrics"}
    magnitude = Column(Float, default=0.0)
    timestamp = Column(DateTime, default=datetime.utcnow)
    uncertainty_metadata = Column(JSON, nullable=True) # e.g., {"std_dev": 0.1, "distribution": "normal"}

    # Relationships for graph traversal
    outgoing_edges = relationship("ImpactEdge", foreign_keys="ImpactEdge.source_node_id", back_populates="source_node")
    incoming_edges = relationship("ImpactEdge", foreign_keys="ImpactEdge.target_node_id", back_populates="target_node")

    def __repr__(self):
        return f"<ImpactNode(id={self.id[:8]}, type={self.outcome_type}, magnitude={self.magnitude})>"

class ImpactEdge(Base):
    """
    Represents a causal connection between two nodes in the Impact Graph.
    """
    __tablename__ = 'impact_edges'

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    source_node_id = Column(String, ForeignKey('impact_nodes.id'), nullable=False)
    target_node_id = Column(String, ForeignKey('impact_nodes.id'), nullable=False)
    
    causal_weight = Column(Float, nullable=False) # Strength of the causal relationship
    confidence_score = Column(Float, default=1.0) # How confident we are in this edge
    propagation_delay = Column(Float, default=0.0) # Delay parameter (e.g., in seconds/units)
    
    edge_metadata = Column(JSON, nullable=True) # Additional edge properties

    # Relationships
    source_node = relationship("ImpactNode", foreign_keys=[source_node_id], back_populates="outgoing_edges")
    target_node = relationship("ImpactNode", foreign_keys=[target_node_id], back_populates="incoming_edges")

    def __repr__(self):
        return f"<ImpactEdge(source={self.source_node_id[:8]}, target={self.target_node_id[:8]}, weight={self.causal_weight})>"

# Database setup helper
def init_db(db_url="sqlite:///impact_graph.db"):
    engine = create_engine(db_url)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)
