from __future__ import annotations

import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI
from pydantic import BaseModel, Field
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from api.metrics_feedback_api import MetricsFeedbackAPI
from models.base import Base


DATABASE_URL = os.getenv("METRICS_DB_URL", "sqlite:///metrics_feedback.db")

_engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)


def init_db() -> None:
    Base.metadata.create_all(bind=_engine)


def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class ActionLink(BaseModel):
    target_node_id: str
    weight: float = 1.0
    confidence: float = 1.0
    delay: float = 0.0


class SubmitActionRequest(BaseModel):
    outcome_type: str
    domain_context: Dict[str, Any]
    magnitude: float = 0.0
    uncertainty: Optional[Dict[str, Any]] = None
    causal_links: Optional[List[ActionLink]] = None
    caller_identity: Optional[str] = None


class RetrieveForecastRequest(BaseModel):
    action_node_id: str
    time_horizon: float = 30.0
    task_sequence_id: Optional[str] = None
    decay_function: str = "exponential"
    decay_rate: float = 0.01
    decay_floor: float = 0.0
    caller_identity: Optional[str] = None


class CounterfactualRequest(BaseModel):
    source_node_id: str
    removed_agent_id: str
    time_horizon: float = 30.0
    caller_identity: Optional[str] = None


class SynergyDensityRequest(BaseModel):
    agent_node_ids: List[str]
    caller_identity: Optional[str] = None


class AgentProfileRequest(BaseModel):
    agent_id: str
    caller_identity: Optional[str] = None


class ProvenanceRequest(BaseModel):
    metric_type: str
    metric_id: str
    caller_identity: Optional[str] = None


class SubmitOutcomeRequest(BaseModel):
    projection_id: str
    realized_impact_vector: Dict[str, float]
    realized_timestamp: Optional[datetime] = None
    run_calibration: bool = True
    caller_identity: Optional[str] = None


class AuditQueryRequest(BaseModel):
    operation: Optional[str] = None
    since: Optional[datetime] = None
    limit: int = Field(default=50, ge=1, le=500)
    caller_identity: Optional[str] = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="Metrics and Feedback API",
    version="1.0.0",
    description=(
        "Structured API for action ingestion, impact forecasting, counterfactual "
        "simulation, synergy metrics, provenance tracing, and realized-outcome updates."
    ),
    lifespan=lifespan,
)


def _service(db: Session, caller_identity: Optional[str]) -> MetricsFeedbackAPI:
    return MetricsFeedbackAPI(session=db, caller_identity=caller_identity)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/actions")
def submit_action(payload: SubmitActionRequest, db: Session = Depends(get_db)) -> Dict[str, Any]:
    service = _service(db, payload.caller_identity)
    response = service.submit_action(
        outcome_type=payload.outcome_type,
        domain_context=payload.domain_context,
        magnitude=payload.magnitude,
        uncertainty=payload.uncertainty,
        causal_links=[x.model_dump() for x in payload.causal_links] if payload.causal_links else None,
    )
    return response.to_dict()


@app.post("/v1/forecasts")
def retrieve_forecast(payload: RetrieveForecastRequest, db: Session = Depends(get_db)) -> Dict[str, Any]:
    service = _service(db, payload.caller_identity)
    response = service.retrieve_forecast(
        action_node_id=payload.action_node_id,
        time_horizon=payload.time_horizon,
        task_sequence_id=payload.task_sequence_id,
        decay_function=payload.decay_function,
        decay_rate=payload.decay_rate,
        decay_floor=payload.decay_floor,
    )
    return response.to_dict()


@app.post("/v1/counterfactuals")
def run_counterfactual(payload: CounterfactualRequest, db: Session = Depends(get_db)) -> Dict[str, Any]:
    service = _service(db, payload.caller_identity)
    response = service.run_counterfactual(
        source_node_id=payload.source_node_id,
        removed_agent_id=payload.removed_agent_id,
        time_horizon=payload.time_horizon,
    )
    return response.to_dict()


@app.post("/v1/synergy-density")
def query_synergy_density(payload: SynergyDensityRequest, db: Session = Depends(get_db)) -> Dict[str, Any]:
    service = _service(db, payload.caller_identity)
    response = service.query_synergy_density(agent_node_ids=payload.agent_node_ids)
    return response.to_dict()


@app.post("/v1/agent-impact-profiles")
def agent_impact_profile(payload: AgentProfileRequest, db: Session = Depends(get_db)) -> Dict[str, Any]:
    service = _service(db, payload.caller_identity)
    response = service.agent_impact_profile(agent_id=payload.agent_id)
    return response.to_dict()


@app.post("/v1/provenance-traces")
def trace_provenance(payload: ProvenanceRequest, db: Session = Depends(get_db)) -> Dict[str, Any]:
    service = _service(db, payload.caller_identity)
    response = service.trace_provenance(metric_type=payload.metric_type, metric_id=payload.metric_id)
    return response.to_dict()


@app.post("/v1/outcomes")
def submit_outcome(payload: SubmitOutcomeRequest, db: Session = Depends(get_db)) -> Dict[str, Any]:
    service = _service(db, payload.caller_identity)
    response = service.submit_outcome(
        projection_id=payload.projection_id,
        realized_impact_vector=payload.realized_impact_vector,
        realized_timestamp=payload.realized_timestamp,
        run_calibration=payload.run_calibration,
    )
    return response.to_dict()


@app.post("/v1/audit-log")
def query_audit_log(payload: AuditQueryRequest, db: Session = Depends(get_db)) -> Dict[str, Any]:
    service = _service(db, payload.caller_identity)
    records = service.query_audit_log(operation=payload.operation, since=payload.since, limit=payload.limit)
    return {"status": "ok", "records": records}
