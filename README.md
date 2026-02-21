# Cooperative Intelligence Metrics System

## Overview

The Cooperative Intelligence Metrics System is an adaptive economic and behavioral control layer designed to govern agent interactions based on real-world causal impact and cooperative synergy. Unlike traditional metrics that rely on simple throughput or token counts, this system utilizes a dynamic Impact Graph to model influence propagation, predictive stability, and collaborative intelligence vectors.

The core objective is to provide a transparent, auditable, and mathematically rigorous framework for evaluating agent contributions within complex, multi-agent environments.

## Core Capabilities

### Impact Graph and Causal Modeling
The system maintains a real-time directed acyclic graph (DAG) representing causal connections between agent actions and real-world outcomes. Each edge in the graph encodes causal weights, confidence scores, and propagation delays, allowing for sophisticated influence tracing.

### Predictive Impact Forecasting
By traversing the Impact Graph, the system generates probabilistic downstream projections for any given action. These forecasts include multi-dimensional impact vectors, uncertainty bounds, and dependency references, moving beyond scalar scoring to provide a nuanced view of potential outcomes.

### Counterfactual Simulation
The system can simulate the removal of specific agents or actions to compute their marginal cooperative influence. This "what-if" analysis isolates the unique value added by an agent by comparing full-system projections against agent-absent counterfactuals.

### Synergy Density Computation
This engine measures the super-additive or sub-additive effects of agent clusters. By comparing the collective impact of a group against the sum of individual contributions in isolation, the system identifies high-performing collaborations and detects destructive interference.

### Impact Provenance Tracing
Transparency is maintained through a robust provenance mechanism. For every metric generated, the system can reconstruct the full causal path, including all nodes, edges, propagation weights, and predictive assumptions, ensuring full reproducibility and explainability.

### Immutable Audit Logging
Every significant operation and calculation is recorded in an immutable audit log. Each entry includes the algorithm version used, the request payload, the structured response, and a unique identifier, facilitating comprehensive auditing and verification.

## Technology Stack

- **API Layer**: FastAPI with Uvicorn for high-performance, asynchronous HTTP interaction.
- **Graph Engine**: NetworkX for complex graph traversals and causal path computation.
- **ORM & Database**: SQLAlchemy for flexible data modeling and support for multiple relational backends.
- **Validation**: Pydantic for rigorous data validation and schema enforcement.
- **Testing**: Pytest for comprehensive unit and integration testing.

## Getting Started

### Prerequisites

- Python 3.9 or higher
- Pip (Python Package Installer)

### Installation

1. Clone the repository to your local machine.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the API

Start the web server using Uvicorn:
```bash
uvicorn api.http_server:app --reload
```

## Testing and Verification

The system includes a comprehensive test suite covering core engines and API integration.

### Running Tests
Execute the test suite from the root directory:
```bash
pytest
```

### Auditing
To verify the integrity of computations, query the `/v1/audit-log` endpoint after performing operations. This will return a versioned record of the logic and data used to arrive at specific results.

## Documentation

For detailed information on testing strategies and system architecture, refer to the following documents:
- `testing_recommendations.md`: Comprehensive guide for unit and integration testing.
- `STRESS_TESTING.md`: Procedures for measuring system stability under heavy load.
