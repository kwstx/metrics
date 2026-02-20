# System Testing Recommendations

## Running the Automated Test Suite
The primary way to verify the system is by using the Python testing framework, Pytest. You can execute the entire suite of tests by running the standard `pytest` command from the root directory of the project. This will automatically discover and run all the scripts located in the `tests/` folder. These tests cover individual components like the Synergy Density engine, the Impact Forecast logic, and the Provenance Tracing mechanism.

## End-to-End API Integration Testing
You can test the full lifecycle of the system through the HTTP layer. The system includes an integration test script specifically designed to simulate real-world usage. This process involves:

1.  **Seeding the Graph**: Creating initial nodes and edges to represent a causal impact network.
2.  **Action Submission**: Sending JSON data to the `/v1/actions` endpoint to record new agent activities.
3.  **Forecasting and Simulations**: Requesting downstream impact projections and running counterfactual scenarios to see how the system predicts influence when certain variables are removed.
4.  **Synergy and Outcome Calibration**: Submitting realized outcomes to the system and triggering the recalibration engine, which updates reliability coefficients based on the accuracy of past predictions.
5.  **Provenance Tracing**: Requesting a full trace for any generated metric to ensure every score can be reconstructed and explained through the causal path.

## Auditing and Verification
Every significant operation in the system generates an audit record. You can test the integrity of the system by querying the `/v1/audit-log` endpoint after performing actions to verify that each computation has been logged with a unique identifier and timestamp.

## Manual API Interaction
If you wish to test manually, you can start the web server using the `uvicorn` command and interact with the API endpoints directly using a browser or a command-line tool. The system is designed to provide descriptive error messages in a structured format, allowing you to test edge cases such as invalid node references or circular dependencies in the impact graph.
