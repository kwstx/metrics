import time
import httpx
import subprocess
import os
import signal
import sys

def run_verification():
    print("Starting Metrics and Feedback HTTP Server...")
    # Start the server
    server_process = subprocess.Popen(
        ["python", "-m", "uvicorn", "api.http_server:app", "--port", "8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Wait for server to start
    time.sleep(3)
    
    try:
        url = "http://127.0.0.1:8000"
        client = httpx.Client(base_url=url)
        
        # 1. Health check
        print("Checking server health...")
        health = client.get("/health")
        print(f"Health Status: {health.json()}")
        
        # 2. Submit an action
        print("\nSubmitting a test action...")
        action_payload = {
            "outcome_type": "ACTION",
            "domain_context": {"agent_id": "auditor-01", "domain": "verification"},
            "magnitude": 15.0,
            "caller_identity": "verification-script"
        }
        action_resp = client.post("/v1/actions", json=action_payload)
        action_data = action_resp.json()
        print(f"Action Response Status: {action_data['status']}")
        print(f"Audit ID: {action_data['audit_id']}")
        
        # 3. Retrieve Forecast (another audit record)
        node_id = action_data["data"]["node"]["id"]
        print(f"\nRetrieving forecast for node {node_id}...")
        forecast_payload = {
            "action_node_id": node_id,
            "time_horizon": 30.0,
            "caller_identity": "verification-script"
        }
        forecast_resp = client.post("/v1/forecasts", json=forecast_payload)
        forecast_data = forecast_resp.json()
        print(f"Forecast Response Status: {forecast_data['status']}")
        
        # 4. Query Audit Log
        print("\nQuerying Audit Log for verification...")
        audit_payload = {
            "operation": "submit_action",
            "limit": 5,
            "caller_identity": "verification-script"
        }
        audit_resp = client.post("/v1/audit-log", json=audit_payload)
        audit_data = audit_resp.json()
        
        print("\n--- Recent Audit Records ---")
        for record in audit_data.get("records", []):
            print(f"ID: {record['id']} | Op: {record['operation']} | Caller: {record['caller_identity']} | TS: {record['timestamp']}")
        
        # Verify specific operation is there
        found = any(r['operation'] == 'submit_action' and r['caller_identity'] == 'verification-script' for r in audit_data['records'])
        if found:
            print("\nVerification SUCCESS: Audit record found for the submitted action.")
        else:
            print("\nVerification FAILURE: Required audit record not found.")

    finally:
        print("\nShutting down server...")
        server_process.terminate()
        try:
            server_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_process.kill()

if __name__ == "__main__":
    run_verification()
