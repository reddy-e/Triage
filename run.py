"""X-RayTriage AI - Main Application Runner.

Launches the complete FastAPI application server.
Serves both the REST API and the clinical frontend at http://localhost:8000.
"""

import sys
import uvicorn

# Ensure utf-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from backend.app.config import settings

def main():
    print("=" * 70)
    print("  X-RayTriage AI - Explainable AI for Chest X-Ray Triage")
    print("=" * 70)
    print(f"  * Model Architecture:    {settings.model_name}")
    print(f"  * Decision Thresholds:   High >= {settings.high_priority_threshold}, Medium >= {settings.medium_priority_threshold}")
    print(f"  * Max Upload Limit:      {settings.max_upload_size_bytes // (1024 * 1024)} MB")
    print(f"  * Medical Disclaimer:    {settings.safety_disclaimer}")
    print("=" * 70)
    print("  Server launching at:     http://localhost:8000")
    print("  Interactive API docs at: http://localhost:8000/docs")
    print("=" * 70)
    
    uvicorn.run(
        "backend.app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )

if __name__ == "__main__":
    main()
