#!/usr/bin/env bash
# run.sh — start all Traffic AI services
# Usage: bash run.sh
# Stop: Ctrl-C (SIGINT is forwarded to all background jobs)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# Activate virtual environment if present
if [ -f "$ROOT/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$ROOT/.venv/bin/activate"
fi

echo "=================================================="
echo "  Traffic AI — Context-Aware Ensemble System"
echo "=================================================="
echo ""

# Trap Ctrl-C and kill all spawned children cleanly
_cleanup() {
    echo ""
    echo "Stopping all services..."
    kill 0
    wait
    echo "All services stopped."
}
trap _cleanup INT TERM

echo "Starting model services..."
uvicorn models.markov_api:app        --host 0.0.0.0 --port 8001 --reload &
uvicorn models.rf_api:app            --host 0.0.0.0 --port 8002 --reload &
uvicorn models.lstm_api:app          --host 0.0.0.0 --port 8003 --reload &
uvicorn models.bayesian_api:app      --host 0.0.0.0 --port 8004 --reload &

# Give models time to finish startup training before the orchestrator starts
echo "Waiting for model services to initialise (~5 s)..."
sleep 5

echo "Starting ensemble orchestrator..."
uvicorn ensemble.main_api:app        --host 0.0.0.0 --port 8000 --reload &

echo ""
echo "  Service           URL"
echo "  ──────────────────────────────────────────────"
echo "  Ensemble API      http://localhost:8000/docs"
echo "  Markov            http://localhost:8001/docs"
echo "  Random Forest     http://localhost:8002/docs"
echo "  LSTM              http://localhost:8003/docs"
echo "  Bayesian Network  http://localhost:8004/docs"
echo ""
echo "Example request:"
echo '  curl -s -X POST http://localhost:8000/predict_traffic \'
echo '       -H "Content-Type: application/json" \'
echo '       -d '"'"'{"road_id":142,"avg_speed":24,"vehicle_count":210,'
echo '              "weather":"rain","hour":8,"previous_state":2}'"'"' | python3 -m json.tool'
echo ""
echo "Press Ctrl-C to stop all services."
echo "=================================================="

wait
