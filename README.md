# FraudShield

End-to-end MLOps project: fraud detection on the PaySim dataset.

## Run the API
    pip install -r requirements.txt
    uvicorn src.serving.app:app --log-level warning
Open http://127.0.0.1:8000/docs

## Run tests
    python -m pytest -q

## Docker
    docker build -f docker/Dockerfile.api -t fraudshield .
    docker run -p 8000:8000 fraudshield

## Layout
- `src/` modules: `features/`, `models/`, `serving/`, `monitoring/`
- `tests/` pytest suite
- `scripts/` analysis and simulation scripts (run from the project root)
- `artifacts/` trained model files and metadata
- `docker/` Dockerfiles