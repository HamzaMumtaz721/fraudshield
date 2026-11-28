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
## API
| Endpoint | Purpose |
|---|---|
| GET /health | Application health |
| GET /model-info | Model version, SHA-256, threshold, features |
| POST /predict | Score one transaction (`/score` is an alias) |
| GET /metrics | Prometheus metrics |
| GET /stats | JSON counters |

## Monitoring stack (Docker Compose)
    cp .env.example .env   # then set GRAFANA_ADMIN_PASSWORD
    docker compose up -d --build
- API: http://127.0.0.1:8000
- Prometheus: http://127.0.0.1:9090 (alerts at /alerts)
- Grafana: http://127.0.0.1:3000 (login: admin / password from .env)
Alert rules: `monitoring/alert_rules.yml` (APIDown, HighErrorRate, HighLatencyP95, DataQualityFailures).
Cleanup: `docker compose down`
## Data versioning (DVC)
The PaySim dataset is tracked by DVC (`data/raw/paysim.csv.dvc`); the CSV is not in Git.

    pip install dvc
    dvc remote list    # the committed remote is a local folder on one laptop;
                       # on another machine set your own: dvc remote modify localstore url <path>
    dvc pull           # downloads the data
    dvc repro          # validates the data and builds data/processed/

Dataset versions are marked with Git tags (`data-v1`). Every MLflow run records
`git_commit` and `dvc_data_md5`. See `docs/dvc_merge_conflicts.md` for merge conflicts.