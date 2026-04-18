# Airflow DAGs for Ad Generator MLOps Pipeline

This directory contains Apache Airflow DAGs that automate the ML pipeline:

## DAGs Overview

### 1. `data_ingestion_dag.py`
Automates data collection and preparation
- **Schedule**: Daily at 2 AM
- **Tasks**:
  - Fetch new product data
  - Validate data quality
  - Store in PostgreSQL
  - Trigger data quality checks

### 2. `model_retraining_dag.py`
Handles model retraining workflow
- **Schedule**: Weekly (Sunday 3 AM)
- **Triggers**: Also triggered when data quality improves significantly
- **Tasks**:
  - Check if retraining needed (data drift, performance drop)
  - Load training data from PostgreSQL
  - Train new model version
  - Log to MLflow with auto-versioning
  - Compare with production model
  - Register best model
  - Notify on completion

### 3. `batch_inference_dag.py`
Generates ads in batch for large product catalogs
- **Schedule**: On-demand or weekly
- **Tasks**:
  - Load products requiring ads
  - Generate ads in batches
  - Store results in database
  - Export to CSV/JSON
  - Quality check generated ads

## Running Airflow Locally

```bash
# Start Airflow (already in docker-compose)
docker-compose up airflow-webserver airflow-scheduler

# Access UI
# URL: http://localhost:8080
# Username: admin
# Password: admin
```

## Configuration

- DAGs are auto-discovered from this directory
- Connection to PostgreSQL: `postgres_default`
- Connection to MLflow: `mlflow_default`
- All configurations in `airflow.cfg`
