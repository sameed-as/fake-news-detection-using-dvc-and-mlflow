# 📖 Runbook - Operations Guide

Quick reference for deploying, operating, and troubleshooting the Ad Generator system.

---

## 🚀 Deployment Procedures

### Local Setup (10 minutes)

```bash
# 1. Clone repository
git clone https://github.com/yourusername/ad-generator.git
cd ad-generator

# 2. Setup environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Copy environment template
cp .env.example .env

# 4. Start infrastructure
docker-compose up -d

# 5. Verify services
docker-compose ps
# All services should show "Up"

# 6. Setup MinIO bucket
python scripts/setup_minio.py

# 7. Train model (optional - can use pretrained)
python src/model/train.py

# 8. Start API
python -m uvicorn src.api.main:app --reload
```

**Verification**:
- MLflow: http://localhost:5000
- Prometheus: http://localhost:9090/targets (all UP)
- Grafana: http://localhost:3000
- API: http://localhost:8000/docs

---

### Production Deployment to GKE (30 minutes)

#### Prerequisites
- Google Cloud account with billing enabled
- gcloud CLI installed
- kubectl installed
- Docker Hub account

#### Step 1: Create GKE Cluster

```bash
# Login to Google Cloud
gcloud auth login

# Set project
gcloud config set project YOUR_PROJECT_ID

# Create cluster (f1-micro for free tier)
gcloud container clusters create ad-generator \
  --zone us-central1-a \
  --num-nodes 2 \
  --machine-type e2-medium \
  --disk-size 10

# Get credentials
gcloud container clusters get-credentials ad-generator
```

#### Step 2: Build and Push Docker Image

```bash
# Build image
docker build -t yourusername/ad-generator-api:v1.0 .

# Login to Docker Hub
docker login

# Push image
docker push yourusername/ad-generator-api:v1.0
```

#### Step 3: Create Kubernetes Secrets

```bash
# Create namespace
kubectl create namespace ad-generator

# Encode secrets (PowerShell)
$mlflowUri = [Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes("http://mlflow:5000"))

# Create secret from file
kubectl create secret generic api-credentials \
  --from-literal=mlflow-uri=$mlflowUri \
  --namespace=ad-generator
```

#### Step 4: Deploy Application

```bash
# Apply all manifests
kubectl apply -f k8s/ --namespace=ad-generator

# Watch deployment
kubectl rollout status deployment/ad-generator-api -n ad-generator

# Get external IP (wait 2-3 minutes)
kubectl get service ad-generator-api -n ad-generator
```

#### Step 5: Verify Deployment

```bash
# Check pods
kubectl get pods -n ad-generator

# Check logs
kubectl logs -f deployment/ad-generator-api -n ad-generator

# Test API
curl http://EXTERNAL_IP:8000/health
```

---

## 🔍 Monitoring & Alerts

### Grafana Dashboard Access

1. Port-forward Grafana (if not exposed):
```bash
kubectl port-forward -n monitoring svc/grafana 3000:3000
```

2. Login: http://localhost:3000
   - Username: `admin`
   - Password: `admin` (change on first login)

3. Navigate to "ML Ad Generator - Metrics Dashboard"

### Key Metrics to Watch

| Metric | Normal | Warning | Critical |
|--------|--------|---------|----------|
| Latency (p50) | <1s | 1-2s | >2s |
| Latency (p99) | <2s | 2-3s | >3s |
| Quality Score | >0.7 | 0.5-0.7 | <0.5 |
| Error Rate | <1% | 1-5% | >5% |
| CPU Usage | <60% | 60-80% | >80% |

### Alert Conditions

**High Latency Alert**:
```promql
histogram_quantile(0.99, rate(ad_generation_latency_bucket[5m])) > 2000
```

**Low Quality Alert**:
```promql
avg(ad_quality_score) < 0.5
```

**Service Down Alert**:
```promql
up{job="api"} == 0
```

---

## 🔧 Common Operations

### Scaling Replicas

```bash
# Manual scaling
kubectl scale deployment ad-generator-api --replicas=5 -n ad-generator

# Check HPA status
kubectl get hpa -n ad-generator

# View autoscaling events
kubectl describe hpa ad-generator-hpa -n ad-generator
```

### Updating Model Version

```bash
# 1. Train new model locally
python src/model/train.py

# 2. Register in MLflow UI
# http://localhost:5000 → Models → ad-generator → New Version

# 3. Update API to load new version (edit src/api/main.py)
# MODEL_VERSION = "2"

# 4. Build new Docker image
docker build -t yourusername/ad-generator-api:v1.1 .
docker push yourusername/ad-generator-api:v1.1

# 5. Update K8s deployment
kubectl set image deployment/ad-generator-api \
  api=yourusername/ad-generator-api:v1.1 \
  -n ad-generator

# 6. Watch rollout
kubectl rollout status deployment/ad-generator-api -n ad-generator
```

### Rolling Back Deployment

```bash
# View rollout history
kubectl rollout history deployment/ad-generator-api -n ad-generator

# Rollback to previous version
kubectl rollout undo deployment/ad-generator-api -n ad-generator

# Rollback to specific revision
kubectl rollout undo deployment/ad-generator-api --to-revision=2 -n ad-generator
```

### Restarting Services

```bash
# Restart API pods
kubectl rollout restart deployment/ad-generator-api -n ad-generator

# Restart specific service (local)
docker-compose restart mlflow

# Restart all services (local)
docker-compose restart
```

---

## 🐛 Troubleshooting

### Issue: Training Loss Not Decreasing

**Symptoms**: Loss stays high (>5.0) after many epochs

**Diagnosis**:
```bash
# Check training logs
tail -f logs/training.log

# Verify data quality
python -c "import pandas as pd; df = pd.read_csv('data/products_sample.csv'); print(df.head()); print(df.shape)"
```

**Solutions**:
1. Check learning rate (try 3e-4 to 5e-4)
2. Increase epochs (5 → 10)
3. Verify data has 1000+ examples
4. Check task prefix: "generate ad:" exists in prompts

---

### Issue: API Returns 500 Error

**Symptoms**: `/generate` endpoint fails with Internal Server Error

**Diagnosis**:
```bash
# Check API logs (local)
docker-compose logs api

# Check API logs (K8s)
kubectl logs -f deployment/ad-generator-api -n ad-generator

# Test model loading
python -c "from transformers import T5Tokenizer, T5ForConditionalGeneration; \
model = T5ForConditionalGeneration.from_pretrained('./models/ad-creative-generator'); \
print('Model loaded!')"
```

**Solutions**:
1. **Model not found**: Train model or download pretrained
2. **Out of memory**: Reduce batch size or use smaller model
3. **CUDA error**: Install CPU-only PyTorch
```bash
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

### Issue: Prometheus Shows No Data

**Symptoms**: Grafana dashboards empty

**Diagnosis**:
```bash
# Check Prometheus targets
# Visit: http://localhost:9090/targets
# All targets should show "UP"

# Test metrics endpoint
curl http://localhost:8000/metrics
```

**Solutions**:
1. **API not exposing metrics**: Add PrometheusMiddleware to FastAPI
2. **Prometheus not scraping**: Check `prometheus.yml` scrape config
3. **Wrong port**: Verify API runs on port 8000

---

### Issue: Kubernetes Pods CrashLooping

**Symptoms**: Pods repeatedly restart

**Diagnosis**:
```bash
# Get pod status
kubectl get pods -n ad-generator

# View pod events
kubectl describe pod POD_NAME -n ad-generator

# Check logs
kubectl logs POD_NAME -n ad-generator --previous
```

**Solutions**:
| Error | Fix |
|-------|-----|
| ImagePullBackOff | Check image name/tag in deployment.yaml |
| CrashLoopBackOff | Check logs for Python errors |
| OOMKilled | Increase memory limits in deployment.yaml |
| InvalidImageName | Push image to Docker Hub first |

---

### Issue: Airflow DAG Not Running

**Symptoms**: DAG shows no runs in Airflow UI

**Diagnosis**:
1. Check Airflow UI: http://localhost:8080
2. Verify DAG is enabled (toggle switch)
3. Check scheduler logs:
```bash
docker-compose logs airflow-scheduler
```

**Solutions**:
1. **DAG not found**: Check file in `airflow/dags/`
2. **Import errors**: Fix Python syntax in DAG file
3. **Scheduler not running**: `docker-compose restart airflow-scheduler`

---

###  Issue: MLflow UI Not Accessible

**Symptoms**: http://localhost:5000 doesn't load

**Diagnosis**:
```bash
# Check MLflow container
docker-compose ps mlflow

# Check logs
docker-compose logs mlflow

# Test port
curl http://localhost:5000
```

**Solutions**:
1. **Container not running**: `docker-compose up -d mlflow`
2. **Port conflict**: Change port in `docker-compose.yaml`
3. **Database connection failed**: Check PostgreSQL is running

---

## 📊 Performance Tuning

### API Optimization

**Increase Throughput**:
```python
# src/api/main.py
# Add to uvicorn startup
workers = 4  # CPU cores
```

**Reduce Latency**:
```python
# Use smaller batch size for beam search
outputs = model.generate(**inputs, num_beams=3)  # Down from 4
```

**Enable Caching**:
```python
# Cache model in memory
@lru_cache()
def load_model():
    return T5ForConditionalGeneration.from_pretrained(...)
```

### Database Optimization

```sql
-- Add indexes for common queries
CREATE INDEX idx_products_category ON products(category);
CREATE INDEX idx_ads_product_id ON generated_ads(product_id);
CREATE INDEX idx_ads_generated_at ON generated_ads(generated_at DESC);
```

### Kubernetes Resource Tuning

```yaml
# k8s/deployment.yaml
resources:
  requests:
    memory: "2Gi"   # Minimum needed
    cpu: "500m"     # 0.5 cores
  limits:
    memory: "4Gi"   # Maximum allowed
    cpu: "2000m"    # 2 cores
```

---

## 🔐 Security Checklist

- [ ] Change default Grafana password
- [ ] Rotate API keys monthly
- [ ] Use TLS for external traffic
- [ ] Enable network policies in K8s
- [ ] Scan Docker images for vulnerabilities
- [ ] Set up authentication for Airflow UI
- [ ] Encrypt secrets at rest
- [ ] Limit egress traffic
- [ ] Enable audit logging
- [ ] Review IAM permissions

---

## 💾 Backup & Recovery

### Backup MLflow Experiments

```bash
# Export runs to CSV
mlflow experiments csv -x EXPERIMENT_ID -o backup.csv

# Backup artifact store
aws s3 sync s3://mlflow-artifacts ./backup/artifacts/
```

### Backup Database

```bash
# PostgreSQL backup
docker exec postgres pg_dump -U mlops adgen > backup_$(date +%Y%m%d).sql

# Restore
docker exec -i postgres psql -U mlops adgen < backup_20250112.sql
```

### Disaster Recovery

1. **Data Loss**: Restore from latest PostgreSQL backup
2. **Model Loss**: Re-download from MLflow artifact store
3. **Cluster Failure**: Redeploy from k8s manifests in Git
4. **Complete Failure**: Follow deployment procedure from scratch

---

## 📞 Escalation Procedures

### Severity Levels

**P0 - Critical** (respond in 15 min):
- Production API down for >5 minutes
- Data loss incident
- Security breach

**P1 - High** (respond in 1 hour):
- Latency >5s sustained
- Error rate >10%
- Autoscaling not working

**P2 - Medium** (respond in 4 hours):
- Dashboard not loading
- Airflow DAG failed
- Model quality degraded

**P3 - Low** (respond in 1 day):
- Documentation updates
- Feature requests
- Performance optimization

### Contact Information

- **On-Call Engineer**: oncall@company.com
- **ML Team**: ml-team@company.com
- **Infrastructure**: infra@company.com
- **Slack Channel**: #ml-ops-alerts

---

## 📋 Maintenance Schedule

### Daily
- [ ] Check Grafana dashboards for anomalies
- [ ] Review failed Airflow DAG runs
- [ ] Monitor disk usage

### Weekly
- [ ] Review model performance metrics
- [ ] Check for security updates
- [ ] Analyze error logs

### Monthly
- [ ] Rotate credentials
- [ ] Update dependencies
- [ ] Review and optimize costs
- [ ] Backup verification test

---

## 🎓 Useful Commands

### Docker

```bash
# View all containers
docker-compose ps

# View logs
docker-compose logs -f SERVICE_NAME

# Restart service
docker-compose restart SERVICE_NAME

# Stop all
docker-compose down

# Remove volumes
docker-compose down -v
```

### Kubernetes

```bash
# Get all resources
kubectl get all -n ad-generator

# Describe resource
kubectl describe pod/deployment/service NAME -n ad-generator

# Execute command in pod
kubectl exec -it POD_NAME -n ad-generator -- /bin/bash

# Port forward
kubectl port-forward svc/SERVICE_NAME LOCAL_PORT:REMOTE_PORT -n ad-generator

# View resource usage
kubectl top pods -n ad-generator
```

### MLflow

```bash
# Start UI
mlflow ui --host 0.0.0.0 --port 5000

# List experiments
mlflow experiments list

# View run
mlflow runs describe RUN_ID
```

---

**For architecture details, see [ARCHITECTURE.md](ARCHITECTURE.md)**
