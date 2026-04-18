# 🏗️ System Architecture

This document describes the technical architecture of the E-Commerce Ad Creative Generator MLOps system.

---

## 📊 System Overview

The system is built around three core pillars:

1. **ML Pipeline**: Fine-tuned language model for ad generation
2. **MLOps Automation**: Airflow orchestration with MLflow tracking
3. **Production Infrastructure**: Kubernetes deployment with monitoring

---

## 🔄 Data Flow Architecture

```mermaid
graph LR
    A[Product Data<br/>CSV/API] --> B[Data Ingestion<br/>Airflow DAG]
    B --> C[PostgreSQL<br/>Product DB]
    C --> D[Training Pipeline<br/>Airflow DAG]
    D --> E[Flan-T5 Model<br/>Fine-tuning]
    E --> F[MLflow<br/>Tracking]
    F --> G[Model Registry<br/>Versioning]
    G --> H[Production API<br/>FastAPI]
    C --> I[Batch Inference<br/>Airflow DAG]
    I --> H
    H --> J[Generated Ads<br/>Database/CSV]
    
    style A fill:#e1f5ff
    style C fill:#fff3cd
    style F fill:#d4edda
    style H fill:#f8d7da
    style J fill:#e1f5ff
```

**Flow Description**:
1. **Ingestion**: Product data → PostgreSQL (daily)
2. **Training**: Data → Model training → MLflow → Registry (weekly)
3. **Inference**: Products → API → Generated ads (real-time/batch)

---

## 🤖 MLOps Pipeline Architecture

```mermaid
graph TB
    subgraph "Development"
        A[Data Scientists] --> B[Jupyter Notebooks]
        B --> C[Experiment Tracking<br/>MLflow]
    end
    
    subgraph "Automation"
        D[Airflow Scheduler] --> E[Data Ingestion DAG]
        D --> F[Model Training DAG]
        D --> G[Batch Inference DAG]
    end
    
    subgraph "Training"
        E --> H[Product Database]
        H --> F
        F --> I[Train Flan-T5]
        I --> J[Log to MLflow]
        J --> K{Better than<br/>Production?}
        K -->|Yes| L[Promote to Prod]
        K -->|No| M[Keep Current]
    end
    
    subgraph "Deployment"
        L --> N[Container Registry<br/>Docker Hub]
        N --> O[GitHub Actions<br/>CI/CD]
        O --> P[Kubernetes<br/>GKE Cluster]
    end
    
    subgraph "Monitoring"
        P --> Q[Prometheus<br/>Metrics]
        Q --> R[Grafana<br/>Dashboards]
        R --> S[Alerts]
    end
    
    style C fill:#d4edda
    style J fill:#d4edda
    style P fill:#f8d7da
    style R fill:#cfe2ff
```

**Pipeline Stages**:
1. **Development**: Experiment → Track → Compare
2. **Automation**: Scheduled DAGs trigger workflows
3. **Training**: Auto-retrain → Compare → Promote
4. **Deployment**: Build → Test → Deploy to K8s
5. **Monitoring**: Collect metrics → Visualize → Alert

---

## 🏛️ System Component Architecture

```mermaid
graph TB
    subgraph "Client Layer"
        A[Web Frontend]
        B[API Clients]
        C[Airflow UI]
    end
    
    subgraph "API Gateway"
        D[Ingress/LoadBalancer]
    end
    
    subgraph "Application Layer - Kubernetes Cluster"
        E[Ad Generator API<br/>FastAPI Pods<br/>Replicas: 2-5]
        F[Prometheus<br/>Metrics Collector]
        G[Grafana<br/>Visualization]
    end
    
    subgraph "Orchestration Layer"
        H[Airflow Webserver]
        I[Airflow Scheduler]
        J[Airflow Workers]
    end
    
    subgraph "ML Platform"
        K[MLflow Tracking<br/>Experiments]
        L[MLflow Registry<br/>Model Versions]
        M[MinIO<br/>Artifact Storage]
    end
    
    subgraph "Data Layer"
        N[PostgreSQL<br/>Products + Ads]
        O[Training Data<br/>CSV Files]
    end
    
    subgraph "Model"
        P[Flan-T5-Small<br/>60M Parameters]
        Q[Tokenizer]
    end
    
    A --> D
    B --> D
    C --> H
    D --> E
    E --> F
    F --> G
    E --> P
    P --> Q
    H --> I
    I --> J
    J --> K
    K --> L
    L --> M
    E --> N
    J --> N
    J --> O
    O --> P
    
    style E fill:#f8d7da
    style H fill:#fff3cd
    style K fill:#d4edda
    style N fill:#cfe2ff
    style P fill:#e7d4f5
```

**Component Descriptions**:

### Client Layer
- **Web Frontend**: React/HTML interface for ad generation
- **API Clients**: External services consuming API
- **Airflow UI**: DAG monitoring and triggering

### API Gateway
- **Ingress/LoadBalancer**: Routes traffic to API pods
- **TLS Termination**: HTTPS support
- **Rate Limiting**: Prevent abuse

### Application Layer (Kubernetes)
- **Ad Generator API**: FastAPI service (2-5 replicas)
  - Health checks: /health, /ready
  - Metrics export: /metrics
  - Auto-scaling via HPA
- **Prometheus**: Scrapes metrics every 15s
- **Grafana**: Real-time dashboards

### Orchestration Layer (Airflow)
- **Webserver**: UI for DAG management
- **Scheduler**: Triggers DAGs on schedule
- **Workers**: Execute tasks (data ingestion, training)

### ML Platform
- **MLflow Tracking**: Log experiments, metrics, params
- **MLflow Registry**: Version control for models
- **MinIO**: S3-compatible artifact storage

### Data Layer
- **PostgreSQL**: Relational database
  - Tables: products, generated_ads, model_metadata
- **Training Data**: CSV files with product→ad examples

### Model
- **Flan-T5-Small**: 60M parameter transformer
- **Tokenizer**: Text preprocessing

---

## 🔐 Security Architecture

```mermaid
graph TB
    A[External Traffic] --> B[Cloud Load Balancer<br/>TLS]
    B --> C[Ingress Controller]
    C --> D{Authentication}
    D -->|Valid| E[API Service]
    D -->|Invalid| F[401 Unauthorized]
    E --> G[Kubernetes Secrets]
    G --> H[MLflow Credentials]
    G --> I[Database Passwords]
    G --> J[MinIO Keys]
    
    style B fill:#d4edda
    style D fill:#fff3cd
    style G fill:#f8d7da
```

**Security Measures**:
- ✅ TLS/HTTPS for all external traffic
- ✅ Kubernetes secrets for sensitive data (base64 encoded)
- ✅ Network policies to isolate services
- ✅ No credentials in code or Docker images
- ✅ Rate limiting on API endpoints
- ✅ CORS configured for web frontend

---

## 📈 Scaling Architecture

```mermaid
graph LR
    A[Traffic Increase] --> B[Metrics Collection<br/>Prometheus]
    B --> C[HPA Controller]
    C --> D{CPU > 70%<br/>or<br/>Memory > 80%?}
    D -->|Yes| E[Scale Up<br/>Add Pods]
    D -->|No| F[Current State]
    E --> G[Load Balancer<br/>Distributes]
    F --> G
    G --> H[API Pods<br/>2-5 replicas]
    
    style C fill:#fff3cd
    style E fill:#d4edda
    style H fill:#f8d7da
```

**Scaling Configuration**:
- **Min Replicas**: 2 (high availability)
- **Max Replicas**: 5 (cost control)
- **Scale Up Trigger**: CPU > 70% OR Memory > 80%
- **Scale Down**: After 5 min below threshold
- **Target**: 60% CPU utilization

---

## 💾 Data Architecture

### Database Schema

```mermaid
erDiagram
    PRODUCTS ||--o{ GENERATED_ADS : has
    PRODUCTS {
        int id PK
        string name UK
        string category
        text description
        timestamp created_at
        timestamp updated_at
    }
    GENERATED_ADS {
        int id PK
        int product_id FK
        text ad_text
        string model_version
        float confidence
        timestamp generated_at
    }
    MODEL_METADATA {
        int id PK
        string version
        float training_loss
        float eval_loss
        string mlflow_run_id
        timestamp trained_at
        boolean is_production
    }
```

### Data Flow

1. **Ingestion**: CSV → Validation → PostgreSQL
2. **Training**: PostgreSQL → pandas DataFrame → Model
3. **Inference**: Product data → Model → Generated ad → PostgreSQL
4. **Export**: PostgreSQL → CSV/JSON for analysis

---

## 🔄 CI/CD Architecture

```mermaid
graph LR
    A[Git Push<br/>main branch] --> B[GitHub Actions<br/>Triggered]
    B --> C[Run Tests<br/>pytest]
    C --> D{Tests Pass?}
    D -->|No| E[Notify Developer]
    D -->|Yes| F[Build Docker Image]
    F --> G[Push to Docker Hub<br/>tagged: latest + SHA]
    G --> H[Update K8s Manifest<br/>new image tag]
    H --> I[kubectl apply<br/>Deploy to GKE]
    I --> J[Rolling Update<br/>Zero Downtime]
    J --> K[Health Checks]
    K --> L{Healthy?}
    L -->|Yes| M[Complete Rollout]
    L -->|No| N[Auto Rollback]
    
    style C fill:#cfe2ff
    style F fill:#fff3cd
    style I fill:#d4edda
    style M fill:#d4edda
    style N fill:#f8d7da
```

**CI/CD Stages**:
1. **Test**: Lint (flake8) + Unit tests (pytest)
2. **Build**: Docker image with  Python 3.9
3. **Push**: Docker Hub with version tags
4. **Deploy**: Kubernetes rolling update
5. **Verify**: Health checks + rollback if failed

---

## 📊 Monitoring Architecture

### Metrics Collection

```mermaid
graph TB
    A[API Pods] --> B[Prometheus Metrics<br/>/metrics endpoint]
    B --> C[Prometheus Server<br/>Scrape every 15s]
    C --> D[Time Series DB]
    D --> E[Grafana]
    E --> F[Dashboards]
    C --> G[Alert Manager]
    G --> H[Notifications<br/>Email/Slack]
    
    style B fill:#cfe2ff
    style C fill:#d4edda
    style E fill:#fff3cd
    style G fill:#f8d7da
```

### Custom Metrics Exported

| Metric | Type | Description |
|--------|------|-------------|
| `ad_generation_requests_total` | Counter | Total API requests |
| `ad_generation_latency` | Histogram | Request latency (ms) |
| `ad_quality_score` | Gauge | Quality score (0-1) |
| `model_version_info` | Gauge | Current model version |
| `http_requests_total` | Counter | HTTP requests by status |

### Alert Rules

- **High Latency**: p99 > 2000ms for 5 min
- **Low Quality**: avg quality < 0.5 for 10 min
- **Service Down**: no requests for 2 min
- **High Errors**: 5xx rate > 5% for 5 min

---

## 🎯 Deployment Topology

### Local Development
```
Docker Desktop
├── MLflow (port 5000)
├── PostgreSQL (port 5432)
├── MinIO (port 9000)
├── Prometheus (port 9090)
└── Grafana (port 3000)

Local Python
└── API (port 8000)
```

### Production (GKE)
```
Google Kubernetes Engine
├── Namespace: ad-generator
│   ├── Deployment: ad-generator-api (2-5 pods)
│   ├── Service: LoadBalancer (external IP)
│   ├── HPA: autoscaling enabled
│   ├── ConfigMap: app-config
│   └── Secret: api-credentials
│
├── Namespace: monitoring
│   ├── Prometheus (scraping)
│   └── Grafana (dashboards)
│
└── Namespace: airflow
    ├── Airflow Webserver
    ├── Airflow Scheduler
    └── Airflow Workers
```

---

## 🚀 Technology Stack

### Core Technologies
- **Language**: Python 3.9+
- **ML Framework**: Hugging Face Transformers
- **Model**: Flan-T5-Small (60M params)
- **API Framework**: FastAPI
- **Database**: PostgreSQL

### MLOps Tools
- **Experiment Tracking**: MLflow
- **Orchestration**: Apache Airflow
- **Monitoring**: Prometheus + Grafana
- **Object Storage**: MinIO (S3-compatible)

### Infrastructure
- **Containerization**: Docker
- **Orchestration**: Kubernetes (GKE)
- **CI/CD**: GitHub Actions
- **Cloud Provider**: Google Cloud Platform

### Development
- **Version Control**: Git + GitHub
- **Testing**: pytest
- **Linting**: flake8, black
- **Documentation**: Markdown, Mermaid

---

## 📦 Container Architecture

### API Container (Dockerfile)
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ ./src/
COPY models/ ./models/
EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0"]
```

**Image Size**: ~1.2GB  
**Layers**: Python base + dependencies + code  
**Health Check**: GET /health every 30s

---

## 🔧 Configuration Management

### Environment Variables
- `.env`: Local development config
- `k8s/configmap.yaml`: Non-sensitive K8s config
- `k8s/secret.yaml`: Sensitive credentials (base64)

### Feature Flags
- Model version selection
- Batch size configuration
- Quality threshold tuning

---

## 📝 API Architecture

### REST Endpoints

```
POST /generate
  - Generate single ad
  - Input: {product_name, category, description}
  - Output: {ad_text, quality_score, model_version}

GET /health
  - Health check
  - Returns: {status: "healthy"}

GET /metrics
  - Prometheus metrics
  - Format: OpenMetrics

GET /docs
  - FastAPI documentation
  - Interactive Swagger UI
```

### Request Flow
1. Client → LoadBalancer
2. LoadBalancer → API Pod (round-robin)
3. API → Load model from cache
4. Model → Generate ad with beam search
5. API → Calculate quality score
6. API → Log metrics to Prometheus
7. Response → Client

---

## 🎓 Design Decisions

### Why Flan-T5-Small?
- ✅ Instruction-tuned (understands "generate ad")
- ✅ Small enough for CPU inference (~1s latency)
- ✅ Good quality with 1000 examples
- ✅ Fits in 2GB RAM

### Why FastAPI?
- ✅ Fast async I/O
- ✅ Auto-generated OpenAPI docs
- ✅ Type safety with Pydantic
- ✅ Built-in metrics support

### Why Kubernetes?
- ✅ Auto-scaling based on load
- ✅ Self-healing (pod restarts)
- ✅ Zero-downtime rolling updates
- ✅ Industry standard for ML deployment

### Why Airflow?
- ✅ Complex DAG dependencies
- ✅ Retry logic built-in
- ✅ Rich UI for monitoring
- ✅ Cron-like scheduling

---

## 🔮 Future Enhancements

1. **A/B Testing**: Canary deployments with traffic splitting
2. **Model Drift Detection**: Statistical tests on input distribution
3. **Multi-language Support**: Generate ads in Spanish, French
4. **Image Generation**: Add product images to ads
5. **Feedback Loop**: User ratings → retrain on best ads

---

**For operations guide, see [RUNBOOK.md](RUNBOOK.md)**
