# Medical Claims Data Engineering Platform

[![CI/CD Pipeline](https://github.com/username/claims-data-engineering/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/username/claims-data-engineering/actions)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![AWS](https://img.shields.io/badge/AWS-Integrated-orange.svg)](https://aws.amazon.com/)
[![DBT](https://img.shields.io/badge/dbt-1.7-green.svg)](https://www.getdbt.com/)

Production-grade ETL/ELT data engineering platform for processing 17M+ medical claims records with AWS cloud infrastructure, automated data quality monitoring, and real-time alerting.

---

## Table of Contents

- [Executive Summary](#executive-summary)
- [Architecture](#architecture)
- [Features](#features)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [ETL Pipeline](#etl-pipeline)
- [Data Quality Framework](#data-quality-framework)
- [AWS Integration](#aws-integration)
- [DBT Models](#dbt-models)
- [Monitoring & Alerting](#monitoring--alerting)
- [CI/CD Pipeline](#cicd-pipeline)
- [Configuration](#configuration)
- [Testing](#testing)
- [Docker Deployment](#docker-deployment)
- [API Reference](#api-reference)
- [Contributing](#contributing)

---

## Executive Summary

### Problem Statement

Healthcare organizations need reliable, scalable data pipelines to process millions of medical claims for analytics, cost estimation, and regulatory compliance. Manual ETL processes are error-prone, lack observability, and cannot scale.

### Solution

This platform provides:

- **Scalable ETL/ELT Pipeline**: Process 17M+ rows (3.7GB) with chunked ingestion and parallel processing
- **Medallion Architecture**: Bronze/Silver/Gold data layers for progressive data refinement
- **Automated Data Quality**: Great Expectations-style validation with 99%+ quality thresholds
- **Real-time Monitoring**: CloudWatch metrics, SNS alerts, and visual dashboards
- **Cloud-Native**: Full AWS integration (S3, RDS, Lambda, Glue, SNS, CloudWatch)
- **CI/CD Ready**: GitHub Actions workflows with automated testing and deployment

### Key Metrics

| Metric | Value |
|--------|-------|
| Raw Data Volume | 16.98M rows, 3.73 GB, 63 columns |
| Unique Claims | ~6.5M medical claims |
| Processing Throughput | 100K rows/chunk |
| Data Quality Score | 99%+ validation pass rate |
| Pipeline Reliability | 99.9% uptime with retry logic |

---

## Architecture

### System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        DATA ENGINEERING PLATFORM                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌───────────┐ │
│  │   SOURCES    │    │   EXTRACT    │    │  TRANSFORM   │    │   LOAD    │ │
│  │              │    │              │    │              │    │           │ │
│  │ • S3 Files   │───▶│ • Chunked    │───▶│ • Clean      │───▶│ • S3      │ │
│  │ • RDS        │    │   Reading    │    │ • Enrich     │    │ • RDS     │ │
│  │ • APIs       │    │ • Validation │    │ • Transform  │    │ • Redshift│ │
│  └──────────────┘    └──────────────┘    └──────────────┘    └───────────┘ │
│         │                   │                   │                  │        │
│         └───────────────────┼───────────────────┼──────────────────┘        │
│                             │                   │                           │
│                             ▼                   ▼                           │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                      DATA QUALITY LAYER                                │ │
│  │  • Expectation Suites  • Validators  • Profilers  • Lineage Tracking  │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                             │                                               │
│                             ▼                                               │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                      MONITORING & ALERTING                             │ │
│  │  • CloudWatch Metrics  • SNS Alerts  • Dashboards  • Logging          │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow (Medallion Architecture)

```
┌─────────────┐      ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│    RAW      │      │   BRONZE    │      │   SILVER    │      │    GOLD     │
│             │      │             │      │             │      │             │
│ • Source    │─────▶│ • Ingested  │─────▶│ • Cleaned   │─────▶│ • Aggregated│
│   files     │      │ • Validated │      │ • Enriched  │      │ • Star      │
│ • APIs      │      │ • Parquet   │      │ • Quality   │      │   Schema    │
│             │      │             │      │   Checked   │      │ • Analytics │
└─────────────┘      └─────────────┘      └─────────────┘      └─────────────┘
     S3 raw/             S3 bronze/          S3 silver/          S3 gold/
                                                                  RDS/Redshift
```

### AWS Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              AWS CLOUD                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐              │
│   │   S3    │     │  GLUE   │     │ LAMBDA  │     │   RDS   │              │
│   │         │     │         │     │         │     │         │              │
│   │ Data    │────▶│ Catalog │     │ ETL     │────▶│ Postgres│              │
│   │ Lake    │     │ Crawler │     │ Trigger │     │ DB      │              │
│   └─────────┘     └─────────┘     └─────────┘     └─────────┘              │
│        │               │               │               │                    │
│        │               │               │               │                    │
│        ▼               ▼               ▼               ▼                    │
│   ┌─────────────────────────────────────────────────────────┐              │
│   │                     CLOUDWATCH                           │              │
│   │  • Logs  • Metrics  • Alarms  • Dashboards              │              │
│   └─────────────────────────────────────────────────────────┘              │
│                              │                                              │
│                              ▼                                              │
│                       ┌─────────┐                                           │
│                       │   SNS   │──────▶ Email/Slack Alerts                │
│                       └─────────┘                                           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Features

### Core Features

| Feature | Description |
|---------|-------------|
| **Chunked Data Ingestion** | Process large files (100K rows/chunk) without memory overflow |
| **Data Quality Checks** | Great Expectations-style validation with customizable suites |
| **Medallion Architecture** | Bronze → Silver → Gold progressive data refinement |
| **Star Schema Modeling** | Dimensional modeling for analytics (fact & dimension tables) |
| **DBT Transformations** | SQL-based transformations with testing and documentation |
| **Real-time Monitoring** | CloudWatch metrics and alarms for pipeline health |
| **SNS Alerting** | Automated notifications for failures and quality issues |
| **Data Lineage** | Track data flow and transformations |
| **Data Catalog** | Asset discovery and governance metadata |
| **CI/CD Pipeline** | Automated testing and deployment via GitHub Actions |
| **Docker Support** | Containerized deployment for consistency |

### AWS Services Used

- **Amazon S3**: Data lake storage (raw, bronze, silver, gold layers)
- **Amazon RDS**: PostgreSQL data warehouse
- **AWS Lambda**: Serverless ETL triggers and processing
- **AWS Glue**: Data catalog and ETL job management
- **Amazon SNS**: Pipeline alerting and notifications
- **Amazon CloudWatch**: Monitoring, logging, and dashboards
- **Amazon Redshift**: (Optional) Scalable analytics warehouse

---

## Quick Start

### Prerequisites

- Python 3.10+
- Docker & Docker Compose
- AWS CLI configured (optional for local development)

### Installation

```bash
# Clone the repository
git clone https://github.com/username/claims-data-engineering.git
cd claims-data-engineering

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v
```

### Running with Docker

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f pipeline

# Run ETL pipeline
docker-compose run pipeline python -m src.etl.run_pipeline

# Access Jupyter notebooks
# Navigate to http://localhost:8888
```

### Running ETL Pipeline

```python
from src.etl.pipeline import ETLPipeline, ETLPipelineBuilder
from src.etl.extract import DataExtractor
from src.etl.transform import DataTransformer
from src.etl.load import DataLoader
from src.data_quality.expectations import DataQualityChecker

# Load configuration
import yaml
with open('config/settings.yaml') as f:
    config = yaml.safe_load(f)

# Build pipeline
pipeline = (
    ETLPipelineBuilder("claims_pipeline", config)
    .add_extract(DataExtractor(name="extract"))
    .add_transform(DataTransformer(name="transform"))
    .add_quality_check(DataQualityChecker(name="quality"))
    .add_load(DataLoader(name="load"))
    .build()
)

# Execute
result = pipeline.run()
print(f"Status: {result.status.value}")
print(f"Rows processed: {result.total_rows_processed:,}")
```

---

## Project Structure

```
claims-data-engineering/
├── .github/
│   └── workflows/
│       └── ci-cd.yml              # GitHub Actions CI/CD
├── config/
│   └── settings.yaml              # Pipeline configuration
├── data/
│   ├── raw/                       # Raw source files
│   ├── bronze/                    # Ingested data (Parquet)
│   ├── silver/                    # Cleaned & transformed
│   ├── gold/                      # Analytics-ready
│   ├── catalog/                   # Data catalog metadata
│   └── lineage/                   # Lineage tracking
├── dbt/
│   ├── models/
│   │   ├── staging/               # Staging models
│   │   ├── intermediate/          # Intermediate transforms
│   │   └── marts/                 # Fact & dimension tables
│   ├── tests/
│   └── dbt_project.yml
├── docker/
├── logs/
│   ├── alerts/
│   └── metrics/
├── notebooks/
│   ├── 01_data_ingestion.ipynb
│   ├── 02_exploratory_data_analysis.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_training.ipynb
│   └── 05_model_evaluation.ipynb
├── reports/
│   ├── dashboards/
│   ├── figures/
│   ├── profiles/
│   └── quality/
├── src/
│   ├── aws/
│   │   ├── lambda/
│   │   │   └── etl_handler.py     # Lambda ETL handler
│   │   ├── s3_handler.py
│   │   ├── glue_handler.py
│   │   └── redshift_handler.py
│   ├── catalog/
│   │   ├── data_catalog.py        # Asset management
│   │   └── lineage.py             # Lineage tracking
│   ├── data_quality/
│   │   ├── expectations.py        # Quality checks
│   │   ├── validators.py          # Schema validation
│   │   └── profiler.py            # Data profiling
│   ├── etl/
│   │   ├── pipeline.py            # Pipeline orchestrator
│   │   ├── extract.py             # Extraction stages
│   │   ├── transform.py           # Transformation stages
│   │   └── load.py                # Loading stages
│   ├── monitoring/
│   │   ├── metrics.py             # CloudWatch metrics
│   │   ├── alerting.py            # SNS alerting
│   │   └── dashboard.py           # Dashboard generation
│   └── utils/
│       ├── logger.py
│       └── helpers.py
├── tests/
│   ├── test_etl_pipeline.py
│   ├── test_data_quality.py
│   └── integration/
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## ETL Pipeline

### Pipeline Architecture

The ETL pipeline follows a modular, stage-based architecture:

```python
# Pipeline stages
class PipelineStage:
    """Base stage with retry logic and metrics."""
    def execute(self, context: Dict) -> StageResult
    def _run(self, context: Dict) -> Dict  # Override in subclasses

# Available stages
- DataExtractor       # Extract from files/S3/databases
- DataTransformer     # Clean, enrich, transform
- DataCleaner         # Remove duplicates, invalid data
- DataAggregator      # Create aggregations
- DataQualityChecker  # Run quality expectations
- DataLoader          # Load to S3/RDS/Redshift
- DimensionBuilder    # Build dimension tables
- FactTableLoader     # Build fact tables
```

### Pipeline Configuration

```yaml
# config/settings.yaml
etl:
  chunk_size: 100000
  sample_size: 1000000
  parallel_workers: 4
  retry_attempts: 3
  retry_delay_seconds: 30
  
  stages:
    extract:
      enabled: true
      validate_schema: true
    transform:
      enabled: true
      apply_quality_checks: true
    load:
      enabled: true
      target: "data_warehouse"
```

### Running the Pipeline

```bash
# Via Python
python -m src.etl.run_pipeline

# Via Docker
docker-compose run pipeline

# Via AWS Lambda (triggered by S3 event or schedule)
aws lambda invoke --function-name claims-etl-trigger response.json
```

---

## Data Quality Framework

### Expectation Suites

```python
from src.data_quality.expectations import ExpectationSuite

# Create custom suite
suite = ExpectationSuite("claims_validation")

# Add expectations
suite.expect_table_row_count_to_be_between(1000, 50000000)
suite.expect_column_to_exist("claim_id_key")
suite.expect_column_values_to_not_be_null("claim_id_key")
suite.expect_column_values_to_be_unique("claim_id_key")
suite.expect_column_values_to_be_positive("amt_paid", mostly=0.99)
suite.expect_column_values_to_be_between("age", 0, 120)
suite.expect_column_values_to_be_in_set("gender", ['M', 'F'])

# Save for reuse
suite.save("config/expectations/claims_suite.json")
```

### Quality Thresholds

| Check Type | Threshold | Action on Failure |
|------------|-----------|-------------------|
| Critical | 99% | Halt pipeline |
| Warning | 95% | Send alert, continue |
| Info | 90% | Log only |

### Data Profiling

```python
from src.data_quality.profiler import DataProfiler

profiler = DataProfiler()
profile = profiler.profile(df, "claims_data")

# Generate HTML report
profiler.save_profile(profile, format="html")

# Compare profiles for drift detection
drift = profiler.compare_profiles(baseline, current)
```

---

## AWS Integration

### S3 Data Lake

```python
from src.aws.s3_handler import S3Handler

s3 = S3Handler(bucket="medical-claims-data-lake")

# Upload data to appropriate layer
s3.upload_dataframe(df, "silver/claims/2024/01/claims.parquet")

# Download for processing
df = s3.download_dataframe("bronze/claims/raw_claims.parquet")
```

### Lambda ETL Trigger

```python
# src/aws/lambda/etl_handler.py

def handler(event, context):
    """
    Supports multiple triggers:
    - S3 file arrival
    - CloudWatch scheduled event
    - API Gateway request
    - Step Functions state machine
    """
    trigger_type = identify_trigger(event)
    
    if trigger_type == "s3":
        return handle_s3_trigger(event)
    elif trigger_type == "schedule":
        return handle_schedule_trigger(event)
```

### CloudWatch Monitoring

```python
from src.monitoring.metrics import CloudWatchMetrics

cw = CloudWatchMetrics(namespace="ClaimsPipeline")

# Publish metrics
cw.put_metric("RowsProcessed", 1000000, "Count")
cw.put_metric("PipelineDuration", 120.5, "Seconds")

# Create alarms
cw.create_alarm(
    alarm_name="HighErrorRate",
    metric_name="ErrorCount",
    threshold=10,
    sns_topic_arn="arn:aws:sns:us-east-1:123456789:alerts"
)
```

### SNS Alerting

```python
from src.monitoring.alerting import AlertManager

alerts = AlertManager(aws_config)

# Send pipeline alerts
alerts.send_pipeline_failure_alert(
    pipeline_name="claims_etl",
    stage_name="transform",
    error_message="Data quality check failed"
)

alerts.send_data_quality_alert(
    table_name="claims",
    quality_score=0.85,
    failed_checks=["null_check", "range_check"]
)
```

---

## DBT Models

### Model Layers

| Layer | Materialization | Description |
|-------|-----------------|-------------|
| Staging | View | Clean raw data |
| Intermediate | Ephemeral | Business logic |
| Marts | Table | Analytics-ready |

### Running DBT

```bash
cd dbt

# Install dependencies
dbt deps

# Run models
dbt run

# Test models
dbt test

# Generate documentation
dbt docs generate
dbt docs serve
```

### Sample Models

```sql
-- models/marts/facts/fact_claims.sql
{{
    config(
        materialized='incremental',
        unique_key='claim_fact_key'
    )
}}

SELECT
    {{ dbt_utils.generate_surrogate_key(['claim_id_key', 'service_date']) }} as claim_fact_key,
    claim_id_key,
    patient_key,
    diagnosis_key,
    amt_billed,
    amt_paid,
    payment_ratio
FROM {{ ref('int_claims_enriched') }}
{% if is_incremental() %}
WHERE _loaded_at > (SELECT MAX(_loaded_at) FROM {{ this }})
{% endif %}
```

---

## Monitoring & Alerting

### Metrics Collected

| Metric | Type | Description |
|--------|------|-------------|
| RowsProcessed | Counter | Total rows processed |
| PipelineDuration | Histogram | Pipeline execution time |
| StageSuccess | Counter | Successful stage executions |
| StageFailure | Counter | Failed stage executions |
| DataQualityScore | Gauge | Quality check pass rate |
| ErrorCount | Counter | Total errors |

### Alert Types

| Alert | Severity | Trigger |
|-------|----------|---------|
| Pipeline Failure | Critical | Stage fails after retries |
| Data Quality | Warning/Error | Quality score < threshold |
| Long Running Job | Warning | Duration > 1 hour |
| High Error Rate | Error | > 10 errors in 5 minutes |

---

## CI/CD Pipeline

### GitHub Actions Workflow

```yaml
# .github/workflows/ci-cd.yml

Jobs:
1. lint          # Code quality (Black, Flake8, MyPy)
2. test          # Unit tests with coverage
3. dbt-test      # DBT model tests
4. build         # Docker image build
5. deploy-staging    # Deploy to staging
6. integration-test  # Run integration tests
7. deploy-production # Deploy to production (manual approval)
```

### Deployment Flow

```
Push to main → Lint → Test → Build → Stage → Integration → Production
                                         ↓
                                 Manual Approval Required
```

---

## Configuration

### Environment Variables

```bash
# AWS Configuration
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret

# Database
DATABASE_URL=postgresql://user:pass@host:5432/claims_warehouse

# Pipeline
ENVIRONMENT=production
CONFIG_PATH=config/settings.yaml

# Alerting
SNS_TOPIC_ARN=arn:aws:sns:us-east-1:123456789:claims-alerts
```

### Settings File

See `config/settings.yaml` for complete configuration options.

---

## Testing

### Running Tests

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html

# Specific module
pytest tests/test_etl_pipeline.py -v

# Integration tests
pytest tests/integration/ -v
```

### Test Coverage

| Module | Coverage |
|--------|----------|
| src/etl | 90%+ |
| src/data_quality | 85%+ |
| src/monitoring | 80%+ |

---

## Docker Deployment

### Building Images

```bash
# Production image
docker build -t claims-pipeline:latest .

# Development image
docker build --target development -t claims-pipeline:dev .

# Lambda image
docker build --target lambda -t claims-pipeline:lambda .
```

### Running Services

```bash
# Full stack
docker-compose up -d

# Individual services
docker-compose up -d postgres localstack
docker-compose run pipeline

# View logs
docker-compose logs -f
```

---

## API Reference

### Lambda Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/run` | POST | Trigger pipeline execution |
| `/status/{id}` | GET | Get pipeline status |
| `/quality` | POST | Run quality checks only |

### Example Request

```bash
curl -X POST https://api.example.com/run \
  -H "Content-Type: application/json" \
  -d '{"action": "run", "config": {"sample_size": 100000}}'
```

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Code Standards

- **Formatting**: Black (line length 100)
- **Linting**: Flake8
- **Type Hints**: Required for all functions
- **Tests**: Required for new features

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- NH DHHS for public claims data
- AWS for cloud infrastructure
- DBT Labs for transformation framework
- Great Expectations for data quality patterns

---

**Built with by the Data Engineering Team**
