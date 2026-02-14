# =============================================================================
# Medical Claims Data Engineering Platform - Makefile
# =============================================================================

.PHONY: help install test lint format run docker-up docker-down dbt-run clean

# Default target
help:
	@echo "Medical Claims Data Engineering Platform"
	@echo "========================================"
	@echo ""
	@echo "Available commands:"
	@echo "  make install      Install Python dependencies"
	@echo "  make test         Run all tests with coverage"
	@echo "  make lint         Run code quality checks"
	@echo "  make format       Format code with Black and isort"
	@echo "  make run          Run ETL pipeline"
	@echo "  make docker-up    Start Docker services"
	@echo "  make docker-down  Stop Docker services"
	@echo "  make dbt-run      Run DBT models"
	@echo "  make clean        Clean temporary files"
	@echo ""

# -----------------------------------------------------------------------------
# Setup & Installation
# -----------------------------------------------------------------------------

install:
	pip install --upgrade pip
	pip install -r requirements.txt
	@echo "✅ Dependencies installed"

install-dev: install
	pip install pytest pytest-cov black flake8 isort mypy
	@echo "✅ Dev dependencies installed"

# -----------------------------------------------------------------------------
# Testing
# -----------------------------------------------------------------------------

test:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing
	@echo "✅ Tests completed. Coverage report: htmlcov/index.html"

test-fast:
	pytest tests/ -v --tb=short -x
	@echo "✅ Fast tests completed"

test-etl:
	pytest tests/test_etl_pipeline.py -v
	@echo "✅ ETL tests completed"

test-quality:
	pytest tests/test_data_quality.py -v
	@echo "✅ Data quality tests completed"

# -----------------------------------------------------------------------------
# Code Quality
# -----------------------------------------------------------------------------

lint:
	flake8 src/ tests/ --max-line-length=100 --exclude=__pycache__
	mypy src/ --ignore-missing-imports
	@echo "✅ Linting completed"

format:
	black src/ tests/ --line-length=100
	isort src/ tests/ --profile=black
	@echo "✅ Code formatted"

check: lint test
	@echo "✅ All checks passed"

# -----------------------------------------------------------------------------
# Pipeline Execution
# -----------------------------------------------------------------------------

run:
	python -m src.etl.run_pipeline
	@echo "✅ Pipeline completed"

run-extract:
	python -c "from src.etl.extract import DataExtractor; e = DataExtractor(); e.execute({})"
	@echo "✅ Extract stage completed"

run-transform:
	python -c "from src.etl.transform import DataTransformer; t = DataTransformer(); t.execute({})"
	@echo "✅ Transform stage completed"

run-quality:
	python -c "from src.data_quality.expectations import DataQualityChecker; q = DataQualityChecker(); q.execute({})"
	@echo "✅ Quality check completed"

# -----------------------------------------------------------------------------
# Docker
# -----------------------------------------------------------------------------

docker-build:
	docker build -t claims-pipeline:latest .
	@echo "✅ Docker image built"

docker-up:
	docker-compose up -d
	@echo "✅ Docker services started"
	@echo "  - PostgreSQL: localhost:5432"
	@echo "  - pgAdmin: localhost:8080"
	@echo "  - Jupyter: localhost:8888"
	@echo "  - LocalStack: localhost:4566"

docker-down:
	docker-compose down
	@echo "✅ Docker services stopped"

docker-logs:
	docker-compose logs -f

docker-shell:
	docker-compose run pipeline /bin/bash

# -----------------------------------------------------------------------------
# DBT
# -----------------------------------------------------------------------------

dbt-deps:
	cd dbt && dbt deps

dbt-run:
	cd dbt && dbt run --profiles-dir profiles/
	@echo "✅ DBT models run"

dbt-test:
	cd dbt && dbt test --profiles-dir profiles/
	@echo "✅ DBT tests completed"

dbt-docs:
	cd dbt && dbt docs generate && dbt docs serve
	@echo "✅ DBT docs generated"

dbt-clean:
	cd dbt && dbt clean
	@echo "✅ DBT artifacts cleaned"

# -----------------------------------------------------------------------------
# AWS
# -----------------------------------------------------------------------------

aws-deploy-lambda:
	cd src/aws/lambda && zip -r function.zip . && \
	aws lambda update-function-code --function-name claims-etl-trigger --zip-file fileb://function.zip
	@echo "✅ Lambda function deployed"

aws-create-bucket:
	aws s3 mb s3://medical-claims-data-lake --region us-east-1
	@echo "✅ S3 bucket created"

aws-sync-data:
	aws s3 sync data/ s3://medical-claims-data-lake/
	@echo "✅ Data synced to S3"

# -----------------------------------------------------------------------------
# Cleanup
# -----------------------------------------------------------------------------

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	rm -rf dbt/target dbt/dbt_packages 2>/dev/null || true
	@echo "✅ Cleaned temporary files"

clean-data:
	rm -rf data/bronze/* data/silver/* data/gold/* 2>/dev/null || true
	@echo "✅ Cleaned processed data"

clean-all: clean clean-data
	docker-compose down -v 2>/dev/null || true
	@echo "✅ Full cleanup completed"

# -----------------------------------------------------------------------------
# Development
# -----------------------------------------------------------------------------

jupyter:
	jupyter notebook --notebook-dir=notebooks

profile:
	python -c "from src.data_quality.profiler import DataProfiler; import pandas as pd; \
		df = pd.read_parquet('data/bronze/claims_ingested.parquet'); \
		p = DataProfiler(); \
		profile = p.profile(df, 'claims'); \
		p.save_profile(profile, 'html')"
	@echo "✅ Data profile generated"

lineage:
	python -c "from src.catalog.lineage import LineageTracker; \
		l = LineageTracker(); \
		l.visualize_lineage('reports/lineage')"
	@echo "✅ Lineage diagram generated"
