
# ============================================================
# INDUSTRIAL PREDICTIVE MAINTENANCE, SAFETY & EMERGENCY AI
# Development, Testing, MLOps & Deployment Makefile
# ============================================================

.DEFAULT_GOAL := help

SHELL := /bin/sh

PYTHON ?= python3
PIP := $(PYTHON) -m pip

PROJECT_NAME := industrial-ai
BACKEND_DIR := backend
FRONTEND_DIR := frontend
TEST_DIR := tests

COMPOSE := docker compose
ALEMBIC := $(PYTHON) -m alembic
PYTEST := $(PYTHON) -m pytest

API_HOST ?= 0.0.0.0
API_PORT ?= 8000

COVERAGE_MIN ?= 80

# Adjust this if your application entrypoint differs.
APP_MODULE ?= app.main:app

# Commands run with backend/ as the Python import root.
export PYTHONPATH := $(BACKEND_DIR):$(PYTHONPATH)

.PHONY: help install install-dev install-ai install-cv \
        install-rag install-all setup env-check \
        run run-prod frontend-install frontend-dev frontend-build \
        docker-build docker-up docker-down docker-restart \
        docker-logs docker-ps docker-clean docker-all \
        docker-ai docker-iot docker-monitoring \
        db-upgrade db-downgrade db-current db-history \
        db-revision db-check \
        test test-unit test-integration test-api \
        test-security test-coverage test-fast \
        test-smoke test-fire test-motion test-overheating \
        test-rul test-anomaly test-failure \
        test-cost test-repair test-health \
        test-telemetry test-safety test-incidents \
        lint format format-check typecheck security-scan \
        check quality \
        ml-train ml-evaluate mlflow \
        cv-evaluate rag-evaluate agent-evaluate \
        monitoring-up monitoring-down \
        clean clean-cache clean-reports \
        health status


# ============================================================
# HELP
# ============================================================

help:
	@echo ""
	@echo "Industrial Predictive Maintenance, Safety & Emergency AI"
	@echo "======================================================="
	@echo ""
	@echo "SETUP"
	@echo "  make install             Install project dependencies"
	@echo "  make install-dev         Install development tools"
	@echo "  make setup               Create local environment"
	@echo "  make env-check           Verify .env exists"
	@echo ""
	@echo "APPLICATION"
	@echo "  make run                 Start FastAPI development server"
	@echo "  make run-prod            Start FastAPI production-style server"
	@echo "  make frontend-install    Install frontend dependencies"
	@echo "  make frontend-dev        Start frontend development server"
	@echo "  make frontend-build      Build frontend"
	@echo ""
	@echo "DOCKER"
	@echo "  make docker-build        Build Docker images"
	@echo "  make docker-up           Start core services"
	@echo "  make docker-down         Stop services"
	@echo "  make docker-logs         View backend logs"
	@echo "  make docker-ps           Show containers"
	@echo "  make docker-ai           Start AI profile"
	@echo "  make docker-iot          Start IoT profile"
	@echo "  make docker-monitoring   Start monitoring profile"
	@echo "  make docker-all          Start all profiles"
	@echo ""
	@echo "DATABASE"
	@echo "  make db-upgrade          Apply database migrations"
	@echo "  make db-downgrade        Roll back one migration"
	@echo "  make db-revision MSG=x   Create migration"
	@echo "  make db-current          Show current revision"
	@echo ""
	@echo "TESTING"
	@echo "  make test                Run all tests"
	@echo "  make test-unit           Run unit tests"
	@echo "  make test-integration    Run integration tests"
	@echo "  make test-api            Run API tests"
	@echo "  make test-security       Run security tests"
	@echo "  make test-coverage       Run tests with coverage"
	@echo ""
	@echo "QUALITY"
	@echo "  make lint                Run Ruff linting"
	@echo "  make format              Format Python code"
	@echo "  make typecheck           Run mypy"
	@echo "  make security-scan       Run Bandit"
	@echo ""
	@echo "CLEANUP"
	@echo "  make clean               Remove local generated files"
	@echo ""


# ============================================================
# ENVIRONMENT SETUP
# ============================================================

setup:
	@test -f .env || cp .env.example .env
	@echo "Environment file ready."
	@echo "Update .env with your local credentials."

env-check:
	@test -f .env || (echo "ERROR: .env is missing"; exit 1)
	@echo ".env exists."


# ============================================================
# DEPENDENCY INSTALLATION
# ============================================================

install:
	$(PIP) install -r requirements.txt

install-dev:
	$(PIP) install pytest pytest-asyncio pytest-cov
	$(PIP) install ruff black mypy bandit

install-ai:
	$(PIP) install transformers sentence-transformers
	$(PIP) install langchain langgraph

install-cv:
	$(PIP) install opencv-python-headless pillow
	$(PIP) install ultralytics

install-rag:
	$(PIP) install qdrant-client rank-bm25
	$(PIP) install sentence-transformers

install-all: install install-dev install-ai install-cv install-rag


# ============================================================
# FASTAPI BACKEND
# ============================================================

run:
	cd $(BACKEND_DIR) && \
	$(PYTHON) -m uvicorn $(APP_MODULE) \
	--host $(API_HOST) \
	--port $(API_PORT) \
	--reload

run-prod:
	cd $(BACKEND_DIR) && \
	$(PYTHON) -m uvicorn $(APP_MODULE) \
	--host $(API_HOST) \
	--port $(API_PORT) \
	--workers 2


# ============================================================
# FRONTEND
# ============================================================

frontend-install:
	cd $(FRONTEND_DIR) && npm install

frontend-dev:
	cd $(FRONTEND_DIR) && npm run dev

frontend-build:
	cd $(FRONTEND_DIR) && npm run build


# ============================================================
# DOCKER
# ============================================================

docker-build:
	$(COMPOSE) build

docker-up:
	$(COMPOSE) up -d --build

docker-down:
	$(COMPOSE) down

docker-restart:
	$(COMPOSE) restart

docker-logs:
	$(COMPOSE) logs -f backend

docker-ps:
	$(COMPOSE) ps

docker-clean:
	$(COMPOSE) down --remove-orphans

docker-ai:
	$(COMPOSE) --profile ai up -d --build

docker-iot:
	$(COMPOSE) --profile iot up -d --build

docker-monitoring:
	$(COMPOSE) --profile monitoring up -d

docker-all:
	$(COMPOSE) \
	--profile ai \
	--profile iot \
	--profile monitoring \
	--profile storage \
	up -d --build


# ============================================================
# DATABASE MIGRATIONS
# ============================================================

db-upgrade:
	$(ALEMBIC) upgrade head

db-downgrade:
	$(ALEMBIC) downgrade -1

db-current:
	$(ALEMBIC) current

db-history:
	$(ALEMBIC) history

db-revision:
	@test -n "$(MSG)" || \
	(echo "Usage: make db-revision MSG='description'"; exit 1)
	$(ALEMBIC) revision --autogenerate -m "$(MSG)"

db-check:
	$(ALEMBIC) check


# ============================================================
# GENERAL TESTING
# ============================================================

test:
	$(PYTEST) $(TEST_DIR) -v

test-unit:
	$(PYTEST) $(TEST_DIR)/unit -v

test-integration:
	$(PYTEST) $(TEST_DIR)/integration -v

test-api:
	$(PYTEST) $(TEST_DIR)/api -v

test-security:
	$(PYTEST) $(TEST_DIR)/security -v

test-fast:
	$(PYTEST) $(TEST_DIR)/unit -q

test-coverage:
	$(PYTEST) $(TEST_DIR) \
	--cov=$(BACKEND_DIR)/app \
	--cov-report=term-missing \
	--cov-report=html \
	--cov-fail-under=$(COVERAGE_MIN)


# ============================================================
# INDUSTRIAL AI UNIT TESTS
# ============================================================

test-smoke:
	$(PYTEST) $(TEST_DIR)/unit/test_smoke_detection.py -v

test-fire:
	$(PYTEST) $(TEST_DIR)/unit/test_fire_detection.py -v

test-motion:
	$(PYTEST) $(TEST_DIR)/unit/test_motion_detection.py -v

test-overheating:
	$(PYTEST) $(TEST_DIR)/unit/test_overheating.py -v

test-rul:
	$(PYTEST) $(TEST_DIR)/unit/test_rul.py -v

test-anomaly:
	$(PYTEST) $(TEST_DIR)/unit/test_anomaly.py -v

test-failure:
	$(PYTEST) $(TEST_DIR)/unit/test_failure_prediction.py -v

test-cost:
	$(PYTEST) $(TEST_DIR)/unit/test_cost_engine.py -v

test-repair:
	$(PYTEST) $(TEST_DIR)/unit/test_repair_replacement.py -v

test-health:
	$(PYTEST) $(TEST_DIR)/unit/test_health_score.py -v


# ============================================================
# INTEGRATION TESTS
# ============================================================

test-telemetry:
	$(PYTEST) $(TEST_DIR)/integration/test_telemetry_pipeline.py -v

test-safety:
	$(PYTEST) $(TEST_DIR)/integration/test_safety_pipeline.py -v

test-incidents:
	$(PYTEST) $(TEST_DIR)/integration/test_incident_pipeline.py -v


# ============================================================
# CODE QUALITY
# ============================================================

lint:
	$(PYTHON) -m ruff check $(BACKEND_DIR) $(TEST_DIR)

format:
	$(PYTHON) -m black $(BACKEND_DIR) $(TEST_DIR)
	$(PYTHON) -m ruff check --fix $(BACKEND_DIR) $(TEST_DIR)

format-check:
	$(PYTHON) -m black --check $(BACKEND_DIR) $(TEST_DIR)

typecheck:
	$(PYTHON) -m mypy $(BACKEND_DIR)/app

security-scan:
	$(PYTHON) -m bandit -r $(BACKEND_DIR)/app

check: lint format-check typecheck

quality: check security-scan test


# ============================================================
# MACHINE LEARNING
# ============================================================

# These targets require the corresponding training and
# evaluation modules to exist in your repository.

ml-train:
	$(PYTHON) -m app.ml.training.train

ml-evaluate:
	$(PYTHON) -m app.ml.evaluation.evaluate

mlflow:
	$(COMPOSE) up -d mlflow


# ============================================================
# COMPUTER VISION EVALUATION
# ============================================================

cv-evaluate:
	$(PYTHON) -m app.cv.evaluation.evaluate


# ============================================================
# RAG EVALUATION
# ============================================================

rag-evaluate:
	$(PYTHON) -m app.rag.evaluation.evaluate


# ============================================================
# MULTI-AGENT EVALUATION
# ============================================================

agent-evaluate:
	$(PYTHON) -m app.agents.evaluation.evaluate


# ============================================================
# MONITORING
# ============================================================

monitoring-up:
	$(COMPOSE) --profile monitoring up -d

monitoring-down:
	$(COMPOSE) --profile monitoring stop prometheus grafana


# ============================================================
# HEALTH & STATUS
# ============================================================

health:
	$(PYTHON) -c "import urllib.request; print(urllib.request.urlopen('http://localhost:8000/health', timeout=5).read().decode())"

status:
	$(COMPOSE) ps


# ============================================================
# CLEANUP
# ============================================================

clean-cache:
	find . -type d -name '__pycache__' -prune -exec rm -rf {} +
	find . -type d -name '.pytest_cache' -prune -exec rm -rf {} +
	find . -type d -name '.mypy_cache' -prune -exec rm -rf {} +
	find . -type d -name '.ruff_cache' -prune -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete

clean-reports:
	rm -rf htmlcov
	rm -f .coverage
	rm -f coverage.xml

clean: clean-cache clean-reports
	@echo "Local generated files cleaned."


# ============================================================
# END OF MAKEFILE
# ============================================================
