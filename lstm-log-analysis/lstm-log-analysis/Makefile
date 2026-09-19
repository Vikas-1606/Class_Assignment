#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = lstm-log-analysis
PYTHON_INTERPRETER = python3
VENV_DIR = .venv
VENV_PYTHON = $(VENV_DIR)/bin/python
VENV_PIP = $(VENV_DIR)/bin/pip

#################################################################################
# COMMANDS                                                                      #
#################################################################################

.PHONY: help setup_environment requirements data features train predict evaluate test lint clean all

help: ## Show this help message
	@echo "Available commands in $(PROJECT_NAME):"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup_environment: ## Set up Python virtual environment
	$(PYTHON_INTERPRETER) -m venv $(VENV_DIR)
	$(VENV_PIP) install --upgrade pip setuptools wheel
	@echo ">>> Virtual environment created at $(VENV_DIR)."

requirements: ## Install Python dependencies
	$(VENV_PIP) install -r requirements.txt
	@echo ">>> Dependencies installed successfully."

data: ## Ingest and parse raw logs into structured event sequences
	$(VENV_PYTHON) src/dataset.py --download-or-generate
	@echo ">>> Raw and interim datasets prepared."

features: ## Extract sliding window sequences and persist train/val/test splits
	$(VENV_PYTHON) src/features.py
	@echo ">>> Feature engineering complete. Processed tensors saved."

train: ## Train DeepLog LSTM model on normal execution traces
	$(VENV_PYTHON) src/modeling/train.py
	@echo ">>> Model training complete. Model saved in models/."

predict: ## Run anomaly detection inference and output performance metrics
	$(VENV_PYTHON) src/modeling/predict.py --top-k 9
	@echo ">>> Inference evaluation complete."

evaluate: predict ## Alias for predict and evaluation

test: ## Run unit and integration tests with pytest
	$(VENV_PYTHON) -m pytest tests/ src/test_scripts/ -v
	@echo ">>> All test suites executed successfully."

lint: ## Check source code formatting and style
	$(VENV_PYTHON) -m flake8 src/ tests/ --count --select=E9,F63,F7,F82 --show-source --statistics

clean: ## Clean up generated temporary files, caches, and build artifacts
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	@echo ">>> Temporary artifacts cleaned."

all: data features train predict test ## Run full pipeline from data ingestion to test suite
	@echo ">>> Full end-to-end pipeline finished successfully!"
