# MIDUS Cognitive Health & Biomarker Analysis - Simple Makefile
# Usage: make <target>

.PHONY: help preprocess train analyze evaluate predict list clean

# Default target
.DEFAULT_GOAL := help

# Colors for output
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
CYAN := \033[0;36m
NC := \033[0m # No Color

# Project configuration
PYTHON_CMD = uv run python
MAIN_FILE = main.py

## Help
help:	## Show available commands
	@echo "$(CYAN)🧠 MIDUS Analysis Tool$(NC)"
	@echo "$(CYAN)=====================$(NC)"
	@echo ""
	@echo "$(YELLOW)Available commands:$(NC)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(GREEN)%-12s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(YELLOW)Examples:$(NC)"
	@echo "  make analyze     # Quick data analysis"
	@echo "  make preprocess  # Preprocess data"
	@echo "  make train       # Train model"
	@echo ""

## Core Commands (1:1 mapping with CLI)
preprocess:	## Preprocess MIDUS data
	@echo "$(BLUE)🔧 Preprocessing data...$(NC)"
	@$(PYTHON_CMD) $(MAIN_FILE) preprocess

train:	## Train ML models
	@echo "$(GREEN)🚀 Training model...$(NC)"
	@$(PYTHON_CMD) $(MAIN_FILE) train

analyze:	## Analyze and visualize data
	@echo "$(BLUE)📈 Analyzing data...$(NC)"
	@$(PYTHON_CMD) $(MAIN_FILE) analyze

evaluate:	## Evaluate trained models
	@echo "$(CYAN)📊 Evaluating models...$(NC)"
	@$(PYTHON_CMD) $(MAIN_FILE) evaluate

predict:	## Make predictions
	@echo "$(YELLOW)🔮 Making predictions...$(NC)"
	@$(PYTHON_CMD) $(MAIN_FILE) predict

list:	## List project contents
	@echo "$(CYAN)📁 Listing contents...$(NC)"
	@$(PYTHON_CMD) $(MAIN_FILE) list

clean:	## Clean generated files
	@echo "$(YELLOW)🧹 Cleaning files...$(NC)"
	@$(PYTHON_CMD) $(MAIN_FILE) clean --all