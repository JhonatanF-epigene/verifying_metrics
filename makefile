RM = rm -rf

PYTHON = python

# Default target (to be invoked if no target is specified)
.DEFAULT_GOAL := install

# Declare targets as PHONY (non-file targets)
.PHONY: install 

check-doppler:
	@TOKEN=$$(doppler configure get token --plain 2>/dev/null); \
	if [ -z "$$TOKEN" ]; then \
		doppler login; \
	else \
		echo "Doppler token is already set."; \
	fi

# Install dependencies
install: check-doppler
	@echo "Installing dependencies..."
	@ doppler run --silent --project epigenelabs --config org -- pdm install
	@echo "Cleaning Doppler specific environment variables..."
	@sh -c 'unset DOPPLER_PROJECT DOPPLER_CONFIG DOPPLER_ENVIRONMENT && doppler setup --project data-integration-omic --config dev'
	@echo "Environment ready for development"

install-dev: check-doppler
	@echo "Installing dependencies..."
	@ doppler run --silent --project epigenelabs --config org -- pdm install -G dev
	@echo "Cleaning Doppler specific environment variables..."
	@sh -c 'unset DOPPLER_PROJECT DOPPLER_CONFIG DOPPLER_ENVIRONMENT && doppler setup --project data-integration-omic --config dev'
	@echo "Environment ready for development"

install-CancerFinder: install
	@bash -c 'source .venv/bin/activate && \
	git clone -b DI-1967_CancerFinder https://github.com/epigenelabs-data/mlflow-models.git && \
	cd mlflow-models/packages/SequencingCancerFinder && pip install -e .'

install-cellnet: install 
	@bash -c 'source .venv/bin/activate && \
	git clone https://github.com/epigenelabs-data/scTab scTab && \
	cd scTab && pip install --no-deps -e .'

# Clean up virtual environment and any other artifacts
clean:
	@echo "Clean environment..."
	@$(RM) venv poetry.lock .python-version htmlcov .coverage .venv .pdm-build
	@echo "Cleaned up all artifacts."