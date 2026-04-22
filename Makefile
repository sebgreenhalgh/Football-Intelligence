SHELL := /bin/bash

DOCS_PORT ?= 8000

.PHONY: help format serve-docs

help:
	@echo "Targets:"
	@echo "  make format        - run ruff (autofix) on src/"
	@echo "  make serve-docs    - serve docs/ at http://localhost:$(DOCS_PORT)"

format:
	uv run ruff check src --fix

serve-docs:
	@echo "Serving docs/ at http://localhost:$(DOCS_PORT)"
	@cd docs && python3 -m http.server $(DOCS_PORT)
