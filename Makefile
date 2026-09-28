.PHONY: help test-gold check-metadata

help:
	@echo "Use the approved Python environment; see docs/DEVELOPMENT_WORKFLOW.md"
	@echo "Targets: test-gold, check-metadata"

test-gold:
	PYTHONPATH=src python -m pytest tests/test_gold_corpus.py -q

check-metadata:
	python scripts/check_project_metadata.py
