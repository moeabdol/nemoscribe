.PHONY: test cov lint fmt

test:
	pytest -v

cov:
	pytest -v --cov=nemoscribe --cov-report=term-missing --cov-report=html

lint:
	ruff check src tests
	ruff format --check src tests
	pyright

fmt:
	ruff format src tests
