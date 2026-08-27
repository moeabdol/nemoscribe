.PHONY: test cov lint fmt floor

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

floor:
	uv venv --python 3.12 /tmp/nemoscribe-floor
	uv pip install --python /tmp/nemoscribe-floor/bin/python -e . --group dev
	/tmp/nemoscribe-floor/bin/pytest -q
