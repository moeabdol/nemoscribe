.PHONY: test cov

test:
	pytest -v

cov:
	pytest -v --cov=nemoscribe --cov-report=term-missing --cov-report=html
