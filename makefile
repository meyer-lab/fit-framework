.PHONY: clean test pyright

all: test

test: .venv
	rye run pytest -s -v -x

.venv:
	rye sync

coverage.xml: .venv
	rye run pytest --junitxml=junit.xml --cov=fit_framework --cov-report xml:coverage.xml

pyright: .venv
	rye run pyright fit_framework

clean:
	rm -rf coverage.xml