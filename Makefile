.PHONY: setup web demo test examples benchmark doctor

setup:
	uv sync --native-tls --locked
	npm --prefix web ci

web:
	npm --prefix web run build

demo: setup web
	uv run qmech serve

test:
	./scripts/test.sh

examples:
	uv run qmech examples

benchmark:
	./scripts/benchmark.sh

doctor:
	uv run qmech doctor
