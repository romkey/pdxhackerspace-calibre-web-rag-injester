.PHONY: install-dev lint test ingest compose-up compose-test

install-dev:
	pip install -e .[dev]

lint:
	ruff check .

test:
	pytest -q

ingest:
	calibre-web2rag

compose-up:
	docker compose up --build ingester

compose-test:
	docker compose -f docker-compose.test.yml up --build --abort-on-container-exit tests
