# Shortcuts for the commands in README.md. Everything runs in Docker.
.PHONY: up test lint secrets prod-build

up:
	docker compose up --build

test:
	docker compose -f docker-compose.test.yaml run --rm test

lint:
	docker compose -f docker-compose.lint.yaml run --rm lint

secrets:
	docker compose -f docker-compose.lint.yaml run --rm secrets

prod-build:
	docker compose -f docker-compose.prod.yaml build
