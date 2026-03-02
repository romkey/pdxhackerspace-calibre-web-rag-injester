# syntax=docker/dockerfile:1.7
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
ARG DOCKER_INSTALL_EXTRAS=""

COPY pyproject.toml README.md /app/
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip
COPY src /app/src
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ -n "$DOCKER_INSTALL_EXTRAS" ]; then \
      pip install ".[${DOCKER_INSTALL_EXTRAS}]"; \
    else \
      pip install .; \
    fi

ENTRYPOINT ["calibre-web2rag"]
