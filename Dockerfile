# syntax=docker/dockerfile:1.7
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md /app/
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip
COPY src /app/src
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install .

ENTRYPOINT ["calibre-web2rag"]
