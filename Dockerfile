FROM python:3.11-slim-buster

WORKDIR /app

COPY src ./pyproject.toml ./poetry.lock ./

ENV PYTHONPATH "${PYTHONPATH}:/app"

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-dev --no-root

ENTRYPOINT [ "python", "-m", "src/llama_runner" ]
