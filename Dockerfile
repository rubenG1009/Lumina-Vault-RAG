FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY . /app
ARG LUMINA_EMBEDDING_BACKEND=hash
RUN if [ "$LUMINA_EMBEDDING_BACKEND" = semantic ]; then \
      pip install --no-cache-dir '.[semantic]'; \
    else pip install --no-cache-dir .; fi \
    && useradd --create-home lumina \
    && mkdir -p /home/lumina/.cache/fastembed \
    && chown -R lumina:lumina /app /home/lumina
USER lumina
EXPOSE 8000
CMD ["sh", "-c", "lumina migrate && uvicorn lumina.api:app_factory --factory --host 0.0.0.0 --port 8000"]
