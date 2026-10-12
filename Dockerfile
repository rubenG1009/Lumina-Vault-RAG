FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir . && useradd --create-home lumina && chown -R lumina:lumina /app
USER lumina
EXPOSE 8000
CMD ["sh", "-c", "lumina migrate && uvicorn lumina.api:app_factory --factory --host 0.0.0.0 --port 8000"]
