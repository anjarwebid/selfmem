FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libpq-dev && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download embedding model into the image
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY config.py embeddings.py db.py auth.py server.py \
     models.py passwords.py tokens.py quotas.py ratelimit.py audit.py \
     billing.py gdpr.py ./
COPY templates/ templates/
COPY static/ static/

EXPOSE 8818

CMD ["python", "server.py"]
