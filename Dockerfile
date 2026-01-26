FROM python:3.11-slim

# Set UV environment variables for GCP authentication
ENV UV_INDEX_GCP_INDEX_USERNAME="oauth2accesstoken"
ARG GCP_TOKEN
ENV UV_INDEX_GCP_INDEX_PASSWORD=$GCP_TOKEN

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_HOME=/app \
    PORT=8000

WORKDIR $APP_HOME

# System deps (only curl for liveness/debug)
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock* ./

# Install uv
RUN pip3 install --upgrade pip && \
    pip3 install uv

# Install dependencies using UV with GCP token
# The token is passed via build arg and used for authentication
RUN if [ -n "$GCP_TOKEN" ]; then \
        echo "Installing with GCP authentication..."; \
        uv pip install --no-cache --system -e .; \
    else \
        echo "WARNING: No GCP_TOKEN provided, attempting install without authentication..."; \
        uv pip install --no-cache --system -e .; \
    fi

COPY elastic_mcp.py ./app.py
COPY post_retriver.py ./
COPY mcp_tools.json ./
COPY adapters ./adapters
COPY common ./common
COPY configuration ./configuration
COPY helpers ./helpers
COPY pocketflows ./pocketflows
COPY templates ./templates

RUN useradd -u 10001 -r -s /usr/sbin/nologin appuser && chown -R appuser:appuser $APP_HOME
USER appuser

EXPOSE 8000

# Set environment variables for GCP authentication at runtime
# GOOGLE_APPLICATION_CREDENTIALS should be set at runtime via docker run -e
# GoogleAuthenticationMethod defaults to APPLICATION_DEFAULT
ENV GoogleAuthenticationMethod="application_default"
ENV GoogleProjectId="hu0092-bus-t-ai"

# Default command: run uvicorn
ENTRYPOINT ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
