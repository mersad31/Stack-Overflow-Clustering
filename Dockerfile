# syntax=docker/dockerfile:1.7
# =============================================================================
# Stack Overflow 2024 Developer Profile Clustering - Docker Image
# =============================================================================
# Multi-stage build:
#   1) builder  -> compile wheels for heavy deps (hdbscan, umap-learn, shap, ...)
#   2) runtime  -> slim image with only runtime libs + installed wheels
# =============================================================================
ARG PYTHON_VERSION=3.13

# -----------------------------------------------------------------------------
# Stage 1: builder
# -----------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS builder

# Build-time system deps for compiling C/Cython extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        git \
        curl \
        libffi-dev \
        libssl-dev \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Upgrade pip / wheel / setuptools first
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy dependency manifests only (better layer caching)
COPY requirements.txt pyproject.toml ./

# Build wheels into /wheels — keeps runtime stage small and reproducible
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

# -----------------------------------------------------------------------------
# Stage 2: runtime
# -----------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS runtime

LABEL org.opencontainers.image.title="stackoverflow-developer-profile-clustering" \
      org.opencontainers.image.description="Clustering analysis of Stack Overflow Developer Survey 2024" \
      org.opencontainers.image.authors="Mersad Ahmadi <40301274>, Ali Hasanzadeh <40304464>" \
      org.opencontainers.image.source="https://github.com/stackoverflow-clustering"

# Runtime shared libraries required by numpy / scikit-learn / matplotlib / pyarrow
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libgl1 \
        libglib2.0-0 \
        libffi8 \
        ca-certificates \
        curl \
        fonts-noto-core \
        fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy pre-built wheels from builder and install them (no compiler needed here)
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir /wheels/*.whl \
    && rm -rf /wheels

# Copy project source
COPY . .

# Make sure src/ is importable (scripts use sys.path.insert for it,
# but setting PYTHONPATH keeps ad-hoc `python -m` calls working too).
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app:/app/src \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    MPLCONFIGDIR=/tmp/matplotlib \
    HF_HUB_DISABLE_TELEMETRY=1

# Pre-create writable dirs (mounted volumes will overlay these)
RUN mkdir -p /app/data/raw /app/data/interim /app/data/processed \
             /app/artifacts/models /app/reports/figures /app/reports/tables \
             /tmp/matplotlib

EXPOSE 8501

# Healthcheck for the Streamlit dashboard
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

# Default entrypoint: Streamlit dashboard.
# Override with `command` (docker run ... <cmd>) to run scripts, e.g.:
#   docker run --rm so-clustering python -m scripts.run_phase2 --config config.yaml
CMD ["streamlit", "run", "dashboard/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
