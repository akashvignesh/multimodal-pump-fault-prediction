FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for Docker layer caching
COPY requirements.txt .

# Install Python dependencies (CPU-only torch via --extra-index-url in requirements.txt)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --timeout 120 -r requirements.txt

# Copy only what's needed at runtime (skip scripts/, notebooks/, tests/, docs/)
COPY src ./src
COPY data ./data
COPY artifacts ./artifacts
COPY app ./app
COPY .streamlit ./.streamlit
COPY streamlit_app.py .

# Copy root-level docs referenced by the Streamlit app
COPY ablation_results.json .
COPY evaluation_report.md optimization_study.md load_scale_report.md ./
COPY AI_USAGE.md DATA_MANIFEST.md README.md ./

# Copy startup script
COPY start.sh .
RUN chmod +x start.sh

# Expose ports (API and Streamlit)
EXPOSE 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import httpx; r = httpx.get('http://localhost:8000/health'); assert r.status_code == 200"

# Run both API and Streamlit
CMD ["./start.sh"]