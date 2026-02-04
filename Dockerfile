# Multi-stage Dockerfile for Audio Transcriber
# Supports both CPU and GPU deployment

# Stage 1: Base image with system dependencies
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies including FFmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    wget \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Stage 2: Python dependencies
FROM base as dependencies

# Copy requirements files
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Stage 3: Final application image
FROM dependencies as app

# Copy application code
COPY transcribe.py .
COPY app.py .
COPY examples.py .
COPY config.yaml .
COPY setup.py .
COPY LICENSE .
COPY README.md .

# Create directories for models and data
RUN mkdir -p /app/models /app/input /app/output

# Set proper permissions
RUN chmod +x transcribe.py app.py

# Expose port for web interface
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Default command: run web interface
CMD ["python", "app.py"]

# Alternative commands:
# For CLI: docker run audio-transcriber python transcribe.py audio.m4a
# For web: docker run -p 7860:7860 audio-transcriber


# Stage 4: GPU-enabled image (optional)
FROM dependencies as gpu

# Install PyTorch with CUDA support
RUN pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# Copy application code
COPY transcribe.py .
COPY app.py .
COPY examples.py .
COPY config.yaml .
COPY setup.py .
COPY LICENSE .
COPY README.md .

# Create directories
RUN mkdir -p /app/models /app/input /app/output

# Set proper permissions
RUN chmod +x transcribe.py app.py

# Expose port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Default command
CMD ["python", "app.py"]
