# =============================================================================
# SAGIN Intelligent Routing - Docker Image
# =============================================================================
# Multi-stage build for optimized image size
# Supports NVIDIA GPU training with CUDA

# -----------------------------------------------------------------------------
# Stage 1: Base image with CUDA and PyTorch
# -----------------------------------------------------------------------------
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# Stage 2: Install Python dependencies
# -----------------------------------------------------------------------------
FROM base AS dependencies

# Copy requirements first for better caching
COPY requirements.txt .

# Upgrade pip
RUN pip install --upgrade pip

# Install PyTorch Geometric and dependencies
# Note: Must match PyTorch and CUDA versions
RUN pip install torch-scatter torch-sparse torch-geometric \
    -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

# Install other Python dependencies
RUN pip install -r requirements.txt

# Install Jupyter and development tools
RUN pip install \
    jupyterlab>=4.0.0 \
    ipywidgets>=8.0.0 \
    jupyter-dash \
    plotly>=5.0.0

# -----------------------------------------------------------------------------
# Stage 3: Production image
# -----------------------------------------------------------------------------
FROM dependencies AS production

# Copy project code
COPY . .

# Create necessary directories
RUN mkdir -p logs checkpoints results data

# Create non-root user for security
RUN useradd -m -u 1000 sagin && \
    chown -R sagin:sagin /app

# Switch to non-root user
USER sagin

# Expose ports
# 8888: Jupyter Lab
# 6006: TensorBoard
EXPOSE 8888 6006

# Default command: run training
CMD ["python", "src/experiments/train.py", "--config", "configs/routing_config.yaml"]

# -----------------------------------------------------------------------------
# Stage 4: Development image with additional tools
# -----------------------------------------------------------------------------
FROM production AS development

# Switch to root for installing additional packages
USER root

# Install development tools
RUN pip install \
    pytest>=7.0.0 \
    pytest-cov>=4.0.0 \
    black>=23.0.0 \
    isort>=5.12.0 \
    mypy>=1.0.0 \
    flake8>=6.0.0

# Switch back to non-root user
USER sagin

# Default command for development: bash shell
CMD ["bash"]
