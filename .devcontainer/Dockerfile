# .devcontainer/Dockerfile
# CORRECTED: Use only the nvidia/cuda base image and install Python within it

# Base image supporting CUDA 12.1.1 on Ubuntu 22.04
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Argument for Python version (optional, could hardcode 3.10)
ARG PYTHON_VERSION=3.10

# Avoid prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies AND Python itself
# Ubuntu 22.04 base includes tools to install specific Python versions
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Python core and package management
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-venv \
    python${PYTHON_VERSION}-dev \
    python3-pip \
    # Naini System Dependencies
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-hin \
    tesseract-ocr-eng \
    libgl1-mesa-glx \
    ghostscript \
    # Build tools often needed for pip packages
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Make python3.10 the default python/python3
RUN update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1

# Upgrade pip and set pip3 as default pip
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Copy the requirements file into the container (from /workspace context)
COPY requirements.txt /tmp/requirements.txt

# Install Python packages using the default pip (pip3)
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Set the working directory
WORKDIR /workspace

# Note: If you need a specific user, add user creation steps here (and adjust devcontainer.json remoteUser)
# RUN useradd -ms /bin/bash vscode
# USER vscode

# Command to keep the container running (useful for devcontainers)
ENTRYPOINT [ "bash" ]