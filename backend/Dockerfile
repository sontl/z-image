FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    git \
    ninja-build \
    curl \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    && rm -rf /var/lib/apt/lists/*

# Install pip for Python 3.11
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# Set python3.11 as default python
RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3

WORKDIR /app

# Install PyTorch with CUDA 12.4 support
# Note: Adjust the version if needed, but this targets the cu124 wheels
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install build dependencies for Flash Attention
RUN pip install packaging ninja

# Install Flash Attention
# Using --no-build-isolation to ensure it uses the installed PyTorch
RUN pip install flash-attn --no-build-isolation

# Install Diffusers from source (latest)
RUN pip install git+https://github.com/huggingface/diffusers

# Install other dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Command to run the API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
