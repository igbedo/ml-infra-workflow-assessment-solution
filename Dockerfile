# Dockerfile for training the MNIST model
# Uses CPU-only PyTorch by default. For GPU, see README instructions.
#LABEL org.opencontainers.image.authors="Damian Igbe"
FROM python:3.11-slim

# Keep Python output unbuffered so logs appear immediately in docker logs/terminal
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install minimal OS dependencies often needed by PyTorch/torchvision
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency file first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy training script
COPY train_mnist.py .

# Default directories inside the container. These should be mounted from host.
RUN mkdir -p /app/data /app/results

# Default command. Override args at docker run time as needed.
CMD ["python", "train_mnist.py", "--data-dir", "/app/data", "--output-dir", "/app/results"]
