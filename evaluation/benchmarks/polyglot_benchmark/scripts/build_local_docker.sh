#!/bin/bash

set -e

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BENCHMARK_DIR="$( cd "${SCRIPT_DIR}/.." && pwd )"
REPO_ROOT="$( cd "${BENCHMARK_DIR}/../../.." && pwd )"

# Create a temporary directory for the Docker build
BUILD_DIR=$(mktemp -d)
trap "rm -rf $BUILD_DIR" EXIT

echo "Creating Docker build context in $BUILD_DIR"

# Create a simple Dockerfile that includes all the necessary tools
cat > "$BUILD_DIR/Dockerfile" << 'EOF'
FROM ubuntu:22.04

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Install common dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    wget \
    software-properties-common \
    apt-transport-https \
    ca-certificates \
    gnupg \
    lsb-release \
    libboost-all-dev \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip3 install --no-cache-dir pytest pytest-timeout

# Install Node.js and npm
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Install Go
RUN wget https://go.dev/dl/go1.20.5.linux-amd64.tar.gz \
    && tar -C /usr/local -xzf go1.20.5.linux-amd64.tar.gz \
    && rm go1.20.5.linux-amd64.tar.gz
ENV PATH="/usr/local/go/bin:${PATH}"

# Install Java
RUN apt-get update && apt-get install -y openjdk-17-jdk \
    && rm -rf /var/lib/apt/lists/*
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

# Install Gradle
RUN wget https://services.gradle.org/distributions/gradle-7.6-bin.zip \
    && mkdir /opt/gradle \
    && unzip -d /opt/gradle gradle-7.6-bin.zip \
    && rm gradle-7.6-bin.zip
ENV PATH="/opt/gradle/gradle-7.6/bin:${PATH}"

# Create workspace directory
RUN mkdir -p /workspace
WORKDIR /workspace

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=UTF-8

CMD ["/bin/bash"]
EOF

# Build the Docker image
IMAGE_NAME="polyglot-benchmark:local"
echo "Building Docker image: $IMAGE_NAME"
docker build -t "$IMAGE_NAME" "$BUILD_DIR"

# Export the image name as an environment variable
echo "export POLYGLOT_DOCKER_IMAGE=$IMAGE_NAME" > "$BENCHMARK_DIR/docker_image.env"

echo "Docker image built successfully: $IMAGE_NAME"
echo "To use this image, run:"
echo "source $BENCHMARK_DIR/docker_image.env"
echo "Then run the benchmark as usual."