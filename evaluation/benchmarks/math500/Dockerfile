FROM python:3.11-bookworm

# Install necessary packages for math calculations
RUN pip install --no-cache-dir numpy matplotlib sympy scipy

# Set up workspace directory
RUN mkdir -p /workspace

# Set working directory
WORKDIR /workspace

# Set environment variables
ENV PYTHONUNBUFFERED=1