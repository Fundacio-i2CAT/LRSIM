FROM python:3.13-slim

# Install git, build dependencies, and certificates
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./

# Install requirements
RUN pip install --no-cache-dir -r requirements.txt

COPY lrsim/ ./lrsim/
COPY pyproject.toml ./
COPY lrsim_logo.png ./
COPY README.md ./

# Create output directories
RUN mkdir -p /app/logs /app/tle_output /app/visualisation_output

# Environment variables for common configuration
ENV CONFIG_FILE=lrsim/config/ether_simple.yaml
ENV PYTHONPATH=/app

# Set the entrypoint to run the simulator with configurable config file
ENTRYPOINT ["python", "-m", "lrsim.main", "--config"]
CMD ["lrsim/config/ether_simple.yaml"]
