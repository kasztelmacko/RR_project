# Use the official R image
FROM rocker/r-ver:4.3.0

# Install system dependencies (Python 3 + virtualenv support)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    python3 \
    python3-venv \
    python3-pip \
    libpython3-dev \
    && rm -rf /var/lib/apt/lists/*

# Ensure consistent Python binary
RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    python3 --version

# Upgrade pip & core Python tools
RUN python3 -m pip install --upgrade pip setuptools wheel

# Create workspace directory
WORKDIR /workspaces

# Set default command to interactive R session
CMD ["R"]
