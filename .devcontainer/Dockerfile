# Use the official R image
FROM rocker/r-ver:4.3.0

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    python3 \
    python3-venv \
    python3-pip \
    libpython3-dev \
    openssh-client \
    && rm -rf /var/lib/apt/lists/*

# Ensure consistent Python binary
RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    python3 --version

# Upgrade pip & core Python tools
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install R languageserver package
RUN R -e "install.packages('languageserver', repos='https://cloud.r-project.org')"

# Set Git user (passed from devcontainer.json)
ARG GIT_USER_NAME
ARG GIT_USER_EMAIL
RUN git config --global user.name "${GIT_USER_NAME}" && \
    git config --global user.email "${GIT_USER_EMAIL}"

# Clone the repository securely using GITHUB_TOKEN
ARG GITHUB_TOKEN
RUN git clone https://${GIT_USER_NAME}:${GITHUB_TOKEN}@github.com/kasztelmacko/RR_project.git /workspaces/RR_project

# Set working directory
WORKDIR /workspaces/RR_project

# Default to R
CMD ["R"]
