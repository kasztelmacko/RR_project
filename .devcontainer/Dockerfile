# Use NVIDIA CUDA base image with Ubuntu 22.04 and CUDA 12.4.1
FROM ubuntu:22.04

# Avoid timezone prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Set locale
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    build-essential \
    software-properties-common \
    apt-transport-https \
    ca-certificates \
    gnupg \
    lsb-release \
    libssl-dev \
    libcurl4-openssl-dev \
    libxml2-dev \
    libfontconfig1-dev \
    libharfbuzz-dev \
    libfribidi-dev \
    libfreetype6-dev \
    libpng-dev \
    libtiff5-dev \
    libjpeg-dev \
    locales \
    && rm -rf /var/lib/apt/lists/*

# Configure locale
RUN echo "C.UTF-8 UTF-8" >> /etc/locale.gen && \
    locale-gen

# Install Python 3.9
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.9 python3.9-dev python3.9-venv python3-pip && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 && \
    python3 -m pip install --upgrade pip setuptools wheel

# Install R 4.3 and required packages for VSCode-R
RUN wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc | gpg --dearmor > /usr/share/keyrings/r-project.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/r-project.gpg] https://cloud.r-project.org/bin/linux/ubuntu jammy-cran40/" > /etc/apt/sources.list.d/r-project.list && \
    apt-get update && \
    apt-get install -y r-base r-base-dev && \
    R -e "install.packages(c('jsonlite', 'rlang'), repos='https://cloud.r-project.org')"

# Create working directory
WORKDIR /workspaces

# Git config and clone repo securely
ARG GIT_USER_NAME
ARG GIT_USER_EMAIL
ARG GITHUB_TOKEN
RUN git config --global user.name "${GIT_USER_NAME}" && \
    git config --global user.email "${GIT_USER_EMAIL}" && \
    git clone https://${GIT_USER_NAME}:${GITHUB_TOKEN}@github.com/kasztelmacko/RR_project.git /workspaces/RR_project

# Set working directory to the cloned repo
WORKDIR /workspaces/RR_project

USER root

# Set default shell to bash
SHELL ["/bin/bash", "-c"]