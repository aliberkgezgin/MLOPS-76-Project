FROM python:3.11.9-slim-bullseye

# Create a new non-root user
RUN groupadd -r appgroup && useradd -r -g appgroup -m -s /bin/bash appuser

# Install system dependencies, clean up package manager caches
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry and configure it
RUN pip install --no-cache-dir poetry==1.8.3 \
    && poetry config virtualenvs.create false

# Set work directory
WORKDIR /app

# Copy only files necessary for installing dependencies
COPY pyproject.toml poetry.lock ./

# Install project dependencies
RUN poetry install --no-root --without lint

# NOTE: Pytorch and Poetry not that consistent
# CUDA:
# RUN poetry run pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
# CPU: 
RUN poetry run pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu

# Fix missing README.md for poetry package
RUN if [ ! -f README.md ]; then echo "Placeholder README" > README.md; fi

# Copy source code
COPY src/ src/

# Create a non-root user and switch to it
USER appuser


EXPOSE 8000

#ENV PYTHONUNBUFFERED=1

CMD ["/usr/local/bin/python", "-m", "src"]
