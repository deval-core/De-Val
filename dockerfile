FROM python:3.11.8-slim


# Install dependencies
RUN apt-get update && apt-get install -y build-essential \ 
    curl \
    libatlas-base-dev \
    liblapack-dev \
    libblas-dev \
    gfortran \ 
    && rm -rf /var/lib/apt/lists/*


# Set environment variables
ENV PATH="${PATH}:/root/.local/bin" \
  PYTHONFAULTHANDLER=1 \
  PYTHONUNBUFFERED=1 \
  PYTHONHASHSEED=random \
  PIP_NO_CACHE_DIR=off \
  PIP_DISABLE_PIP_VERSION_CHECK=on \
  PIP_DEFAULT_TIMEOUT=300 \
  # Poetry's configuration:
  POETRY_NO_INTERACTION=1 \
  POETRY_VIRTUALENVS_CREATE=false \
  # HF
  TRANSFORMERS_CACHE=/tmp

# Copy all of deval over and initialize
WORKDIR /app

COPY --chown=miner:miner . ./
RUN pip install poetry 
RUN POETRY_VIRTUALENVS_CREATE=false poetry install --only main

CMD ["poetry", "run", "python3", "neurons/validator.py", "--host", "0.0.0.0", "--port", "8000"]
