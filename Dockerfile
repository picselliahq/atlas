FROM python:3.12-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ARG HOME_DIRECTORY=/usr/local/src
WORKDIR $HOME_DIRECTORY/app

# Install system dependencies required by ultralytics and OpenCV,
# and create unprivileged user
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/* && \
    useradd --create-home --home-dir $HOME_DIRECTORY --uid 1000 user && \
    chmod 750 $HOME_DIRECTORY && \
    mkdir -p $HOME_DIRECTORY/data $HOME_DIRECTORY/models $HOME_DIRECTORY/.cache/huggingface && \
    chown -R user:user $HOME_DIRECTORY
USER user

# Set environment variables
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="$HOME_DIRECTORY/.venv/bin:$PATH"

# Install dependencies
COPY --chown=user pyproject.toml uv.lock ${HOME_DIRECTORY}/
RUN --mount=type=cache,target=$HOME_DIRECTORY/.cache/uv,uid=1000 \
    uv sync --frozen --no-dev

# Copy application code
COPY --chown=user app/ $HOME_DIRECTORY/app/
