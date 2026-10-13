FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# The container runs as SANDBOX_USER_UID:SANDBOX_USER_GID (server settings). Build with the same
# IDs as the host user that runs the server, so the container can write the session folders:
#   docker build --build-arg SANDBOX_UID=$(id -u) --build-arg SANDBOX_GID=$(id -g) ...
# A group or user that already has the ID in the base image is reused (e.g. GID 20 on macOS,
# GID 100 'users', or root's 0); otherwise a 'sandbox' group or user is created with it.
ARG SANDBOX_UID=1000
ARG SANDBOX_GID=999
RUN if ! getent group "${SANDBOX_GID}" >/dev/null; then \
        groupadd -g "${SANDBOX_GID}" sandbox; \
    fi && \
    if ! getent passwd "${SANDBOX_UID}" >/dev/null; then \
        useradd -m -u "${SANDBOX_UID}" -g "${SANDBOX_GID}" -s /bin/bash sandbox; \
    fi

# Work dir owned by the sandbox user
RUN mkdir -p /work
RUN chown -R ${SANDBOX_UID}:${SANDBOX_GID} /work

# Build deps for native extensions (pybedtools, numba, etc.)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    g++ \
    gcc \
    make \
    libz-dev \
    libbz2-dev \
    liblzma-dev \
    bedtools && \
    rm -rf /var/lib/apt/lists/*

# Deps
COPY server_requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && pip install -r /tmp/requirements.txt

# Runner
COPY runner_daemon.py /srv/runner_daemon.py
RUN chown ${SANDBOX_UID}:${SANDBOX_GID} /srv/runner_daemon.py

USER ${SANDBOX_UID}:${SANDBOX_GID}
WORKDIR /work

# Keep it simple: runner polls /work/inbox and writes /work/outbox
ENTRYPOINT ["python", "/srv/runner_daemon.py"]
