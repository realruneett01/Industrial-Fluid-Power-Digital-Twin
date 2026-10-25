# Runtime image: inference + OPC-UA advisor only (no training stack). Build from a repo that has
# models/online populated:   docker build -t hydro:0.2 .
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HYDRO_ROOT=/opt/hydro \
    HYDRO_SITE_DIR=/var/lib/hydro/site

RUN useradd --system --create-home --uid 10001 hydro \
    && mkdir -p /opt/hydro /var/lib/hydro/site \
    && chown -R hydro:hydro /var/lib/hydro

WORKDIR /opt/hydro
COPY pyproject.toml LICENSE THIRD_PARTY_NOTICES.md ./
COPY src ./src
COPY configs ./configs
RUN pip install .
COPY models ./models

USER hydro
VOLUME ["/var/lib/hydro/site"]
HEALTHCHECK --interval=60s --timeout=10s --start-period=120s --retries=3 \
    CMD ["hydro", "health"]

# The gate runs first; the service starts only if it passes. Live sensors need
# HYDRO_OPCUA_USER / HYDRO_OPCUA_PASSWORD and --endpoint; without them it exits non-zero.
ENTRYPOINT ["hydro"]
CMD ["deploy", "--start"]
