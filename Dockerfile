FROM python:3.11-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:0.6.14 /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_LINK_MODE=copy \
    HF_HOME=/tmp \
    TORCH_HOME=/tmp \
    DOCLING_ARTIFACTS_PATH=/root/.cache/docling/models \
    OMP_NUM_THREADS=4

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY open_deep_research/tools ./open_deep_research/tools
COPY scripts ./scripts

RUN uv pip install --system \
        python-dotenv \
        langchain-core \
        langchain-docling \
        langchain-text-splitters \
    && uv pip install --system \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        docling

# Bake Docling models into image for standalone/offline runs.
RUN docling-tools models download

ENTRYPOINT ["uv", "run", "--no-project", "python", "scripts/pdf_chunk_count.py"]
CMD ["--help"]