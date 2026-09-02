# CPU-oriented runtime for the DeepONet CFD surrogate workflow.
# Large datasets and generated artifacts remain on the host and are mounted
# when a training or evaluation command needs them.
FROM python:3.11-slim-bookworm

LABEL org.opencontainers.image.title="DeepONet CFD Surrogate" \
      org.opencontainers.image.description="CPU scientific-ML runtime for a reproducible CFD surrogate workflow"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    PYTHONPATH=/workspace/deeponet-cfd-surrogate/src \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/deeponet-cfd-surrogate

COPY requirements.txt ./
# Select the CPU PyTorch index explicitly; the project does not require CUDA.
RUN python -m pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        "torch>=2.2" \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY . ./

# Fail the build early if dependencies or the project package are incomplete.
RUN python -c "import h5py, matplotlib, numpy, requests, scipy, torch, yaml; import cfd_pretrain; print('DeepONet runtime OK; torch=' + torch.__version__)"

CMD ["python", "src/train_v2.py", "--help"]
