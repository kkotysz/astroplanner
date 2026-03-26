FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/data/hf \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:${PATH}"

ARG BITNET_REF=404980e

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    clang \
    cmake \
    curl \
    git \
    git-lfs \
    make \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt
RUN git clone --recursive https://github.com/microsoft/BitNet.git

WORKDIR /opt/BitNet
RUN git checkout "${BITNET_REF}" && \
    git submodule update --init --recursive

RUN python3 -m venv "${VIRTUAL_ENV}" && \
    "${VIRTUAL_ENV}/bin/pip" install --no-cache-dir --upgrade pip && \
    "${VIRTUAL_ENV}/bin/pip" install --no-cache-dir -r requirements.txt

# Compatibility patch: official microsoft/BitNet-b1.58-2B-4T config uses
# "BitNetForCausalLM" while upstream converter registers "BitnetForCausalLM".
RUN sed -i 's/@Model.register("BitnetForCausalLM")/@Model.register("BitnetForCausalLM", "BitNetForCausalLM")/' \
    utils/convert-hf-to-gguf-bitnet.py

COPY docker/bitnet/entrypoint.sh /usr/local/bin/bitnet-entrypoint
RUN chmod +x /usr/local/bin/bitnet-entrypoint

EXPOSE 8080

ENTRYPOINT ["/usr/local/bin/bitnet-entrypoint"]
