# AstroPlanner AI Setup (BitNet + Docker Compose)

This setup is BitNet-only and uses Docker Compose.
It follows the official BitNet repository workflow:
[microsoft/BitNet](https://github.com/microsoft/BitNet?tab=readme-ov-file).

AstroPlanner talks to a local OpenAI-compatible endpoint:

- `POST /v1/chat/completions`

The files are already included in this repo:

- [docker-compose.bitnet.yml](docker-compose.bitnet.yml)
- [docker/bitnet/Dockerfile](docker/bitnet/Dockerfile)
- [docker/bitnet/entrypoint.sh](docker/bitnet/entrypoint.sh)

The container relies on official BitNet scripts:

- `setup_env.py`
- `run_inference.py` (for manual CLI checks with `-cnv`)

## 1. Start BitNet server

From project root:

```bash
docker compose -f docker-compose.bitnet.yml up -d --build
```

On first run, container will:

1. clone and prepare official `microsoft/BitNet`,
2. download ready GGUF model,
3. compile runtime,
4. start API server on port `8080`.

First startup can take several minutes.
This image is pinned to BitNet commit `404980e`, which is a practical workaround
for the current `i2_s` ARM regression reported upstream.

Default model repo in Compose is:

- `microsoft/bitnet-b1.58-2B-4T-gguf`

For alternatives, use models compatible with BitNet's official supported families:
[Supported Models](https://github.com/microsoft/BitNet?tab=readme-ov-file#supported-models).

Check logs:

```bash
docker compose -f docker-compose.bitnet.yml logs -f bitnet
```

## 2. Verify API is up

```bash
curl http://localhost:8080/v1/models
```

Optional chat test:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<MODEL_ID_FROM_/v1/models>",
    "messages": [{"role":"user","content":"Say hi in one short sentence."}]
  }'
```

Optional CLI conversation check (BitNet `-cnv` mode):

```bash
docker exec -it astroplanner-bitnet bash -lc '
  cd /opt/BitNet &&
  python3 run_inference.py \
    -m /data/models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
    -p "You are a helpful assistant." \
    -cnv
'
```

Note: `-cnv` is a `llama-cli`/`run_inference.py` option.  
For OpenAI API (`/v1/chat/completions`) `llama-server` uses the model's chat metadata,
or an explicit chat template only when you pass one.

## 3. Configure AstroPlanner

In AstroPlanner:

1. open `Settings -> General Settings`,
2. set `LLM server URL` to `http://localhost:8080`,
3. set `LLM model` to the model id returned by `GET /v1/models`,
4. open `AI Assistant` and test.

## 4. Useful operations

Stop:

```bash
docker compose -f docker-compose.bitnet.yml down
```

Stop and remove volumes (models/cache):

```bash
docker compose -f docker-compose.bitnet.yml down -v
```

## 5. Common tweaks

Edit `docker-compose.bitnet.yml` environment variables:

- `BITNET_THREADS` for CPU utilization,
- `BITNET_CTX_SIZE` for context window,
- `BITNET_GGUF_REPO` to change GGUF repo,
- `BITNET_MODEL_NAME` local model directory name (leave default unless needed),
- `BITNET_N_PREDICT` global generation cap (default `256`),
- `BITNET_QUANT_TYPE` for quantization type,
- `BITNET_CHAT_TEMPLATE` optional forced chat template for `/v1/chat/completions`,
- `BITNET_OVERRIDE_KV` optional metadata override for advanced debugging only,
- `BITNET_ENABLE_CONT_BATCH` enable/disable continuous batching (`1`/`0`),
- `BITNET_USE_PRETUNED` (keep `0` by default; set `1` only for models with available preset kernels).

If a model requires Hugging Face auth, set:

- `HUGGING_FACE_HUB_TOKEN`.

## Troubleshooting

- `Cannot reach LLM server` in AstroPlanner:
  - verify container status: `docker compose -f docker-compose.bitnet.yml ps`,
  - verify API: `curl http://localhost:8080/v1/models`.

- First boot seems stuck:
  - initial compile/model prep is heavy; check live logs.

- Wrong model/quantization:
  - adjust `BITNET_GGUF_REPO` and `BITNET_QUANT_TYPE`, then recreate:
  - `docker compose -f docker-compose.bitnet.yml down -v`
  - `docker compose -f docker-compose.bitnet.yml up -d --build`

- Model answers with garbage like `GGGG...` on Apple Silicon / ARM:
  - this is a known upstream BitNet `i2_s` regression on newer commits,
  - this Docker image already pins BitNet to `404980e` as a workaround,
  - `-cnv` alone does not fix that broken upstream build path.
