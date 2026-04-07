# AstroPlanner LLM Setup (Ollama + Gemma 4)

AstroPlanner talks to a local OpenAI-compatible endpoint:

- `POST /v1/chat/completions`
- `GET /v1/models`

The recommended setup is host-managed [Ollama](https://ollama.com/download) with Google Gemma 4.

Default endpoint and model used by AstroPlanner:

- `LLM server URL`: `http://localhost:11434`
- `LLM model`: `gemma4:e4b`

## 1. Install Ollama

Install Ollama on the host machine:

- macOS: [Ollama download for macOS](https://ollama.com/download/mac)
- Linux: `curl -fsSL https://ollama.com/install.sh | sh`

On macOS:

1. download `Ollama.app`,
2. move it to `/Applications`,
3. launch it once so the local API and CLI become available.

Verify the CLI is available:

```bash
ollama --version
```

If the local API is not already running, start it manually:

```bash
ollama serve
```

Quick helper from the repo:

```bash
make llm-install-help
```

## 2. Pull the default model

From the project root:

```bash
make llm-pull
```

This pulls:

- `gemma4:e4b`

List installed models:

```bash
make llm-models
```

## 3. Verify the OpenAI-compatible API

Check the endpoint and confirm the default model is available:

```bash
make llm-check
```

Manual checks:

```bash
curl http://localhost:11434/v1/models
```

```bash
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma4:e4b",
    "messages": [{"role":"user","content":"Say hi in one short sentence."}]
  }'
```

## 4. Configure AstroPlanner

In AstroPlanner:

1. open `Settings -> General Settings`,
2. set `LLM server URL` to `http://localhost:11434`,
3. set `LLM model` to `gemma4:e4b`,
4. open `AI Assistant` and test.

## 5. Optional larger Gemma 4 variants

If your machine has more RAM/VRAM, you can use larger models by pulling them in Ollama and changing only the model name in AstroPlanner settings.

Examples:

- `gemma4:26b`
- `gemma4:31b`

Example pull:

```bash
ollama pull gemma4:26b
```

AstroPlanner does not need code changes for a different Ollama model. Only the `LLM model` setting changes.

## 6. Optional Docker profile (Linux / CPU tests)

The repo includes an optional Compose profile named `ollama`.

Use it when:

- you are on Linux,
- you want a containerized local endpoint for CPU tests,
- you do not want to install host Ollama.

Start Dockerized Ollama:

```bash
make llm-up-docker
```

Pull the default model inside the container:

```bash
make llm-pull-docker
```

Verify the endpoint:

```bash
make llm-check
```

Other useful commands:

```bash
make llm-models-docker
make llm-logs-docker
make llm-stop-docker
```

Important:

- the Docker profile uses the same API endpoint: `http://localhost:11434`
- host Ollama and Dockerized Ollama should not be run on the same port at the same time
- on macOS, Ollama's official guidance is to use the standalone app outside Docker rather than Docker Desktop

## Troubleshooting

- `Cannot reach LLM server` in AstroPlanner:
  - start Ollama,
  - verify API: `curl http://localhost:11434/v1/models`,
  - verify settings in `Settings -> General Settings`.

- `make llm-check` says the default model is missing:
  - run `make llm-pull`, or `make llm-pull-docker` if using the Docker profile.

- You want a different local backend:
  - AstroPlanner only expects an OpenAI-compatible endpoint,
  - Ollama is just the default recommended runtime.
