# AstroPlanner LLM Setup

AstroPlanner's AI Assistant talks to a local or desktop inference server through an
OpenAI-compatible chat API. The backend is optional: planning, scoring,
`Describe Object`, `Suggest Targets`, and resolver flows stay deterministic and do
not require an LLM.

AstroPlanner expects:

- `POST /v1/chat/completions` for chat responses
- `GET /v1/models` or an Ollama-compatible model list endpoint for `Detect models`

Model discovery is helpful but not mandatory. If your backend does not expose a
model-list endpoint, type the model name manually in `Settings -> General Settings`.

## Recommended Path: Jan

The recommended desktop setup is Jan with its built-in local API and the model Jan
currently marks as default or active.

1. Install and open Jan.
2. Download or activate Jan's recommended/default local model.
3. Enable Jan's local OpenAI-compatible API server.
4. Copy the server base URL from Jan's local API settings.
5. In AstroPlanner, open `Settings -> General Settings -> AI`.
6. Paste the Jan server URL into `LLM server URL`.
7. Click `Detect models`.
8. Pick the model Jan reports as the active/default model, or type that model name
   manually if model discovery is disabled in Jan.
9. Keep `Enable model thinking / reasoning` off unless the chosen model/backend
   explicitly supports it and you want longer reasoning traces.

AstroPlanner accepts either a plain server base URL, such as `http://localhost:1337`,
or an OpenAI-style base URL ending in `/v1`, such as `http://localhost:1337/v1`.
Use the exact host and port shown by Jan on your machine.

## Other Supported Backends

Any backend can work if it exposes an OpenAI-compatible chat-completions API. Common
options include:

- Jan
- Ollama
- Docker Model Runner
- LM Studio
- llama.cpp server
- vLLM or another OpenAI-compatible service

Backend examples:

- Jan:
  - `LLM server URL`: copy from Jan's local API settings
  - `LLM model`: use Jan's active/default downloaded model
- Ollama:
  - `LLM server URL`: `http://localhost:11434`
  - `LLM model`: for example `gemma4:e4b`
- Docker Model Runner:
  - `LLM server URL`: `http://localhost:12434/engines`
  - `LLM model`: for example `docker.io/ai/gemma4:E4B`

The model name must match the backend's own model identifier. When in doubt, use
`Detect models` in AstroPlanner or copy the model ID from the backend UI.

## Optional Ollama Helpers

The repository still includes Make targets for an Ollama-based local test setup.
They are convenience helpers, not a requirement for the AI Assistant.

Show host install hints:

```bash
make llm-install-help
```

Pull the repository's example Ollama model:

```bash
make llm-pull
```

List installed Ollama models:

```bash
make llm-models
```

Check the default Ollama endpoint and example model:

```bash
make llm-check
```

Manual Ollama checks:

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

## Optional Dockerized Ollama

The repo includes an optional Compose profile named `ollama`. Use it only when you
want a containerized local endpoint for Linux/CPU tests or do not want to install
host Ollama.

Start Dockerized Ollama:

```bash
make llm-up-docker
```

Pull the example model inside the container:

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

Important details:

- The Docker profile uses `http://localhost:11434`.
- Host Ollama and Dockerized Ollama should not run on the same port at the same time.
- Docker Model Runner is separate from Dockerized Ollama; use
  `http://localhost:12434/engines` for Docker Model Runner's OpenAI-compatible API.

## Troubleshooting

- `Cannot reach LLM server` in AstroPlanner:
  - start Jan, Ollama, Docker Model Runner, or your chosen backend,
  - verify the URL and port in that backend's UI,
  - try `Detect models`,
  - verify settings in `Settings -> General Settings`.

- `Detect models` returns no models:
  - confirm the backend exposes `GET /v1/models` or an Ollama-compatible model list,
  - type the model name manually if the backend supports chat but not model listing.

- Chat returns HTTP 404:
  - check whether you pasted the backend root URL or an OpenAI base URL ending in `/v1`,
  - avoid pasting unrelated web UI URLs,
  - confirm the backend supports `POST /v1/chat/completions`.

- Responses include unwanted reasoning traces or are very slow:
  - turn off `Enable model thinking / reasoning`,
  - use a smaller model,
  - reduce `LLM max tokens`.

- You use a different local backend:
  - no code changes should be needed if it is OpenAI-compatible,
  - set the backend URL and model name in AstroPlanner settings.
