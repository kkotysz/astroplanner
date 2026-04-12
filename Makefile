.PHONY: help ps llm-install-help llm-pull llm-check llm-models llm-up-docker llm-pull-docker llm-models-docker llm-logs-docker llm-stop-docker up-seestar up-seestar-sim logs-seestar logs-seestar-sim stop-seestar down

COMPOSE := ./scripts/astroplanner_compose.sh
SEESTAR_REAL_CONFIG := ./docker/seestar_alp/config.toml
SEESTAR_SIM_CONFIG := ./docker/seestar_alp/config.simulator.toml
LLM_URL := http://localhost:11434
LLM_MODEL := gemma4:e4b
OLLAMA_SERVICE := ollama

help:
	@printf '%s\n' \
		'Available targets:' \
		'  make ps                Show AstroPlanner stack status' \
		'  make llm-install-help  Show how to install Ollama before using the AI panel' \
		'  make llm-pull          Pull the default Gemma 4 model into Ollama' \
		'  make llm-check         Check Ollama OpenAI-compatible endpoint and default model' \
		'  make llm-models        List locally available Ollama models' \
		'  make llm-up-docker     Start optional Ollama Docker profile (Linux / CPU tests)' \
		'  make llm-pull-docker   Pull the default Gemma 4 model inside the Docker profile' \
		'  make llm-models-docker List models inside the Dockerized Ollama profile' \
		'  make llm-logs-docker   Follow Dockerized Ollama logs' \
		'  make llm-stop-docker   Stop the Dockerized Ollama profile' \
		'  make up-seestar        Start Seestar ALP service for a real device' \
		'  make up-seestar-sim    Start Seestar ALP + simulator' \
		'  make logs-seestar      Follow Seestar ALP logs' \
		'  make logs-seestar-sim  Follow simulator logs' \
		'  make stop-seestar      Stop Seestar ALP and simulator' \
		'  make down              Stop the whole AstroPlanner stack'

ps:
	@$(COMPOSE) ps

llm-install-help:
	@printf '%s\n' \
		'Ollama is required for the local AI panel.' \
		'' \
		'macOS:' \
		'  1. Download Ollama: https://ollama.com/download/mac' \
		'  2. Move Ollama.app to /Applications and launch it once' \
		'  3. Verify the CLI is available: ollama --version' \
		'  4. Pull the default model: make llm-pull' \
		'' \
		'Linux:' \
		'  1. Host install: curl -fsSL https://ollama.com/install.sh | sh' \
		'  2. Start the server if needed: ollama serve' \
		'  3. Pull the default model: make llm-pull' \
		'' \
		'Optional Linux Docker profile:' \
		'  1. Start Dockerized Ollama: make llm-up-docker' \
		'  2. Pull the default model in the container: make llm-pull-docker' \
		'  3. Verify the endpoint: make llm-check'

llm-pull:
	@command -v ollama >/dev/null 2>&1 || { echo 'ollama is not installed.'; echo 'Run: make llm-install-help'; exit 1; }
	@ollama pull $(LLM_MODEL)

llm-check:
	@command -v curl >/dev/null 2>&1 || { echo 'curl is required for llm-check.'; exit 1; }
	@command -v python3 >/dev/null 2>&1 || { echo 'python3 is required for llm-check.'; exit 1; }
	@curl -fsS $(LLM_URL)/v1/models | python3 -c 'import json,sys; data=json.load(sys.stdin); models=[item.get("id","") for item in data.get("data",[]) if isinstance(item,dict)]; target="$(LLM_MODEL)";\
print(f"Ollama endpoint OK: $(LLM_URL)");\
print("Available models:", ", ".join(models) if models else "<none>");\
raise SystemExit(0 if target in models else (print(f"Default model missing: {target}. Run: ollama pull {target} or make llm-pull-docker") or 1))' || { echo 'Cannot query Ollama at $(LLM_URL). Start Ollama or run make llm-up-docker.'; exit 1; }

llm-models:
	@command -v ollama >/dev/null 2>&1 || { echo 'ollama is not installed.'; echo 'Run: make llm-install-help'; exit 1; }
	@ollama list

llm-up-docker:
	@$(COMPOSE) --profile ollama up -d $(OLLAMA_SERVICE)

llm-pull-docker:
	@$(COMPOSE) --profile ollama up -d $(OLLAMA_SERVICE)
	@$(COMPOSE) exec -T $(OLLAMA_SERVICE) ollama pull $(LLM_MODEL)

llm-models-docker:
	@$(COMPOSE) --profile ollama up -d $(OLLAMA_SERVICE)
	@$(COMPOSE) exec -T $(OLLAMA_SERVICE) ollama list

llm-logs-docker:
	@$(COMPOSE) --profile ollama logs -f $(OLLAMA_SERVICE)

llm-stop-docker:
	@$(COMPOSE) --profile ollama stop $(OLLAMA_SERVICE)

up-seestar:
	@SEESTAR_ALP_CONFIG=$(SEESTAR_REAL_CONFIG) $(COMPOSE) up -d --force-recreate seestar-alp

up-seestar-sim:
	@SEESTAR_ALP_CONFIG=$(SEESTAR_SIM_CONFIG) $(COMPOSE) --profile simulator up -d --build seestar-alp seestar-simulator

logs-seestar:
	@$(COMPOSE) logs -f seestar-alp

logs-seestar-sim:
	@$(COMPOSE) logs -f seestar-simulator

stop-seestar:
	@$(COMPOSE) stop seestar-alp seestar-simulator

down:
	@$(COMPOSE) down
