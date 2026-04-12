# Seestar ALP Service

AstroPlanner can talk to `seestar_alp` as an external HTTP service.
This repo does not vendor `seestar_alp`; the Docker Compose setup builds it directly from the upstream GitHub repository by default.

Files included here:

- `docker-compose.yml`
- `docker/seestar_alp/config.toml`
- `docker/seestar_alp/config.simulator.toml`
- `docker/seestar_alp/simulator.config.toml`
- `scripts/astroplanner_compose.sh`

## 1. Default build: no local clone needed

By default, Compose uses a remote Git build context:

```text
https://github.com/smart-underworld/seestar_alp.git#main
```

So the standard startup path does not require any local clone of `seestar_alp`.

Start the service:

```bash
make up-seestar
```

## 2. Optional overrides

If you want to use:

- a local clone,
- your own fork,
- or a pinned branch/tag/commit,

override `SEESTAR_ALP_BUILD_CONTEXT`.

Local clone example:

```bash
export SEESTAR_ALP_BUILD_CONTEXT=/Users/krzkot/git-repos/seestar_alp
```

Pinned commit example:

```bash
export SEESTAR_ALP_BUILD_CONTEXT=https://github.com/smart-underworld/seestar_alp.git#<commit-or-tag>
```

The wrapper forces the Compose project name to `astroplanner`, so it does not inherit an unrelated global `COMPOSE_PROJECT_NAME` from your shell.

## 3. Real device setup

Edit `docker/seestar_alp/config.toml` before first run:

- set `[[seestars]].ip_address` to the real Seestar IP on your LAN,
- keep `device_num = 1` unless you intentionally use a different device number,
- adjust site latitude/longitude if you want ALP startup defaults to match your location.

Important: inside Docker, do not rely on broadcast or mDNS discovery. Use a fixed telescope IP.

`[network].ip_address` should stay `0.0.0.0` in Docker. That is the bind address for the ALP service itself, not the telescope IP.

Start the service:

```bash
make up-seestar
```

If you were previously running the simulator, `make up-seestar` now force-recreates `seestar-alp` with the real-device config so it does not keep using `config.simulator.toml`.
It does not force a rebuild on every run anymore.

## 4. Simulator setup

For local testing without hardware, use the simulator profile and the simulator config:

```bash
make up-seestar-sim
```

That starts:

- `seestar-alp` on `http://localhost:5555`
- ALP web UI on `http://localhost:5432`
- simulator TCP/UDP endpoints on `localhost:4700` and `localhost:4720`

## 5. Verify the service

Check container state:

```bash
make ps
```

Check ALP API:

```bash
curl "http://localhost:5555/api/v1/telescope/1/connected?ClientID=1&ClientTransactionID=1"
```

Open the ALP web UI:

```text
http://localhost:5432
```

Logs:

```bash
make logs-seestar
```

Simulator logs:

```bash
make logs-seestar-sim
```

## 6. Configure AstroPlanner

In AstroPlanner:

1. open `Settings -> General Settings`,
2. set `Backend` to `Seestar ALP service`,
3. set `ALP base URL` to `http://localhost:5555`,
4. set `ALP device #` to `1`,
5. click `Test ALP connection`,
6. open `Seestar Session...` and use `Refresh Status`, `Push Queue`, or `Push + Start`.

For dev/debug only, you can expose the extra sample button with:

```bash
export ASTROPLANNER_SEESTAR_DEBUG=1
python astro_planner.py
```

## 7. Stop and clean up

Stop services:

```bash
make stop-seestar
```

Stop and remove the persisted ALP data volume contents from this repo:

```bash
rm -rf docker/seestar_alp/data/*
```

## Notes

- The shared `docker-compose.yml` is cross-platform and uses published ports instead of host networking.
- The default image build pulls `seestar_alp` source from GitHub at build time.
- `docker/seestar_alp/data/` stores ALP logs and runtime data from the mounted config.
- If you switch between real hardware and simulator, change `SEESTAR_ALP_CONFIG` accordingly before `up`.
- If you want fully reproducible builds, set `SEESTAR_ALP_BUILD_CONTEXT` to a specific tag or commit instead of `#main`.
- Prefer `make ...` or `./scripts/astroplanner_compose.sh ...` over raw `docker compose ...`, because the wrapper pins the project to `astroplanner`.
