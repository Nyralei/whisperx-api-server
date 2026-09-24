# Web UI (optional)

A single-page browser client for the API lives in [`webui/`](../webui/) (React + Vite + TypeScript + Tailwind). It is a thin client over the public endpoints — it adds no API routes and no Python dependencies, and the core API behaves identically whether it is enabled or not (covered by a test).

Features: drag-and-drop upload with progress, all transcription form parameters, a live pipeline-stage timeline (client-generated `X-Request-ID` + status polling, working in both direct and Kafka modes), an interactive transcript with click-to-seek, word-level highlighting and color-coded speakers, export buttons for every response format, one-click re-run of the same file with adjusted parameters (the file stays loaded in the browser — no re-selection), links to the interactive API docs (Swagger UI / ReDoc), and a light/dark theme toggle (follows the OS by default).

**Enable it** with `WEBUI__ENABLED=true`. The UI is then served at `/webui/` (with `/` redirecting to it); when the flag is off — the default — neither route exists. The server fails fast at startup if the flag is on but no build output is found.

**Docker images** build the UI via an opt-in build arg (default `skip` — no bun stage runs, existing profiles build exactly as before):

```bash
# Standalone; same BUILD_WEBUI=build works for compose-kafka.yaml
BUILD_WEBUI=build docker compose --profile cuda build
# then run with WEBUI__ENABLED=true in .env
docker compose --profile cuda up
```

**Without Docker**, build the assets once (Bun ≥ 1.3; no Bun process is needed at runtime):

```bash
cd webui && bun install --frozen-lockfile && bun run build
WEBUI__ENABLED=true whisperx-api
```

The server looks for `webui/dist` relative to the working directory, then the repo root; point `WEBUI__DIST_DIR` at the build output for non-standard layouts. For UI development, `bun run dev` starts Vite on `:5173` and proxies API calls to `http://localhost:8000` (override with `WEBUI_DEV_API`). The frontend carries its own quality gates — `bun run lint` (Biome) and `bun run test` (Vitest) — which run in CI on every push.

**Auth:** the static assets themselves are served without an API key (they contain nothing sensitive); every API call the UI makes carries the key the user enters. The UI probes `GET /info` on load — a 200 without credentials means no key is configured and the key field is hidden.
