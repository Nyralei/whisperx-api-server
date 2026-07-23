# WhisperX Web UI

Optional browser UI for whisperx-api-server. A thin client over the public
OpenAI-compatible endpoints — it holds no state of its own and adds no API
routes; the built assets are served by FastAPI under `/webui/` when
`WEBUI__ENABLED=true`.

## Build

```bash
bun install --frozen-lockfile
bun run build     # emits dist/, picked up by the server at startup
```

## Develop

```bash
bun run dev       # Vite dev server on :5173, proxying API calls
```

API requests are proxied to `http://localhost:8000` (override with the
`WEBUI_DEV_API` environment variable), so run the API server alongside.

## Quality

```bash
bun run lint      # Biome — lint + format check
bun run format    # Biome — apply formatting
bun run test      # Vitest — unit + component tests
```

Requires Bun ≥ 1.3. Stack: React 19, Vite 8, TypeScript 7,
Tailwind CSS v4 (semantic design-token theme with a light/dark toggle), Biome,
Vitest + Testing Library. Colours are driven by role tokens (`bg-surface`,
`text-fg`, `border-border`, …) defined in `src/index.css`, which flip under the
`.dark` class — components carry no per-element `dark:` variants.
