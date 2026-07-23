"""Optional bundled web UI: a static mount of the webui/ build output.

The UI is a thin client over the public API — it holds no server-side state
and registers no API routes. Everything here is skipped entirely unless
WEBUI__ENABLED=true (see config.WebUIConfig).
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from whisperx_api_server.config import Config

logger = logging.getLogger(__name__)

MOUNT_PATH = "/webui"


def resolve_dist_dir(config: Config) -> Path:
    """Locate the built UI assets (a directory containing index.html).

    WEBUI__DIST_DIR wins when set; otherwise ./webui/dist relative to the
    working directory (the Docker image layout), then webui/dist at the repo
    root (source checkouts / editable installs).
    """
    if config.webui.dist_dir:
        candidates = [Path(config.webui.dist_dir)]
    else:
        candidates = [
            Path.cwd() / "webui" / "dist",
            Path(__file__).resolve().parents[2] / "webui" / "dist",
        ]

    for candidate in candidates:
        if (candidate / "index.html").is_file():
            return candidate

    tried = ", ".join(str(c) for c in candidates)
    raise RuntimeError(
        "WEBUI__ENABLED is set but no built UI was found (looked for index.html "
        f"in: {tried}). Build it with `cd webui && npm ci && npm run build`, "
        "rebuild the Docker image with `--build-arg BUILD_WEBUI=build`, or point "
        "WEBUI__DIST_DIR at an existing build output."
    )


def mount_webui(app: FastAPI, config: Config) -> None:
    """Serve the built UI at /webui/ and redirect / to it.

    The mount is intentionally outside API-key auth: it serves only the public
    HTML/JS bundle, and every API call the UI makes carries the key entered by
    the user. Nothing here is added to the OpenAPI schema.
    """
    dist_dir = resolve_dist_dir(config)

    @app.get("/", include_in_schema=False)
    def webui_root_redirect() -> RedirectResponse:
        return RedirectResponse(url=f"{MOUNT_PATH}/", status_code=307)

    app.mount(MOUNT_PATH, StaticFiles(directory=dist_dir, html=True), name="webui")
    logger.info("Web UI enabled at %s/ (assets: %s)", MOUNT_PATH, dist_dir)
