#!/usr/bin/env python3
"""
proxy.py
--------
FastAPI proxy that forwards requests to worker processes.
"""
import asyncio
import json
import pathlib
import os
import logging
import sys
from typing import List, Tuple, Optional

import aiohttp
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import Response, JSONResponse
from whisperx_api_server.routers.misc import router as misc_router

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("proxy")

# Configuration from environment
WORKERS_CFG = os.environ.get("WORKERS_CFG", "/tmp/whisperx_workers.json")
WORKERS_POLL_INTERVAL = float(os.environ.get("WORKERS_POLL_INTERVAL", "2.0"))
REQUEST_TIMEOUT = int(os.environ.get("WORKER_REQUEST_TIMEOUT", "3600"))

app = FastAPI(
    title="whisperx-proxy",
    description="Production proxy for WhisperX workers"
)
app.include_router(misc_router)

# Runtime state
_workers_lock = asyncio.Lock()
_workers: List[dict] = []
_outstanding = {}  # worker_key -> outstanding request count
_failed_workers = set()  # track temporarily failed workers


async def _load_workers_from_file():
    """Load worker configuration from file."""
    global _workers, _outstanding
    p = pathlib.Path(WORKERS_CFG)
    
    if not p.exists():
        logger.warning(f"Workers config file not found: {WORKERS_CFG}")
        return
    
    try:
        data = json.loads(p.read_text())
        async with _workers_lock:
            old_count = len(_workers)
            _workers = data
            
            # Initialize outstanding counters
            for w in _workers:
                key = f"{w['host']}:{w['port']}"
                _outstanding.setdefault(key, 0)
            
            if len(_workers) != old_count:
                logger.info(f"Workers updated: {old_count} -> {len(_workers)}")
                logger.debug(f"Current workers: {json.dumps(_workers, indent=2)}")
                
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in workers config: {e}")
    except Exception as e:
        logger.error(f"Failed to load workers config: {e}", exc_info=True)


@app.on_event("startup")
async def startup():
    """Initialize proxy on startup."""
    logger.info("=== WhisperX Proxy Starting ===")
    logger.info(f"Workers config: {WORKERS_CFG}")
    logger.info(f"Poll interval: {WORKERS_POLL_INTERVAL}s")
    logger.info(f"Request timeout: {REQUEST_TIMEOUT}s")
    
    await _load_workers_from_file()
    
    if not _workers:
        logger.warning("No workers configured at startup - waiting for workers file")
    else:
        logger.info(f"Loaded {len(_workers)} worker(s)")
    
    asyncio.create_task(_periodic_reload_workers())
    asyncio.create_task(_periodic_health_check())


async def _periodic_reload_workers():
    """Periodically reload worker configuration if file changes."""
    last_mtime = None
    p = pathlib.Path(WORKERS_CFG)
    
    while True:
        try:
            if p.exists():
                mtime = p.stat().st_mtime
                if last_mtime is None or mtime != last_mtime:
                    await _load_workers_from_file()
                    last_mtime = mtime
        except Exception as e:
            logger.error(f"Error in periodic reload: {e}")
        
        await asyncio.sleep(WORKERS_POLL_INTERVAL)


async def _periodic_health_check():
    """Periodically check worker health and clear failed workers."""
    while True:
        await asyncio.sleep(30)
        
        if _failed_workers:
            logger.info(f"Clearing {len(_failed_workers)} failed worker(s) for retry")
            _failed_workers.clear()


async def _choose_worker() -> Optional[dict]:
    """Choose the best available worker based on load."""
    async with _workers_lock:
        if not _workers:
            logger.error("No workers configured")
            return None
        
        # Filter out failed workers
        available = [w for w in _workers if f"{w['host']}:{w['port']}" not in _failed_workers]
        
        if not available:
            logger.warning("All workers marked as failed, retrying all")
            _failed_workers.clear()
            available = _workers
        
        # Prefer idle workers
        for w in available:
            key = f"{w['host']}:{w['port']}"
            if _outstanding.get(key, 0) == 0:
                logger.debug(f"Selected idle worker: {key}")
                return w
        
        # Find least busy worker
        best = None
        best_count = float('inf')
        for w in available:
            key = f"{w['host']}:{w['port']}"
            cnt = _outstanding.get(key, 0)
            if cnt < best_count:
                best_count = cnt
                best = w
        
        if best:
            logger.debug(f"Selected least busy worker: {best['host']}:{best['port']} (load: {best_count})")
        
        return best


async def _forward_form_to_worker(
    worker: dict, 
    endpoint: str, 
    form_items: List[Tuple[str, object]]
) -> Tuple[bytes, int, dict]:
    """Forward multipart form data to worker and return response."""
    key = f"{worker['host']}:{worker['port']}"
    url = f"http://{key}{endpoint}"
    
    _outstanding[key] = _outstanding.get(key, 0) + 1
    logger.info(f"Forwarding request to {key}{endpoint} (load: {_outstanding[key]})")
    
    try:
        timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            data = aiohttp.FormData()
            
            for (field_name, value) in form_items:
                if isinstance(value, (bytes, bytearray)):
                    data.add_field(
                        field_name, 
                        value, 
                        filename="upload", 
                        content_type="application/octet-stream"
                    )
                else:
                    data.add_field(field_name, str(value))
            
            async with session.post(url, data=data) as resp:
                body = await resp.read()
                headers = dict(resp.headers)
                status = resp.status
                
                logger.info(f"Worker {key} responded: status={status}, size={len(body)} bytes")
                return body, status, headers
                
    except asyncio.TimeoutError:
        logger.error(f"Worker {key} timed out after {REQUEST_TIMEOUT}s")
        _failed_workers.add(key)
        raise HTTPException(status_code=504, detail=f"Worker timeout after {REQUEST_TIMEOUT}s")
    except aiohttp.ClientError as e:
        logger.error(f"Worker {key} connection error: {e}")
        _failed_workers.add(key)
        raise HTTPException(status_code=502, detail=f"Worker connection error: {str(e)}")
    except Exception as e:
        logger.error(f"Worker {key} unexpected error: {e}", exc_info=True)
        _failed_workers.add(key)
        raise HTTPException(status_code=500, detail=f"Worker error: {str(e)}")
    finally:
        _outstanding[key] = max(0, _outstanding.get(key, 1) - 1)


async def _try_forward(form_items: List[Tuple[str, object]], endpoint: str):
    """Try to forward request to available workers with retries."""
    max_attempts = 3
    
    for attempt in range(max_attempts):
        worker = await _choose_worker()
        
        if not worker:
            if attempt < max_attempts - 1:
                logger.warning(f"No workers available, retry {attempt + 1}/{max_attempts}")
                await asyncio.sleep(1)
                continue
            else:
                logger.error("No workers available after all retries")
                raise HTTPException(
                    status_code=503, 
                    detail="No workers available"
                )
        
        try:
            return await _forward_form_to_worker(worker, endpoint, form_items)
        except HTTPException:
            if attempt < max_attempts - 1:
                logger.warning(f"Retrying with different worker (attempt {attempt + 2}/{max_attempts})")
                continue
            else:
                raise
    
    raise HTTPException(status_code=503, detail="All workers failed")


def _extract_form_items(request_form) -> List[Tuple[str, object]]:
    """Extract form items from request, reading files into memory."""
    items = []
    
    for k, v in request_form.multi_items():
        # Handle file uploads
        if hasattr(v, "file") and hasattr(v, "filename"):
            try:
                file_bytes = v.file.read()
                items.append((k, file_bytes))
                logger.debug(f"Form field '{k}': file, {len(file_bytes)} bytes")
            except Exception as e:
                logger.error(f"Failed to read file field '{k}': {e}")
                raise HTTPException(status_code=400, detail=f"Failed to read file: {str(e)}")
        else:
            items.append((k, v))
            logger.debug(f"Form field '{k}': {v}")
    
    return items


@app.post("/v1/audio/transcriptions")
async def proxy_transcribe(request: Request):
    """Proxy transcription requests to workers."""
    logger.info("Received transcription request")
    
    try:
        form = await request.form()
        form_items = _extract_form_items(form)
        body, status, headers = await _try_forward(form_items, "/v1/audio/transcriptions")
        
        resp_headers = {}
        if headers.get("content-type"):
            resp_headers["content-type"] = headers["content-type"]
        
        return Response(content=body, status_code=status, headers=resp_headers)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transcription request failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/audio/translations")
async def proxy_translate(request: Request):
    """Proxy translation requests to workers."""
    logger.info("Received translation request")
    
    try:
        form = await request.form()
        form_items = _extract_form_items(form)
        body, status, headers = await _try_forward(form_items, "/v1/audio/translations")
        
        resp_headers = {}
        if headers.get("content-type"):
            resp_headers["content-type"] = headers["content-type"]
        
        return Response(content=body, status_code=status, headers=resp_headers)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Translation request failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/workers")
async def list_workers():
    """Get current worker status."""
    async with _workers_lock:
        return JSONResponse({
            "workers": _workers,
            "outstanding": _outstanding,
            "failed": list(_failed_workers),
            "total_workers": len(_workers),
            "available_workers": len(_workers) - len(_failed_workers)
        })


@app.get("/metrics")
async def metrics():
    """Get proxy metrics in Prometheus format."""
    async with _workers_lock:
        total_outstanding = sum(_outstanding.values())
        available = len(_workers) - len(_failed_workers)
        
        metrics_text = f"""# HELP whisperx_workers_total Total number of configured workers
# TYPE whisperx_workers_total gauge
whisperx_workers_total {len(_workers)}

# HELP whisperx_workers_available Number of available workers
# TYPE whisperx_workers_available gauge
whisperx_workers_available {available}

# HELP whisperx_requests_outstanding Outstanding requests across all workers
# TYPE whisperx_requests_outstanding gauge
whisperx_requests_outstanding {total_outstanding}
"""
        return Response(content=metrics_text, media_type="text/plain")

@app.get("/healthcheck")
async def healthcheck():
    """Health check endpoint."""
    async with _workers_lock:
        available_workers = len(_workers) - len(_failed_workers)
        status = "healthy" if available_workers > 0 else "degraded"
        return JSONResponse({
            "status": status,
            "total_workers": len(_workers),
            "available_workers": available_workers,
            "failed_workers": list(_failed_workers)
        })

if __name__ == "__main__":
    uvicorn.run(
        "proxy:app",
        host="0.0.0.0",
        port=int(os.environ.get("PROXY_PORT", 8000)),
        log_level="info"
    )