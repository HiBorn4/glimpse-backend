"""
Wedding Photo AI — FastAPI Backend (R2 Edition)
DEBUG BUILD — verbose print statements on every request and lifecycle event.
"""
import sys
import os
import time
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from config import CORS_ORIGINS, API_HOST, API_PORT, USE_GPU
from cache import load_classification_from_r2

from dotenv import load_dotenv

load_dotenv()

print("=" * 60)
print("  GLIMPSE BACKEND — DEBUG MODE")
print("=" * 60)
print(f"  Python      : {sys.version.split()[0]}")
print(f"  Working dir : {os.getcwd()}")
print(f"  API_HOST    : {os.getenv('API_HOST', '0.0.0.0')}")
print(f"  API_PORT    : {os.getenv('API_PORT', '8000')}")
print(f"  ENV         : {os.getenv('ENV', 'dev')}")
print(f"  R2_PUBLIC_URL  : {os.getenv('R2_PUBLIC_URL', '(NOT SET)')}")
print(f"  FRONTEND_URL   : {os.getenv('FRONTEND_URL', 'http://localhost:3000')}")
print(f"  USE_GPU        : {os.getenv('USE_GPU', 'false')}")
print("=" * 60)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n[STARTUP] ── Lifespan begin ──────────────────────────────")

    # ── 1. Load classification data from R2 ──────────────────────────────
    print("[STARTUP] Step 1/2 — Loading classification data from R2…")
    t0 = time.perf_counter()
    ok = await load_classification_from_r2()
    elapsed = time.perf_counter() - t0
    if ok:
        print(f"[STARTUP] ✅ Classification loaded in {elapsed:.2f}s")
    else:
        print(f"[STARTUP] ❌ Classification FAILED after {elapsed:.2f}s")

    # ── 2. Load face search engine ────────────────────────────────────────
    print("[STARTUP] Step 2/2 — Loading face search engine from R2…")
    t0 = time.perf_counter()
    try:
        from r2_face_search import R2FaceSearchEngine
        engine = R2FaceSearchEngine(use_gpu=USE_GPU)
        await engine.load_from_r2()
        app.state.search_engine = engine
        elapsed = time.perf_counter() - t0
        print(f"[STARTUP] ✅ Face search engine ready in {elapsed:.2f}s")
    except Exception as e:
        elapsed = time.perf_counter() - t0
        print(f"[STARTUP] ⚠️  Face search NOT available ({elapsed:.2f}s): {e}")
        print(f"[STARTUP]    → selfie search endpoint will return 503")
        app.state.search_engine = None

    print("[STARTUP] ── Server is READY — accepting requests ──────────\n")
    yield

    print("\n[SHUTDOWN] ── Lifespan end ──────────────────────────────────")
    print("[SHUTDOWN] Closing HTTP client pool…")
    try:
        from routers.images import _client
        if _client and not _client.is_closed:
            await _client.aclose()
            print("[SHUTDOWN] ✅ HTTP client closed")
    except Exception as e:
        print(f"[SHUTDOWN] ⚠️  Could not close HTTP client: {e}")
    print("[SHUTDOWN] Bye.\n")


app = FastAPI(
    title="Wedding Photo AI",
    version="4.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url=None,
)

# ── Request logging middleware ─────────────────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    method = request.method
    path   = request.url.path
    query  = str(request.url.query)
    client = request.client.host if request.client else "unknown"

    print(f"[REQ]  {method} {path}{'?' + query if query else ''} — from {client}")

    try:
        response: Response = await call_next(request)
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000
        print(f"[ERR]  {method} {path} → EXCEPTION after {elapsed_ms:.1f}ms: {exc}")
        raise

    elapsed_ms = (time.perf_counter() - start) * 1000
    status = response.status_code

    # colour-code by status
    if status < 300:
        tag = "✅ OK "
    elif status < 400:
        tag = "↩️  RDR"
    elif status < 500:
        tag = "⚠️  4XX"
    else:
        tag = "❌ 5XX"

    print(f"[RES]  {tag} {status} ← {method} {path} ({elapsed_ms:.1f}ms)")
    return response


# ── GZip ──────────────────────────────────────────────────────────────────────
app.add_middleware(GZipMiddleware, minimum_size=1000)

# ── CORS ──────────────────────────────────────────────────────────────────────
print(f"[CONFIG] CORS allow_origins = ['*']  (all origins permitted in debug mode)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    max_age=86400,
)

# ── Routers ───────────────────────────────────────────────────────────────────
print("[CONFIG] Registering routers…")
from routers import events, gallery, search, images

app.include_router(events.router,  prefix="/api/events",  tags=["Events"])
app.include_router(gallery.router, prefix="/api/gallery", tags=["Gallery"])
app.include_router(search.router,  prefix="/api/search",  tags=["Face Search"])
app.include_router(images.router,  prefix="/api/images",  tags=["Images"])
print("[CONFIG] ✅ Routers registered: /api/events, /api/gallery, /api/search, /api/images")


@app.get("/")
async def root():
    print("[ROOT] / hit — returning service info")
    return {"service": "Wedding Photo AI", "version": "4.0.0", "debug": True}


@app.get("/health")
async def health():
    print("[HEALTH] /health hit — checking cache state…")
    from cache import get_cache
    data = get_cache()
    loaded = data is not None
    count  = len(data.get("photos", {})) if data else 0
    print(f"[HEALTH] classification_loaded={loaded}, photo_count={count}")
    return {
        "status": "ok",
        "classification_loaded": loaded,
        "photo_count": count,
    }


if __name__ == "__main__":
    import uvicorn

    is_prod = os.getenv("ENV", "dev").lower() == "prod"
    print(f"[MAIN] Starting uvicorn — prod={is_prod}")
    uvicorn.run(
        "main:app",
        host=API_HOST,
        port=API_PORT,
        reload=not is_prod,
        workers=1,          # always 1 in debug so prints aren't interleaved
        loop="uvloop",
        http="httptools",
        access_log=True,    # uvicorn's own access log ON in debug
    )
