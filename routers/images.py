"""
Image Router — R2 Edition (Zero-Latency)
All images served from Cloudflare R2.

Routes:
    GET /api/images/thumbnail/{person_id}           → 302 → R2 (NO CACHE)
    GET /api/images/viewing/{ceremony}/{filename}   → 302 → R2 CDN
    GET /api/images/download/{ceremony}/{filename}  → Proxied stream + attachment
"""

import os
import time
from pathlib import PurePosixPath
import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse, RedirectResponse

from config import r2_url, r2_url_with_subfolder

router = APIRouter()

_PROXY_VIEWING = os.getenv("PROXY_VIEWING", "false").lower() == "true"

_IMG_CACHE      = "public, max-age=604800, immutable"  # photos: content-addressed filenames, safe to cache
_REDIRECT_CACHE = "public, max-age=86400"
_DL_CACHE       = "public, max-age=3600"

# THUMBNAILS: strongest possible no-cache combo.
# person_N.jpg maps to a different face in every client's bucket, and the
# URL path is identical across clients (e.g. /api/images/thumbnail/48),
# so browser memory/disk cache would happily reuse one client's face for
# another client. no-store + no-cache + must-revalidate + Pragma + Expires
# defeats every caching layer including Chrome's aggressive memory cache.
_THUMB_NOCACHE_HEADERS = {
    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
    "Pragma":        "no-cache",
    "Expires":       "0",
}

print(f"[images] PROXY_VIEWING={_PROXY_VIEWING}")

_client: httpx.AsyncClient | None = None


def get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        print("[images] Creating new shared httpx AsyncClient (HTTP/2 enabled)")
        _client = httpx.AsyncClient(
            timeout=httpx.Timeout(60.0, connect=5.0),
            follow_redirects=True,
            limits=httpx.Limits(max_connections=200, max_keepalive_connections=50),
            http2=True,
        )
    return _client


def _content_type(filename: str) -> str:
    ext = filename.rsplit(".", 1)[-1].lower()
    return {
        "jpg":  "image/jpeg",
        "jpeg": "image/jpeg",
        "png":  "image/png",
        "webp": "image/webp",
        "gif":  "image/gif",
        "avif": "image/avif",
    }.get(ext, "application/octet-stream")



async def _stream_r2(url: str, filename: str, attachment: bool = False, extra_headers: dict | None = None) -> StreamingResponse:
    print(f"[images/_stream_r2] fetching from R2: {url}  attachment={attachment}")
    t0 = time.perf_counter()
    client = get_client()
    try:
        req = client.build_request("GET", url)
        r2_resp = await client.send(req, stream=True)
    except httpx.RequestError as e:
        print(f"[images/_stream_r2] ❌ R2 request error: {e}")
        raise HTTPException(status_code=502, detail=f"Could not reach image storage: {e}")

    connect_ms = (time.perf_counter() - t0) * 1000
    print(f"[images/_stream_r2] R2 responded with HTTP {r2_resp.status_code} in {connect_ms:.1f}ms")

    if r2_resp.status_code == 404:
        await r2_resp.aclose()
        print(f"[images/_stream_r2] ❌ 404 from R2 — file not found: {url}")
        raise HTTPException(status_code=404, detail="Image not found")
    if r2_resp.status_code != 200:
        await r2_resp.aclose()
        print(f"[images/_stream_r2] ❌ Unexpected R2 status {r2_resp.status_code} for: {url}")
        raise HTTPException(status_code=502, detail=f"Storage returned {r2_resp.status_code}")

    content_length = r2_resp.headers.get("content-length")
    print(f"[images/_stream_r2] streaming {filename} — content-length={content_length or 'unknown'}")

    headers: dict[str, str] = {
        "Cache-Control": _DL_CACHE if attachment else _IMG_CACHE,
        "Vary": "Accept-Encoding",
        "Access-Control-Expose-Headers": "Content-Length",
    }
    if extra_headers:
        headers.update(extra_headers)
    if content_length:
        headers["Content-Length"] = content_length
    if attachment:
        safe_name = PurePosixPath(filename).name
        headers["Content-Disposition"] = f'attachment; filename="{safe_name}"'
        print(f"[images/_stream_r2] Content-Disposition: attachment; filename={safe_name!r}")

    async def stream_body():
        bytes_sent = 0
        chunk_count = 0
        try:
            async for chunk in r2_resp.aiter_bytes(chunk_size=131072):
                bytes_sent += len(chunk)
                chunk_count += 1
                yield chunk
        finally:
            await r2_resp.aclose()
            total_ms = (time.perf_counter() - t0) * 1000
            print(f"[images/_stream_r2] ✅ streamed {bytes_sent:,} bytes "
                  f"in {chunk_count} chunks for {filename} ({total_ms:.0f}ms total)")

    return StreamingResponse(
        stream_body(),
        media_type=_content_type(filename),
        headers=headers,
    )


async def _redirect_or_proxy(url: str, filename: str) -> RedirectResponse | StreamingResponse:
    if _PROXY_VIEWING:
        print(f"[images/_redirect_or_proxy] PROXY_VIEWING=true → proxying {filename}")
        return await _stream_r2(url, filename)

    # All filenames are normalised to .webp by the gallery router before URLs are built,
    # so no HEAD-check fallback is needed. Go straight to the redirect.
    print(f"[images/_redirect_or_proxy] → 302 redirect to {url}")
    return RedirectResponse(
        url=url, status_code=302,
        headers={"Cache-Control": _REDIRECT_CACHE},
    )


# ── Thumbnail ──────────────────────────────────────────────────────────────────
#
# Now returns a 302 redirect directly to R2 (NOT a proxied stream).
# The browser will cache the image under its final R2 URL, which contains
# the per-client r2.dev subdomain — so two different clients never share a
# cache entry. The 302 response itself carries no-store headers so the
# redirect decision is always re-made on the current backend's R2_PUBLIC_URL.

@router.get("/thumbnail/{person_id}")
async def proxy_thumbnail(person_id: int):
    filename = f"person_{person_id}.webp"
    url = r2_url("thumbnails", filename)
    print(f"[images/thumbnail] person_id={person_id} → 302 to {url}")
    return RedirectResponse(
        url=url,
        status_code=302,
        headers=_THUMB_NOCACHE_HEADERS,
    )


# ── Viewing ───────────────────────────────────────────────────────────────────

@router.get("/viewing/{ceremony}/{filename}")
async def serve_viewing_with_ceremony(ceremony: str, filename: str):
    url = r2_url_with_subfolder("viewing", ceremony, filename)
    print(f"[images/viewing] ceremony={ceremony!r} filename={filename!r} → {url}")
    return await _redirect_or_proxy(url, filename)


# ── Download ──────────────────────────────────────────────────────────────────

@router.get("/download/{filename}")
async def serve_download(filename: str):
    url = r2_url("downloading", filename)
    print(f"[images/download] flat path, filename={filename!r} → {url}")
    return await _stream_r2(url, filename, attachment=True)


@router.get("/download/{ceremony}/{filename}")
async def serve_download_with_ceremony(ceremony: str, filename: str):
    url = r2_url_with_subfolder("downloading", ceremony, filename)
    print(f"[images/download] ceremony={ceremony!r} filename={filename!r} → {url}")
    return await _stream_r2(url, filename, attachment=True)
