"""
Image Router — R2 Edition (Zero-Navigation Downloads)

WHY DOWNLOADS ARE PROXIED:
    <a download> only forces a save when the FINAL URL is same-origin.
    After a cross-origin 302 to R2, browsers strip `download` and navigate
    to the image — the page disappears.

    Fix: Railway proxies the bytes and sets Content-Disposition: attachment.
    Browser sees same-origin response → saves file, stays on page.

    Viewing still 302-redirects to R2 (cross-origin fine for <img> tags).
    Set PROXY_DOWNLOADS=false in env to revert to redirect (not recommended).
"""

import os
import time
from pathlib import PurePosixPath
import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse, RedirectResponse

from config import r2_url, r2_url_with_subfolder

router = APIRouter()

_PROXY_VIEWING   = os.getenv("PROXY_VIEWING",   "false").lower() == "true"
_PROXY_DOWNLOADS = os.getenv("PROXY_DOWNLOADS", "true").lower() == "true"   # NOW DEFAULT TRUE

_IMG_CACHE      = "public, max-age=604800, immutable"
_REDIRECT_CACHE = "public, max-age=86400"
_DL_CACHE       = "no-store"

_THUMB_NOCACHE_HEADERS = {
    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
    "Pragma":        "no-cache",
    "Expires":       "0",
}

print(f"[images] PROXY_VIEWING={_PROXY_VIEWING} PROXY_DOWNLOADS={_PROXY_DOWNLOADS}")

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


async def _stream_r2(
    url: str,
    filename: str,
    attachment: bool = False,
    extra_headers: dict | None = None,
) -> StreamingResponse:
    print(f"[images/_stream_r2] fetching from R2: {url}  attachment={attachment}")
    t0 = time.perf_counter()
    client = get_client()
    try:
        req = client.build_request("GET", url)
        r2_resp = await client.send(req, stream=True)
    except httpx.RequestError as e:
        print(f"[images/_stream_r2] R2 request error: {e}")
        raise HTTPException(status_code=502, detail=f"Could not reach image storage: {e}")

    connect_ms = (time.perf_counter() - t0) * 1000
    print(f"[images/_stream_r2] R2 HTTP {r2_resp.status_code} in {connect_ms:.1f}ms")

    if r2_resp.status_code == 404:
        await r2_resp.aclose()
        raise HTTPException(status_code=404, detail="Image not found")
    if r2_resp.status_code != 200:
        await r2_resp.aclose()
        raise HTTPException(status_code=502, detail=f"Storage returned {r2_resp.status_code}")

    content_length = r2_resp.headers.get("content-length")
    headers: dict[str, str] = {
        "Cache-Control": _DL_CACHE if attachment else _IMG_CACHE,
        "Vary": "Accept-Encoding",
        "Access-Control-Expose-Headers": "Content-Length, Content-Disposition",
    }
    if extra_headers:
        headers.update(extra_headers)
    if content_length:
        headers["Content-Length"] = content_length
    if attachment:
        safe_name = PurePosixPath(filename).name
        # THE FIX: same-origin response + this header = browser saves the file,
        # never navigates away. No JS blob tricks needed.
        headers["Content-Disposition"] = f'attachment; filename="{safe_name}"'

    async def stream_body():
        bytes_sent = 0
        try:
            async for chunk in r2_resp.aiter_bytes(chunk_size=131072):
                bytes_sent += len(chunk)
                yield chunk
        finally:
            await r2_resp.aclose()
            total_ms = (time.perf_counter() - t0) * 1000
            print(f"[images/_stream_r2] streamed {bytes_sent:,} bytes for {filename} in {total_ms:.0f}ms")

    return StreamingResponse(
        stream_body(),
        media_type=_content_type(filename),
        headers=headers,
    )


async def _redirect_or_proxy(url: str, filename: str) -> RedirectResponse | StreamingResponse:
    if _PROXY_VIEWING:
        return await _stream_r2(url, filename)
    return RedirectResponse(url=url, status_code=302, headers={"Cache-Control": _REDIRECT_CACHE})


async def _download(url: str, filename: str) -> StreamingResponse | RedirectResponse:
    if _PROXY_DOWNLOADS:
        return await _stream_r2(url, filename, attachment=True)
    print(f"[images/download] PROXY_DOWNLOADS=false -> 302 to {url}")
    return RedirectResponse(url=url, status_code=302, headers={"Cache-Control": "no-store"})


# ── Thumbnail ──────────────────────────────────────────────────────────────────

@router.get("/thumbnail/{person_id}")
async def proxy_thumbnail(person_id: int):
    filename = f"person_{person_id}.webp"
    url = r2_url("thumbnails", filename)
    return RedirectResponse(url=url, status_code=302, headers=_THUMB_NOCACHE_HEADERS)


# ── Viewing ───────────────────────────────────────────────────────────────────

@router.get("/viewing/{ceremony}/{filename}")
async def serve_viewing_with_ceremony(ceremony: str, filename: str):
    url = r2_url_with_subfolder("viewing", ceremony, filename)
    print(f"[images/viewing] {ceremony!r}/{filename!r}")
    return await _redirect_or_proxy(url, filename)


# ── Download ──────────────────────────────────────────────────────────────────

@router.get("/download/{filename}")
async def serve_download(filename: str):
    url = r2_url("downloading", filename)
    print(f"[images/download] flat filename={filename!r}")
    return await _download(url, filename)


@router.get("/download/{ceremony}/{filename}")
async def serve_download_with_ceremony(ceremony: str, filename: str):
    url = r2_url_with_subfolder("downloading", ceremony, filename)
    print(f"[images/download] {ceremony!r}/{filename!r}")
    return await _download(url, filename)
