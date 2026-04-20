"""
Gallery Router — Instant responses with ETag + Cache-Control.

CHANGE (ultra-fast downloads):
    download_url stays as a same-origin /api/images/download/... path,
    but the images router now issues a 302 redirect to R2 instead of
    proxy-streaming the bytes.

    Why same-origin + 302 instead of a direct-to-R2 URL?
      1. The frontend's <a download="filename.jpg"> attribute reliably
         renames the saved file ONLY for same-origin URLs.  For
         cross-origin URLs, the browser requires Content-Disposition
         from the server — and R2's response-content-disposition query
         param support has been unreliable historically.
      2. A 302 costs ~200 bytes of Railway bandwidth per download.  The
         actual image bytes flow from Cloudflare's nearest edge POP
         straight to the user's disk.  Zero Railway egress on the payload.
      3. The browser's <a download> click fires instantly; the 302 is
         sub-20ms; the real bottleneck (bytes over the wire) runs at
         full CDN speed with HTTP/2 or HTTP/3.

    End result: clicking Download opens the save dialog in well under
    a second, the browser shows its own progress bar, and the file
    travels one hop from the nearest Cloudflare POP.  No blob buffering.
"""
import time
from typing import Optional
from fastapi import APIRouter, HTTPException, Query, Request, Response
from cache import get_cache, get_photos_filtered
from config import r2_url, r2_url_with_subfolder

router = APIRouter()
_API_CACHE = "private, no-cache"


def _extract_ceremony(photo: dict) -> str:
    ceremony = photo.get("ceremony", "")
    if ceremony and ceremony != "all":
        return ceremony
    rp = photo.get("relative_path", "")
    if "/" in rp:
        return rp.split("/")[0]
    return ""


def _to_webp(filename: str) -> str:
    """Force any image filename to .webp — all viewing/thumbnail files in R2 are webp."""
    if not filename:
        return filename
    stem, _, ext = filename.rpartition(".")
    if ext.lower() == "webp":
        return filename
    return f"{stem}.webp"


def _viewing_filename(photo: dict) -> str:
    """Return the webp viewing filename (R2 stores only webp for viewing)."""
    return _to_webp(photo.get("filename", ""))


def _build_urls(photo: dict) -> dict:
    original_filename = photo.get("filename", "")   # e.g. "DSC00120.jpg" — for downloads
    viewing_filename  = _to_webp(original_filename) # e.g. "DSC00120.webp" — for viewing
    ceremony = _extract_ceremony(photo)
    photo = dict(photo)

    # viewing_url goes DIRECT to R2 — browser skips the backend 302 entirely.
    # download_url stays same-origin so the frontend's <a download> attribute
    # reliably controls the saved filename; the images router issues a 302
    # to R2 on request, so the bytes still skip Railway.
    if ceremony:
        photo["viewing_url"]  = r2_url_with_subfolder("viewing", ceremony, viewing_filename)
        photo["download_url"] = f"/api/images/download/{ceremony}/{original_filename}"
    else:
        photo["viewing_url"]  = r2_url("viewing", viewing_filename)
        photo["download_url"] = f"/api/images/download/{original_filename}"
    return photo


@router.get("/photos")
async def get_photos(
    request:     Request,
    response:    Response,
    ceremony:    Optional[str] = Query(None),
    person_ids:  Optional[str] = Query(None, description="Comma-separated person IDs"),
    photo_type:  Optional[str] = Query(None),
    orientation: Optional[str] = Query(None, description="portrait | landscape"),
    exact_only:  bool          = Query(False),
    only_me:     bool          = Query(False),
    page:        int           = Query(1, ge=1),
    per_page:    int           = Query(80, ge=1, le=500),
):
    t0 = time.perf_counter()
    print(f"[gallery/photos] ── incoming request ──────────────────────")
    print(f"[gallery/photos]   page={page}, per_page={per_page}")
    print(f"[gallery/photos]   ceremony={ceremony!r}")
    print(f"[gallery/photos]   person_ids={person_ids!r}")
    print(f"[gallery/photos]   photo_type={photo_type!r}")
    print(f"[gallery/photos]   orientation={orientation!r}")
    print(f"[gallery/photos]   exact_only={exact_only}, only_me={only_me}")

    # Parse person IDs
    requested_people: Optional[set[int]] = None
    if person_ids:
        try:
            requested_people = {int(x.strip()) for x in person_ids.split(",") if x.strip()}
            print(f"[gallery/photos]   parsed person_ids set: {requested_people}")
        except ValueError:
            print(f"[gallery/photos] ❌ Invalid person_ids value: {person_ids!r}")
            raise HTTPException(status_code=400, detail="person_ids must be comma-separated integers")

    effective_exact = exact_only or (only_me and bool(requested_people) and len(requested_people) == 1)
    if effective_exact != exact_only:
        print(f"[gallery/photos]   effective_exact overridden to True (only_me=True + single person)")

    # Filter
    print(f"[gallery/photos] → querying cache with filters…")
    t_filter = time.perf_counter()
    filtered, etag = get_photos_filtered(
        person_ids=requested_people,
        ceremony=ceremony,
        photo_type=photo_type,
        exact_only=effective_exact,
        orientation=orientation,
    )
    filter_ms = (time.perf_counter() - t_filter) * 1000
    print(f"[gallery/photos]   filter returned {len(filtered)} photos in {filter_ms:.1f}ms (etag={etag})")

    # Sort + paginate
    filtered_sorted = sorted(filtered, key=lambda x: x.get("filename", ""))
    total  = len(filtered_sorted)
    start  = (page - 1) * per_page
    end    = start + per_page
    page_photos = [_build_urls(p) for p in filtered_sorted[start:end]]
    has_more = end < total

    print(f"[gallery/photos]   page {page}: showing [{start}:{end}] of {total} total, has_more={has_more}")

    # ETag check
    client_etag = request.headers.get("if-none-match", "")
    page_etag   = f'"{etag}-p{page}"'
    response.headers["ETag"]          = page_etag
    response.headers["Cache-Control"] = _API_CACHE

    if client_etag == page_etag:
        total_ms = (time.perf_counter() - t0) * 1000
        print(f"[gallery/photos] ↩️  304 Not Modified (ETag match) — {total_ms:.1f}ms total")
        return Response(status_code=304, headers=dict(response.headers))

    total_ms = (time.perf_counter() - t0) * 1000
    print(f"[gallery/photos] ✅ returning {len(page_photos)} photos — {total_ms:.1f}ms total")

    return {
        "photos":   page_photos,
        "total":    total,
        "page":     page,
        "per_page": per_page,
        "has_more": has_more,
        "etag":     etag,
    }


@router.get("/stats")
async def get_gallery_stats():
    print("[gallery/stats] → computing gallery stats from cache")
    data = get_cache()
    if not data:
        print("[gallery/stats] ❌ No cache data")
        raise HTTPException(status_code=404, detail="No classification data")

    type_counts:     dict[str, int] = {}
    ceremony_counts: dict[str, int] = {}

    for photo in data["photos"].values():
        t = photo["photo_type"]
        c = photo["ceremony"]
        type_counts[t]     = type_counts.get(t, 0) + 1
        ceremony_counts[c] = ceremony_counts.get(c, 0) + 1

    top_people = sorted(
        data["people"].values(),
        key=lambda x: x["total_photos"],
        reverse=True,
    )[:30]

    print(f"[gallery/stats] ✅ photo_types={type_counts}")
    print(f"[gallery/stats]    ceremony_counts={ceremony_counts}")
    print(f"[gallery/stats]    top_people count={len(top_people)}")

    return {
        "total_photos":    data["event_info"]["total_photos"],
        "total_people":    data["event_info"]["total_people"],
        "photo_types":     type_counts,
        "ceremony_counts": ceremony_counts,
        "top_people":      top_people,
    }
