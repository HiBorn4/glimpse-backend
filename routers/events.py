"""Events Router — pre-cached event info and people list.

Each person returned by /api/events/people gets a `thumbnail_url` that points
DIRECTLY at R2 (not the backend proxy). This makes thumbnails behave exactly
like the hero image and the photo viewing_urls — the browser caches by unique
R2 URL, so switching the backend's .env (i.e. switching clients) produces an
entirely different URL and the memory-cache collision disappears.

Additionally, a per-process `_CACHE_NONCE` is appended as a query string.
Every backend restart gets a fresh nonce, so even two clients that somehow
shared the same R2 URL would still produce distinct thumbnail URLs across
`.env` swaps during local development.
"""
import os
import time
from fastapi import APIRouter, HTTPException, Response
from cache import get_event_info, get_people_sorted, get_cache
from config import r2_url

router = APIRouter()

# No-cache on the JSON response itself — the browser must revalidate every
# time so a newly-restarted backend immediately ships fresh thumbnail URLs.
_INFO_CACHE   = "no-store, no-cache, must-revalidate"
_PEOPLE_CACHE = "no-store, no-cache, must-revalidate"

# Per-process nonce. Changes on every backend restart — which is what happens
# whenever you change .env and reload. Brand-new URLs for every client switch.
_CACHE_NONCE = str(int(time.time()))
print(f"[events] CACHE_NONCE for this process = {_CACHE_NONCE}")


def _thumbnail_url_for(person_id: int) -> str:
    """
    Build a public R2 thumbnail URL with a per-boot cache-buster.

    Bypasses the backend proxy entirely — the browser fetches directly from R2,
    which means two different clients have two different r2.dev subdomains and
    therefore two different <img> src URLs. No path collision, no memory cache
    reuse.
    """
    base_url = r2_url("thumbnails", f"person_{person_id}.jpg")
    return f"{base_url}?v={_CACHE_NONCE}"


@router.get("/info")
async def get_event_info_route(response: Response):
    print("[events/info] → fetching event info from cache")
    info = get_event_info()
    if not info:
        print("[events/info] ❌ No event info in cache — R2 load may have failed")
        raise HTTPException(status_code=404, detail="No event data. Check R2 config.")
    print(f"[events/info] ✅ returning: total_photos={info.get('total_photos')}, "
          f"total_people={info.get('total_people')}, "
          f"ceremonies={info.get('ceremonies')}")
    response.headers["Cache-Control"] = _INFO_CACHE
    response.headers["Pragma"]        = "no-cache"
    response.headers["Expires"]       = "0"
    return info


@router.get("/people")
async def get_people(response: Response):
    print("[events/people] → fetching people list from cache")
    people = get_people_sorted()

    # Inject a fresh, direct-to-R2 thumbnail_url on every person.
    # (We copy rather than mutate the cached dict so subsequent requests
    # re-build with the current nonce — though the nonce is process-wide
    # constant, copying keeps the cache immutable which is less surprising.)
    enriched = []
    for p in people:
        pid = p.get("person_id")
        if pid is None:
            enriched.append(p)
            continue
        q = dict(p)
        q["thumbnail_url"] = _thumbnail_url_for(int(pid))
        enriched.append(q)

    print(f"[events/people] ✅ returning {len(enriched)} people "
          f"(each with thumbnail_url, nonce={_CACHE_NONCE})")
    if enriched:
        top3 = [f"#{p.get('person_id')} {p.get('name','?')} "
                f"({p.get('total_photos',0)} photos) → {p.get('thumbnail_url')}"
                for p in enriched[:3]]
        print(f"[events/people]    top 3 with thumb URLs:")
        for line in top3:
            print(f"[events/people]      {line}")

    response.headers["Cache-Control"] = _PEOPLE_CACHE
    response.headers["Pragma"]        = "no-cache"
    response.headers["Expires"]       = "0"
    return {"people": enriched, "total": len(enriched)}


@router.get("/ceremonies")
async def get_ceremonies(response: Response):
    print("[events/ceremonies] → fetching ceremonies from cache")
    info = get_event_info()
    if not info:
        print("[events/ceremonies] ❌ No event info in cache")
        raise HTTPException(status_code=404, detail="No event data")
    ceremonies = info.get("ceremonies", [])
    print(f"[events/ceremonies] ✅ returning {len(ceremonies)} ceremonies: {ceremonies}")
    response.headers["Cache-Control"] = _INFO_CACHE
    response.headers["Pragma"]        = "no-cache"
    response.headers["Expires"]       = "0"
    return {"ceremonies": ceremonies}
