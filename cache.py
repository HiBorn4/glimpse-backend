"""
In-memory classification cache — R2 edition.
Loads classification.json directly from R2 at startup.
"""
import json
import hashlib
import time
import httpx
from typing import Optional

from config import r2_url

_cache: Optional[dict] = None
_photo_list: Optional[list] = None

_person_photo_sets:   dict[int, set[str]] = {}
_ceremony_photo_sets: dict[str, set[str]] = {}
_type_photo_sets:     dict[str, set[str]] = {}

_result_cache: dict[str, tuple[str, list]] = {}
_people_sorted: Optional[list] = None
_event_info: Optional[dict] = None


async def load_classification_from_r2() -> bool:
    global _cache, _photo_list, _people_sorted, _event_info
    global _person_photo_sets, _ceremony_photo_sets, _type_photo_sets, _result_cache

    url = r2_url("metadata", "classification.json")
    print(f"[cache] Fetching classification.json from R2…")
    print(f"[cache]   URL: {url}")

    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            resp = await client.get(url)
            fetch_ms = (time.perf_counter() - t0) * 1000
            print(f"[cache]   HTTP {resp.status_code} in {fetch_ms:.0f}ms — "
                  f"content-length={resp.headers.get('content-length', 'unknown')} bytes")
            resp.raise_for_status()
            _cache = resp.json()
        except httpx.HTTPStatusError as e:
            print(f"[cache] ❌ R2 returned HTTP {e.response.status_code} for classification.json")
            print(f"[cache]    URL was: {url}")
            return False
        except httpx.TimeoutException:
            print(f"[cache] ❌ Timeout fetching classification.json (>60s)")
            return False
        except Exception as e:
            print(f"[cache] ❌ Unexpected error: {type(e).__name__}: {e}")
            return False

    photos_dict = _cache.get("photos", {})
    people_dict = _cache.get("people", {})
    print(f"[cache]   Raw data: {len(photos_dict)} photos, {len(people_dict)} people in JSON")

    # Build indexes
    t_idx = time.perf_counter()
    _photo_list = list(photos_dict.values())
    _person_photo_sets.clear()
    _ceremony_photo_sets.clear()
    _type_photo_sets.clear()
    _result_cache.clear()

    for filename, record in photos_dict.items():
        cer   = record.get("ceremony", "")
        ptype = record.get("photo_type", "")
        _ceremony_photo_sets.setdefault(cer, set()).add(filename)
        _type_photo_sets.setdefault(ptype, set()).add(filename)
        for pid in record.get("people", []):
            _person_photo_sets.setdefault(pid, set()).add(filename)

    idx_ms = (time.perf_counter() - t_idx) * 1000
    print(f"[cache]   Index built in {idx_ms:.1f}ms:")
    print(f"[cache]     person sets  : {len(_person_photo_sets)} people")
    print(f"[cache]     ceremony sets: {len(_ceremony_photo_sets)} → {list(_ceremony_photo_sets.keys())}")
    print(f"[cache]     type sets    : {len(_type_photo_sets)} → {list(_type_photo_sets.keys())}")

    _people_sorted = sorted(
        people_dict.values(),
        key=lambda p: p.get("total_photos", 0),
        reverse=True,
    )

    _event_info = _cache.get("event_info", {
        "total_photos": len(_photo_list),
        "total_people": len(people_dict),
        "ceremonies":   list(_ceremony_photo_sets.keys()),
    })

    print(f"[cache] ✅ Ready: {len(_photo_list)} photos | "
          f"{len(_person_photo_sets)} people | "
          f"{len(_ceremony_photo_sets)} ceremonies")
    if _people_sorted:
        top3 = [f"{p.get('name','?')}({p.get('total_photos',0)})" for p in _people_sorted[:3]]
        print(f"[cache]   Top people by photo count: {top3}")

    return True


def get_cache() -> Optional[dict]:
    return _cache


def get_event_info() -> Optional[dict]:
    return _event_info


def get_people_sorted() -> list:
    return _people_sorted or []


def _make_query_key(
    person_ids:  Optional[set],
    ceremony:    Optional[str],
    photo_type:  Optional[str],
    exact_only:  bool,
    orientation: Optional[str],
) -> str:
    parts = [
        ",".join(str(p) for p in sorted(person_ids)) if person_ids else "",
        ceremony    or "",
        photo_type  or "",
        "1" if exact_only else "0",
        orientation or "",
    ]
    return "|".join(parts)


def get_photos_filtered(
    person_ids:  Optional[set[int]] = None,
    ceremony:    Optional[str]      = None,
    photo_type:  Optional[str]      = None,
    exact_only:  bool               = False,
    orientation: Optional[str]      = None,
) -> tuple[list[dict], str]:
    if not _cache:
        print("[cache/filter] ❌ cache is empty — startup may have failed")
        return [], "empty"

    query_key = _make_query_key(person_ids, ceremony, photo_type, exact_only, orientation)

    if query_key in _result_cache:
        etag, cached = _result_cache[query_key]
        print(f"[cache/filter] ✅ cache HIT for key={query_key!r} → {len(cached)} photos (etag={etag})")
        return cached, etag

    print(f"[cache/filter] cache MISS for key={query_key!r} — computing…")
    t0 = time.perf_counter()

    photos_dict = _cache["photos"]
    result_set: Optional[set[str]] = None

    if person_ids:
        for pid in person_ids:
            pid_set = _person_photo_sets.get(pid, set())
            print(f"[cache/filter]   person_id={pid} → {len(pid_set)} photos in index")
            result_set = pid_set.copy() if result_set is None else result_set & pid_set
        print(f"[cache/filter]   after person intersection: {len(result_set or set())} photos")

        if exact_only and result_set:
            before = len(result_set)
            result_set = {
                fn for fn in result_set
                if set(photos_dict[fn].get("people", [])) == person_ids
            }
            print(f"[cache/filter]   exact_only filter: {before} → {len(result_set)} photos")

    if ceremony:
        cer_set    = _ceremony_photo_sets.get(ceremony, set())
        print(f"[cache/filter]   ceremony={ceremony!r} → {len(cer_set)} photos in index")
        result_set = cer_set.copy() if result_set is None else result_set & cer_set
        print(f"[cache/filter]   after ceremony intersection: {len(result_set)} photos")

    if photo_type:
        type_set   = _type_photo_sets.get(photo_type, set())
        print(f"[cache/filter]   photo_type={photo_type!r} → {len(type_set)} photos in index")
        result_set = type_set.copy() if result_set is None else result_set & type_set
        print(f"[cache/filter]   after type intersection: {len(result_set)} photos")

    if result_set is None:
        photo_list = _photo_list or []
        print(f"[cache/filter]   no filters applied → returning all {len(photo_list)} photos")
    else:
        photo_list = [photos_dict[fn] for fn in result_set if fn in photos_dict]

    if orientation:
        before = len(photo_list)
        photo_list = [p for p in photo_list if p.get("orientation") == orientation]
        print(f"[cache/filter]   orientation={orientation!r}: {before} → {len(photo_list)} photos")

    fingerprint = "|".join(sorted(p.get("filename", "") for p in photo_list))
    etag = hashlib.md5(fingerprint.encode(), usedforsecurity=False).hexdigest()[:16]

    _result_cache[query_key] = (etag, photo_list)
    compute_ms = (time.perf_counter() - t0) * 1000
    print(f"[cache/filter] ✅ computed {len(photo_list)} photos in {compute_ms:.1f}ms, etag={etag}")
    print(f"[cache/filter]   result_cache now has {len(_result_cache)} entries")

    return photo_list, etag


def invalidate_cache():
    global _result_cache
    before = len(_result_cache)
    _result_cache.clear()
    print(f"[cache] 🔄 Result cache invalidated — cleared {before} entries")
