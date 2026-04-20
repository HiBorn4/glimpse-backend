"""
Face Search Router — Guest selfie → matching photos.
"""
import time
from typing import Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, Request, Query
from config import SEARCH_THRESHOLD, SEARCH_TOP_K, r2_url, r2_url_with_subfolder

router = APIRouter()


def _proxy_urls(photos_by_ceremony: dict) -> dict:
    # Direct-to-R2 viewing URLs — no backend 302 round-trip.
    return {
        ceremony: [r2_url_with_subfolder("viewing", ceremony, fn) for fn in filenames]
        for ceremony, filenames in photos_by_ceremony.items()
    }


def _photo_urls(photo: dict) -> dict:
    fn  = photo.get("filename", "")
    cer = photo.get("ceremony", "")
    if (not cer or cer == "all") and "/" in photo.get("relative_path", ""):
        cer = photo["relative_path"].split("/")[0]

    photo = dict(photo)
    if cer and cer != "all":
        # viewing = direct R2; download = backend (needs attachment header)
        photo["viewing_url"]  = r2_url_with_subfolder("viewing", cer, fn)
        photo["download_url"] = f"/api/images/download/{cer}/{fn}"
    else:
        photo["viewing_url"]  = r2_url("viewing", fn)
        photo["download_url"] = f"/api/images/download/{fn}"
    return photo


@router.post("/selfie")
async def search_by_selfie(
    request:   Request,
    selfie:    UploadFile = File(..., description="Guest selfie image"),
    threshold: float      = Query(SEARCH_THRESHOLD, description="Min similarity (0.3–0.8)"),
    top_k:     int        = Query(SEARCH_TOP_K, description="Max matches to return"),
):
    t0 = time.perf_counter()
    print(f"[search/selfie] ── selfie upload received ─────────────────")
    print(f"[search/selfie]   filename={selfie.filename!r}, content_type={selfie.content_type!r}")
    print(f"[search/selfie]   threshold={threshold}, top_k={top_k}")

    engine = request.app.state.search_engine
    if engine is None:
        print("[search/selfie] ❌ search_engine is None — face search not loaded at startup")
        raise HTTPException(status_code=503, detail="Face search not available.")

    contents = await selfie.read()
    size_kb = len(contents) / 1024
    print(f"[search/selfie]   file size: {size_kb:.1f} KB ({len(contents)} bytes)")

    if not contents:
        print("[search/selfie] ❌ empty file uploaded")
        raise HTTPException(status_code=400, detail="Empty file uploaded")
    if len(contents) > 10 * 1024 * 1024:
        print(f"[search/selfie] ❌ file too large: {size_kb:.0f} KB > 10 MB")
        raise HTTPException(status_code=400, detail="File too large. Max 10MB.")

    print(f"[search/selfie] → running face detection + embedding search…")
    t_search = time.perf_counter()
    results = engine.search_by_selfie_bytes(contents, top_k=top_k, threshold=threshold)
    search_ms = (time.perf_counter() - t_search) * 1000
    print(f"[search/selfie]   search completed in {search_ms:.0f}ms")

    if results.get("error"):
        print(f"[search/selfie] ⚠️  no face found: {results['error']}")
        raise HTTPException(status_code=400, detail=results["error"])

    faces_detected = results.get("faces_detected", 0)
    matches        = results.get("matches", [])
    print(f"[search/selfie]   faces_detected={faces_detected}, matches_found={len(matches)}")

    for i, match in enumerate(matches):
        pid  = match.get("person_id")
        name = match.get("person_name", "?")
        sim  = match.get("similarity", 0)
        photos_total = match.get("total_photos", 0)
        print(f"[search/selfie]   match[{i}]: person_id={pid} name={name!r} "
              f"similarity={sim:.4f} total_photos={photos_total}")

        if "photos_by_ceremony" in match:
            pbc = match["photos_by_ceremony"]
            before_count = sum(len(v) for v in pbc.values())
            match["photos_by_ceremony"] = _proxy_urls(pbc)
            after_count = sum(len(v) for v in match["photos_by_ceremony"].values())
            print(f"[search/selfie]   match[{i}]: rewritten {before_count} photo URLs across "
                  f"{len(pbc)} ceremonies")

    total_ms = (time.perf_counter() - t0) * 1000
    print(f"[search/selfie] ✅ done — {total_ms:.0f}ms total")
    return results


@router.get("/person-photos")
async def get_person_photos(
    request:     Request,
    person_id:   int            = Query(...),
    ceremony:    Optional[str]  = Query(None),
    photo_type:  Optional[str]  = Query(None),
    with_people: Optional[str]  = Query(None, description="Additional person IDs comma-separated"),
):
    t0 = time.perf_counter()
    print(f"[search/person-photos] person_id={person_id}, ceremony={ceremony!r}, "
          f"photo_type={photo_type!r}, with_people={with_people!r}")

    engine = request.app.state.search_engine
    if engine is None:
        print("[search/person-photos] ❌ search_engine is None")
        raise HTTPException(status_code=503, detail="Search not available")

    person_ids = [person_id]
    if with_people:
        try:
            extra = [int(x.strip()) for x in with_people.split(",")]
            person_ids.extend(extra)
            print(f"[search/person-photos]   expanded person_ids: {person_ids}")
        except ValueError:
            print(f"[search/person-photos] ❌ invalid with_people: {with_people!r}")
            raise HTTPException(status_code=400, detail="with_people must be comma-separated integers")

    print(f"[search/person-photos] → querying engine for {len(person_ids)} person(s)…")
    photos = engine.get_photos_for_people(
        person_ids=person_ids,
        ceremony=ceremony,
        photo_type=photo_type,
    )
    print(f"[search/person-photos]   engine returned {len(photos)} raw photos")

    photos = [_photo_urls(p) for p in photos]
    total_ms = (time.perf_counter() - t0) * 1000
    print(f"[search/person-photos] ✅ returning {len(photos)} photos — {total_ms:.0f}ms")

    return {
        "person_ids": person_ids,
        "ceremony":   ceremony,
        "photo_type": photo_type,
        "total":      len(photos),
        "photos":     photos,
    }