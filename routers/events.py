"""Events Router — pre-cached event info and people list."""
from fastapi import APIRouter, HTTPException, Response
from cache import get_event_info, get_people_sorted, get_cache

router = APIRouter()

_INFO_CACHE   = "public, max-age=300, stale-while-revalidate=60"
_PEOPLE_CACHE = "public, max-age=300, stale-while-revalidate=60"


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
    return info


@router.get("/people")
async def get_people(response: Response):
    print("[events/people] → fetching people list from cache")
    people = get_people_sorted()
    print(f"[events/people] ✅ returning {len(people)} people")
    if people:
        top3 = [f"#{p.get('person_id')} {p.get('name','?')} ({p.get('total_photos',0)} photos)"
                for p in people[:3]]
        print(f"[events/people]    top 3: {top3}")
    response.headers["Cache-Control"] = _PEOPLE_CACHE
    return {"people": people, "total": len(people)}


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
    return {"ceremonies": ceremonies}
