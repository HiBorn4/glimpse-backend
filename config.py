"""
Configuration for Wedding Photo AI Backend
Zero local file dependencies — everything served from R2.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── Server ────────────────────────────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

CORS_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    os.getenv("FRONTEND_URL", "http://localhost:3000"),
]

# ── Search tuning ─────────────────────────────────────────────────────────────
SEARCH_THRESHOLD = float(os.getenv("SEARCH_THRESHOLD", "0.35"))
SEARCH_TOP_K     = int(os.getenv("SEARCH_TOP_K", "3"))
USE_GPU          = os.getenv("USE_GPU", "false").lower() == "true"

# ── R2 Cloudflare Configuration ───────────────────────────────────────────────
R2_ACCOUNT_ID  = os.getenv("R2_ACCOUNT_ID", "")
R2_ACCESS_KEY  = os.getenv("R2_ACCESS_KEY", "")
R2_SECRET_KEY  = os.getenv("R2_SECRET_KEY", "")
R2_BUCKET_NAME = os.getenv("R2_BUCKET_NAME", "")

# Public R2 URL — your Cloudflare custom domain or the per-bucket r2.dev URL.
# e.g. https://pub-xxxxxxxxxxxxxxxxxxxxxxxx.r2.dev
#
# Two deployment modes are supported:
#
# 1. ONE BUCKET PER CLIENT with its own pub-*.r2.dev URL  (your current setup)
#    The bucket name is the subdomain — the key path inside starts at "photos/".
#    Leave R2_EVENT_PREFIX empty.
#    Final URL: {R2_PUBLIC_URL}/photos/{folder}/{filename}
#
# 2. ONE SHARED BUCKET with a top-level folder per event
#    Set R2_EVENT_PREFIX to that folder name.
#    Final URL: {R2_PUBLIC_URL}/{R2_EVENT_PREFIX}/photos/{folder}/{filename}
R2_PUBLIC_URL = os.getenv("R2_PUBLIC_URL", "")

# Top-level prefix inside the bucket. OPTIONAL — leave empty when each client has
# its own bucket (mode 1 above). Set only when multiple events live in one bucket.
R2_EVENT_PREFIX = os.getenv("R2_EVENT_PREFIX", "").strip("/")

if not R2_PUBLIC_URL:
    raise RuntimeError(
        "R2_PUBLIC_URL must be set. "
        "Example: R2_PUBLIC_URL=https://pub-xxxx.r2.dev"
    )

# R2_BUCKET_NAME and R2_EVENT_PREFIX are no longer required at import time —
# they're only meaningful for signed-URL writes (bucket name) or multi-event
# buckets (event prefix). Per-bucket r2.dev URLs already encode the bucket.

print(f"[config] R2_PUBLIC_URL   = {R2_PUBLIC_URL}")
print(f"[config] R2_EVENT_PREFIX = {R2_EVENT_PREFIX!r} "
      f"({'single-bucket-per-client mode' if not R2_EVENT_PREFIX else 'shared-bucket mode'})")


def _join(*parts: str) -> str:
    """Join URL path segments, skipping empty ones, with exactly one slash between each."""
    clean = [p.strip("/") for p in parts if p and p.strip("/")]
    return "/".join(clean)


def r2_url(folder: str, filename: str) -> str:
    """
    Build a public R2 URL.

    When R2_EVENT_PREFIX is empty (one bucket per client):
        {R2_PUBLIC_URL}/photos/{folder}/{filename}

    When R2_EVENT_PREFIX is set (shared bucket, folder per event):
        {R2_PUBLIC_URL}/{R2_EVENT_PREFIX}/photos/{folder}/{filename}
    """
    base = R2_PUBLIC_URL.rstrip("/")
    tail = _join("photos", folder, filename)
    return f"{base}/{tail}"


def r2_url_with_subfolder(folder: str, subfolder: str, filename: str) -> str:
    """
    Build a public R2 URL with a subfolder (e.g. ceremony name).

    When R2_EVENT_PREFIX is empty:
        {R2_PUBLIC_URL}/photos/{folder}/{subfolder}/{filename}

    When R2_EVENT_PREFIX is set:
        {R2_PUBLIC_URL}/{R2_EVENT_PREFIX}/photos/{folder}/{subfolder}/{filename}
    """
    base = R2_PUBLIC_URL.rstrip("/")
    tail = _join(R2_EVENT_PREFIX, "photos", folder, subfolder, filename)
    return f"{base}/{tail}"
