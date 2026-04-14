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
# These must be set in your environment / .env file.
R2_ACCOUNT_ID  = os.getenv("R2_ACCOUNT_ID", "")
R2_ACCESS_KEY  = os.getenv("R2_ACCESS_KEY", "")
R2_SECRET_KEY  = os.getenv("R2_SECRET_KEY", "")
R2_BUCKET_NAME = os.getenv("R2_BUCKET_NAME", "pranay-weds-vaishnavi-2026")

# Public R2 URL — your Cloudflare custom domain or the R2 public bucket URL.
# e.g. https://photos.yourdomain.com
# All asset URLs (viewing, downloading, thumbnails) are built from this.
R2_PUBLIC_URL = os.getenv("R2_PUBLIC_URL", "")

# Top-level prefix inside the bucket.
# Bucket layout:
#   pranay-weds-vaishnavi-2026/photos/
#     thumbnails/   → person_0.jpg, person_1.jpg …
#     viewing/      → flat: IMG_001.jpg …       (display quality)
#     downloading/  → flat: IMG_001.jpg …       (full-res originals)
#     embeddings/   → person_centroids.pkl, all_face_embeddings.pkl
#     metadata/     → classification.json, face_cache_v5.pkl
R2_EVENT_PREFIX = os.getenv("R2_EVENT_PREFIX", "pranay-weds-vaishnavi-2026")

if not R2_PUBLIC_URL:
    raise RuntimeError(
        "R2_PUBLIC_URL must be set. "
        "Example: R2_PUBLIC_URL=https://pub-xxxx.r2.dev"
    )


def r2_url(folder: str, filename: str) -> str:
    """
    Build a public R2 URL.

    Folder is relative to photos/:
        thumbnails  → .../photos/thumbnails/{filename}
        viewing     → .../photos/viewing/{filename}
        downloading → .../photos/downloading/{filename}
        embeddings  → .../photos/embeddings/{filename}
        metadata    → .../photos/metadata/{filename}
    """
    base   = R2_PUBLIC_URL.rstrip("/")
    return f"{base}/photos/{folder}/{filename}"


def r2_url_with_subfolder(folder: str, subfolder: str, filename: str) -> str:
    """
    Build a public R2 URL with a subfolder (e.g. ceremony name).

    Example:
        r2_url_with_subfolder("viewing", "wedding", "IMG_001.jpg")
        → .../photos/viewing/wedding/IMG_001.jpg
    """
    base = R2_PUBLIC_URL.rstrip("/")
    return f"{base}/photos/{folder}/{subfolder}/{filename}"
