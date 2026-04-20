"""
R2 Face Search Engine — HARDENED
================================

Loads person_centroids.pkl, all_face_embeddings.pkl, and classification.json
directly from Cloudflare R2. Zero local file dependencies.

This rewrite fixes the "No face detected" / 30-second-timeout failures on
real-world selfies — especially high-megapixel DSLR portraits where the face
is small relative to the frame and the lighting is heavily warm/backlit.

Key changes vs. the previous version
────────────────────────────────────
1.  EXIF-AWARE DECODING via PIL's ImageOps.exif_transpose. cv2.imdecode
    ignores EXIF orientation, so iPhone/Android portraits arrive sideways
    and RetinaFace misses them. We now auto-rotate before detection.

2.  SHORTER, SMARTER CASCADE. The old code ran up to 10 sequential detection
    attempts on a 1600x2400 image; each attempt on CPU cost 2–3 s, so a
    full miss blew the 30 s request budget. The new cascade uses 4–5
    targeted passes with a hard wall-clock budget (default 12 s).

3.  CORRECT REGION CROPPING. The old "top 55 %" crop assumed a head-shot
    selfie, but for seated full-body portraits (lehenga, saree, formal
    wedding shots) the face is mid-frame. We now use a 3-tile vertical
    sweep (top, middle, bottom thirds) plus the full frame.

4.  RELAXED DETECTION SCORE FLOOR (0.15, was 0.30). RetinaFace scores
    drop under dramatic warm lighting; 0.30 was rejecting real faces.
    We also RANK candidates by (score × face_area) so a real face with
    score 0.22 beats a background smudge with score 0.18.

5.  ROBUST EMBEDDING NORMALISATION. `normed_embedding` is sometimes
    zero-vectored when the face crop is extreme; we fall back to
    L2-normalising `embedding` ourselves to avoid NaN cosine sims.

6.  TOLERANT PICKLE SCHEMAS. person_centroids.pkl and
    all_face_embeddings.pkl come in several shapes across pipeline
    versions — dict-of-arrays, dict-of-lists, list-of-dicts. We sniff
    and handle all of them.

7.  WALL-CLOCK BUDGET. A top-level `deadline` stops the cascade early
    if we're close to the 30 s request ceiling, so the client always
    sees either a real answer or a clean "no face" — never a hang.
"""

from __future__ import annotations

import io
import pickle
import time
import asyncio
import warnings
from typing import Any, Optional

import httpx
import numpy as np
import cv2
from PIL import Image, ImageOps
from sklearn.preprocessing import normalize

import insightface
from insightface.app import FaceAnalysis

warnings.filterwarnings("ignore")

from config import r2_url


# ── Tuning knobs ──────────────────────────────────────────────────────────────

# Photo-type classifications (reused by get_photos_for_people)
SOLO_MAX_EFFECTIVE_PEOPLE = 1
GROUP_MIN_FACES = 5

# Minimum det_score to accept a detection. 0.15 is generous — RetinaFace scores
# drop on dramatic / warm / backlit lighting, and the downstream cosine match
# is the real quality gate anyway. A false-positive face with garbage embedding
# will produce low cosine sim and be filtered by `threshold` in the router.
MIN_DET_SCORE = 0.15

# Minimum face size (pixels, long edge of bbox) to accept. Anything smaller
# is usually a background face or noise — its embedding won't be reliable.
MIN_FACE_PIXELS = 40

# Centroid similarity below this → also scan all_face_embeddings for a
# deeper per-photo match.
CENTROID_FALLBACK_THRESHOLD = 0.45

# Hard wall-clock budget for the detection cascade (seconds). The router has
# a ~30 s ceiling; we leave ~15 s headroom for network + embedding matching
# + URL rewriting + response encoding.
DETECT_BUDGET_SECS = 12.0

# Cap the long edge of the uploaded image before the detector ever sees it.
# 2000 px is plenty — a face 8 % of frame (typical full-body portrait) is
# still 160 px, and RetinaFace detects reliably down to ~30 px.
MAX_DIM_FOR_DETECTION = 2000


# ── Async helper ──────────────────────────────────────────────────────────────

async def _gather(*coros):
    return await asyncio.gather(*coros)


# ── Image loading (EXIF-aware) ────────────────────────────────────────────────

def _decode_exif_aware(image_bytes: bytes) -> Optional[np.ndarray]:
    """
    Decode image bytes into a BGR numpy array, honouring EXIF orientation.

    cv2.imdecode does NOT read EXIF — iPhone/Android portraits carry
    orientation=6 (rotate 270° CW to display upright), so faces come out
    sideways and RetinaFace misses them. We route through PIL first, apply
    ImageOps.exif_transpose, then convert to BGR for cv2/insightface.
    """
    try:
        pil_img = Image.open(io.BytesIO(image_bytes))
        # Apply EXIF rotation in-place (returns a new image, original untouched)
        pil_img = ImageOps.exif_transpose(pil_img)
        # Convert palette / RGBA / grayscale → RGB
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        rgb = np.array(pil_img)                      # HxWx3 RGB
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)   # HxWx3 BGR (for cv2/insightface)
        return bgr
    except Exception as exc:
        print(f"  [decode] PIL decode failed ({exc}); falling back to cv2.imdecode")
        try:
            arr = np.frombuffer(image_bytes, np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception as exc2:
            print(f"  [decode] cv2 fallback also failed: {exc2}")
            return None


def _clahe(src: np.ndarray) -> np.ndarray:
    """CLAHE contrast enhancement on the L channel — helps warm/backlit shots."""
    lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return cv2.cvtColor(cv2.merge([cl.apply(l), a, b]), cv2.COLOR_LAB2BGR)


def _resize_long_edge(src: np.ndarray, long_edge: int) -> np.ndarray:
    """Resize so max(h,w) == long_edge. No upscale if already smaller."""
    h, w = src.shape[:2]
    le = max(h, w)
    if le <= long_edge:
        return src
    scale = long_edge / le
    return cv2.resize(src, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def _safe_normed_embedding(face) -> Optional[np.ndarray]:
    """
    Return a unit-length 1xD embedding for a detected face.

    Prefers face.normed_embedding; falls back to L2-normalising
    face.embedding; returns None if neither is usable.
    """
    emb = getattr(face, "normed_embedding", None)
    if emb is not None:
        arr = np.asarray(emb, dtype=np.float32).reshape(1, -1)
        n = np.linalg.norm(arr)
        if n > 1e-6 and np.isfinite(n):
            # normed_embedding is already unit-length; guard against edge cases
            return arr / n if abs(n - 1.0) > 1e-3 else arr

    emb = getattr(face, "embedding", None)
    if emb is not None:
        arr = np.asarray(emb, dtype=np.float32).reshape(1, -1)
        n = np.linalg.norm(arr)
        if n > 1e-6 and np.isfinite(n):
            return arr / n

    return None


def _face_area(face) -> float:
    """Area of the face bbox in pixels (of whatever image it was detected on)."""
    try:
        x1, y1, x2, y2 = face.bbox
        return max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))
    except Exception:
        return 0.0


def _face_long_edge(face) -> float:
    try:
        x1, y1, x2, y2 = face.bbox
        return max(float(x2 - x1), float(y2 - y1))
    except Exception:
        return 0.0


# ── Main engine ───────────────────────────────────────────────────────────────

class R2FaceSearchEngine:
    """
    Face search engine that loads all data from Cloudflare R2.

    R2 paths:
        photos/embeddings/person_centroids.pkl     (required)
        photos/embeddings/all_face_embeddings.pkl  (optional, for fallback)
        photos/metadata/classification.json         (required)
    """

    def __init__(self, use_gpu: bool = False):
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if use_gpu else ["CPUExecutionProvider"]
        )

        # ONE FaceAnalysis instance is enough — we change det_size on demand
        # via prepare(). This saves ~400 MB of RAM vs. the previous version's
        # three separate instances.
        self.app = FaceAnalysis(name="buffalo_l", providers=providers)
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self._current_det_size = (640, 640)

        # Populated by load_from_r2()
        self.centroids:       dict  = {}
        self.person_ids:      list  = []
        self.centroid_matrix: Optional[np.ndarray] = None
        self.classification:  dict  = {}
        self.all_embeddings:  Any   = None   # shape-tolerant

    # ── det_size switching ────────────────────────────────────────────────────

    def _set_det_size(self, size: tuple[int, int]):
        """Re-prepare the detector at a new det_size only if it changed."""
        if size == self._current_det_size:
            return
        self.app.prepare(ctx_id=0, det_size=size)
        self._current_det_size = size

    # ── Startup: load embeddings from R2 ──────────────────────────────────────

    async def load_from_r2(self):
        """
        Fetch centroids + classification (blocking — required for search),
        then all_face_embeddings (best-effort, non-blocking fallback data).
        """
        centroids_url      = r2_url("embeddings", "person_centroids.pkl")
        classification_url = r2_url("metadata",   "classification.json")
        all_emb_url        = r2_url("embeddings", "all_face_embeddings.pkl")

        print(f"  Fetching centroids:      {centroids_url}")
        print(f"  Fetching classification: {classification_url}")

        async with httpx.AsyncClient(timeout=120.0) as client:
            centroids_resp, class_resp = await _gather(
                client.get(centroids_url),
                client.get(classification_url),
            )

        centroids_resp.raise_for_status()
        class_resp.raise_for_status()

        raw_centroids = pickle.loads(centroids_resp.content)
        self.classification = class_resp.json()

        # Normalise the centroids pickle shape. We accept:
        #   A) {person_id: np.ndarray}                     ← most common
        #   B) {person_id: {"centroid": np.ndarray, ...}}
        #   C) {"centroids": {person_id: np.ndarray}, ...}
        if isinstance(raw_centroids, dict) and "centroids" in raw_centroids:
            raw_centroids = raw_centroids["centroids"]

        flat: dict[int, np.ndarray] = {}
        for pid, val in raw_centroids.items():
            try:
                pid_int = int(pid)
            except (TypeError, ValueError):
                continue
            if isinstance(val, dict):
                vec = val.get("centroid") or val.get("embedding") or val.get("vector")
            else:
                vec = val
            if vec is None:
                continue
            arr = np.asarray(vec, dtype=np.float32).ravel()
            if arr.size == 0 or not np.isfinite(arr).all():
                continue
            flat[pid_int] = arr

        if not flat:
            raise RuntimeError("person_centroids.pkl did not yield any usable centroids")

        self.centroids = flat
        self.person_ids = sorted(flat.keys())

        # Stack and L2-normalise once at startup so search is pure matmul.
        mat = np.stack([flat[pid] for pid in self.person_ids])
        self.centroid_matrix = normalize(mat)

        print(f"  ✅ FaceSearch ready: {len(self.person_ids)} people indexed "
              f"(embedding dim = {self.centroid_matrix.shape[1]})")

        # Optional: deeper per-face embeddings. Best-effort only.
        try:
            print(f"  Fetching all_face_embeddings (optional): {all_emb_url}")
            async with httpx.AsyncClient(timeout=300.0) as client:
                emb_resp = await client.get(all_emb_url)
            if emb_resp.status_code == 200:
                self.all_embeddings = pickle.loads(emb_resp.content)
                try:
                    if isinstance(self.all_embeddings, dict):
                        total = sum(
                            len(v) if hasattr(v, "__len__") else 1
                            for v in self.all_embeddings.values()
                        )
                    else:
                        total = len(self.all_embeddings)
                except Exception:
                    total = -1
                print(f"  ✅ all_face_embeddings loaded: {total} records")
            else:
                print(f"  ⚠️  all_face_embeddings HTTP {emb_resp.status_code} — centroid-only mode")
        except Exception as exc:
            print(f"  ⚠️  all_face_embeddings load failed: {exc} — centroid-only mode")

    # ── Detection (the part that was failing) ─────────────────────────────────

    def _detect_faces(self, img: np.ndarray, deadline: float) -> list:
        """
        Multi-pass face detection with a wall-clock budget.

        Returns a list of Face objects (may be empty). The returned faces are
        ranked so `max(faces, key=...)` picks the most plausible one.

        Strategy
        ────────
        Each pass calls self.app.get(variant). Variants sweep across:
          • det_size       — 640 (fast), 1024 (rescues mid-frame faces)
          • image size     — full-res and 1280-long-edge downscale
          • image regions  — full / top-third / middle-third / bottom-third
          • enhancements   — CLAHE and brightness boost for warm lighting

        We stop as soon as we have a face that's unambiguously good
        (score ≥ 0.55 AND long-edge ≥ 80 px) to save time. Otherwise we run
        the cascade and return the best candidate across all passes, ranked
        by (det_score × face_long_edge) so a plausible mid-confidence detect
        beats a tiny background blip.
        """
        all_candidates: list = []
        h, w = img.shape[:2]
        print(f"  [detect] input image: {w}x{h}")

        def _budget_left() -> float:
            return max(0.0, deadline - time.time())

        def _collect(faces, tag: str):
            """Keep faces that pass minimum thresholds and log."""
            kept = []
            for f in faces:
                score = float(getattr(f, "det_score", 0.0))
                le = _face_long_edge(f)
                if score < MIN_DET_SCORE:
                    continue
                if le < MIN_FACE_PIXELS:
                    continue
                if _safe_normed_embedding(f) is None:
                    continue
                kept.append(f)
            if kept:
                best = max(kept, key=lambda f: f.det_score)
                print(f"  [detect] {tag}: +{len(kept)} face(s), best score={best.det_score:.3f}, "
                      f"long_edge={_face_long_edge(best):.0f}px")
            all_candidates.extend(kept)
            return kept

        def _have_strong_hit() -> bool:
            """Early-exit criterion: at least one clearly-valid face."""
            for f in all_candidates:
                if f.det_score >= 0.55 and _face_long_edge(f) >= 80:
                    return True
            return False

        def _try(variant: np.ndarray, tag: str):
            """Run one detection attempt. Returns True to short-circuit cascade."""
            if _budget_left() < 1.5:
                print(f"  [detect] ⏱ budget exhausted, skipping {tag}")
                return False
            t0 = time.time()
            try:
                faces = self.app.get(variant)
            except Exception as exc:
                print(f"  [detect] {tag}: app.get crashed: {exc}")
                return False
            dt = time.time() - t0
            print(f"  [detect] {tag}: app.get took {dt*1000:.0f}ms, {len(faces)} raw face(s)")
            _collect(faces, tag)
            return _have_strong_hit()

        # ── Pass A: det_size 640 on a 1280-long-edge downscale ────────────────
        # This is the sweet spot: fast (~1 s CPU) and catches faces that are
        # ≥ 8 % of the frame, which covers selfies, head-shots, and
        # reasonably-framed full-body portraits.
        self._set_det_size((640, 640))
        img_1280 = _resize_long_edge(img, 1280)
        if _try(img_1280, "A[640@1280]"):
            return all_candidates

        # ── Pass B: det_size 1024 on the full-res image ───────────────────────
        # For full-body portraits where the face is ~5-8 % of the frame,
        # a larger det_size finds it at original resolution.
        self._set_det_size((1024, 1024))
        if _try(img, "B[1024@full]"):
            return all_candidates

        # ── Pass C: CLAHE + brightness on the 1280 downscale ──────────────────
        # Rescues warm/backlit shots (chandelier lighting, golden-hour,
        # sangeet / reception interiors).
        if _budget_left() > 2.5:
            self._set_det_size((640, 640))
            enhanced = cv2.convertScaleAbs(_clahe(img_1280), alpha=1.2, beta=15)
            if _try(enhanced, "C[640@1280+CLAHE+bright]"):
                return all_candidates

        # ── Pass D: vertical thirds — scan top / middle / bottom of the frame ─
        # For portrait-orientation full-body shots the face lives in whichever
        # third the subject's head happens to land in. We detect in each
        # third at det_size 640 — cheap, because each crop is 1/3 the area.
        # This REPLACES the old "top 55 %" hard-coded crop which literally
        # cropped the face off full-body portraits.
        if _budget_left() > 3.0 and h >= w:  # only bother for portrait images
            self._set_det_size((640, 640))
            third = h // 3
            # Overlap the thirds by 10 % so faces on the boundary aren't split.
            overlap = int(0.1 * third)
            slices = [
                ("top",    slice(0,                     min(h, third + overlap))),
                ("middle", slice(max(0, third - overlap), min(h, 2 * third + overlap))),
                ("bottom", slice(max(0, 2 * third - overlap), h)),
            ]
            for name, sl in slices:
                if _budget_left() < 1.5:
                    break
                crop = img[sl, :, :]
                crop_1280 = _resize_long_edge(crop, 1280)
                if _try(crop_1280, f"D[640@1280-{name}]"):
                    return all_candidates

        # ── Pass E: last-resort larger det_size on downscaled image ───────────
        if _budget_left() > 2.5:
            self._set_det_size((1280, 1280))
            if _try(img_1280, "E[1280@1280]"):
                return all_candidates

        return all_candidates

    # ── Public search entry points ────────────────────────────────────────────

    def search_by_selfie_bytes(self, image_bytes: bytes, top_k: int = 3, threshold: float = 0.1):
        """Accept raw bytes from an HTTP upload. Handles EXIF + size capping."""
        t0 = time.time()
        deadline = t0 + DETECT_BUDGET_SECS

        img = _decode_exif_aware(image_bytes)
        if img is None:
            return {"error": "Could not decode image — is this a valid JPEG/PNG?", "matches": []}

        h, w = img.shape[:2]
        print(f"  [selfie] decoded: {w}x{h} (EXIF-corrected)")

        # Cap the long edge before detection. A 4000x6000 DSLR portrait is
        # overkill for matching — the embedding is built from a 112x112 crop.
        if max(h, w) > MAX_DIM_FOR_DETECTION:
            img = _resize_long_edge(img, MAX_DIM_FOR_DETECTION)
            print(f"  [selfie] resized for detection: {img.shape[1]}x{img.shape[0]}")

        return self.search_by_selfie(img, top_k=top_k, threshold=threshold, deadline=deadline)

    def search_by_selfie(self, selfie_image, top_k: int = 3, threshold: float = 0.1,
                         deadline: Optional[float] = None):
        """
        Core search. Accepts a BGR numpy array (already EXIF-corrected and
        size-capped) or a file path string.
        """
        if self.centroid_matrix is None:
            return {"error": "Face search not loaded", "matches": []}

        if isinstance(selfie_image, str):
            # Direct path: load via PIL for EXIF safety, not cv2.imread.
            try:
                with open(selfie_image, "rb") as f:
                    selfie_image = _decode_exif_aware(f.read())
            except Exception as exc:
                return {"error": f"Could not read image: {exc}", "matches": []}

        if selfie_image is None:
            return {"error": "Could not read image", "matches": []}

        if deadline is None:
            deadline = time.time() + DETECT_BUDGET_SECS

        # Detect
        faces = self._detect_faces(selfie_image, deadline=deadline)

        if not faces:
            return {
                "error": (
                    "No face detected in the photo. Try a clearer shot of your face — "
                    "better lighting, facing the camera, and your full face visible in frame."
                ),
                "faces_detected": 0,
                "matches": [],
            }

        # Rank candidates: (score × long_edge) — penalises both tiny faces
        # and low-confidence background detections. Among equally good
        # candidates we prefer the single largest face in the image, which
        # on a selfie is the one we want.
        faces.sort(key=lambda f: f.det_score * _face_long_edge(f), reverse=True)
        best_face = faces[0]
        print(f"  [search] chose best face: score={best_face.det_score:.3f}, "
              f"long_edge={_face_long_edge(best_face):.0f}px, bbox={best_face.bbox}")

        query_emb = _safe_normed_embedding(best_face)
        if query_emb is None:
            return {
                "error": "Face detected but embedding extraction failed — try another photo.",
                "faces_detected": len(faces),
                "matches": [],
            }

        # Cosine similarity against all person centroids (single matmul).
        similarities = (query_emb @ self.centroid_matrix.T).flatten()
        top_indices = np.argsort(similarities)[::-1][:max(top_k, 5)]

        top_sim = float(similarities[top_indices[0]]) if len(top_indices) else 0.0
        print(f"  [search] top centroid sim={top_sim:.3f}, "
              f"top5={[round(float(similarities[i]), 3) for i in top_indices[:5]]}")

        pci = self.classification.get("person_ceremony_index", {})
        people_dict = self.classification.get("people", {})

        matches: list[dict] = []
        for idx in top_indices[:top_k]:
            sim = float(similarities[idx])
            if sim < threshold:
                continue
            person_id = self.person_ids[idx]
            person_key = str(person_id)
            person_info = people_dict.get(person_key, {})
            matches.append({
                "person_id":          person_id,
                "similarity":         round(sim, 4),
                "person_name":        person_info.get("name", f"Person_{person_id}"),
                "total_photos":       person_info.get("total_photos", 0),
                "ceremonies":         person_info.get("ceremonies", {}),
                "photos_by_ceremony": pci.get(person_key, {}),
            })

        # Weak top match → run per-embedding fallback
        if top_sim < CENTROID_FALLBACK_THRESHOLD and self.all_embeddings is not None:
            print(f"  [search] top centroid sim={top_sim:.3f} < {CENTROID_FALLBACK_THRESHOLD} "
                  f"— running per-embedding fallback")
            fallback = self._search_all_embeddings(query_emb, top_k=top_k, threshold=threshold)
            if fallback:
                existing = {m["person_id"] for m in matches}
                for fm in fallback:
                    if fm["person_id"] not in existing:
                        matches.append(fm)
                    else:
                        for m in matches:
                            if m["person_id"] == fm["person_id"] and fm["similarity"] > m["similarity"]:
                                m["similarity"] = fm["similarity"]
                matches.sort(key=lambda m: m["similarity"], reverse=True)
                matches = matches[:top_k]

        return {
            "error":          None,
            "faces_detected": len(faces),
            "matches":        matches,
        }

    # ── Per-embedding fallback search ─────────────────────────────────────────

    def _search_all_embeddings(self, query_emb: np.ndarray, top_k: int, threshold: float):
        """
        Compare query_emb against every individual face embedding and aggregate
        by person_id. Returns the best match per person, ranked.

        Supports multiple pickle schemas:
          A) {person_id (int): [emb, emb, ...]}
          B) {photo_filename (str): [(person_id, emb), (person_id, emb), ...]}
          C) {photo_filename (str): [{"person_id": int, "embedding": arr}, ...]}
          D) list of {"person_id": int, "embedding": arr}
        """
        if self.all_embeddings is None:
            return []

        emb_data = self.all_embeddings
        person_best: dict[int, float] = {}

        # Helper to compare a single candidate embedding and update person_best
        def _score(pid_raw, emb_raw):
            try:
                pid = int(pid_raw)
                arr = np.asarray(emb_raw, dtype=np.float32).ravel()
                n = np.linalg.norm(arr)
                if n < 1e-6 or not np.isfinite(n):
                    return
                e = (arr / n).reshape(1, -1)
                sim = float((query_emb @ e.T).flatten()[0])
                if sim > person_best.get(pid, -1.0):
                    person_best[pid] = sim
            except Exception:
                return

        try:
            # Schema D: flat list of dicts
            if isinstance(emb_data, list):
                for rec in emb_data:
                    if isinstance(rec, dict):
                        pid = rec.get("person_id")
                        emb = rec.get("embedding") or rec.get("emb") or rec.get("normed_embedding")
                        if pid is not None and emb is not None:
                            _score(pid, emb)
                    elif isinstance(rec, (tuple, list)) and len(rec) >= 2:
                        _score(rec[0], rec[1])
            elif isinstance(emb_data, dict) and emb_data:
                first_key = next(iter(emb_data))
                first_val = emb_data[first_key]

                # Schema A: {person_id: [emb, ...]}
                if isinstance(first_key, (int, np.integer)) or (
                    isinstance(first_key, str) and first_key.isdigit()
                ):
                    if isinstance(first_val, (list, tuple)) or (
                        isinstance(first_val, np.ndarray) and first_val.ndim >= 2
                    ):
                        for pid_raw, embs in emb_data.items():
                            if isinstance(embs, np.ndarray) and embs.ndim == 1:
                                _score(pid_raw, embs)
                            else:
                                for emb in embs:
                                    # Each item might be a bare array or a dict
                                    if isinstance(emb, dict):
                                        _score(pid_raw, emb.get("embedding") or emb.get("emb"))
                                    else:
                                        _score(pid_raw, emb)
                    else:
                        # {person_id: single_emb}
                        for pid_raw, emb in emb_data.items():
                            _score(pid_raw, emb)

                # Schema B/C: {photo_filename: [...]}
                elif isinstance(first_key, str) and isinstance(first_val, list):
                    for _fn, entries in emb_data.items():
                        for entry in entries:
                            if isinstance(entry, dict):
                                pid = entry.get("person_id")
                                emb = entry.get("embedding") or entry.get("emb") or entry.get("normed_embedding")
                                if pid is not None and emb is not None:
                                    _score(pid, emb)
                            elif isinstance(entry, (tuple, list)) and len(entry) >= 2:
                                _score(entry[0], entry[1])
                else:
                    print(f"  [fallback] Unrecognised pkl shape: "
                          f"first_key type={type(first_key).__name__}, "
                          f"first_val type={type(first_val).__name__}")
                    return []
            else:
                return []

        except Exception as exc:
            print(f"  [fallback] Error during per-embedding search: {exc}")
            return []

        pci = self.classification.get("person_ceremony_index", {})
        people_dict = self.classification.get("people", {})

        matches = []
        for pid, sim in sorted(person_best.items(), key=lambda x: x[1], reverse=True)[:top_k]:
            if sim < threshold:
                continue
            person_key = str(pid)
            person_info = people_dict.get(person_key, {})
            matches.append({
                "person_id":          pid,
                "similarity":         round(sim, 4),
                "person_name":        person_info.get("name", f"Person_{pid}"),
                "total_photos":       person_info.get("total_photos", 0),
                "ceremonies":         person_info.get("ceremonies", {}),
                "photos_by_ceremony": pci.get(person_key, {}),
            })

        print(f"  [fallback] found {len(matches)} match(es) from {len(person_best)} people")
        return matches

    # ── Photo retrieval (unchanged semantics) ─────────────────────────────────

    def get_photos_for_people(self, person_ids, ceremony=None, photo_type=None):
        person_ids_set = set(person_ids)
        results = []

        for filename, record in self.classification["photos"].items():
            photo_people = set(record.get("people", []))

            if not person_ids_set.issubset(photo_people):
                continue
            if ceremony and record.get("ceremony") != ceremony:
                continue

            if photo_type is not None:
                eff      = record.get("effective_people", record.get("face_count", 0))
                rec_type = record.get("photo_type", "")

                if photo_type == "solo":
                    if eff > SOLO_MAX_EFFECTIVE_PEOPLE:
                        continue
                elif photo_type == "couple":
                    if eff != 2:
                        continue
                elif photo_type == "small_group":
                    if eff < 3 or eff >= GROUP_MIN_FACES:
                        continue
                elif photo_type == "group":
                    if eff < GROUP_MIN_FACES:
                        continue
                elif photo_type == "any_group":
                    if eff < 2:
                        continue
                elif photo_type in ("portrait", "candid"):
                    if rec_type != photo_type:
                        continue
                else:
                    if rec_type != photo_type:
                        continue

            results.append(record)

        return results
