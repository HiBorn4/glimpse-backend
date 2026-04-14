"""
R2 Face Search Engine
=====================
Loads person_centroids.pkl and classification.json directly from Cloudflare R2.
Zero local file dependencies. Everything streamed into memory at startup.
"""

import io
import pickle
import json
import httpx
import numpy as np
import cv2
from sklearn.preprocessing import normalize

import insightface
from insightface.app import FaceAnalysis

import warnings
warnings.filterwarnings("ignore")

from config import r2_url


SOLO_MAX_EFFECTIVE_PEOPLE = 1
GROUP_MIN_FACES = 5


class R2FaceSearchEngine:
    """
    Face search engine that loads all data from Cloudflare R2.

    R2 paths:
        photos/embeddings/person_centroids.pkl
        photos/metadata/classification.json   (shared with cache.py)
    """

    def __init__(self, use_gpu: bool = False):
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if use_gpu
            else ["CPUExecutionProvider"]
        )
        self.app = FaceAnalysis(name="buffalo_l", providers=providers)
        self.app.prepare(ctx_id=0, det_size=(640, 640))

        # Populated by load_from_r2()
        self.centroids:       dict       = {}
        self.person_ids:      list       = []
        self.centroid_matrix: np.ndarray | None = None
        self.classification:  dict       = {}

    async def load_from_r2(self):
        """
        Fetch embeddings + classification from R2 concurrently.
        Must be called once at startup (inside async context).
        """
        centroids_url      = r2_url("embeddings", "person_centroids.pkl")
        classification_url = r2_url("metadata",   "classification.json")

        print(f"  Fetching centroids: {centroids_url}")
        print(f"  Fetching classification: {classification_url}")

        async with httpx.AsyncClient(timeout=120.0) as client:
            centroids_resp, class_resp = await _gather(
                client.get(centroids_url),
                client.get(classification_url),
            )

        centroids_resp.raise_for_status()
        class_resp.raise_for_status()

        self.centroids      = pickle.loads(centroids_resp.content)
        self.classification = class_resp.json()

        self.person_ids      = sorted(self.centroids.keys())
        self.centroid_matrix = normalize(
            np.array([self.centroids[pid] for pid in self.person_ids])
        )

        print(f"  ✅ FaceSearch ready: {len(self.person_ids)} people indexed")

    # ── Core search ───────────────────────────────────────────────────────────

    def search_by_selfie(self, selfie_image, top_k=3, threshold=0.35):
        """
        Accepts a BGR numpy array. Returns matches dict.
        Four-attempt detection pipeline for robustness.
        """
        if self.centroid_matrix is None:
            return {"error": "Face search not loaded", "matches": []}

        if isinstance(selfie_image, str):
            selfie_image = cv2.imread(selfie_image)
        if selfie_image is None:
            return {"error": "Could not read image", "matches": []}

        faces = self._detect_faces(selfie_image)

        if not faces:
            return {
                "error": (
                    "No face detected in selfie. Please try: better lighting, "
                    "face the camera directly, ensure your full face is visible."
                ),
                "faces_detected": 0,
                "matches": [],
            }

        best_face  = max(faces, key=lambda x: x.det_score)
        query_emb  = normalize(best_face.normed_embedding.reshape(1, -1))
        similarities = (query_emb @ self.centroid_matrix.T).flatten()
        top_indices  = similarities.argsort()[::-1][:top_k]

        matches = []
        pci = self.classification.get("person_ceremony_index", {})

        for idx in top_indices:
            sim = float(similarities[idx])
            if sim < threshold:
                continue
            person_id   = self.person_ids[idx]
            person_key  = str(person_id)
            person_info = self.classification["people"].get(person_key, {})

            matches.append({
                "person_id":          person_id,
                "similarity":         round(sim, 4),
                "person_name":        person_info.get("name", f"Person_{person_id}"),
                "total_photos":       person_info.get("total_photos", 0),
                "ceremonies":         person_info.get("ceremonies", {}),
                "photos_by_ceremony": pci.get(person_key, {}),
            })

        return {
            "error":          None,
            "faces_detected": len(faces),
            "matches":        matches,
        }

    def search_by_selfie_bytes(self, image_bytes, top_k=3, threshold=0.35):
        """Accept raw bytes (from API upload)."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return self.search_by_selfie(img, top_k=top_k, threshold=threshold)

    def _detect_faces(self, img):
        """
        Four-attempt detection pipeline.
        Returns face list (may be empty).
        """
        # Attempt 1: original
        faces = self.app.get(img)
        if faces:
            return faces

        # Attempt 2: resize to 1280px wide
        h, w = img.shape[:2]
        if w < 1280:
            scale   = 1280 / w
            resized = cv2.resize(img, (1280, int(h * scale)), interpolation=cv2.INTER_LINEAR)
            faces   = self.app.get(resized)
            if faces:
                return faces

        # Attempt 3: mirror flip
        flipped = cv2.flip(img, 1)
        faces   = self.app.get(flipped)
        if faces:
            return faces

        # Attempt 4: CLAHE contrast enhancement
        lab    = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe  = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l      = clahe.apply(l)
        enh    = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        return self.app.get(enh) or []

    # ── Photo retrieval ───────────────────────────────────────────────────────

    def get_photos_for_people(self, person_ids, ceremony=None, photo_type=None):
        person_ids_set = set(person_ids)
        GROUP_TYPES    = {"couple", "small_group", "group"}
        results        = []

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


# ── Async gather helper ────────────────────────────────────────────────────────

import asyncio

async def _gather(*coros):
    return await asyncio.gather(*coros)
