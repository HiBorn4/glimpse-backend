"""Pydantic models for Wedding Photo AI API"""
from pydantic import BaseModel
from typing import Optional


class EventInfo(BaseModel):
    total_photos: int
    total_people: int
    total_ceremonies: int
    ceremonies: list[str]
    created_at: Optional[str] = None


class PersonInfo(BaseModel):
    person_id: int
    name: str
    total_photos: int
    total_faces: int
    ceremonies: dict[str, int]
    best_face_photo: str
    best_face_bbox: list[int]


class PhotoRecord(BaseModel):
    filename: str
    relative_path: str
    ceremony: str
    people: list[int]
    photo_type: str
    face_count: int


class SearchMatch(BaseModel):
    person_id: int
    similarity: float
    person_name: str
    total_photos: int
    ceremonies: dict[str, int]
    photos_by_ceremony: dict[str, list[str]]


class SearchResponse(BaseModel):
    error: Optional[str]
    faces_detected: int = 0
    matches: list[SearchMatch]


class GalleryFilter(BaseModel):
    ceremony: Optional[str] = None
    person_ids: Optional[list[int]] = None
    photo_type: Optional[str] = None
    page: int = 1
    per_page: int = 50
