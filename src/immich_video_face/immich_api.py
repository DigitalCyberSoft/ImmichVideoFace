"""Immich REST client.

Every *write* goes through here. The database layer is strictly read-only.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Iterator

import httpx

from .config import Config

log = logging.getLogger(__name__)

RETRY_STATUS = {429, 500, 502, 503, 504}


@dataclass(frozen=True)
class VideoAsset:
    id: str
    original_path: str
    # Original media dimensions, needed to scale bounding boxes on write.
    width: int
    height: int
    duration_ms: int
    # Camera filenames collide constantly (~14% of videos in one test library),
    # so reports need the date and id to identify an asset unambiguously.
    taken_at: str = ""
    # Set when this is a Live Photo's motion clip: the visible still it belongs
    # to, which is where any face found here should actually be written.
    still_asset_id: str | None = None
    still_filename: str = ""

    @property
    def write_target(self) -> str:
        """The asset a detected face should be attached to."""
        return self.still_asset_id or self.id


class ImmichError(RuntimeError):
    pass


class ImmichClient:
    def __init__(self, config: Config) -> None:
        self.config = config
        base = config.immich.url.rstrip("/")
        if not base.endswith("/api"):
            base += "/api"
        self._client = httpx.Client(
            base_url=base,
            headers={
                "x-api-key": config.immich.api_key,
                "Accept": "application/json",
            },
            timeout=config.immich.timeout,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "ImmichClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        last_error: Exception | None = None
        for attempt in range(5):
            try:
                response = self._client.request(method, path, **kwargs)
            except httpx.TransportError as exc:  # network blip
                last_error = exc
                time.sleep(2**attempt * 0.25)
                continue
            if response.status_code in RETRY_STATUS:
                last_error = ImmichError(f"{response.status_code} on {method} {path}")
                time.sleep(2**attempt * 0.25)
                continue
            if response.status_code >= 400:
                raise ImmichError(
                    f"{method} {path} -> {response.status_code}: {response.text[:400]}"
                )
            return response
        raise ImmichError(f"{method} {path} failed after retries: {last_error}")

    # ---- identity -------------------------------------------------------

    def my_user_id(self) -> str:
        return self._request("GET", "/users/me").json()["id"]

    def face_model(self) -> str | None:
        """The recognition model Immich is configured to use.

        Every embedding in `face_search` was produced by this model. Ours must
        match or the two live in different vector spaces and every comparison is
        meaningless. Returns None when the key lacks admin rights to read it.
        """
        try:
            data = self._request("GET", "/system-config").json()
        except ImmichError:
            return None
        facial = (data.get("machineLearning") or {}).get("facialRecognition") or {}
        return facial.get("modelName")

    def clip_model(self) -> str | None:
        """Immich's smart search (CLIP) model, or None without admin rights."""
        try:
            data = self._request("GET", "/system-config").json()
        except ImmichError:
            return None
        clip = (data.get("machineLearning") or {}).get("clip") or {}
        return clip.get("modelName")

    def server_version(self) -> str:
        data = self._request("GET", "/server/version").json()
        return f"{data.get('major')}.{data.get('minor')}.{data.get('patch')}"

    # ---- assets ---------------------------------------------------------

    def iter_videos(self, page_size: int = 250) -> Iterator[VideoAsset]:
        """Page through every non-trashed video via POST /search/metadata."""
        page = 1
        while True:
            body = {
                "type": "VIDEO",
                "withExif": True,
                "withDeleted": False,
                "page": page,
                "size": page_size,
            }
            payload = self._request("POST", "/search/metadata", json=body).json()
            assets = payload["assets"]
            for item in assets["items"]:
                yield video_asset_from_dict(item)
            next_page = assets.get("nextPage")
            if not next_page:
                return
            page = int(next_page)

    def get_asset(self, asset_id: str) -> dict:
        return self._request("GET", f"/assets/{asset_id}").json()

    def get_video_asset(self, asset_id: str) -> VideoAsset:
        return video_asset_from_dict(self.get_asset(asset_id))

    def get_preview(self, asset_id: str, size: str = "preview") -> bytes:
        """The rendered preview image.

        Immich runs its own face detection against this, not the original, so
        the parity gate must compare against these exact pixels.
        """
        response = self._request(
            "GET", f"/assets/{asset_id}/thumbnail", params={"size": size}
        )
        return response.content

    # ---- faces ----------------------------------------------------------

    def get_faces(self, asset_id: str) -> list[dict]:
        """All faces on an asset. Used as the idempotency check before writing."""
        return self._request("GET", "/faces", params={"id": asset_id}).json()

    def create_face(
        self,
        *,
        asset_id: str,
        person_id: str,
        image_width: int,
        image_height: int,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> None:
        """Create a manual face. Returns nothing -- Immich gives us no face id back,
        which is why writeback records its own (asset, person, bbox) tuples for undo."""
        self._request(
            "POST",
            "/faces",
            json={
                "assetId": asset_id,
                "personId": person_id,
                "imageWidth": image_width,
                "imageHeight": image_height,
                "x": x,
                "y": y,
                "width": width,
                "height": height,
            },
        )

    def delete_face(self, face_id: str, force: bool = True) -> None:
        self._request("DELETE", f"/faces/{face_id}", json={"force": force})

    # ---- people ---------------------------------------------------------

    def get_people(self, with_hidden: bool = True) -> list[dict]:
        people: list[dict] = []
        page = 1
        while True:
            payload = self._request(
                "GET",
                "/people",
                params={"page": page, "size": 500, "withHidden": str(with_hidden).lower()},
            ).json()
            people.extend(payload.get("people", []))
            if not payload.get("hasNextPage"):
                return people
            page += 1

    def search_person(self, name: str) -> list[dict]:
        return self._request("GET", "/search/person", params={"name": name}).json()

    def create_person(self, name: str = "") -> str:
        return self._request("POST", "/people", json={"name": name}).json()["id"]

    def delete_person(self, person_id: str) -> None:
        self._request("DELETE", f"/people/{person_id}")

    def get_person_face_count(self, person_id: str) -> int:
        """Assets attributed to a person. Undo uses this to avoid deleting a
        person the user has since attached real faces to."""
        payload = self._request("GET", f"/people/{person_id}/statistics").json()
        return int(payload.get("assets", 0))

    # ---- tags -----------------------------------------------------------

    def upsert_tag(self, name: str) -> str:
        """Create or fetch a (possibly nested, slash-separated) tag."""
        payload = self._request("PUT", "/tags", json={"tags": [name]}).json()
        for tag in payload:
            if tag["value"] == name:
                return tag["id"]
        raise ImmichError(f"Tag upsert did not return {name!r}")

    def tag_assets(self, tag_id: str, asset_ids: list[str]) -> None:
        if not asset_ids:
            return
        self._request("PUT", f"/tags/{tag_id}/assets", json={"ids": asset_ids})

    def untag_assets(self, tag_id: str, asset_ids: list[str]) -> None:
        if not asset_ids:
            return
        self._request("DELETE", f"/tags/{tag_id}/assets", json={"ids": asset_ids})


def video_asset_from_dict(item: dict) -> VideoAsset:
    exif = item.get("exifInfo") or {}
    taken = item.get("localDateTime") or item.get("fileCreatedAt") or ""
    return VideoAsset(
        id=item["id"],
        original_path=item.get("originalPath", ""),
        width=int(exif.get("exifImageWidth") or 0),
        height=int(exif.get("exifImageHeight") or 0),
        duration_ms=_duration_ms(item.get("duration")),
        taken_at=str(taken)[:10],
    )


def _duration_ms(duration: Any) -> int:
    """Immich returns duration over the API as 'HH:MM:SS.ffffff'."""
    if not duration:
        return 0
    if isinstance(duration, (int, float)):
        return int(duration)
    try:
        hours, minutes, seconds = str(duration).split(":")
        return int((int(hours) * 3600 + int(minutes) * 60 + float(seconds)) * 1000)
    except (ValueError, TypeError):
        return 0
