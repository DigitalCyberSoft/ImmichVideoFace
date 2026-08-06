"""Read-only access to Immich's Postgres.

This module never writes. It exists because Immich does not expose face
embeddings over the REST API, and re-deriving them would be both slower and
less accurate than reading the ones Immich already computed.

The schema is an internal Immich detail, so `assert_schema` fails loudly and
specifically rather than letting a rename silently produce wrong matches.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import psycopg
from psycopg.rows import dict_row

from .config import Config

log = logging.getLogger(__name__)

# (table, required columns) -- checked at startup.
REQUIRED_SCHEMA: dict[str, set[str]] = {
    "asset": {
        "id",
        "type",
        "originalPath",
        "deletedAt",
        "ownerId",
        "duration",
        "localDateTime",
        "livePhotoVideoId",
        "visibility",
    },
    "asset_face": {
        "id",
        "assetId",
        "personId",
        "deletedAt",
        "isVisible",
        "sourceType",
        "imageWidth",
        "imageHeight",
        "boundingBoxX1",
        "boundingBoxY1",
        "boundingBoxX2",
        "boundingBoxY2",
    },
    "person": {"id", "name", "ownerId", "isHidden", "faceAssetId"},
    "face_search": {"faceId", "embedding"},
}


@dataclass
class Gallery:
    """Reference embeddings for every person, grouped by person id."""

    person_ids: list[str]
    names: dict[str, str]
    # Row-normalised (n_faces, 512); `owners[i]` is the person for row i.
    embeddings: np.ndarray
    owners: np.ndarray

    def __len__(self) -> int:
        return len(self.person_ids)

    @property
    def face_count(self) -> int:
        return int(self.embeddings.shape[0])


@dataclass
class ReferenceFace:
    """A single labelled face, used by the embedding-parity gate."""

    face_id: str
    person_id: str
    person_name: str
    asset_id: str
    original_path: str
    # Immich runs its own detection against the preview, so parity must compare
    # against these exact pixels rather than the original file.
    preview_path: str | None
    image_width: int
    image_height: int
    box: tuple[int, int, int, int]
    embedding: np.ndarray


class ImmichDb:
    def __init__(self, config: Config) -> None:
        self.config = config
        self._conn = psycopg.connect(
            config.db.dsn(),
            row_factory=dict_row,
            # Belt and braces: the app must never mutate Immich's database.
            autocommit=True,
        )
        self._conn.read_only = True

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "ImmichDb":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ---- schema guard ---------------------------------------------------

    def assert_schema(self) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT table_name, column_name
                FROM information_schema.columns
                WHERE table_schema = 'public' AND table_name = ANY(%s)
                """,
                (list(REQUIRED_SCHEMA),),
            )
            found: dict[str, set[str]] = {}
            for row in cur.fetchall():
                found.setdefault(row["table_name"], set()).add(row["column_name"])

        problems: list[str] = []
        for table, columns in REQUIRED_SCHEMA.items():
            if table not in found:
                problems.append(f"missing table {table!r}")
                continue
            if missing := columns - found[table]:
                problems.append(f"{table} is missing column(s): {', '.join(sorted(missing))}")

        if problems:
            raise RuntimeError(
                "Immich database schema does not match what this tool expects "
                "(likely an Immich upgrade).\n  - " + "\n  - ".join(problems)
            )

    # ---- reference gallery ----------------------------------------------

    def load_gallery(self, owner_id: str) -> Gallery:
        """Every labelled face embedding for this owner.

        Manual faces are excluded for free: `createFace` performs no inference,
        so faces this tool writes never get a `face_search` row and therefore
        can never feed back into future matching.
        """
        sql = """
            SELECT af."personId" AS person_id, p.name AS name, fs.embedding AS embedding
            FROM face_search fs
            JOIN asset_face af ON af.id = fs."faceId"
            JOIN person p      ON p.id = af."personId"
            WHERE af."personId" IS NOT NULL
              AND af."deletedAt" IS NULL
              AND af."isVisible" IS TRUE
              AND p."isHidden" IS FALSE
              AND p."ownerId" = %(owner_id)s
        """
        if self.config.match.named_people_only:
            sql += " AND p.name <> ''"

        with self._conn.cursor() as cur:
            cur.execute(sql, {"owner_id": owner_id})
            rows = cur.fetchall()

        if not rows:
            raise RuntimeError(
                "Reference gallery is empty -- no labelled faces found for this owner. "
                "Has Immich finished its own face detection?"
            )

        names: dict[str, str] = {}
        owners: list[str] = []
        vectors: list[np.ndarray] = []
        for row in rows:
            person_id = str(row["person_id"])
            names.setdefault(person_id, row["name"] or "")
            owners.append(person_id)
            vectors.append(parse_vector(row["embedding"]))

        embeddings = normalise(np.vstack(vectors))
        return Gallery(
            person_ids=sorted(names),
            names=names,
            embeddings=embeddings,
            owners=np.array(owners, dtype=object),
        )

    # ---- helpers used by the parity gate --------------------------------

    def sample_reference_faces(self, owner_id: str, limit: int = 5) -> list[ReferenceFace]:
        """Named faces on *image* assets, for embedding-space verification."""
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT af.id AS face_id, af."personId" AS person_id, p.name AS name,
                       a.id AS asset_id, a."originalPath" AS original_path,
                       pf.path AS preview_path,
                       af."imageWidth" AS image_width, af."imageHeight" AS image_height,
                       af."boundingBoxX1" AS x1, af."boundingBoxY1" AS y1,
                       af."boundingBoxX2" AS x2, af."boundingBoxY2" AS y2,
                       fs.embedding AS embedding
                FROM face_search fs
                JOIN asset_face af ON af.id = fs."faceId"
                JOIN person p      ON p.id = af."personId"
                JOIN asset a       ON a.id = af."assetId"
                LEFT JOIN asset_file pf ON pf."assetId" = a.id AND pf.type = 'preview'
                WHERE p.name <> ''
                  AND p."ownerId" = %(owner_id)s
                  AND a.type = 'IMAGE'
                  AND a."deletedAt" IS NULL
                  AND af."deletedAt" IS NULL
                  AND af."sourceType" = 'machine-learning'
                  AND (af."boundingBoxX2" - af."boundingBoxX1") > 120
                ORDER BY random()
                LIMIT %(limit)s
                """,
                {"owner_id": owner_id, "limit": limit},
            )
            rows = cur.fetchall()

        return [
            ReferenceFace(
                face_id=str(row["face_id"]),
                person_id=str(row["person_id"]),
                person_name=row["name"],
                asset_id=str(row["asset_id"]),
                original_path=row["original_path"],
                preview_path=row["preview_path"],
                image_width=row["image_width"],
                image_height=row["image_height"],
                box=(row["x1"], row["y1"], row["x2"], row["y2"]),
                embedding=parse_vector(row["embedding"]),
            )
            for row in rows
        ]

    def list_videos(
        self, owner_id: str, limit: int | None = None, newest_first: bool = False
    ) -> list:
        """Enumerate videos without the API.

        Dry runs need nothing Immich only exposes over REST -- the write path
        reports bounding boxes in our own frame space, so stored dimensions are
        never needed. This lets the whole scan-and-tune loop run before an API
        key exists.
        """
        from .immich_api import VideoAsset

        # localDateTime is when the video was actually taken, which is what
        # "recent" should mean -- not when it happened to be imported.
        order = 'v."localDateTime" DESC NULLS LAST' if newest_first else 'v."originalPath"'
        # `still.id` is non-null exactly when this video is a Live Photo's motion
        # clip, which is both the filter and the redirect target.
        mode = self.config.select.live_photos
        live_photo_filter = "AND still.id IS NULL" if mode == "skip" else ""
        sql = f"""
            SELECT v.id, v."originalPath", v.duration, v."localDateTime",
                   still.id AS still_id, still."originalPath" AS still_path
            FROM asset v
            -- LATERAL, not a plain join: duplicated stills can reference the
            -- same motion clip, which would otherwise scan that video twice.
            LEFT JOIN LATERAL (
                SELECT s.id, s."originalPath"
                FROM asset s
                WHERE s."livePhotoVideoId" = v.id AND s."deletedAt" IS NULL
                ORDER BY s."fileCreatedAt"
                LIMIT 1
            ) still ON TRUE
            WHERE v.type = 'VIDEO' AND v."deletedAt" IS NULL
              AND v."ownerId" = %(owner_id)s
              {live_photo_filter}
            ORDER BY {order}
        """
        params: dict = {"owner_id": owner_id}
        if limit:
            sql += " LIMIT %(limit)s"
            params["limit"] = limit

        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            return [
                VideoAsset(
                    id=str(row["id"]),
                    original_path=row["originalPath"],
                    width=0,
                    height=0,
                    duration_ms=int(row["duration"] or 0),
                    taken_at=str(row["localDateTime"] or "")[:10],
                    still_asset_id=(
                        str(row["still_id"])
                        if row["still_id"] and mode == "redirect"
                        else None
                    ),
                    still_filename=(
                        str(row["still_path"] or "").rsplit("/", 1)[-1]
                        if row["still_id"] and mode == "redirect"
                        else ""
                    ),
                )
                for row in cur.fetchall()
            ]

    def person_centroids(self, gallery: Gallery) -> tuple[list[str], np.ndarray, dict[str, int]]:
        """One mean embedding per person, plus their face counts."""
        counts: dict[str, int] = {}
        sums: dict[str, np.ndarray] = {}
        for person_id, embedding in zip(gallery.owners.astype(str), gallery.embeddings):
            counts[person_id] = counts.get(person_id, 0) + 1
            if person_id in sums:
                sums[person_id] += embedding
            else:
                sums[person_id] = embedding.copy()
        ids = sorted(sums)
        matrix = normalise(np.vstack([sums[i] for i in ids]))
        return ids, matrix, counts

    def person_stats(self, owner_id: str) -> dict[str, int]:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT count(*) FILTER (WHERE name <> '') AS named,
                       count(*)                           AS total
                FROM person WHERE "ownerId" = %(owner_id)s AND "isHidden" IS FALSE
                """,
                {"owner_id": owner_id},
            )
            return dict(cur.fetchone())


def parse_vector(raw: object) -> np.ndarray:
    """pgvector returns `vector` columns as '[1,2,3]' text over the wire."""
    if isinstance(raw, (list, tuple)):
        return np.asarray(raw, dtype=np.float32)
    if isinstance(raw, np.ndarray):
        return raw.astype(np.float32)
    text = str(raw).strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    return np.fromstring(text, sep=",", dtype=np.float32)


def normalise(matrix: np.ndarray) -> np.ndarray:
    """L2-normalise rows so cosine distance is 1 - dot product."""
    matrix = np.atleast_2d(matrix).astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms
