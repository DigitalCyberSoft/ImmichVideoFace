"""The video index: frame embeddings and appearance times, published to Postgres.

Immich has nowhere to keep either. A face row has no timestamp, and smart search
holds exactly one embedding per asset. So both go into a schema of their own,
which this tool's database role owns:

    <schema>.frame_embedding  one CLIP vector per sampled frame, for object search
    <schema>.appearance       when each person is on screen in a video
    <schema>.video            what was indexed, with which settings

Immich's own tables are still never written; faces and people go through the
REST API as before. Stock Immich ignores this schema. A build carrying the
companion patch reads it: smart search ranks a video by its best-matching frame,
and the video viewer shows who is on screen when.

Setup (once, as a Postgres superuser), with `ivf` the role in `[db] user`:

    CREATE SCHEMA ivf AUTHORIZATION ivf;
    GRANT USAGE ON SCHEMA ivf TO immich;   -- the role Immich connects as
    ALTER DEFAULT PRIVILEGES FOR ROLE ivf IN SCHEMA ivf GRANT SELECT ON TABLES TO immich;

The tables themselves are created on first use.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import numpy as np
import psycopg
from psycopg import sql

from .config import Config

log = logging.getLogger(__name__)

DDL = """
CREATE TABLE IF NOT EXISTS {schema}.frame_embedding (
    asset_id   uuid    NOT NULL,
    time_ms    integer NOT NULL,
    model      text    NOT NULL,
    -- Unconstrained so any CLIP model fits; searched exhaustively, not via an index.
    embedding  vector  NOT NULL,
    PRIMARY KEY (asset_id, time_ms)
);
CREATE TABLE IF NOT EXISTS {schema}.appearance (
    asset_id    uuid  NOT NULL,
    -- Immich's person id (the API id; a person-group id from Immich 3.3 on).
    person_id   uuid  NOT NULL,
    -- The face carrying this person on the video, so the timeline follows the
    -- face through merges and reassignments. NULL if it could not be resolved.
    face_id     uuid,
    -- [[start_ms, end_ms], ...], sorted and non-overlapping.
    segments    jsonb NOT NULL,
    distance    real,
    updated_at  timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (asset_id, person_id)
);
CREATE INDEX IF NOT EXISTS appearance_face_id ON {schema}.appearance (face_id);
CREATE TABLE IF NOT EXISTS {schema}.video (
    asset_id     uuid PRIMARY KEY,
    duration_ms  integer,
    frames       integer,
    clip_frames  integer,
    fingerprint  text,
    indexed_at   timestamptz NOT NULL DEFAULT now()
);
"""


@dataclass
class IndexStats:
    videos: int
    frames: int
    appearances: int


class VideoIndex:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.schema = config.index.schema_name
        self._conn = psycopg.connect(config.db.conninfo(), autocommit=False)
        self._ensure_tables()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "VideoIndex":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _ensure_tables(self) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM information_schema.schemata WHERE schema_name = %s",
                (self.schema,),
            )
            if cur.fetchone() is None:
                raise RuntimeError(
                    f"Schema {self.schema!r} does not exist. Create it once as a superuser: "
                    f"CREATE SCHEMA {self.schema} AUTHORIZATION {self.config.db.user};"
                )
            cur.execute(sql.SQL(DDL).format(schema=sql.Identifier(self.schema)))
        self._conn.commit()

    def _t(self, name: str) -> sql.Composed:
        return sql.SQL("{}.{}").format(sql.Identifier(self.schema), sql.Identifier(name))

    # ---- objects --------------------------------------------------------

    def replace_video(
        self,
        *,
        asset_id: str,
        duration_s: float,
        frames: int,
        frame_embeddings: list[tuple[float, np.ndarray]],
        model: str,
        fingerprint: str,
    ) -> None:
        """Replace everything indexed for one video's frames, atomically."""
        rows = []
        seen: set[int] = set()
        for time_s, vector in frame_embeddings:
            time_ms = int(round(time_s * 1000))
            if time_ms in seen:
                continue
            seen.add(time_ms)
            rows.append((asset_id, time_ms, model, _vector_literal(vector)))
        with self._conn.transaction(), self._conn.cursor() as cur:
            cur.execute(
                sql.SQL("DELETE FROM {} WHERE asset_id = %s").format(self._t("frame_embedding")),
                (asset_id,),
            )
            if rows:
                cur.executemany(
                    sql.SQL(
                        "INSERT INTO {} (asset_id, time_ms, model, embedding) "
                        "VALUES (%s, %s, %s, %s::vector)"
                    ).format(self._t("frame_embedding")),
                    rows,
                )
            cur.execute(
                sql.SQL(
                    "INSERT INTO {} (asset_id, duration_ms, frames, clip_frames, fingerprint) "
                    "VALUES (%s, %s, %s, %s, %s) "
                    "ON CONFLICT (asset_id) DO UPDATE SET duration_ms = EXCLUDED.duration_ms, "
                    "frames = EXCLUDED.frames, clip_frames = EXCLUDED.clip_frames, "
                    "fingerprint = EXCLUDED.fingerprint, indexed_at = now()"
                ).format(self._t("video")),
                (asset_id, int(round(duration_s * 1000)), frames, len(rows), fingerprint),
            )

    def frame_models(self) -> set[str]:
        """Search models that frames are currently embedded with."""
        with self._conn.cursor() as cur:
            cur.execute(sql.SQL("SELECT DISTINCT model FROM {}").format(self._t("frame_embedding")))
            models = {row[0] for row in cur.fetchall()}
        self._conn.rollback()
        return models

    def stale_videos(self, model: str) -> list[str]:
        """Videos with frames embedded by a model other than `model`."""
        with self._conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    "SELECT DISTINCT asset_id::text FROM {} WHERE model <> %s ORDER BY 1"
                ).format(self._t("frame_embedding")),
                (model,),
            )
            ids = [row[0] for row in cur.fetchall()]
        self._conn.rollback()
        return ids

    def replace_frames(
        self, *, asset_id: str, frame_embeddings: list[tuple[float, np.ndarray]], model: str
    ) -> int:
        """Replace one video's frame embeddings, leaving its scan record alone."""
        rows = []
        seen: set[int] = set()
        for time_s, vector in frame_embeddings:
            time_ms = int(round(time_s * 1000))
            if time_ms not in seen:
                seen.add(time_ms)
                rows.append((asset_id, time_ms, model, _vector_literal(vector)))
        with self._conn.transaction(), self._conn.cursor() as cur:
            cur.execute(
                sql.SQL("DELETE FROM {} WHERE asset_id = %s").format(self._t("frame_embedding")),
                (asset_id,),
            )
            if rows:
                cur.executemany(
                    sql.SQL(
                        "INSERT INTO {} (asset_id, time_ms, model, embedding) "
                        "VALUES (%s, %s, %s, %s::vector)"
                    ).format(self._t("frame_embedding")),
                    rows,
                )
            cur.execute(
                sql.SQL("UPDATE {} SET clip_frames = %s, indexed_at = now() WHERE asset_id = %s").format(
                    self._t("video")
                ),
                (len(rows), asset_id),
            )
        return len(rows)

    # ---- appearances ----------------------------------------------------

    def upsert_appearance(
        self,
        *,
        asset_id: str,
        person_id: str,
        face_id: str | None,
        segments: list[tuple[float, float]],
        distance: float | None,
    ) -> None:
        if not segments:
            return
        payload = json.dumps([[int(round(a * 1000)), int(round(b * 1000))] for a, b in segments])
        with self._conn.transaction(), self._conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    "INSERT INTO {} (asset_id, person_id, face_id, segments, distance) "
                    "VALUES (%s, %s, %s, %s::jsonb, %s) "
                    "ON CONFLICT (asset_id, person_id) DO UPDATE SET "
                    "face_id = EXCLUDED.face_id, segments = EXCLUDED.segments, "
                    "distance = EXCLUDED.distance, updated_at = now()"
                ).format(self._t("appearance")),
                (asset_id, person_id, face_id, payload, distance),
            )

    def delete_appearance(self, asset_id: str, person_id: str) -> None:
        with self._conn.transaction(), self._conn.cursor() as cur:
            cur.execute(
                sql.SQL("DELETE FROM {} WHERE asset_id = %s AND person_id = %s").format(
                    self._t("appearance")
                ),
                (asset_id, person_id),
            )

    def stats(self) -> IndexStats:
        with self._conn.cursor() as cur:
            counts = []
            for table in ("video", "frame_embedding", "appearance"):
                cur.execute(sql.SQL("SELECT count(*) FROM {}").format(self._t(table)))
                counts.append(int(cur.fetchone()[0]))
        self._conn.rollback()
        return IndexStats(*counts)


def resolve_face_ids(faces: list[dict]) -> dict[str, str]:
    """{person id: face id} for one asset, from `GET /faces?id=<asset>`.

    Prefers Immich's own detections, which carry embeddings, over manual faces.
    """
    chosen: dict[str, tuple[int, str]] = {}
    for face in faces:
        person = face.get("person") or {}
        person_id = person.get("id")
        if not person_id or not face.get("id"):
            continue
        rank = 0 if face.get("sourceType") == "machine-learning" else 1
        if person_id not in chosen or rank < chosen[person_id][0]:
            chosen[person_id] = (rank, face["id"])
    return {person_id: face_id for person_id, (_, face_id) in chosen.items()}


def _vector_literal(vector: np.ndarray) -> str:
    return "[" + ",".join(f"{float(v):.8g}" for v in np.asarray(vector).ravel()) + "]"
