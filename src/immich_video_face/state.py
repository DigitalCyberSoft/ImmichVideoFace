"""Local SQLite state: resume, and the audit trail that makes undo possible.

Immich cannot hold this for us -- `AssetMetadataKey` is a locked enum whose only
member is `mobile-app`, so `PUT /assets/{id}/metadata` rejects arbitrary keys.

The written_faces table matters more than it looks: `POST /api/faces` returns
204 with no body, so Immich never tells us the id of the face it just created.
Recording (asset, person, bbox) here is the only way to identify our own writes
later and take them back.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at    REAL NOT NULL,
    finished_at   REAL,
    fingerprint   TEXT NOT NULL,
    mode          TEXT NOT NULL,
    notes         TEXT
);

CREATE TABLE IF NOT EXISTS scanned_assets (
    asset_id      TEXT PRIMARY KEY,
    fingerprint   TEXT NOT NULL,
    run_id        INTEGER,
    scanned_at    REAL NOT NULL,
    frames        INTEGER NOT NULL DEFAULT 0,
    observations  INTEGER NOT NULL DEFAULT 0,
    clusters      INTEGER NOT NULL DEFAULT 0,
    matches       INTEGER NOT NULL DEFAULT 0,
    status        TEXT NOT NULL DEFAULT 'ok',
    error         TEXT,
    -- 0 until the batch's results have been dealt with. Only settled assets
    -- count as done, so an uncommitted batch is re-selected next time rather
    -- than silently skipped.
    settled       INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS written_faces (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        INTEGER NOT NULL,
    asset_id      TEXT NOT NULL,
    person_id     TEXT NOT NULL,
    image_width   INTEGER NOT NULL,
    image_height  INTEGER NOT NULL,
    x             INTEGER NOT NULL,
    y             INTEGER NOT NULL,
    width         INTEGER NOT NULL,
    height        INTEGER NOT NULL,
    distance      REAL,
    created_at    REAL NOT NULL,
    revoked_at    REAL
);
CREATE INDEX IF NOT EXISTS written_faces_run ON written_faces(run_id);
CREATE INDEX IF NOT EXISTS written_faces_asset ON written_faces(asset_id);

CREATE TABLE IF NOT EXISTS created_people (
    person_id     TEXT PRIMARY KEY,
    run_id        INTEGER NOT NULL,
    created_at    REAL NOT NULL,
    revoked_at    REAL
);

-- Staged writes awaiting human review. A scan never touches Immich; it records
-- what it *would* write here and renders an HTML report. `ivf commit` applies
-- them afterwards, so nothing reaches the library unseen.
CREATE TABLE IF NOT EXISTS pending_writes (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        INTEGER NOT NULL,
    asset_id      TEXT NOT NULL,
    person_id     TEXT NOT NULL,
    person_name   TEXT NOT NULL DEFAULT '',
    image_width   INTEGER NOT NULL,
    image_height  INTEGER NOT NULL,
    x             INTEGER NOT NULL,
    y             INTEGER NOT NULL,
    width         INTEGER NOT NULL,
    height        INTEGER NOT NULL,
    distance      REAL,
    margin        REAL,
    observations  INTEGER,
    created_at    REAL NOT NULL,
    committed_at  REAL,
    UNIQUE (run_id, asset_id, person_id)
);
CREATE INDEX IF NOT EXISTS pending_uncommitted ON pending_writes(run_id, committed_at);

-- Centroids of clusters that matched nobody. Kept so `promote` can cluster
-- them *across* videos: a face seen in one clip is noise, the same stranger
-- across eight clips is a person worth creating.
CREATE TABLE IF NOT EXISTS unmatched_clusters (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        INTEGER NOT NULL,
    asset_id      TEXT NOT NULL,
    embedding     BLOB NOT NULL,
    observations  INTEGER NOT NULL,
    image_width   INTEGER NOT NULL,
    image_height  INTEGER NOT NULL,
    x             INTEGER NOT NULL,
    y             INTEGER NOT NULL,
    width         INTEGER NOT NULL,
    height        INTEGER NOT NULL,
    best_distance REAL,
    promoted_to   TEXT,
    created_at    REAL NOT NULL,
    -- Small JPEG of the clearest frame, so `promote` can show what it is about
    -- to turn into a person rather than asking for blind trust.
    crop_jpeg     BLOB
);
CREATE INDEX IF NOT EXISTS unmatched_pending ON unmatched_clusters(promoted_to);

-- Keyed by run as well as asset: a later run re-tagging the same asset must
-- record it, otherwise undoing that run silently leaves the tag behind.
-- Stable numbering for new-person candidates.
--
-- Numbers were previously assigned by sorting groups by size on every listing,
-- so the same person changed number between reports and it was never clear
-- which one to promote. A group is now identified by its earliest-banked
-- member, and keeps whatever number that anchor was first given.
CREATE TABLE IF NOT EXISTS promote_groups (
    group_no   INTEGER PRIMARY KEY AUTOINCREMENT,
    anchor_id  INTEGER NOT NULL UNIQUE,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS tagged_assets (
    asset_id      TEXT NOT NULL,
    run_id        INTEGER NOT NULL,
    tag_id        TEXT NOT NULL,
    created_at    REAL NOT NULL,
    PRIMARY KEY (run_id, asset_id, tag_id)
);
-- Sensitive-content tags this tool put on (see sensitive.py). cleared_at is set
-- when someone took the tag off by hand, so it is never put back.
CREATE TABLE IF NOT EXISTS sensitive_tags (
    category      TEXT NOT NULL,
    asset_id      TEXT NOT NULL,
    tag_id        TEXT NOT NULL,
    tagged_at     REAL NOT NULL,
    cleared_at    REAL,
    PRIMARY KEY (category, asset_id)
);
"""


@dataclass
class WrittenFace:
    id: int
    asset_id: str
    person_id: str
    box: tuple[int, int, int, int]  # x, y, width, height


@dataclass
class PendingWrite:
    id: int
    asset_id: str
    person_id: str
    person_name: str
    image_size: tuple[int, int]
    box: tuple[int, int, int, int]
    distance: float
    # On-screen intervals in seconds; empty when the timeline does not apply
    # (a Live Photo clip whose face is written onto the still).
    segments: list[tuple[float, float]] = field(default_factory=list)


@dataclass
class UnmatchedCluster:
    id: int
    asset_id: str
    embedding: "np.ndarray"
    observations: int
    image_size: tuple[int, int]
    box: tuple[int, int, int, int]
    crop_jpeg: bytes | None = None
    group_no: int | None = None
    segments: list[tuple[float, float]] = field(default_factory=list)


class State:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._conn = sqlite3.connect(path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._migrate()
        self._conn.commit()

    def _migrate(self) -> None:
        """Bring databases created by earlier versions up to date."""
        existing = {
            row["name"]
            for row in self._conn.execute("PRAGMA table_info(unmatched_clusters)")
        }
        if "crop_jpeg" not in existing:
            self._conn.execute("ALTER TABLE unmatched_clusters ADD COLUMN crop_jpeg BLOB")
        if "group_no" not in existing:
            self._conn.execute("ALTER TABLE unmatched_clusters ADD COLUMN group_no INTEGER")
        if "segments" not in existing:
            self._conn.execute("ALTER TABLE unmatched_clusters ADD COLUMN segments TEXT")
        pending_cols = {
            row["name"] for row in self._conn.execute("PRAGMA table_info(pending_writes)")
        }
        if "segments" not in pending_cols:
            self._conn.execute("ALTER TABLE pending_writes ADD COLUMN segments TEXT")

        scanned_cols = {
            row["name"] for row in self._conn.execute("PRAGMA table_info(scanned_assets)")
        }
        if scanned_cols and "settled" not in scanned_cols:
            self._conn.execute(
                "ALTER TABLE scanned_assets ADD COLUMN settled INTEGER NOT NULL DEFAULT 0"
            )
            # Anything already committed counts as settled.
            self._conn.execute(
                "UPDATE scanned_assets SET settled=1 WHERE asset_id IN "
                "(SELECT asset_id FROM written_faces WHERE revoked_at IS NULL)"
            )

        # tagged_assets gained run_id in its primary key; rebuild if needed.
        keyed = [
            row["name"]
            for row in self._conn.execute("PRAGMA table_info(tagged_assets)")
            if row["pk"]
        ]
        if keyed and "run_id" not in keyed:
            self._conn.executescript(
                """
                ALTER TABLE tagged_assets RENAME TO tagged_assets_old;
                CREATE TABLE tagged_assets (
                    asset_id   TEXT NOT NULL,
                    run_id     INTEGER NOT NULL,
                    tag_id     TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    PRIMARY KEY (run_id, asset_id, tag_id)
                );
                INSERT INTO tagged_assets SELECT asset_id, run_id, tag_id, created_at
                    FROM tagged_assets_old;
                DROP TABLE tagged_assets_old;
                """
            )

    # ---- sensitive content ---------------------------------------------

    def sensitive_tagged(self, category: str) -> dict[str, bool]:
        """{asset id: cleared by hand?} for tags this tool put on in `category`."""
        rows = self._conn.execute(
            "SELECT asset_id, cleared_at FROM sensitive_tags WHERE category = ?", (category,)
        )
        return {row["asset_id"]: row["cleared_at"] is not None for row in rows}

    def sensitive_categories(self) -> set[str]:
        rows = self._conn.execute("SELECT DISTINCT category FROM sensitive_tags")
        return {row["category"] for row in rows}

    def sensitive_record(self, category: str, asset_ids: list[str], tag_id: str) -> None:
        now = time.time()
        self._conn.executemany(
            "INSERT OR REPLACE INTO sensitive_tags (category, asset_id, tag_id, tagged_at) "
            "VALUES (?, ?, ?, ?)",
            [(category, asset_id, tag_id, now) for asset_id in asset_ids],
        )
        self._conn.commit()

    def sensitive_mark_cleared(self, category: str, asset_id: str) -> None:
        self._conn.execute(
            "UPDATE sensitive_tags SET cleared_at = ? WHERE category = ? AND asset_id = ?",
            (time.time(), category, asset_id),
        )
        self._conn.commit()

    def sensitive_forget(self, category: str, asset_ids: list[str]) -> None:
        self._conn.executemany(
            "DELETE FROM sensitive_tags WHERE category = ? AND asset_id = ?",
            [(category, asset_id) for asset_id in asset_ids],
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "State":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @contextlib.contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        with self._conn:
            yield self._conn

    # ---- runs -----------------------------------------------------------

    def start_run(self, fingerprint: str, mode: str, notes: str = "") -> int:
        with self._write() as conn:
            cursor = conn.execute(
                "INSERT INTO runs (started_at, fingerprint, mode, notes) VALUES (?,?,?,?)",
                (time.time(), fingerprint, mode, notes),
            )
            return int(cursor.lastrowid or 0)

    def finish_run(self, run_id: int) -> None:
        with self._write() as conn:
            conn.execute("UPDATE runs SET finished_at=? WHERE id=?", (time.time(), run_id))

    def run_fingerprint(self, run_id: int) -> str | None:
        row = self._conn.execute(
            "SELECT fingerprint FROM runs WHERE id=?", (run_id,)
        ).fetchone()
        return row["fingerprint"] if row else None

    def latest_write_run(self) -> int | None:
        """Most recent run that actually created something, regardless of mode."""
        row = self._conn.execute(
            "SELECT max(run_id) AS run_id FROM ("
            "  SELECT run_id FROM written_faces WHERE revoked_at IS NULL"
            "  UNION ALL"
            "  SELECT run_id FROM created_people WHERE revoked_at IS NULL"
            ")"
        ).fetchone()
        return int(row["run_id"]) if row and row["run_id"] is not None else None

    def latest_run(self, mode: str | None = None) -> int | None:
        sql = "SELECT id FROM runs"
        params: tuple = ()
        if mode:
            sql += " WHERE mode=?"
            params = (mode,)
        sql += " ORDER BY id DESC LIMIT 1"
        row = self._conn.execute(sql, params).fetchone()
        return int(row["id"]) if row else None

    # ---- scan bookkeeping ----------------------------------------------

    def rekey(self, old: set[str], new: str) -> int:
        """Move scans and runs recorded under fingerprints `old` to `new`.

        For a fingerprint whose definition changed but whose results did not
        (the search model left it): the work done stays done, and staged runs
        stay committable. Returns the number of scanned assets moved.
        """
        old = {fp for fp in old if fp and fp != new}
        if not old:
            return 0
        marks = ",".join("?" * len(old))
        with self._conn:
            moved = self._conn.execute(
                f"UPDATE scanned_assets SET fingerprint=? WHERE fingerprint IN ({marks})",
                (new, *old),
            ).rowcount
            self._conn.execute(
                f"UPDATE runs SET fingerprint=? WHERE fingerprint IN ({marks})", (new, *old)
            )
        return moved

    def already_scanned(self, fingerprint: str, include_failed: bool = True) -> set[str]:
        """Assets that are genuinely done: scanned under the current settings
        *and* settled.

        A changed threshold changes the fingerprint, which correctly makes
        previously-scanned assets eligible again.

        Assets that failed under the current settings count as done unless
        `include_failed` is false. Otherwise an unreadable file (a truncated
        clip, say) is picked again by every batch, crowding out new videos, and
        a batched backfill never reaches the end. `ivf scan --retry-failed`
        gives them another try.
        """
        status = "status IN ('ok', 'error')" if include_failed else "status='ok'"
        rows = self._conn.execute(
            "SELECT asset_id FROM scanned_assets "
            f"WHERE fingerprint=? AND settled=1 AND {status}",
            (fingerprint,),
        ).fetchall()
        return {row["asset_id"] for row in rows}

    def failed_assets(self) -> list[tuple[str, str]]:
        rows = self._conn.execute(
            "SELECT asset_id, coalesce(error, '') AS error FROM scanned_assets "
            "WHERE status<>'ok' ORDER BY asset_id"
        ).fetchall()
        return [(row["asset_id"], row["error"]) for row in rows]

    def discard_unsettled(self) -> tuple[int, int]:
        """Throw away any batch that was scanned but never committed.

        Keeps exactly one batch outstanding: starting a new one abandons the
        previous report and puts those videos back in the queue, so the newest
        report is always the only thing that matters.

        Returns (staged faces dropped, assets returned to the queue).
        """
        with self._write() as conn:
            returned = conn.execute(
                "SELECT count(*) AS n FROM scanned_assets WHERE settled=0"
            ).fetchone()["n"]
            if not returned:
                return 0, 0
            # Every uncommitted row, not just ones matching an unsettled asset:
            # a Live Photo is scanned as the clip but staged against its still,
            # so the two id sets do not line up.
            dropped = conn.execute(
                "DELETE FROM pending_writes WHERE committed_at IS NULL"
            ).rowcount or 0
            conn.execute("DELETE FROM scanned_assets WHERE settled=0")
            return dropped, int(returned)

    def settle_run(self, run_id: int) -> int:
        """Mark a scan run's assets as dealt with, so later batches move on."""
        with self._write() as conn:
            cursor = conn.execute(
                "UPDATE scanned_assets SET settled=1 WHERE run_id=?", (run_id,)
            )
            return cursor.rowcount or 0

    def record_scan(
        self,
        *,
        asset_id: str,
        fingerprint: str,
        run_id: int,
        frames: int = 0,
        observations: int = 0,
        clusters: int = 0,
        matches: int = 0,
        status: str = "ok",
        error: str | None = None,
    ) -> None:
        with self._write() as conn:
            conn.execute(
                """
                INSERT INTO scanned_assets
                    (asset_id, fingerprint, run_id, scanned_at, frames, observations,
                     clusters, matches, status, error)
                VALUES (?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(asset_id) DO UPDATE SET
                    fingerprint=excluded.fingerprint, run_id=excluded.run_id,
                    scanned_at=excluded.scanned_at, frames=excluded.frames,
                    observations=excluded.observations, clusters=excluded.clusters,
                    matches=excluded.matches, status=excluded.status, error=excluded.error
                """,
                (
                    asset_id,
                    fingerprint,
                    run_id,
                    time.time(),
                    frames,
                    observations,
                    clusters,
                    matches,
                    status,
                    error,
                ),
            )

    # ---- write audit ----------------------------------------------------

    def record_face(
        self,
        *,
        run_id: int,
        asset_id: str,
        person_id: str,
        image_width: int,
        image_height: int,
        box: tuple[int, int, int, int],
        distance: float,
    ) -> None:
        with self._write() as conn:
            conn.execute(
                """
                INSERT INTO written_faces
                    (run_id, asset_id, person_id, image_width, image_height,
                     x, y, width, height, distance, created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    run_id,
                    asset_id,
                    person_id,
                    image_width,
                    image_height,
                    *box,
                    distance,
                    time.time(),
                ),
            )

    # ---- staged writes ---------------------------------------------------

    def record_pending(
        self,
        *,
        run_id: int,
        asset_id: str,
        person_id: str,
        person_name: str,
        image_width: int,
        image_height: int,
        box: tuple[int, int, int, int],
        distance: float,
        margin: float,
        observations: int,
        segments: list[tuple[float, float]] | None = None,
    ) -> None:
        with self._write() as conn:
            conn.execute(
                """
                INSERT INTO pending_writes
                    (run_id, asset_id, person_id, person_name, image_width, image_height,
                     x, y, width, height, distance, margin, observations, created_at,
                     segments)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(run_id, asset_id, person_id) DO NOTHING
                """,
                (
                    run_id,
                    asset_id,
                    person_id,
                    person_name,
                    image_width,
                    image_height,
                    *box,
                    distance,
                    # SQLite has no infinity literal; store the sentinel as NULL.
                    None if margin == float("inf") else margin,
                    observations,
                    time.time(),
                    _dump_segments(segments),
                ),
            )

    def supersede_pending(self, asset_id: str, keep_run: int) -> int:
        """Drop older uncommitted staging for an asset that has just been rescanned.

        Re-scanning happens routinely -- `--force`, a settings change, an undo --
        and without this the same asset sits staged in two runs at once with
        different results. Committing both is harmless thanks to the idempotency
        check, but the review page then double-counts and it is unclear which
        run is authoritative. The newest scan wins.
        """
        with self._write() as conn:
            cursor = conn.execute(
                "DELETE FROM pending_writes "
                "WHERE asset_id=? AND run_id<>? AND committed_at IS NULL",
                (asset_id, keep_run),
            )
            return cursor.rowcount or 0

    def pending_writes(self, run_id: int) -> list[PendingWrite]:
        rows = self._conn.execute(
            "SELECT * FROM pending_writes WHERE run_id=? AND committed_at IS NULL "
            "ORDER BY asset_id, distance",
            (run_id,),
        ).fetchall()
        return [
            PendingWrite(
                id=int(row["id"]),
                asset_id=row["asset_id"],
                person_id=row["person_id"],
                person_name=row["person_name"],
                image_size=(row["image_width"], row["image_height"]),
                box=(row["x"], row["y"], row["width"], row["height"]),
                distance=float(row["distance"] or 0.0),
                segments=_load_segments(row["segments"]),
            )
            for row in rows
        ]

    def mark_committed(self, pending_ids: list[int]) -> None:
        if not pending_ids:
            return
        with self._write() as conn:
            conn.executemany(
                "UPDATE pending_writes SET committed_at=? WHERE id=?",
                [(time.time(), pid) for pid in pending_ids],
            )

    def committed_people(self, commit_run_id: int) -> list[tuple[str, str]]:
        """(asset, person) pairs a commit run applied, whether written or already there.

        Staged rows belong to the scan run that produced them; a commit run marks
        them with its own time window, which is how they are found again here.
        """
        rows = self._conn.execute(
            "SELECT DISTINCT p.asset_id, p.person_id FROM pending_writes p "
            "JOIN runs r ON r.id = ? AND r.mode = 'commit' "
            "WHERE p.committed_at BETWEEN r.started_at AND coalesce(r.finished_at, 1e18)",
            (commit_run_id,),
        ).fetchall()
        return [(row["asset_id"], row["person_id"]) for row in rows]

    def runs_awaiting_review(self) -> list[tuple[int, int]]:
        """(run_id, pending count) for runs with staged, uncommitted writes."""
        rows = self._conn.execute(
            "SELECT run_id, count(*) AS n FROM pending_writes "
            "WHERE committed_at IS NULL GROUP BY run_id ORDER BY run_id DESC"
        ).fetchall()
        return [(int(row["run_id"]), int(row["n"])) for row in rows]

    def record_unmatched(
        self,
        *,
        run_id: int,
        asset_id: str,
        embedding: "np.ndarray",
        observations: int,
        image_width: int,
        image_height: int,
        box: tuple[int, int, int, int],
        best_distance: float,
        crop_jpeg: bytes | None = None,
        segments: list[tuple[float, float]] | None = None,
    ) -> None:
        with self._write() as conn:
            conn.execute(
                """
                INSERT INTO unmatched_clusters
                    (run_id, asset_id, embedding, observations, image_width, image_height,
                     x, y, width, height, best_distance, created_at, crop_jpeg, segments)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    run_id,
                    asset_id,
                    embedding.astype("float32").tobytes(),
                    observations,
                    image_width,
                    image_height,
                    *box,
                    best_distance,
                    time.time(),
                    crop_jpeg,
                    _dump_segments(segments),
                ),
            )

    def clear_unmatched(self, asset_id: str) -> int:
        """Drop an asset's banked clusters before re-banking it.

        Scans re-run often -- `--force`, a threshold change, an undo -- and
        without this each pass appends a fresh copy of the same faces, inflating
        appearance counts and, once promoted, writing duplicate faces onto the
        same asset. Clusters already promoted are left alone: they belong to a
        person that now exists.
        """
        with self._write() as conn:
            cursor = conn.execute(
                "DELETE FROM unmatched_clusters WHERE asset_id=? AND promoted_to IS NULL",
                (asset_id,),
            )
            return cursor.rowcount or 0

    def pending_unmatched(self, group_nos: list[int] | None = None) -> list["UnmatchedCluster"]:
        import numpy as np

        sql = "SELECT * FROM unmatched_clusters WHERE promoted_to IS NULL"
        params: tuple = ()
        if group_nos:
            placeholders = ",".join("?" * len(group_nos))
            sql += f" AND group_no IN ({placeholders})"
            params = tuple(group_nos)
        sql += " ORDER BY id"

        rows = self._conn.execute(sql, params).fetchall()
        return [
            UnmatchedCluster(
                id=int(row["id"]),
                asset_id=row["asset_id"],
                embedding=np.frombuffer(row["embedding"], dtype="float32"),
                observations=int(row["observations"]),
                image_size=(row["image_width"], row["image_height"]),
                box=(row["x"], row["y"], row["width"], row["height"]),
                crop_jpeg=row["crop_jpeg"],
                group_no=row["group_no"],
                segments=_load_segments(row["segments"]),
            )
            for row in rows
        ]

    def stable_group_no(self, anchor_id: int) -> int:
        """The number for the group anchored on this cluster, allocating once.

        The anchor is the group's earliest-banked member, so a group keeps its
        number as later scans add more appearances to it.
        """
        row = self._conn.execute(
            "SELECT group_no FROM promote_groups WHERE anchor_id=?", (anchor_id,)
        ).fetchone()
        if row:
            return int(row["group_no"])
        with self._write() as conn:
            cursor = conn.execute(
                "INSERT INTO promote_groups (anchor_id, created_at) VALUES (?,?)",
                (anchor_id, time.time()),
            )
            return int(cursor.lastrowid or 0)

    def assign_groups(self, assignments: dict[int, int]) -> None:
        """Record which stable group each pending cluster currently belongs to."""
        with self._write() as conn:
            conn.execute("UPDATE unmatched_clusters SET group_no=NULL WHERE promoted_to IS NULL")
            conn.executemany(
                "UPDATE unmatched_clusters SET group_no=? WHERE id=?",
                [(group, cid) for cid, group in assignments.items()],
            )

    def reset_promotions(self, person_ids: list[str]) -> int:
        """Return clusters to the candidate pool when their person is deleted.

        Without this, undoing a promote strands them: still flagged as promoted,
        but pointing at a person that no longer exists, so they would never be
        offered again.
        """
        if not person_ids:
            return 0
        placeholders = ",".join("?" * len(person_ids))
        with self._write() as conn:
            cursor = conn.execute(
                f"UPDATE unmatched_clusters SET promoted_to=NULL, group_no=NULL "
                f"WHERE promoted_to IN ({placeholders})",
                tuple(person_ids),
            )
            return cursor.rowcount or 0

    def mark_promoted(self, cluster_ids: list[int], person_id: str) -> None:
        with self._write() as conn:
            conn.executemany(
                "UPDATE unmatched_clusters SET promoted_to=? WHERE id=?",
                [(person_id, cid) for cid in cluster_ids],
            )

    def record_person(self, run_id: int, person_id: str) -> None:
        with self._write() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO created_people (person_id, run_id, created_at) "
                "VALUES (?,?,?)",
                (person_id, run_id, time.time()),
            )

    def record_tag(self, run_id: int, asset_id: str, tag_id: str) -> None:
        with self._write() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO tagged_assets (asset_id, run_id, tag_id, created_at) "
                "VALUES (?,?,?,?)",
                (asset_id, run_id, tag_id, time.time()),
            )

    # ---- undo -----------------------------------------------------------

    def faces_for_run(self, run_id: int) -> list[WrittenFace]:
        rows = self._conn.execute(
            "SELECT id, asset_id, person_id, x, y, width, height FROM written_faces "
            "WHERE run_id=? AND revoked_at IS NULL ORDER BY asset_id",
            (run_id,),
        ).fetchall()
        return [
            WrittenFace(
                id=int(row["id"]),
                asset_id=row["asset_id"],
                person_id=row["person_id"],
                box=(row["x"], row["y"], row["width"], row["height"]),
            )
            for row in rows
        ]

    def mark_face_revoked(self, face_row_id: int) -> None:
        with self._write() as conn:
            conn.execute(
                "UPDATE written_faces SET revoked_at=? WHERE id=?", (time.time(), face_row_id)
            )

    def people_for_run(self, run_id: int) -> list[str]:
        rows = self._conn.execute(
            "SELECT person_id FROM created_people WHERE run_id=? AND revoked_at IS NULL",
            (run_id,),
        ).fetchall()
        return [row["person_id"] for row in rows]

    def mark_person_revoked(self, person_id: str) -> None:
        with self._write() as conn:
            conn.execute(
                "UPDATE created_people SET revoked_at=? WHERE person_id=?",
                (time.time(), person_id),
            )

    def tags_for_run(self, run_id: int) -> list[tuple[str, str]]:
        rows = self._conn.execute(
            "SELECT asset_id, tag_id FROM tagged_assets WHERE run_id=?", (run_id,)
        ).fetchall()
        return [(row["asset_id"], row["tag_id"]) for row in rows]

    def forget_assets(self, asset_ids: list[str]) -> None:
        """Make assets eligible for re-scanning after an undo."""
        if not asset_ids:
            return
        with self._write() as conn:
            conn.executemany(
                "DELETE FROM scanned_assets WHERE asset_id=?", [(a,) for a in asset_ids]
            )

    # ---- reporting ------------------------------------------------------

    def summary(self) -> dict[str, int]:
        def scalar(sql: str) -> int:
            return int(self._conn.execute(sql).fetchone()[0])

        return {
            "scanned": scalar("SELECT count(*) FROM scanned_assets WHERE status='ok'"),
            "failed": scalar("SELECT count(*) FROM scanned_assets WHERE status<>'ok'"),
            "faces_written": scalar(
                "SELECT count(*) FROM written_faces WHERE revoked_at IS NULL"
            ),
            "people_created": scalar(
                "SELECT count(*) FROM created_people WHERE revoked_at IS NULL"
            ),
            "runs": scalar("SELECT count(*) FROM runs"),
        }


def _dump_segments(segments: list[tuple[float, float]] | None) -> str | None:
    return json.dumps([[float(a), float(b)] for a, b in segments]) if segments else None


def _load_segments(raw: str | None) -> list[tuple[float, float]]:
    if not raw:
        return []
    return [(float(a), float(b)) for a, b in json.loads(raw)]
