"""Write results into Immich, and take them back again.

Everything here goes through the REST API. Two Immich behaviours shape this
module:

1. `createFace` stores `sourceType: Manual`, and the facial-recognition job
   skips any face that is not `machine-learning`. Our writes are therefore
   stable -- Immich will never re-cluster or reassign them.
2. `createFace` calls `createNewFeaturePhoto` when the person has no feature
   photo yet, so the *first* face written for a brand-new person becomes their
   avatar -- cropped from the video's thumbnail frame, not from the frame we
   detected in. Hence `_anchor_to_preview`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace

import numpy as np

from .config import Config
from .faces import FaceDetector
from .immich_api import ImmichClient
from .match import MatchResult
from .scanner import ScanResult
from .state import State

log = logging.getLogger(__name__)

# How close a preview detection must be to a cluster centroid to be considered
# the same person for thumbnail anchoring.
ANCHOR_MAX_DISTANCE = 0.45
# Immich rounds stored coordinates, so undo matches boxes with a tolerance.
BOX_TOLERANCE = 2


def box_for(result: ScanResult, match: MatchResult) -> tuple[tuple[int, int, int, int], tuple[int, int]]:
    """Return ((x, y, w, h), (image_width, image_height)) for a matched cluster.

    Coordinates stay in the space we detected in and Immich rescales them
    itself, so no conversion to the original resolution is needed.
    """
    x1, y1, x2, y2 = match.cluster.best.box
    box = (
        max(0, int(round(x1))),
        max(0, int(round(y1))),
        max(1, int(round(x2 - x1))),
        max(1, int(round(y2 - y1))),
    )
    return box, (result.frame_width, result.frame_height)


@dataclass
class WriteOutcome:
    faces_written: int = 0
    faces_skipped: int = 0
    faces_failed: int = 0
    failures: list[str] = field(default_factory=list)
    people_created: int = 0
    unmatched_banked: int = 0
    anchored: int = 0
    unanchored_people: list[str] = field(default_factory=list)
    tagged: bool = False


class Writer:
    def __init__(
        self,
        config: Config,
        client: ImmichClient,
        state: State,
        run_id: int,
        detector: FaceDetector | None = None,
    ) -> None:
        self.config = config
        self.client = client
        self.state = state
        self.run_id = run_id
        self.detector = detector
        self._tag_id: str | None = None

    # ---- tags -----------------------------------------------------------

    def tag_id(self) -> str:
        if self._tag_id is None:
            self._tag_id = self.client.upsert_tag(self.config.writeback.tag)
        return self._tag_id

    # ---- commit staged writes --------------------------------------------

    def commit(self, pending: list) -> WriteOutcome:
        """Apply writes that a scan staged and a human has since reviewed.

        Idempotency is re-checked here rather than at scan time, because the
        library may have changed between staging and review.
        """
        outcome = WriteOutcome()

        by_asset: dict[str, list] = {}
        for item in pending:
            by_asset.setdefault(item.asset_id, []).append(item)

        tag_id = self.tag_id() if self.config.writeback.tag else None
        committed: list[int] = []

        for asset_id, items in by_asset.items():
            try:
                existing = {
                    face["person"]["id"]
                    for face in self.client.get_faces(asset_id)
                    if face.get("person")
                }
            except Exception as exc:
                log.warning("Skipping %s, could not read faces: %s", asset_id, exc)
                continue

            wrote_here = False
            for item in items:
                if item.person_id in existing:
                    outcome.faces_skipped += 1
                    committed.append(item.id)
                    continue

                x, y, width, height = item.box
                try:
                    self.client.create_face(
                        asset_id=asset_id,
                        person_id=item.person_id,
                        image_width=item.image_size[0],
                        image_height=item.image_size[1],
                        x=x,
                        y=y,
                        width=width,
                        height=height,
                    )
                except Exception as exc:
                    # A person merged away in Immich, or a deleted asset. One
                    # stale row must not abort the whole batch, so record it
                    # and carry on; it stays uncommitted for a later attempt.
                    outcome.faces_failed += 1
                    outcome.failures.append(f"{asset_id[:8]}/{item.person_id[:8]}: {exc}")
                    log.warning("Could not write face on %s: %s", asset_id, exc)
                    continue

                self.state.record_face(
                    run_id=self.run_id,
                    asset_id=asset_id,
                    person_id=item.person_id,
                    image_width=item.image_size[0],
                    image_height=item.image_size[1],
                    box=item.box,
                    distance=item.distance,
                )
                outcome.faces_written += 1
                committed.append(item.id)
                wrote_here = True

            if tag_id and wrote_here:
                self.client.tag_assets(tag_id, [asset_id])
                self.state.record_tag(self.run_id, asset_id, tag_id)
                outcome.tagged = True

            # Flush per asset so an interruption keeps the progress made.
            self.state.mark_committed(committed)
            committed.clear()

        return outcome

    def _preview_anchors(
        self, result: ScanResult
    ) -> dict[bytes, tuple[tuple[int, int, int, int], tuple[int, int]]]:
        """Find each cluster's face in the asset's own preview image.

        Rather than guessing which timestamp Immich used for the video
        thumbnail, we detect on the preview itself and match by embedding. When
        a cluster is present there, we write *that* box, so the avatar Immich
        generates for a new person is a real face.
        """
        if self.detector is None:
            return {}

        try:
            import cv2

            raw = self.client.get_preview(result.asset.id)
            image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        except Exception as exc:  # preview may not exist yet
            log.debug("No preview for %s: %s", result.asset.id, exc)
            return {}

        if image is None:
            return {}

        detections = self.detector.detect(image)
        if not detections:
            return {}

        height, width = image.shape[:2]
        anchors: dict[bytes, tuple[tuple[int, int, int, int], tuple[int, int]]] = {}
        for cluster in result.clusters:
            distances = [
                (1.0 - float(np.dot(cluster.centroid, det.embedding)), det)
                for det in detections
            ]
            distance, best = min(distances, key=lambda pair: pair[0])
            if distance > ANCHOR_MAX_DISTANCE:
                continue
            x1, y1, x2, y2 = best.box
            anchors[cluster.centroid.tobytes()] = (
                (
                    max(0, int(round(x1))),
                    max(0, int(round(y1))),
                    max(1, int(round(x2 - x1))),
                    max(1, int(round(y2 - y1))),
                ),
                (width, height),
            )
        return anchors


# ---- promotion of unmatched clusters --------------------------------------


@dataclass
class PromoteOutcome:
    groups_considered: int = 0
    people_created: int = 0
    faces_written: int = 0
    skipped_small: int = 0
    anchored: int = 0
    unanchored_people: list[str] = field(default_factory=list)


def anchor_box(
    client: ImmichClient,
    detector: FaceDetector,
    asset_id: str,
    embedding: "np.ndarray",
) -> tuple[tuple[int, int, int, int], tuple[int, int]] | None:
    """Locate this face in the asset's own preview image.

    `createFace` calls `createNewFeaturePhoto` when a person has no feature
    photo, and Immich crops that avatar from the asset's *thumbnail* frame --
    not the frame we detected in. So for a brand-new person we detect on the
    preview itself and, if the same face is there, write that box instead.
    Guessing which timestamp Immich used for the thumbnail is unnecessary.
    """
    import numpy as np

    try:
        import cv2

        raw = client.get_preview(asset_id)
        image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception as exc:
        log.debug("No preview for %s: %s", asset_id, exc)
        return None
    if image is None:
        return None

    detections = detector.detect(image)
    if not detections:
        return None

    best = min(detections, key=lambda d: 1.0 - float(np.dot(embedding, d.embedding)))
    if 1.0 - float(np.dot(embedding, best.embedding)) > ANCHOR_MAX_DISTANCE:
        return None

    height, width = image.shape[:2]
    x1, y1, x2, y2 = best.box
    return (
        (
            max(0, int(round(x1))),
            max(0, int(round(y1))),
            max(1, int(round(x2 - x1))),
            max(1, int(round(y2 - y1))),
        ),
        (width, height),
    )


def group_unmatched(
    pending: list, *, min_assets: int, distance: float, state: State
) -> tuple[list[tuple[int, list]], int]:
    """Cluster banked unmatched faces *across* videos.

    The bar is deliberately high: a group must span at least `min_assets`
    distinct videos. A stranger who walks through one clip stays unnamed noise;
    someone recurring across several is worth a person record.

    `state` is required, deliberately. Numbers must come from the persistent
    anchor mapping, never from position in this particular list: a caller that
    numbered positionally would both disagree with the report and overwrite the
    stored mapping, so `ivf promote 6` could act on a different group than the
    report showed.
    """
    import numpy as np

    if not pending:
        return [], 0

    from .immich_db import normalise

    matrix = normalise(np.vstack([c.embedding for c in pending]))
    if len(pending) == 1:
        labels = np.array([0])
    else:
        from sklearn.cluster import AgglomerativeClustering

        labels = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance,
            metric="cosine",
            linkage="average",
        ).fit_predict(matrix)

    raw: list[list] = []
    for label in sorted(set(labels.tolist())):
        raw.append([c for c, lab in zip(pending, labels) if lab == label])

    qualifying = [m for m in raw if len({c.asset_id for c in m}) >= min_assets]
    qualifying.sort(key=lambda m: (-len({c.asset_id for c in m}), -len(m)))

    numbered = [
        (state.stable_group_no(min(c.id for c in members)), members)
        for members in qualifying
    ]
    numbered.sort(key=lambda pair: pair[0])
    return numbered, len(raw)


def promote_unmatched(
    config: Config,
    client: ImmichClient,
    state: State,
    run_id: int,
    *,
    min_assets: int = 3,
    distance: float = 0.35,
    dry_run: bool = True,
    detector: FaceDetector | None = None,
    only_groups: list[int] | None = None,
    into_person: str | None = None,
) -> tuple[PromoteOutcome, list[tuple[int, list]]]:
    """Attach qualifying cross-video groups to a person.

    By default each group becomes a new unnamed person. With `into_person`,
    the groups are attached to that existing person instead -- the answer to
    "this candidate is obviously Alex".

    With `only_groups`, acts on exactly the group numbers shown in the review
    page and leaves everything else banked.
    """
    outcome = PromoteOutcome()

    if only_groups:
        # Act on the numbers the review page recorded, never on a fresh
        # recomputation. The pool grows with every scan, so regrouping here
        # would silently renumber and promote the wrong people.
        members_by_group: dict[int, list] = {}
        for cluster in state.pending_unmatched(group_nos=only_groups):
            members_by_group.setdefault(cluster.group_no, []).append(cluster)
        groups = sorted(members_by_group.items())
    else:
        pending = state.pending_unmatched()
        if not pending:
            return outcome, []
        groups, raw_count = group_unmatched(
            pending, min_assets=min_assets, distance=distance, state=state
        )
        outcome.groups_considered = raw_count
        outcome.skipped_small = raw_count - len(groups)
        # Listing refreshes the stored numbering, so a bare `ivf promote`
        # and the numbers it prints stay in step.
        state.assign_groups({c.id: no for no, mem in groups for c in mem})

    if dry_run:
        return outcome, groups

    # Same marker commit applies, so "has this video been written to?" has one
    # answer regardless of which path wrote it.
    tag_id = client.upsert_tag(config.writeback.tag) if config.writeback.tag else None

    for _, members in groups:
        if into_person:
            # Attaching to somebody who already exists: they have an avatar and
            # a name, so neither creation nor anchoring applies.
            person_id = into_person
        else:
            person_id = client.create_person("")
            state.record_person(run_id, person_id)
            outcome.people_created += 1

        # The first face written becomes a *new* person's avatar, cropped from
        # the asset's thumbnail frame. Lead with one we can anchor to a real
        # face in that preview, so the avatar isn't a patch of carpet.
        ordered = list(members)
        if detector is not None and not into_person:
            for index, member in enumerate(ordered):
                anchored = anchor_box(client, detector, member.asset_id, member.embedding)
                if anchored is not None:
                    ordered.insert(0, ordered.pop(index))
                    ordered[0] = replace(
                        ordered[0], box=anchored[0], image_size=anchored[1]
                    )
                    outcome.anchored += 1
                    break
            else:
                outcome.unanchored_people.append(person_id)

        seen_assets: set[str] = set()
        for member in ordered:
            # One face per asset per person: a group can legitimately hold
            # several appearances from the same video, and Immich only needs
            # one to associate the two.
            if member.asset_id in seen_assets:
                state.mark_promoted([member.id], person_id)
                continue
            seen_assets.add(member.asset_id)

            x, y, width, height = member.box
            client.create_face(
                asset_id=member.asset_id,
                person_id=person_id,
                image_width=member.image_size[0],
                image_height=member.image_size[1],
                x=x,
                y=y,
                width=width,
                height=height,
            )
            state.record_face(
                run_id=run_id,
                asset_id=member.asset_id,
                person_id=person_id,
                image_width=member.image_size[0],
                image_height=member.image_size[1],
                box=member.box,
                distance=-1.0,
            )
            outcome.faces_written += 1

            if tag_id:
                client.tag_assets(tag_id, [member.asset_id])
                state.record_tag(run_id, member.asset_id, tag_id)

        state.mark_promoted([m.id for m in members], person_id)

    return outcome, groups


# ---- undo ----------------------------------------------------------------


@dataclass
class UndoOutcome:
    faces_deleted: int = 0
    faces_missing: int = 0
    people_deleted: int = 0
    people_kept: int = 0
    assets_untagged: int = 0
    clusters_returned: int = 0


def undo_run(
    config: Config, client: ImmichClient, state: State, run_id: int
) -> UndoOutcome:
    """Remove everything a run wrote.

    Immich returns no id from `createFace`, so we re-read each asset's faces and
    identify ours by person, manual source type, and the bounding box we
    recorded.
    """
    outcome = UndoOutcome()
    written = state.faces_for_run(run_id)

    by_asset: dict[str, list] = {}
    for face in written:
        by_asset.setdefault(face.asset_id, []).append(face)

    for asset_id, records in by_asset.items():
        try:
            current = client.get_faces(asset_id)
        except Exception as exc:
            log.warning("Could not read faces for %s: %s", asset_id, exc)
            continue

        for record in records:
            target = _find_face(current, record)
            if target is None:
                outcome.faces_missing += 1
                state.mark_face_revoked(record.id)
                continue
            client.delete_face(target["id"])
            state.mark_face_revoked(record.id)
            outcome.faces_deleted += 1

    deleted_people: list[str] = []
    for person_id in state.people_for_run(run_id):
        try:
            remaining = client.get_person_face_count(person_id)
        except Exception:
            remaining = None
        if remaining:
            # The user has since attached real faces; leave the person alone.
            outcome.people_kept += 1
            continue
        try:
            client.delete_person(person_id)
            state.mark_person_revoked(person_id)
            deleted_people.append(person_id)
            outcome.people_deleted += 1
        except Exception as exc:
            log.warning("Could not delete person %s: %s", person_id, exc)
            outcome.people_kept += 1

    # Their source clusters become candidates again on the next scan.
    outcome.clusters_returned = state.reset_promotions(deleted_people)

    tagged = state.tags_for_run(run_id)
    by_tag: dict[str, list[str]] = {}
    for asset_id, tag_id in tagged:
        by_tag.setdefault(tag_id, []).append(asset_id)
    for tag_id, asset_ids in by_tag.items():
        try:
            client.untag_assets(tag_id, asset_ids)
            outcome.assets_untagged += len(asset_ids)
        except Exception as exc:
            log.warning("Could not untag for tag %s: %s", tag_id, exc)

    state.forget_assets(sorted(by_asset))
    return outcome


def _find_face(faces: list[dict], record) -> dict | None:
    x, y, width, height = record.box
    for face in faces:
        person = face.get("person") or {}
        if person.get("id") != record.person_id:
            continue
        if face.get("sourceType") != "manual":
            continue
        if (
            abs(face.get("boundingBoxX1", -999) - x) <= BOX_TOLERANCE
            and abs(face.get("boundingBoxY1", -999) - y) <= BOX_TOLERANCE
            and abs(face.get("boundingBoxX2", -999) - (x + width)) <= BOX_TOLERANCE
            and abs(face.get("boundingBoxY2", -999) - (y + height)) <= BOX_TOLERANCE
        ):
            return face
    return None
