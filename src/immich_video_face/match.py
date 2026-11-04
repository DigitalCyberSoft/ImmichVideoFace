"""Match video face clusters against Immich's labelled people."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from .cluster import FaceCluster
from .config import MatchSettings
from .immich_db import Gallery

log = logging.getLogger(__name__)


@dataclass
class MatchResult:
    cluster: FaceCluster
    person_id: str | None
    person_name: str
    distance: float
    # Distance to the next-best *different* person; a small gap means the
    # identification was nearly a coin flip and is worth reviewing.
    runner_up_distance: float
    # On-screen intervals, merged across every cluster that matched this person.
    segments: list[tuple[float, float]] = field(default_factory=list)

    @property
    def matched(self) -> bool:
        return self.person_id is not None

    @property
    def margin(self) -> float:
        if np.isinf(self.runner_up_distance):
            return float("inf")
        return self.runner_up_distance - self.distance


class Matcher:
    def __init__(self, gallery: Gallery, settings: MatchSettings) -> None:
        self.gallery = gallery
        self.settings = settings
        # Precompute the index ranges for each person so scoring is a couple of
        # vectorised ops rather than a Python loop over 29k faces per cluster.
        order = np.argsort(gallery.owners.astype(str), kind="stable")
        self._sorted_embeddings = gallery.embeddings[order]
        sorted_owners = gallery.owners[order].astype(str)
        self._person_ids, starts = np.unique(sorted_owners, return_index=True)
        self._slices = list(zip(starts, list(starts[1:]) + [len(sorted_owners)]))

    def _score(self, probe: np.ndarray) -> np.ndarray:
        """Mean distance to each person's k nearest faces, for one query vector."""
        distances = 1.0 - self._sorted_embeddings @ probe
        top_k = max(1, self.settings.top_k)
        scores = np.empty(len(self._person_ids), dtype=np.float32)
        for index, (start, end) in enumerate(self._slices):
            person_distances = distances[start:end]
            if person_distances.size > top_k:
                # Mean of the k nearest faces, rather than a single nearest
                # neighbour: one mislabelled face in the library shouldn't be
                # able to claim a whole video on its own.
                person_distances = np.partition(person_distances, top_k - 1)[:top_k]
            scores[index] = float(person_distances.mean())
        return scores

    def match(self, cluster: FaceCluster) -> MatchResult:
        # Score the centroid *and* the sharpest individual observations, then
        # keep whichever is closest per person.
        #
        # A centroid averages frontal, profile and motion-blurred views of the
        # same face across a whole video, which can drag it past the threshold
        # even when several frames match cleanly. Observed on real footage:
        # single frames matching at 0.29-0.35 whose cluster centroid landed at
        # 0.46 and was rejected.
        probes = [cluster.centroid]
        if self.settings.probe_observations > 0:
            sharpest = sorted(cluster.observations, key=lambda o: -o.quality)
            probes.extend(
                o.embedding for o in sharpest[: self.settings.probe_observations]
            )

        scores = self._score(probes[0])
        for probe in probes[1:]:
            np.minimum(scores, self._score(probe), out=scores)

        best = int(np.argmin(scores))
        best_distance = float(scores[best])
        runner_up = float("inf")
        if len(scores) > 1:
            runner_up = float(np.partition(scores, 1)[1])

        person_id = str(self._person_ids[best])

        # Prefer a named identity over an unnamed cluster when the two are
        # statistically indistinguishable. Libraries routinely hold the same
        # person as both a curated name and leftover unnamed clusters; when
        # they tie, attaching the video to the name is the useful answer.
        epsilon = self.settings.prefer_named_within
        if epsilon > 0 and not self.gallery.names.get(person_id):
            for index in np.argsort(scores)[: 5]:
                candidate = str(self._person_ids[int(index)])
                if not self.gallery.names.get(candidate):
                    continue
                if float(scores[int(index)]) - best_distance <= epsilon:
                    person_id = candidate
                    best_distance = float(scores[int(index)])
                break

        if best_distance > self.settings.max_distance:
            return MatchResult(cluster, None, "", best_distance, runner_up)

        # An ambiguous lead means two people scored almost identically; that is
        # a coin flip, and a wrong face on someone's page is worse than a miss.
        margin = runner_up - best_distance
        if margin < self.settings.min_margin:
            log.debug(
                "Rejecting %s: distance %.3f but margin only %.3f",
                person_id,
                best_distance,
                margin,
            )
            return MatchResult(cluster, None, "", best_distance, runner_up)

        return MatchResult(
            cluster=cluster,
            person_id=person_id,
            person_name=self.gallery.names.get(person_id, ""),
            distance=best_distance,
            runner_up_distance=runner_up,
        )

    def match_all(self, clusters: list[FaceCluster]) -> list[MatchResult]:
        results = [self.match(cluster) for cluster in clusters]
        for result in results:
            result.segments = list(result.cluster.segments)
        return _resolve_duplicates(results)


def nearest_people(
    gallery: Gallery, settings: MatchSettings, groups: list[tuple[int, list]]
) -> dict[int, tuple[str, str, float]]:
    """Closest known person to each promote candidate: {group_no: (name, id, distance)}.

    These groups failed the matching threshold, but "failed" is not the same as
    "unknown" -- a candidate sitting just past the cut-off is usually somebody
    already in the library. Showing the nearest person turns a blind
    create-a-stranger decision into an informed one.
    """
    from .immich_db import normalise

    matcher = Matcher(gallery, settings)
    result: dict[int, tuple[str, str, float]] = {}
    for number, members in groups:
        centroid = normalise(
            np.mean(np.vstack([m.embedding for m in members]), axis=0)
        )[0]
        scores = matcher._score(centroid)
        best = int(np.argmin(scores))
        person_id = str(matcher._person_ids[best])
        result[number] = (
            gallery.names.get(person_id) or "",
            person_id,
            float(scores[best]),
        )
    return result


def _resolve_duplicates(results: list[MatchResult]) -> list[MatchResult]:
    """Keep only the best cluster per person within a single video.

    Two clusters matching the same person usually means one appearance was
    split (a lighting or angle change). Writing both would create duplicate
    faces on the same asset for the same person.
    """
    from .cluster import merge_segments

    best_by_person: dict[str, MatchResult] = {}
    segments_by_person: dict[str, list] = {}
    unmatched: list[MatchResult] = []
    for result in results:
        if not result.matched:
            unmatched.append(result)
            continue
        assert result.person_id is not None
        # The split clusters are all this person, so all of their screen time counts.
        segments_by_person.setdefault(result.person_id, []).extend(result.segments)
        current = best_by_person.get(result.person_id)
        if current is None or result.distance < current.distance:
            if current is not None:
                log.debug(
                    "Merging duplicate cluster for person %s (%.3f vs %.3f)",
                    result.person_id,
                    result.distance,
                    current.distance,
                )
            best_by_person[result.person_id] = result
    for person_id, kept in best_by_person.items():
        kept.segments = merge_segments(segments_by_person[person_id])
    return sorted(best_by_person.values(), key=lambda r: r.distance) + unmatched
