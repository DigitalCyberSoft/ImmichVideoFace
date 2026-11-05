"""Flag sensitive content with a tag, from embeddings Immich already has.

Every photo has a smart search embedding, and every indexed video has one per
sampled frame (see video_index). Each flagged category is a set of prompts and
a set of look-alikes (`contrast`): an asset scores how much closer it is to the
nearest prompt than to the nearest look-alike, a video by its best frame. Only
the prompts go through the model; the scoring is one query in Postgres.

Assets that pass get the tag `<tag>/<category>`. A patched Immich serves
anything tagged under `<tag>` blurred until the session is unlocked with the
PIN. The tool remembers what it tagged, so:

- a tag removed by hand stays removed (the person looking decided),
- a tag added by hand is never touched,
- a tag the tool added is taken off again when its category stops being
  flagged or the asset stops passing (after a settings change).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field

import numpy as np

from .config import Config, SensitiveCategory

log = logging.getLogger(__name__)


@dataclass
class CategoryPlan:
    category: str
    tag_name: str
    scored: int = 0
    passing: set[str] = field(default_factory=set)
    to_tag: list[str] = field(default_factory=list)
    to_untag: list[str] = field(default_factory=list)
    cleared_by_hand: list[str] = field(default_factory=list)
    already: int = 0
    scores: dict[str, float] = field(default_factory=dict)


def embed_texts(config: Config, texts: list[str]) -> np.ndarray:
    """Text embeddings from Immich's ML service, with the smart search model."""
    import httpx

    entries = json.dumps({"clip": {"textual": {"modelName": config.objects.model}}})
    vectors = []
    with httpx.Client(base_url=config.detect.ml_url.rstrip("/"), timeout=300.0) as client:
        for text in texts:
            for attempt in range(4):
                try:
                    response = client.post("/predict", data={"entries": entries, "text": text})
                    response.raise_for_status()
                    break
                except httpx.HTTPError:
                    if attempt == 3:
                        raise
                    time.sleep(2**attempt)
            raw = response.json()["clip"]
            vector = np.asarray(json.loads(raw) if isinstance(raw, str) else raw, dtype=np.float32)
            vectors.append(vector / np.linalg.norm(vector))
    return np.stack(vectors)


def _literal(vector: np.ndarray) -> str:
    return "[" + ",".join(f"{float(v):.8g}" for v in vector) + "]"


def score_assets(
    conn, config: Config, owner_id: str, category: SensitiveCategory
) -> dict[str, float]:
    """{asset id: score} for every photo and indexed video of the owner.

    The score is how far the asset clears the nearer of its two cut-offs
    (`min_similarity`, `min_margin`), so it is flagged when the score is 0 or more.
    A video scores by its `min_frames`-th best sampled frame (or its worst, when it
    has fewer), or by its thumbnail if that is higher.
    """
    prompts = embed_texts(config, category.prompts)
    contrast = embed_texts(config, category.contrast)

    def nearest(column: str, vectors: np.ndarray) -> tuple[str, list[str]]:
        terms = ", ".join(f"1 - ({column} <=> %s::vector)" for _ in vectors)
        return f"GREATEST({terms})", [_literal(v) for v in vectors]

    def frames(column: str, id_sql: str, source: str, where: str) -> tuple[str, list]:
        """Per row: the asset id, its similarity to the prompts and to the contrast."""
        pos_sql, pos_args = nearest(column, prompts)
        neg_sql, neg_args = nearest(column, contrast)
        sql = (
            f"SELECT {id_sql} AS asset_id, {pos_sql} AS sim, {neg_sql} AS neg "
            f"FROM {source} WHERE {where}"
        )
        return sql, [*pos_args, *neg_args]

    clears = "LEAST(sim - %s, sim - neg - %s)"
    cuts = [category.min_similarity, category.min_margin]
    queries = [
        frames(
            "s.embedding",
            "a.id::text",
            'smart_search s JOIN asset a ON a.id = s."assetId"',
            'a."ownerId" = %s AND a."deletedAt" IS NULL',
        )
    ]
    queries[0][1].append(owner_id)
    if config.index.enabled:
        schema = config.index.schema_name.replace('"', '""')
        sql, args = frames(
            "f.embedding",
            "f.asset_id::text",
            f'"{schema}".frame_embedding f JOIN asset a ON a.id = f.asset_id',
            'f.model = %s AND a."ownerId" = %s AND a."deletedAt" IS NULL',
        )
        queries.append((sql, [*args, config.objects.model, owner_id]))

    scores: dict[str, float] = {}
    with conn.cursor() as cur:
        for sql, args in queries:
            cur.execute(
                f"""
                SELECT asset_id, score FROM (
                    SELECT asset_id, score,
                           row_number() OVER (PARTITION BY asset_id ORDER BY score DESC) AS rank,
                           count(*) OVER (PARTITION BY asset_id) AS frames
                    FROM (SELECT asset_id, {clears} AS score FROM ({sql}) sims) scored
                ) ranked
                WHERE rank = LEAST(%s, frames)
                """,
                [*cuts, *args, max(1, category.min_frames)],
            )
            for row in cur.fetchall():
                score = float(row["score"])
                if score > scores.get(row["asset_id"], -np.inf):
                    scores[row["asset_id"]] = score
    return scores


def tagged_with(conn, owner_id: str, tag_name: str) -> set[str]:
    """Assets currently carrying exactly this tag."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT ta."assetId"::text AS asset_id
            FROM tag_asset ta JOIN tag t ON t.id = ta."tagId"
            WHERE t."userId" = %s AND t.value = %s
            """,
            (owner_id, tag_name),
        )
        return {row["asset_id"] for row in cur.fetchall()}


def sensitive_assets(conn, owner_id: str, root: str) -> set[str]:
    """Assets tagged with `root` or anything under it, by hand or by this tool."""
    under = root.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "/%"
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT DISTINCT ta."assetId"::text AS asset_id
            FROM tag_asset ta JOIN tag t ON t.id = ta."tagId"
            WHERE t."userId" = %s AND (t.value = %s OR t.value LIKE %s)
            """,
            (owner_id, root, under),
        )
        return {row["asset_id"] for row in cur.fetchall()}


def plan_category(
    conn, state, config: Config, owner_id: str, category: str, flagged: bool
) -> CategoryPlan:
    """Work out the tag changes for one category, without making any."""
    settings = config.sensitive
    plan = CategoryPlan(category=category, tag_name=f"{settings.tag}/{category}")
    current = tagged_with(conn, owner_id, plan.tag_name)
    ours = state.sensitive_tagged(category)  # {asset id: cleared?}

    # Tags we put on that are gone now were taken off by hand: remember that.
    for asset_id, cleared in ours.items():
        if not cleared and asset_id not in current:
            plan.cleared_by_hand.append(asset_id)
    cleared = {a for a, c in ours.items() if c} | set(plan.cleared_by_hand)

    if flagged:
        plan.scores = score_assets(conn, config, owner_id, settings.categories[category])
        plan.scored = len(plan.scores)
        plan.passing = {a for a, score in plan.scores.items() if score >= 0}
    for asset_id in sorted(plan.passing):
        if asset_id in current:
            plan.already += 1
        elif asset_id not in cleared:
            plan.to_tag.append(asset_id)
    for asset_id, was_cleared in ours.items():
        if not was_cleared and asset_id in current and asset_id not in plan.passing:
            plan.to_untag.append(asset_id)
    return plan


def apply_plan(client, state, plan: CategoryPlan) -> None:
    for asset_id in plan.cleared_by_hand:
        state.sensitive_mark_cleared(plan.category, asset_id)
    if not (plan.to_tag or plan.to_untag):
        return
    tag_id = client.upsert_tag(plan.tag_name)
    for start in range(0, len(plan.to_tag), 500):
        chunk = plan.to_tag[start : start + 500]
        client.tag_assets(tag_id, chunk)
        state.sensitive_record(plan.category, chunk, tag_id)
    for start in range(0, len(plan.to_untag), 500):
        chunk = plan.to_untag[start : start + 500]
        client.untag_assets(tag_id, chunk)
        state.sensitive_forget(plan.category, chunk)


def cover_face_fixes(db, owner_id: str, hidden: set[str]) -> list[tuple[str, str, str]]:
    """(person id, name, replacement asset id) for each person whose cover face is on
    a sensitive asset, choosing their largest face on an asset that is not.

    People with no other face are left as they are."""
    if not hidden:
        return []
    face, person = db._face_col, db._person_col
    with db._conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT p."{person}"::text AS person_id, p.name, cover."assetId"::text AS asset_id
            FROM person p JOIN asset_face cover ON cover.id = p."faceAssetId"
            WHERE p."ownerId" = %s AND cover."assetId"::text = ANY(%s)
            """,
            (owner_id, list(hidden)),
        )
        stuck = cur.fetchall()
        fixes = []
        for row in stuck:
            cur.execute(
                f"""
                SELECT af."assetId"::text AS asset_id
                FROM asset_face af JOIN asset a ON a.id = af."assetId"
                WHERE af."{face}"::text = %s AND af."deletedAt" IS NULL AND af."isVisible"
                  AND a."deletedAt" IS NULL AND a.visibility = 'timeline'
                  AND NOT (af."assetId"::text = ANY(%s))
                ORDER BY (af."boundingBoxX2" - af."boundingBoxX1")::float / greatest(af."imageWidth", 1)
                       * (af."boundingBoxY2" - af."boundingBoxY1") / greatest(af."imageHeight", 1) DESC
                LIMIT 1
                """,
                (row["person_id"], list(hidden)),
            )
            replacement = cur.fetchone()
            if replacement:
                fixes.append((row["person_id"], row["name"] or "", replacement["asset_id"]))
    return fixes
