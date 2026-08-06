"""Dry-run reporting.

Precision is the thing that matters here: a wrong face on someone's people page
is worse than a missed one. The report exists so matches can be eyeballed
against real crops before anything is written to Immich.
"""

from __future__ import annotations

import base64
import html
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from .scanner import ScanResult


def write_json(results: list[ScanResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = []
    for result in results:
        payload.append(
            {
                "asset_id": result.asset.id,
                # Where the faces actually go; differs for Live Photo clips.
                "write_target": result.asset.write_target,
                "taken_at": result.asset.taken_at,
                "path": str(result.path),
                "frames": result.frames,
                "observations": result.observations,
                "clusters": len(result.clusters),
                "error": result.error,
                "matches": [
                    {
                        "person_id": m.person_id,
                        "person_name": m.person_name,
                        "distance": round(m.distance, 4),
                        "margin": None if np.isinf(m.margin) else round(m.margin, 4),
                        "observations": m.cluster.size,
                        "frames": list(m.cluster.frame_range()),
                    }
                    for m in result.matches
                ],
            }
        )
    path.write_text(json.dumps(payload, indent=2))


def write_html(
    results: list[ScanResult],
    path: Path,
    run_id: int | None = None,
    promote_groups: list[tuple[int, list]] | None = None,
    min_assets: int = 3,
    immich_url: str = "",
    promote_hints: dict[int, tuple[str, str, float]] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[str] = []

    for result in results:
        heading = _asset_heading(result, immich_url)
        if not result.ok:
            rows.append(
                f'<div class="asset error">{heading}'
                f"<p>{html.escape(result.error or '')}</p></div>"
            )
            continue

        cards: list[str] = []
        for index, cluster in enumerate(result.clusters):
            match = next((m for m in result.matches if m.cluster is cluster), None)
            crop = result.crops.get(index)
            image = _img_tag(crop)
            if match is None:
                # Dropped by duplicate resolution: the same person was matched
                # more strongly by another cluster in this video.
                label = "merged"
                detail = "duplicate appearance"
                css = "merged"
            elif match.matched:
                label = html.escape(match.person_name or match.person_id or "?")
                detail = f"d={match.distance:.3f}"
                if not np.isinf(match.margin):
                    detail += f" margin={match.margin:.3f}"
                css = "match" if match.margin > 0.05 else "match narrow"
            else:
                label = "unmatched"
                detail = f"best d={match.distance:.3f}"
                css = "unmatched"
            cards.append(
                f'<figure class="{css}">{image}'
                f"<figcaption><strong>{label}</strong><br>{detail}<br>"
                f"{cluster.size} obs</figcaption></figure>"
            )

        rows.append(
            f'<div class="asset">{heading}'
            f"<p class='meta'>{result.frames} frames &middot; {result.observations} faces "
            f"&middot; {len(result.clusters)} clusters</p>"
            f'<div class="cards">{"".join(cards) or "<p>no faces</p>"}</div></div>'
        )

    total_matches = sum(len(r.matched) for r in results)
    commit_line = (
        f'<p class="commit">Nothing has been written to Immich. To apply these '
        f"{total_matches} faces: <code>ivf commit --run {run_id}</code></p>"
        if run_id is not None
        else ""
    )
    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>ImmichVideoFace review</title>
<style>
 body {{ font-family: system-ui, sans-serif; margin: 2rem; background:#111; color:#eee; }}
 h1 {{ font-size:1.4rem; }}
 .asset {{ border-top:1px solid #333; padding:1rem 0; }}
 .asset.error {{ color:#f88; }}
 .meta {{ color:#888; font-size:.85rem; margin:.2rem 0 .6rem; }}
 .cards {{ display:flex; flex-wrap:wrap; gap:.75rem; }}
 figure {{ margin:0; padding:.4rem; border-radius:6px; background:#1c1c1c; width:130px;
           border:2px solid transparent; }}
 figure.match {{ border-color:#3a7; }}
 figure.match.narrow {{ border-color:#da3; }}
 figure.unmatched {{ border-color:#555; }}
 figure.merged {{ border-color:#357; opacity:.7; }}
 figcaption {{ font-size:.75rem; margin-top:.3rem; line-height:1.3; }}
 img {{ width:100%; border-radius:4px; display:block; }}
 .commit {{ background:#182; color:#fff; padding:.6rem .8rem; border-radius:6px;
            font-size:.9rem; }}
 .commit code {{ background:#0006; padding:.15rem .4rem; border-radius:3px; }}
 h1.section {{ margin-top:2.5rem; padding-top:1.5rem; border-top:3px solid #444; }}
 .group {{ border-top:1px solid #333; padding:.8rem 0; }}
 .ghead {{ display:flex; align-items:center; gap:.6rem; margin-bottom:.4rem; }}
 .num {{ background:#3a7; color:#062; font-weight:700; border-radius:5px;
         padding:.1rem .55rem; font-size:1.05rem; }}
 .hint {{ font-size:.8rem; color:#aaa; }}
 .hint.close {{ color:#dc6; }}
 .into {{ font-size:.75rem; background:#0006; padding:.1rem .4rem;
          border-radius:3px; margin-left:.5rem; color:#9cf; }}
 .group img, .blank {{ width:78px; height:78px; border-radius:4px; background:#222; }}
 h2 a {{ color:#7cf; text-decoration:none; }}
 h2 a:hover {{ text-decoration:underline; }}
 .date, .aid {{ font-size:.72rem; font-weight:400; color:#888; margin-left:.5rem; }}
 .aid {{ font-family:ui-monospace,monospace; }}
 .via {{ font-size:.75rem; font-weight:400; color:#888; }}
 .badge {{ font-size:.68rem; font-weight:600; background:#345; color:#9cf;
           padding:.1rem .4rem; border-radius:3px; margin-left:.5rem; }}
</style></head><body>
<h1>ImmichVideoFace review</h1>
<p class="meta">{datetime.now():%Y-%m-%d %H:%M} &middot; {len(results)} assets &middot;
{total_matches} matches. Green = confident, amber = narrow margin, grey = unmatched,
blue = merged duplicate.</p>
{commit_line}
{"".join(rows)}
{_promote_section(promote_groups or [], min_assets, promote_hints)}
</body></html>"""
    path.write_text(document)


def encode_crop(crop: np.ndarray | None) -> bytes | None:
    """Small JPEG for storage alongside a banked cluster."""
    if crop is None or crop.size == 0:
        return None
    try:
        import cv2

        resized = cv2.resize(crop, (112, 112), interpolation=cv2.INTER_AREA)
        ok, buffer = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        return buffer.tobytes() if ok else None
    except Exception:
        return None


def _asset_heading(result: ScanResult, immich_url: str) -> str:
    """Identify the asset, and link to whatever the faces will be written to.

    Filename alone is not an identifier -- roughly one in seven videos in a real
    library shares its name with another, so searching Immich for
    "IMG_2482.mov" can easily land on the wrong clip.

    For a Live Photo the scanned clip and the write target are *different
    assets*: we decode the hidden motion clip but attach faces to the visible
    still. Showing only the clip would send a reviewer to a hidden asset that
    never receives anything, so the still is named and linked instead.
    """
    scanned = html.escape(result.path.name)
    target_id = result.asset.write_target
    date = html.escape(result.asset.taken_at or "")

    redirected = result.asset.still_asset_id is not None
    if redirected:
        still = html.escape(result.asset.still_filename or "the still")
        label = f"{still} <span class='via'>&larr; motion of {scanned}</span>"
    else:
        label = scanned

    bits = [f"<span class='date'>{date}</span>"] if date else []
    bits.append(f"<span class='aid'>{html.escape(target_id[:8])}</span>")
    if redirected:
        bits.append("<span class='badge'>Live Photo</span>")

    if immich_url:
        link = f"{immich_url.rstrip('/')}/photos/{target_id}"
        title = f'<a href="{html.escape(link)}" target="_blank">{label}</a>'
    else:
        title = label
    return f"<h2>{title} {' '.join(bits)}</h2>"


def _promote_section(
    groups: list[tuple[int, list]],
    min_assets: int,
    hints: dict[int, tuple[str, str, float]] | None = None,
) -> str:
    """Numbered new-person candidates, appended to the review page.

    Numbers are stable across reports, so `ivf promote 4` means the same group
    tomorrow as it does today. Each block also names the closest known person,
    because a candidate just past the matching threshold is usually somebody
    already in the library rather than a stranger.
    """
    hints = hints or {}
    if not groups:
        return (
            '<h1 class="section">New people</h1>'
            f"<p class='meta'>No unknown face recurs across {min_assets}+ videos yet. "
            "Scan more batches and they will appear here.</p>"
        )

    blocks = []
    for number, members in groups:
        videos = len({m.asset_id for m in members})
        tiles = []
        for member in sorted(members, key=lambda m: -m.observations)[:14]:
            if member.crop_jpeg:
                encoded = base64.b64encode(member.crop_jpeg).decode()
                tiles.append(f'<img src="data:image/jpeg;base64,{encoded}" alt="face">')
            else:
                tiles.append('<div class="blank"></div>')
        hint = ""
        name, person_id, distance = hints.get(number, ("", "", 0.0))
        if person_id:
            who = html.escape(name) if name else f"unnamed {person_id[:8]}"
            css = "close" if name and distance < 0.55 else ""
            hint = (
                f"<span class='hint {css}'>closest: <strong>{who}</strong> "
                f"({distance:.2f})</span>"
            )
            if name:
                hint += (
                    f"<code class='into'>ivf promote {number} "
                    f"--into {html.escape(name)}</code>"
                )

        blocks.append(
            f'<div class="group"><div class="ghead"><span class="num">{number}</span>'
            f"<span class='meta'>{videos} videos &middot; {len(members)} appearances</span>"
            f'{hint}</div><div class="cards">{"".join(tiles)}</div></div>'
        )

    numbers = ", ".join(str(n) for n, _ in groups[:4])
    return (
        '<h1 class="section">New people</h1>'
        f"<p class='meta'>{len(groups)} unknown faces recur across at least {min_assets} "
        "videos. Every face within a numbered block should be the same person. "
        "These numbers are stable across reports.</p>"
        f'<p class="commit">New person: <code>ivf promote {numbers}</code> '
        "&nbsp;&middot;&nbsp; existing person: "
        "<code>ivf promote &lt;n&gt; --into &lt;name&gt;</code></p>"
        f'{"".join(blocks)}'
    )


def _img_tag(crop: np.ndarray | None) -> str:
    if crop is None or crop.size == 0:
        return '<div style="height:110px;background:#222;border-radius:4px"></div>'
    try:
        import cv2

        resized = cv2.resize(crop, (112, 112), interpolation=cv2.INTER_AREA)
        ok, buffer = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            raise ValueError("encode failed")
        encoded = base64.b64encode(buffer.tobytes()).decode()
        return f'<img src="data:image/jpeg;base64,{encoded}" alt="face crop">'
    except Exception:
        return '<div style="height:110px;background:#222;border-radius:4px"></div>'
