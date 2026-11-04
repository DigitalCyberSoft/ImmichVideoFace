"""Command line interface."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn
from rich.table import Table

from .config import Config, load_config

app = typer.Typer(add_completion=False, help="Whole-video facial recognition for Immich.")
console = Console()

_config_option = typer.Option(None, "--config", "-c", help="Path to config.toml")


def _setup(config_path: Optional[Path], verbose: bool = False) -> Config:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if verbose:
        logging.getLogger("immich_video_face").setLevel(logging.DEBUG)
    return load_config(config_path)


def _open_index(config: Config):
    """The video index, when publishing is switched on."""
    if not config.index.enabled:
        return None
    from .video_index import VideoIndex

    return VideoIndex(config)


def _rekey(config: Config, state) -> None:
    """Carry finished scans over from fingerprints that still included the
    search model (before it was left out), so a model change rescans nothing."""
    if not config.objects.enabled:
        return
    models = {config.objects.model}
    index = _open_index(config)
    if index is not None:
        try:
            models |= index.frame_models()
        finally:
            index.close()
    moved = state.rekey({config.legacy_fingerprint(m) for m in models}, config.fingerprint)
    if moved:
        console.print(f"Carried {moved} finished scans over to fingerprint {config.fingerprint}.")


def _owner_id(config: Config, client) -> str:
    return config.immich.owner_id or client.my_user_id()


def _check_model(config: Config, client) -> bool:
    """Refuse to run when our model differs from the one Immich used.

    Immich's stored embeddings come from its configured recognition model.
    A different model produces a different vector space, so matching would
    return confident nonsense rather than failing visibly.
    """
    theirs = client.face_model()
    if theirs is None:
        console.print(
            "[yellow]Could not read Immich's face model (API key lacks admin "
            "rights). Run `ivf parity` to confirm the models agree.[/yellow]"
        )
        return True
    if theirs == config.detect.model:
        if config.objects.enabled:
            clip = client.clip_model()
            if clip is not None and clip != config.objects.model:
                console.print(
                    f"[red]Search model mismatch:[/red] Immich uses [bold]{clip}[/bold], "
                    f"this tool is set to [bold]{config.objects.model}[/bold]. "
                    f"Set objects.model = \"{clip}\" in config.toml."
                )
                return False
        return True
    console.print(
        f"[red]Model mismatch:[/red] Immich uses [bold]{theirs}[/bold], "
        f"this tool is set to [bold]{config.detect.model}[/bold]."
    )
    console.print(
        f"Their embeddings are not comparable. Set detect.model = \"{theirs}\" "
        "in config.toml."
    )
    return False


def _resolve_person(client, reference: str) -> Optional[str]:
    """Accept a person's name or id, insisting on an unambiguous answer."""
    if len(reference) == 36 and reference.count("-") == 4:
        return reference

    matches = [
        person
        for person in client.search_person(reference)
        if (person.get("name") or "").lower() == reference.lower()
    ]
    if not matches:
        console.print(f"[red]No person named {reference!r}.[/red]")
        return None
    if len(matches) > 1:
        console.print(f"[red]{len(matches)} people are named {reference!r}:[/red]")
        for person in matches:
            console.print(f"  {person['id']}")
        console.print("Pass the id instead.")
        return None
    console.print(f"Resolved {reference!r} -> {matches[0]['id']}")
    return matches[0]["id"]


@app.command()
def check(config: Optional[Path] = _config_option, verbose: bool = False) -> None:
    """Verify config, API access, database access and schema."""
    from .immich_api import ImmichClient
    from .immich_db import ImmichDb

    cfg = _setup(config, verbose)
    table = Table(title="ImmichVideoFace preflight", show_header=False)

    with ImmichClient(cfg) as client:
        table.add_row("Immich server", client.server_version())
        owner = _owner_id(cfg, client)
        table.add_row("Owner id", owner)

        theirs = client.face_model()
        if theirs is None:
            table.add_row("Face model", f"{cfg.detect.model} (Immich's not readable)")
        elif theirs == cfg.detect.model:
            table.add_row("Face model", f"[green]{theirs}[/green] (matches Immich)")
        else:
            table.add_row(
                "Face model",
                f"[red]{cfg.detect.model} vs Immich's {theirs} - MISMATCH[/red]",
            )

        if cfg.objects.enabled:
            clip = client.clip_model()
            if clip is None:
                table.add_row("Search model", f"{cfg.objects.model} (Immich's not readable)")
            elif clip == cfg.objects.model:
                table.add_row("Search model", f"[green]{clip}[/green] (matches Immich)")
            else:
                table.add_row(
                    "Search model",
                    f"[red]{cfg.objects.model} vs Immich's {clip} - MISMATCH[/red]",
                )

        with ImmichDb(cfg) as db:
            db.assert_schema()
            table.add_row("DB schema", f"[green]ok[/green] ({db.layout})")
            stats = db.person_stats(owner)
            table.add_row("People", f"{stats['total']} ({stats['named']} named)")
            gallery = db.load_gallery(owner)
            table.add_row(
                "Reference gallery",
                f"{gallery.face_count} faces across {len(gallery)} people",
            )

    if cfg.index.enabled:
        index = _open_index(cfg)
        stats = index.stats()
        index.close()
        table.add_row(
            "Video index",
            f"schema {cfg.index.schema_name}: {stats.videos} videos, "
            f"{stats.frames} frames, {stats.appearances} appearances",
        )
    device = {True: "GPU", False: "CPU", None: "set explicitly"}[cfg.gpu_resolved]
    table.add_row(
        "Sampling",
        f"{cfg.sampling.min_fps:g} fps for faces ({device}); "
        f"min {cfg.cluster.min_observations} sighting(s); new appearance after {cfg.timeline.gap_s:g} s"
        + (f"; objects every {cfg.objects.interval_s:g} s" if cfg.objects.enabled else ""),
    )
    table.add_row("Library root", cfg.paths.data_root_host)
    table.add_row("Config fingerprint", cfg.fingerprint)
    console.print(table)


@app.command()
def parity(
    config: Optional[Path] = _config_option,
    samples: int = typer.Option(8, help="Number of faces to verify"),
    verbose: bool = False,
) -> None:
    """Gate 1: prove our embeddings match the ones Immich stored.

    Pulls known faces from `face_search`, re-runs our detection and embedding
    over the same preview image Immich used, and reports cosine distance. If
    these are not near zero, every downstream match is meaningless.
    """
    import cv2
    import numpy as np

    from .faces import FaceDetector, iou
    from .immich_db import ImmichDb, normalise

    cfg = _setup(config, verbose)
    detector = FaceDetector(cfg.detect)

    # Previews are readable straight off the mounted library, so this gate needs
    # no API key -- useful, because it is the one to run before anything else.
    client = None
    if cfg.immich.api_key:
        from .immich_api import ImmichClient

        client = ImmichClient(cfg)

    with ImmichDb(cfg) as db:
        db.assert_schema()
        owner = cfg.immich.owner_id or (client.my_user_id() if client else "")
        if not owner:
            console.print("[red]Set immich.owner_id or an api_key in config.[/red]")
            raise typer.Exit(1)

        references = db.sample_reference_faces(owner, limit=samples)
        if not references:
            console.print("[red]No suitable reference faces found.[/red]")
            raise typer.Exit(1)

        table = Table(title="Embedding parity vs Immich")
        table.add_column("Person")
        table.add_column("IoU", justify="right")
        table.add_column("Cosine distance", justify="right")
        table.add_column("Verdict")

        distances: list[float] = []
        for reference in references:
            image = None
            if reference.preview_path:
                on_disk = cfg.paths.to_host(reference.preview_path)
                if on_disk.exists():
                    image = cv2.imread(str(on_disk), cv2.IMREAD_COLOR)
            if image is None and client is not None:
                raw = client.get_preview(reference.asset_id)
                image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                table.add_row(reference.person_name, "-", "-", "[yellow]no preview[/yellow]")
                continue

            # Immich's stored box lives in the space recorded on the face row;
            # rescale if the preview we fetched came back at another size.
            height, width = image.shape[:2]
            sx = width / reference.image_width if reference.image_width else 1.0
            sy = height / reference.image_height if reference.image_height else 1.0
            expected = (
                reference.box[0] * sx,
                reference.box[1] * sy,
                reference.box[2] * sx,
                reference.box[3] * sy,
            )

            observations = detector.detect(image)
            if not observations:
                table.add_row(reference.person_name, "-", "-", "[yellow]no detection[/yellow]")
                continue

            best = max(observations, key=lambda o: iou(o.box, expected))
            overlap = iou(best.box, expected)
            stored = normalise(reference.embedding)[0]
            distance = float(1.0 - np.dot(stored, best.embedding))
            distances.append(distance)

            verdict = (
                "[green]match[/green]"
                if distance < 0.05
                else "[yellow]close[/yellow]"
                if distance < 0.2
                else "[red]MISMATCH[/red]"
            )
            table.add_row(reference.person_name, f"{overlap:.2f}", f"{distance:.4f}", verdict)

        console.print(table)
        if not distances:
            console.print("[red]No comparisons completed.[/red]")
            raise typer.Exit(1)

        if client is not None:
            client.close()

        median = float(np.median(distances))
        console.print(f"median distance {median:.4f}, worst {max(distances):.4f}")
        if median < 0.05:
            console.print("[green]PASS[/green] - our embeddings share Immich's vector space.")
        else:
            console.print("[red]FAIL[/red] - alignment or model mismatch.")
            raise typer.Exit(1)


@app.command()
def scan(
    config: Optional[Path] = _config_option,
    asset: Optional[str] = typer.Option(None, "--asset", help="Scan a single asset id"),
    limit: Optional[int] = typer.Option(None, "--limit", "-n", help="Max assets to scan"),
    newest: bool = typer.Option(
        False, "--newest", help="Take the most recent videos rather than the oldest"
    ),
    report: bool = typer.Option(True, "--report/--no-report", help="Emit an HTML report"),
    force: bool = typer.Option(False, "--force", help="Re-scan assets already done"),
    retry_failed: bool = typer.Option(
        False, "--retry-failed", help="Also retry videos that failed to scan before"
    ),
    verbose: bool = False,
) -> None:
    """Scan videos and stage the results for review.

    Nothing is written to Immich here. Matches are staged locally and rendered
    as an HTML report; `ivf commit --run N` applies them once you've looked.
    """
    from .faces import FaceDetector
    from .immich_db import ImmichDb
    from .match import Matcher
    from .report import write_html, write_json
    from .scanner import Scanner
    from .state import State
    from .writeback import box_for

    cfg = _setup(config, verbose)

    client = None
    if cfg.immich.api_key:
        from .immich_api import ImmichClient

        client = ImmichClient(cfg)
        if not _check_model(cfg, client):
            raise typer.Exit(1)

    with ImmichDb(cfg) as db, State(cfg.state_db) as state:
        db.assert_schema()
        _rekey(cfg, state)
        owner = cfg.immich.owner_id or (client.my_user_id() if client else "")
        if not owner:
            console.print("[red]Set immich.owner_id or an api_key in config.[/red]")
            raise typer.Exit(1)
        gallery = db.load_gallery(owner)
        console.print(
            f"Gallery: {gallery.face_count} faces / {len(gallery)} people "
            f"({'named only' if cfg.match.named_people_only else 'incl. unnamed'})"
        )

        if asset:
            assets = [a for a in db.list_videos(owner) if a.id == asset]
        else:
            # One outstanding batch at a time: starting a new one abandons any
            # uncommitted batch and returns its videos to the queue, so they are
            # picked up again here rather than skipped.
            dropped, returned = state.discard_unsettled()
            if returned:
                console.print(
                    f"[yellow]Discarded the previous uncommitted batch[/yellow] "
                    f"({dropped} staged faces, {returned} videos back in the queue)"
                )
            done = (
                set()
                if force
                else state.already_scanned(cfg.fingerprint, include_failed=not retry_failed)
            )
            assets = []
            for candidate in db.list_videos(owner, newest_first=newest):
                if candidate.id in done:
                    continue
                assets.append(candidate)
                if limit and len(assets) >= limit:
                    break

        if not assets:
            console.print("Nothing to scan.")
            return

        console.print(
            f"Scanning {len(assets)} asset(s) "
            f"[{'newest first' if newest else 'oldest first'}]"
        )

        detector = FaceDetector(
            cfg.detect, clip_model=cfg.objects.model if cfg.objects.enabled else None
        )
        scanner = Scanner(cfg, detector, Matcher(gallery, cfg.match))
        index = _open_index(cfg)
        run_id = state.start_run(cfg.fingerprint, "scan")

        results = []
        totals = {"staged": 0, "banked": 0, "matched": 0, "superseded": 0, "clip_frames": 0}

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("scanning", total=len(assets))
            for candidate in assets:
                result = scanner.scan(candidate, keep_crops=report)
                results.append(result)
                totals["matched"] += len(result.matched)

                # Object search needs no review: frame embeddings only change which
                # videos a search can find, so they are published straight away.
                if index is not None and result.ok and not candidate.still_asset_id:
                    index.replace_video(
                        asset_id=candidate.id,
                        duration_s=result.duration_s,
                        frames=result.frames,
                        frame_embeddings=result.frame_embeddings,
                        model=cfg.objects.model,
                        fingerprint=cfg.fingerprint,
                    )
                    totals["clip_frames"] += len(result.frame_embeddings)

                # A Live Photo clip's faces land on the still, which has no timeline.
                timeline = not candidate.still_asset_id

                # This scan's result for the asset replaces any older staging of
                # it, so an asset is never awaiting review in two runs at once.
                superseded = state.supersede_pending(candidate.write_target, run_id)
                totals["superseded"] += superseded

                for match in result.matched:
                    assert match.person_id is not None
                    box, dims = box_for(result, match)
                    # For a Live Photo clip this is the visible still, not the
                    # hidden clip we actually decoded.
                    state.record_pending(
                        run_id=run_id,
                        asset_id=candidate.write_target,
                        person_id=match.person_id,
                        person_name=match.person_name,
                        image_width=dims[0],
                        image_height=dims[1],
                        box=box,
                        distance=match.distance,
                        margin=match.margin,
                        observations=match.cluster.size,
                        segments=match.segments if timeline else None,
                    )
                    totals["staged"] += 1

                if cfg.writeback.bank_unmatched:
                    from .report import encode_crop

                    # Re-scanning replaces this asset's banked clusters rather
                    # than adding a second copy. Keyed on the write target, so a
                    # Live Photo clip clears what was banked against its still.
                    state.clear_unmatched(candidate.write_target)
                    for match in result.unmatched:
                        box, dims = box_for(result, match)
                        state.record_unmatched(
                            run_id=run_id,
                            asset_id=candidate.write_target,
                            embedding=match.cluster.centroid,
                            observations=match.cluster.size,
                            image_width=dims[0],
                            image_height=dims[1],
                            box=box,
                            best_distance=match.distance,
                            crop_jpeg=encode_crop(match.cluster.best.crop),
                            segments=match.segments if timeline else None,
                        )
                        totals["banked"] += 1

                state.record_scan(
                    asset_id=candidate.id,
                    fingerprint=cfg.fingerprint,
                    run_id=run_id,
                    frames=result.frames,
                    observations=result.observations,
                    clusters=len(result.clusters),
                    matches=len(result.matched),
                    status="ok" if result.ok else "error",
                    error=result.error,
                )
                progress.advance(task)

        state.finish_run(run_id)

        # Nothing to review means nothing to commit, so settle immediately --
        # otherwise a batch with no matches would repeat forever.
        if not totals["staged"]:
            state.settle_run(run_id)

        failures = [r for r in results if not r.ok]
        console.print(
            f"\nrun {run_id}: {len(results)} scanned, {totals['matched']} matches, "
            f"{len(failures)} failed"
        )
        console.print(f"staged {totals['staged']} face(s) for review - nothing written yet")
        if index is not None:
            console.print(
                f"indexed {totals['clip_frames']} frame(s) for object search"
                if cfg.objects.enabled
                else "object search off; videos recorded in the index"
            )
            index.close()
        if totals["superseded"]:
            console.print(
                f"replaced {totals['superseded']} face(s) staged for these assets "
                f"by an earlier run"
            )
        if totals["banked"]:
            console.print(
                f"banked {totals['banked']} unmatched clusters - "
                f"review with [bold]ivf promote[/bold]"
            )
        for failure in failures[:5]:
            console.print(f"  [red]{failure.path.name}[/red]: {failure.error}")

        # Group every banked unknown face across all scans so far, and number
        # the groups so the report and `ivf promote <n>` agree.
        groups: list = []
        if cfg.writeback.bank_unmatched:
            from .writeback import group_unmatched

            pending = state.pending_unmatched()
            groups, _ = group_unmatched(
                pending,
                min_assets=cfg.writeback.promote_min_assets,
                distance=cfg.cluster.distance,
                state=state,
            )
            state.assign_groups(
                {c.id: no for no, members in groups for c in members}
            )

        if report:
            cfg.report_dir.mkdir(parents=True, exist_ok=True)
            html_path = cfg.report_dir / f"run-{run_id}.html"
            from .match import nearest_people

            write_html(
                results,
                html_path,
                run_id=run_id,
                promote_groups=groups,
                min_assets=cfg.writeback.promote_min_assets,
                immich_url=cfg.immich.browse_url,
                promote_hints=nearest_people(gallery, cfg.match, groups),
            )
            write_json(results, cfg.report_dir / f"run-{run_id}.json")
            console.print(f"\nreview: [bold]{html_path}[/bold]")
        if totals["staged"]:
            console.print(f"then apply: [bold]ivf commit --run {run_id}[/bold]")
        if groups:
            console.print(
                f"{len(groups)} new-person candidate(s) at the bottom of the report; "
                f"create with [bold]ivf promote <numbers>[/bold]"
            )

        if client is not None:
            client.close()


@app.command()
def reembed(
    config: Optional[Path] = _config_option,
    limit: Optional[int] = typer.Option(None, "--limit", "-n", help="Max videos to re-embed"),
    verbose: bool = False,
) -> None:
    """Re-embed scanned videos' frames with the current search model.

    After a search model change (Immich's Smart Search model, mirrored in
    objects.model), videos scanned before keep frames from the old model, and
    search ignores them. This replaces them. Frames only: faces and timelines
    are left alone, so there is nothing to review.
    """
    from .faces import FaceDetector
    from .immich_db import ImmichDb
    from .scanner import Scanner
    from .state import State

    cfg = _setup(config, verbose)
    if not (cfg.objects.enabled and cfg.index.enabled):
        console.print("[red]reembed needs objects.enabled and index.enabled.[/red]")
        raise typer.Exit(1)
    client = None
    if cfg.immich.api_key:
        from .immich_api import ImmichClient

        client = ImmichClient(cfg)
        if not _check_model(cfg, client):
            raise typer.Exit(1)

    index = _open_index(cfg)
    with ImmichDb(cfg) as db, State(cfg.state_db) as state:
        _rekey(cfg, state)
        stale = index.stale_videos(cfg.objects.model)
        if not stale:
            console.print(f"Nothing to re-embed: every frame uses {cfg.objects.model}.")
            return
        owner = cfg.immich.owner_id or (client.my_user_id() if client else "")
        videos = {video.id: video for video in db.list_videos(owner)}
        # Frames of videos Immich no longer lists (deleted) can only go.
        gone = [asset_id for asset_id in stale if asset_id not in videos]
        for asset_id in gone:
            index.replace_frames(asset_id=asset_id, frame_embeddings=[], model=cfg.objects.model)
        todo = [videos[asset_id] for asset_id in stale if asset_id in videos]
        if limit:
            todo = todo[:limit]
        console.print(f"Re-embedding {len(todo)} of {len(stale)} video(s) with {cfg.objects.model}")

        scanner = Scanner(cfg, FaceDetector(cfg.detect, clip_model=cfg.objects.model), None)
        done = frames = 0
        failed: list[tuple[str, str]] = []
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("re-embedding", total=len(todo))
            for video in todo:
                result = scanner.embed(video)
                if result.ok:
                    frames += index.replace_frames(
                        asset_id=video.id,
                        frame_embeddings=result.frame_embeddings,
                        model=cfg.objects.model,
                    )
                    done += 1
                else:
                    failed.append((video.id, result.error or "unknown error"))
                progress.advance(task)
    console.print(f"Re-embedded {done} video(s), {frames} frames.")
    for asset_id, error in failed:
        console.print(f"  [yellow]{asset_id}: {error}[/yellow]")
    left = len(stale) - len(gone) - done
    if left:
        console.print(f"{left} video(s) still on an older model.")


@app.command()
def commit(
    config: Optional[Path] = _config_option,
    run: Optional[int] = typer.Option(None, "--run", help="Run id to commit"),
    yes: bool = typer.Option(False, "--yes", help="Skip confirmation"),
    allow_stale: bool = typer.Option(
        False, "--allow-stale", help="Commit even if settings changed since staging"
    ),
    verbose: bool = False,
) -> None:
    """Apply a reviewed scan's staged faces to Immich."""
    from .immich_api import ImmichClient
    from .state import State
    from .writeback import Writer

    cfg = _setup(config, verbose)
    if not cfg.immich.api_key:
        console.print("[red]commit needs immich.api_key (or $IMMICH_API_KEY).[/red]")
        raise typer.Exit(1)

    with ImmichClient(cfg) as client, State(cfg.state_db) as state:
        _rekey(cfg, state)
        awaiting = state.runs_awaiting_review()
        if run is None:
            if not awaiting:
                console.print("Nothing staged. Run [bold]ivf scan[/bold] first.")
                raise typer.Exit(1)
            if len(awaiting) > 1:
                console.print("Several runs are staged; pass --run:")
                for run_id, count in awaiting:
                    console.print(f"  run {run_id}: {count} faces")
                raise typer.Exit(1)
            run = awaiting[0][0]

        pending = state.pending_writes(run)
        if not pending:
            console.print(f"Run {run} has nothing staged to commit.")
            raise typer.Exit(1)

        # Settings changed since staging means these rows describe results the
        # current configuration would not produce.
        staged_fingerprint = state.run_fingerprint(run)
        if staged_fingerprint and staged_fingerprint != cfg.fingerprint:
            console.print(
                f"[red]Run {run} was staged under different settings[/red] "
                f"({staged_fingerprint} vs current {cfg.fingerprint})."
            )
            console.print(
                "Its matches reflect the old thresholds. Re-scan those assets, "
                "or pass --allow-stale to commit them anyway."
            )
            if not allow_stale:
                raise typer.Exit(1)

        people = {p.person_name or p.person_id[:8] for p in pending}
        assets = {p.asset_id for p in pending}
        console.print(
            f"Run {run}: {len(pending)} faces across {len(assets)} videos, "
            f"{len(people)} distinct people."
        )
        console.print(f"Report: {cfg.report_dir / f'run-{run}.html'}")
        if not yes and not typer.confirm("Write these to Immich?"):
            raise typer.Abort()

        commit_run = state.start_run(cfg.fingerprint, "commit", notes=f"from scan {run}")
        index = _open_index(cfg)
        writer = Writer(cfg, client, state, commit_run, index=index)
        outcome = writer.commit(pending)
        if index is not None:
            index.close()
        state.finish_run(commit_run)

        # The batch is dealt with, so the next scan moves on to new videos.
        settled = state.settle_run(run)
        console.print(f"batch settled: {settled} videos marked done")

        console.print(
            f"wrote {outcome.faces_written} faces, "
            f"skipped {outcome.faces_skipped} already present"
        )
        if outcome.faces_failed:
            console.print(
                f"[yellow]{outcome.faces_failed} failed and remain staged "
                f"(person merged away, or asset deleted)[/yellow]"
            )
            for failure in outcome.failures[:5]:
                console.print(f"  {failure}")
        console.print(f"undo with: [bold]ivf undo --run {commit_run}[/bold]")


@app.command()
def promote(
    groups: Optional[list[int]] = typer.Argument(
        None, help="Group numbers from the review page, e.g. 1 4 6 12"
    ),
    config: Optional[Path] = _config_option,
    into: Optional[str] = typer.Option(
        None,
        "--into",
        help="Attach to an existing person (name or id) instead of creating one",
    ),
    min_assets: Optional[int] = typer.Option(
        None, help="Distinct videos a group must span"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="List the groups without creating anyone"
    ),
    verbose: bool = False,
) -> None:
    """Turn numbered candidates from the review page into people.

    `ivf promote` lists candidates with the closest known person for each.
    `ivf promote 1 4` creates a new person per group.
    `ivf promote 1 --into Alex` attaches group 1 to Alex instead.
    """
    from .immich_api import ImmichClient
    from .immich_db import ImmichDb
    from .match import nearest_people
    from .state import State
    from .writeback import promote_unmatched

    cfg = _setup(config, verbose)
    min_assets = min_assets or cfg.writeback.promote_min_assets
    selected = list(groups) if groups else None
    # No numbers means "show me what's there", never "create everything".
    listing = selected is None or dry_run

    with ImmichClient(cfg) as client, State(cfg.state_db) as state:
        _rekey(cfg, state)
        owner = cfg.immich.owner_id or client.my_user_id()

        into_person = None
        if into:
            into_person = _resolve_person(client, into)
            if into_person is None:
                raise typer.Exit(1)

        run_id = state.start_run(
            cfg.fingerprint, "promote-dry" if listing else "promote"
        )
        detector = None
        if not listing and not into_person:
            from .faces import FaceDetector

            # Needed to anchor each new person's avatar to a real face.
            detector = FaceDetector(cfg.detect)

        outcome, chosen = promote_unmatched(
            cfg,
            client,
            state,
            run_id,
            min_assets=min_assets,
            distance=cfg.cluster.distance,
            dry_run=listing,
            detector=detector,
            only_groups=selected,
            into_person=into_person,
            index=None if listing else _open_index(cfg),
        )
        state.finish_run(run_id)

        with ImmichDb(cfg) as db:
            gallery = db.load_gallery(owner)
        hints = nearest_people(gallery, cfg.match, chosen)

        table = Table(title="New-person candidates")
        table.add_column("#", justify="right")
        table.add_column("Videos", justify="right")
        table.add_column("Appearances", justify="right")
        table.add_column("Closest known person")
        table.add_column("Distance", justify="right")
        for number, members in chosen[:60]:
            name, person_id, distance = hints.get(number, ("", "", float("nan")))
            table.add_row(
                str(number),
                str(len({m.asset_id for m in members})),
                str(len(members)),
                f"[bold]{name}[/bold]" if name else f"(unnamed) {person_id[:8]}",
                f"{distance:.3f}",
            )
        console.print(table)

        if selected:
            missing = set(selected) - {n for n, _ in chosen}
            if missing:
                console.print(
                    f"[yellow]no such group(s): {', '.join(map(str, sorted(missing)))}"
                    f" - numbers come from the latest scan's report[/yellow]"
                )

        if listing:
            console.print(
                f"{outcome.groups_considered} raw groups, {len(chosen)} qualify "
                f"(>= {min_assets} videos), {outcome.skipped_small} too small"
            )
            console.print(
                "New person:      [bold]ivf promote 1 4[/bold]\n"
                "Existing person: [bold]ivf promote 1 --into Alex[/bold]"
            )
            return

        if into_person:
            console.print(
                f"attached {outcome.faces_written} faces to the existing person"
            )
        else:
            console.print(
                f"created {outcome.people_created} people, "
                f"wrote {outcome.faces_written} faces, "
                f"anchored {outcome.anchored} avatars"
            )
        if outcome.unanchored_people:
            console.print(
                f"[yellow]{len(outcome.unanchored_people)} people need a manual "
                f"thumbnail (not visible in any video preview)[/yellow]"
            )
        console.print(f"undo with: [bold]ivf undo --run {run_id}[/bold]")


@app.command()
def undo(
    config: Optional[Path] = _config_option,
    run: Optional[int] = typer.Option(None, "--run", help="Run id (default: latest write run)"),
    yes: bool = typer.Option(False, "--yes", help="Skip confirmation"),
    verbose: bool = False,
) -> None:
    """Remove every face, person and tag a run created."""
    from .immich_api import ImmichClient
    from .state import State
    from .writeback import undo_run

    cfg = _setup(config, verbose)

    with ImmichClient(cfg) as client, State(cfg.state_db) as state:
        run_id = run or state.latest_write_run()
        if run_id is None:
            console.print("Nothing has been written, so there is nothing to undo.")
            raise typer.Exit(1)

        faces = state.faces_for_run(run_id)
        people = state.people_for_run(run_id)
        console.print(
            f"Run {run_id}: {len(faces)} faces and {len(people)} created people to remove."
        )
        if not yes and not typer.confirm("Proceed?"):
            raise typer.Abort()

        outcome = undo_run(cfg, client, state, run_id, index=_open_index(cfg))
        console.print(
            f"deleted {outcome.faces_deleted} faces "
            f"({outcome.faces_missing} already gone), "
            f"{outcome.people_deleted} people ({outcome.people_kept} kept), "
            f"untagged {outcome.assets_untagged} assets"
        )
        if outcome.clusters_returned:
            console.print(
                f"returned {outcome.clusters_returned} clusters to the candidate pool"
            )


@app.command()
def duplicates(
    config: Optional[Path] = _config_option,
    distance: float = typer.Option(0.35, help="How close two people must be to flag"),
    limit: int = typer.Option(40, help="Rows to show"),
    verbose: bool = False,
) -> None:
    """Find people Immich has split into several person records.

    Read-only. When the same face exists as both a name and stray unnamed
    clusters, videos attach to whichever is marginally nearer, so they never
    show up under the name you use. Merging them in Immich fixes that and
    deepens the reference gallery for everything else.
    """
    import numpy as np

    from .immich_db import ImmichDb

    cfg = _setup(config, verbose)

    with ImmichDb(cfg) as db:
        db.assert_schema()
        owner = cfg.immich.owner_id
        if not owner:
            from .immich_api import ImmichClient

            with ImmichClient(cfg) as client:
                owner = client.my_user_id()

        gallery = db.load_gallery(owner)
        ids, centroids, counts = db.person_centroids(gallery)

        pair_distance = 1.0 - centroids @ centroids.T
        np.fill_diagonal(pair_distance, 9.0)
        rows: list[tuple[float, str, str]] = []
        for i, j in zip(*np.where(pair_distance <= distance)):
            if i >= j:
                continue
            rows.append((float(pair_distance[i, j]), ids[int(i)], ids[int(j)]))
        rows.sort()

        table = Table(title=f"Likely duplicate people (centroid distance <= {distance})")
        table.add_column("Distance", justify="right")
        table.add_column("Person A")
        table.add_column("faces", justify="right")
        table.add_column("Person B")
        table.add_column("faces", justify="right")

        named_unnamed = 0
        for dist, a, b in rows[:limit]:
            name_a = gallery.names.get(a) or ""
            name_b = gallery.names.get(b) or ""
            if bool(name_a) != bool(name_b):
                named_unnamed += 1
            table.add_row(
                f"{dist:.3f}",
                f"[bold]{name_a}[/bold]" if name_a else f"(unnamed) {a[:8]}",
                str(counts.get(a, 0)),
                f"[bold]{name_b}[/bold]" if name_b else f"(unnamed) {b[:8]}",
                str(counts.get(b, 0)),
            )
        console.print(table)
        console.print(
            f"{len(rows)} candidate pairs; {named_unnamed} of those shown pair a named "
            f"person with an unnamed cluster."
        )
        console.print(
            "Merge them in Immich (people page -> select -> merge). "
            "This tool never merges people itself."
        )


@app.command()
def status(config: Optional[Path] = _config_option) -> None:
    """Show local scan state."""
    from .state import State

    cfg = _setup(config)
    with State(cfg.state_db) as state:
        _rekey(cfg, state)
        summary = state.summary()
        failed = state.failed_assets()
        awaiting = state.runs_awaiting_review()
        stale = {
            run_id: state.run_fingerprint(run_id)
            for run_id, _ in awaiting
            if state.run_fingerprint(run_id) != cfg.fingerprint
        }

    table = Table(title="Scan state", show_header=False)
    for key, value in summary.items():
        table.add_row(key.replace("_", " "), str(value))
    table.add_row("fingerprint", cfg.fingerprint)
    console.print(table)
    if failed:
        console.print(f"[yellow]{len(failed)} video(s) failed to scan[/yellow] and are skipped:")
        for asset_id, error in failed[:10]:
            console.print(f"  {asset_id}: {error[:120]}")
        console.print("Retry them with [bold]ivf scan --retry-failed[/bold].")

    if awaiting:
        pending = Table(title="Awaiting review")
        pending.add_column("Run", justify="right")
        pending.add_column("Faces staged", justify="right")
        pending.add_column("Settings")
        for run_id, count in awaiting:
            pending.add_row(
                str(run_id),
                str(count),
                "[red]stale[/red]" if run_id in stale else "[green]current[/green]",
            )
        console.print(pending)
        if stale:
            console.print(
                "Stale runs were staged under different settings; re-scan those "
                "assets rather than committing them."
            )


if __name__ == "__main__":
    app()
