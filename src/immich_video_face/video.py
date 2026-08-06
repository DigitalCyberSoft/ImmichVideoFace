"""Frame extraction.

One ffmpeg process per video, raw BGR straight to stdout. Nothing is written to
disk -- a 9,500 video backfill would otherwise churn millions of throwaway JPEGs.

Sampling is keyframes *plus* a minimum-rate floor, expressed as a single select
filter so the video is decoded only once:

    select='eq(pict_type,I)+gte(t-prev_selected_t,<interval>)'

In ffmpeg's expression language `+` is logical OR, and `prev_selected_t` refers
to the previously *kept* frame, so the two conditions deduplicate naturally.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np

from .config import SamplingSettings

log = logging.getLogger(__name__)

_SHOWINFO_PTS = re.compile(rb"pts_time:\s*([0-9.]+)")


class VideoError(RuntimeError):
    pass


@dataclass
class VideoInfo:
    width: int
    height: int
    duration_s: float
    rotation: int

    @property
    def display_size(self) -> tuple[int, int]:
        """Dimensions after ffmpeg applies rotation metadata (autorotate is on)."""
        if self.rotation in (90, 270):
            return self.height, self.width
        return self.width, self.height


@dataclass
class FrameBatch:
    """Frames plus the mapping back to presentation timestamps.

    Timestamps land after the ffmpeg process exits, so observations carry a
    frame index during scanning and get resolved via `timestamp_for` at the end.
    """

    width: int
    height: int
    # Multiply detected coordinates by this to return to source resolution.
    scale_to_source: float
    timestamps: list[float] = field(default_factory=list)

    def timestamp_for(self, index: int) -> float:
        if index < len(self.timestamps):
            return self.timestamps[index]
        return float("nan")


def probe(path: Path) -> VideoInfo:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,duration:stream_side_data=rotation:format=duration",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise VideoError(f"ffprobe failed for {path}: {result.stderr.strip()[:300]}")

    payload = json.loads(result.stdout or "{}")
    streams = payload.get("streams") or []
    if not streams:
        raise VideoError(f"No video stream in {path}")
    stream = streams[0]

    duration = stream.get("duration") or payload.get("format", {}).get("duration") or 0.0

    rotation = 0
    for side_data in stream.get("side_data_list") or []:
        if "rotation" in side_data:
            rotation = abs(int(side_data["rotation"])) % 360

    return VideoInfo(
        width=int(stream.get("width") or 0),
        height=int(stream.get("height") or 0),
        duration_s=float(duration),
        rotation=rotation,
    )


def _target_size(info: VideoInfo, max_long_edge: int) -> tuple[int, int, float]:
    width, height = info.display_size
    if width <= 0 or height <= 0:
        raise VideoError("Video reports zero dimensions")

    long_edge = max(width, height)
    if long_edge <= max_long_edge:
        return width - (width % 2), height - (height % 2), 1.0

    ratio = max_long_edge / long_edge
    out_w = max(2, int(round(width * ratio)))
    out_h = max(2, int(round(height * ratio)))
    out_w -= out_w % 2
    out_h -= out_h % 2
    # Detected boxes are in scaled space; this converts them back.
    return out_w, out_h, width / out_w


def iter_frames(
    path: Path, settings: SamplingSettings, info: VideoInfo | None = None
) -> tuple[Iterator[tuple[int, np.ndarray]], FrameBatch]:
    """Return a frame iterator and the batch metadata it will populate.

    The iterator must be fully consumed (or closed) before `batch.timestamps`
    is complete.
    """
    info = info or probe(path)
    out_w, out_h, scale_to_source = _target_size(info, settings.max_long_edge)
    batch = FrameBatch(width=out_w, height=out_h, scale_to_source=scale_to_source)

    interval = 1.0 / settings.min_fps if settings.min_fps > 0 else 0.0
    conditions = []
    if settings.keyframes:
        conditions.append("eq(pict_type,I)")
    if interval > 0:
        conditions.append(f"gte(t-prev_selected_t,{interval:.6f})")
    if not conditions:
        raise VideoError("Sampling settings select no frames")
    # '+' is logical OR here; also keep the very first frame.
    select_expr = "+".join(conditions) + "+isnan(prev_selected_t)"

    command = [
        "ffmpeg",
        "-nostdin",
        # showinfo logs at info level, so this cannot be quietened to 'error'
        # without losing every frame timestamp.
        "-v",
        "info",
        "-i",
        str(path),
        "-vf",
        f"select='{select_expr}',scale={out_w}:{out_h},showinfo",
        "-fps_mode",
        "passthrough",
        "-frames:v",
        str(settings.max_frames),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-",
    ]

    return _stream(command, batch), batch


def _stream(command: list[str], batch: FrameBatch) -> Iterator[tuple[int, np.ndarray]]:
    frame_bytes = batch.width * batch.height * 3
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=frame_bytes * 2,
    )

    errors: list[bytes] = []

    def drain_stderr() -> None:
        # showinfo emits one line per frame, in output order, so timestamps
        # align with frame indices.
        assert process.stderr is not None
        for line in process.stderr:
            if match := _SHOWINFO_PTS.search(line):
                batch.timestamps.append(float(match.group(1)))
            elif b"error" in line.lower() or b"invalid" in line.lower():
                errors.append(line)

    reader = threading.Thread(target=drain_stderr, daemon=True)
    reader.start()

    index = 0
    try:
        assert process.stdout is not None
        while True:
            raw = process.stdout.read(frame_bytes)
            if not raw or len(raw) < frame_bytes:
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(batch.height, batch.width, 3)
            yield index, frame
            index += 1
    finally:
        if process.stdout:
            process.stdout.close()
        process.wait(timeout=30)
        reader.join(timeout=10)
        if process.stderr:
            process.stderr.close()
        if process.returncode not in (0, None) and errors:
            log.warning("ffmpeg reported issues: %s", b" ".join(errors[:3]).decode(errors="replace"))


def extract_frame_at(path: Path, timestamp_s: float, max_long_edge: int) -> np.ndarray | None:
    """Decode a single frame at a timestamp.

    Used for thumbnail anchoring: to give a newly created person a sensible
    avatar we need the exact frame Immich generated the video thumbnail from.
    """
    info = probe(path)
    out_w, out_h, _ = _target_size(info, max_long_edge)
    command = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-ss",
        f"{max(0.0, timestamp_s):.3f}",
        "-i",
        str(path),
        "-vf",
        f"scale={out_w}:{out_h}",
        "-frames:v",
        "1",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-",
    ]
    result = subprocess.run(command, capture_output=True, check=False)
    expected = out_w * out_h * 3
    if result.returncode != 0 or len(result.stdout) < expected:
        return None
    return np.frombuffer(result.stdout[:expected], dtype=np.uint8).reshape(out_h, out_w, 3)
