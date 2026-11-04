# Immich patch: the video index in Immich itself

These patches make Immich use what `ivf` publishes to the video index (see the main README,
"Appearance times and object search"). Without the index schema they change nothing.

| Patch | What it does |
|---|---|
| `0001-server-…` | `GET /api/assets/{id}/appearances`: who is on screen when in a video. Smart search also matches a video through its sampled frames, ranked by the best one. |
| `0002-web-…` | The video player gets a row per person above the time range, marking when they are on screen. Clicking a mark seeks there. |

They are made against a specific Immich release; the directory name says which.

```bash
git clone --depth 1 --branch v3.3.5 https://github.com/immich-app/immich.git
cd immich
git apply /path/to/ImmichVideoFace/immich-patch/v3.3.5/*.patch
# then build the server and web as you normally build Immich
```

Notes:

- **The server reads schema `ivf`.** Set `IMMICH_VIDEO_INDEX_SCHEMA` if you chose another
  `[index] schema_name`. The role Immich connects as needs `USAGE` on that schema and
  `SELECT` on its tables; the setup SQL in the main README grants both.
- **Search compares every embedding** — photos and sampled frames — instead of using
  Immich's vector index. That is exact, and about 0.2 s per page on a library of 34,000
  assets. The index-backed search is used unchanged while no frames have been published.
- **Only frames embedded with the current smart search model are used**, so changing the
  model never mixes vector spaces; re-scan to bring frames up to date.
- **The timeline is web only.** The mobile apps are unchanged.
- Docker users need to build their own images with the patches applied.
