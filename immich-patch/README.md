# Immich patch: the video index in Immich itself

These patches make Immich use what `ivf` publishes to the video index (see the main README,
"Appearance times and object search"). Without the index schema they change nothing.

| Patch | What it does |
|---|---|
| `0001-server-…` | `GET /api/assets/{id}/appearances`: who is on screen when in a video. Smart search also matches a video through its sampled frames, ranked by the best one. |
| `0002-web-…` | The video player gets a row per person above the time range, marking when they are on screen. Clicking a mark seeks there. |
| `0003-server-…` | Sensitive content: anything tagged `Sensitive` (or a tag under it), or showing a person marked sensitive, is served as a blurred placeholder, with its video and original refused, until the session is unlocked with the PIN. `GET /api/assets/{id}/sensitive`, `GET`/`PUT /api/people/{id}/sensitive`. |
| `0004-web-…` | The viewer offers the PIN on a hidden asset, and a way to hide it all again once unlocked. A person's menu gains *Hide behind PIN*. |
| `0005-search-…` | Search can leave people out (`excludePersonIds`). In the search filters, clicking a selected person again leaves them out. |

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
- **Sensitive content** uses the PIN and unlock of Immich's Locked Folder: 15 minutes,
  extended while in use. Set `IMMICH_SENSITIVE_TAG` to use another tag, or empty to hide
  marked people only. `ivf sensitive` applies the tags; tagging by hand works the same.
  The mobile apps get the placeholders too (the server sends them), but offer no unlock
  outside the Locked Folder, and keep thumbnails they cached before an asset was tagged.
  A browser may also show such a cached thumbnail for up to a day.
- **Hidden people** are kept in `ivf.sensitive_person`, which Immich writes, so its role
  needs `GRANT INSERT, DELETE ON ivf.sensitive_person`. Showing a person again needs the PIN.
- Docker users need to build their own images with the patches applied.
