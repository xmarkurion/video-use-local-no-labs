# LUTs

Drop a 3D LUT file (`.cube` — the DaVinci Resolve / Premiere / most camera
manufacturer export format) in this folder, then reference it by filename
(no extension needed) from `render.py` or the EDL:

```bash
python helpers/render.py edit/edl.json -o edit/final.mp4 --lut "DJI OSMO Action 4 D-Log M to Rec.709 V1"
```

or in the EDL itself, alongside `"grade"`:

```json
{
  "lut": "DJI OSMO Action 4 D-Log M to Rec.709 V1",
  "grade": "warm_cinematic"
}
```

`--lut` on the command line overrides the EDL's `"lut"` field. A LUT and a
`grade` filter can be combined — the grade (exposure/contrast/saturation
correction) runs first, the LUT (the creative "look") runs last, matching how
a colorist layers a corrective pass under a creative one.

You can also pass a path directly instead of a bare name:

```bash
--lut /absolute/path/to/some_other.cube
--lut ../shared-luts/kodak_2383.cube
```

**Where to get LUTs:**
- Your camera/drone manufacturer usually ships one for its log profile (the
  file already in this folder, `DJI OSMO Action 4 D-Log M to Rec.709 V1.cube`,
  is exactly that — converts DJI's flat D-Log M footage to standard Rec.709).
- Free creative LUT packs exist from most colorist communities; any `.cube`
  file works.

**Format support:** `.cube` is the one every tool tested here exports and the
one ffmpeg's `lut3d` filter reads without conversion. If you only have a
`.3dl`/`.csp`/other 3D LUT format, convert it to `.cube` first (most grading
software round-trips it, as does the free tool `lutcalc`).

This folder is yours — nothing in `helpers/` cares what's in it beyond
matching a `--lut` name against a file here.
