# Known gaps

An honest account of where INK TIDE sits against the target, written against
measurements and captured frames rather than intentions. A visual critic ran two
passes over 15 retina frames; its verdict after the second pass was still
**"would not ship"**, and that assessment is recorded here rather than softened.

Frames referenced live in `shots/r3/` and `shots/r4/`. Regenerate with
`node harness/capture.mjs --out=shots/<name>`.

## What is closed, with numbers

| Item | Evidence |
|---|---|
| Water quantisation | 6–123 unique colours per 400×400 open-water patch (was 20,128); 0.5–0.7% of pixels differ from all four neighbours (was 88%). Resolves to five hard bands. |
| Constant screen-space outline width | Median ink run 3–4 device px whether the boat fills the frame or is 50 px wide. The 13 px vs 1 px range is gone. |
| Ramp lighting on hulls, gates, helmet, sky | Measured hard steps with zero interpolation, e.g. helmet (255,23,19)→(228,38,45)→(200,72,90)→(97,42,62) plus a (252,133,135) rim. |
| Value ladder | Non-HUD luminance spans L9–L255 with eight distinct clusters. The frame does not flatten to one mid-tone. |
| Mid-distance aliasing | Edge density rises monotonically toward camera (3.5 → 14 changes per 100 px), so minification is well behaved. |
| Performance | p95 9.20 ms, 0 of 1675 frames over 16.9 ms, at full dpr 2.0. See README. |
| AI race integrity | Full 3-lap race completes; minimum pairwise hull distance 2.05 m; lateral error < 7 m; checkpoints advance in order. |

## What is still open

### 1. Water band *shapes* — the biggest remaining gap (blocker)

The ocean is correctly banded but the band silhouettes are **decorative rather
than aquatic**. Blob analysis of the bright band over a clean 900×900 patch:
211 blobs, median elongation 2.89:1, and orientation still clustered — the top
15° bin holds 31.3% of blobs and three adjacent bins hold 59.2%. Wake troughs
contain nested concentric contour rings that read as a topographic map or
marbled endpaper. At speed this risks reading as scrolling patterned wallpaper.

Decorrelating the wave table (ratios moved off 2.0 toward φ, direction spread
108° → 219°) reduced the single-angle read but did not remove it. The remaining
work is in the band-selection function in `oceanMaterial.ts`, not the wave table:
the bands need a crest-relative silhouette with a real size hierarchy, not one
characteristic dash size at one preferred angle.

### 2. Nothing connects a hull to the water (blocker)

There is no foam ring, no bow wave and no contact darkening on any hull in any
captured frame. `HullCollars` is built, wired and in the draw-call count, but no
frame distinguishes it from the wake ribbon's transom churn — so treat "foam ring
around every hull" as **written but unproven**. The depth-difference mask cannot
carry this alone: a depth mask only marks water in front of recorded geometry,
which from a chase camera is a 2–3 px sliver.

### 3. Ocean foam has no age

Whitecaps appear and vanish with the instantaneous Jacobian. Real foam persists
and slides down the back of a wave after it breaks. Only the wake ribbon
dissipates over time.

### 4. The sun is absent from the composition

Searching every shot for the sun-disc palette finds 10,442 px in `sky.png` and
effectively zero in the other thirteen framings. `SUN_DIR` sits at ~41°
elevation, above almost every gameplay framing. Where the sun does appear it
reads weakly: a flat pale donut with 13 L of contrast against the cloud behind
it and no ink outline.

Deliberately **not** changed late in the build: `SUN_DIR` is a shared uniform
that every cel material's terminator and the water's glitter path key off, so
lowering it needs a full re-tune round rather than a one-line edit.

### 5. Rider body is not ramped

The helmet is correctly banded, but the suit and skin give six near-tones between
L85 and L171 with no readable ramp. The rider reads as a prop at gameplay
distance even though the rig, IK and animation state machine are working.

### 6. HUD crowds gameplay

The four corner clusters occupy roughly 29% of frame area, and the standings
panel's lower edge sits at ~y780 device px while the horizon runs y640–800 in
racing shots — so it permanently occupies racing space rather than sky. An AI
boat is ~25% occluded in `ocean_low.png`.

### 7. Chase-camera framing flattens the sea

`hero.png` skyline deviation is 48 px against 90–190 px in every other framing
using the same shader. The chase camera sits high and looks down, foreshortening
the swell into markings on a plane, and a ~40 px pale haze band on the horizon
erases wave detail there. This is a framing problem, not a displacement problem.

### 8. Audio is unverified by ear

The graph builds, does not throw, and is fully synthesised — engine tone tracking
RPM, speed-scaled water rush, impact thuds, horn, boost, checkpoint blip. **Nobody
has listened to it.** A screenshot cannot verify audio and no automated check
here substitutes for hearing it.

### 9. Smaller items

- Deliberate AI mistakes are implemented but never captured in a frame.
- Landing-impact spray is implemented; no shot proves the burst.
- Sea-state response (`setSeaState`) is untested outside `seaState = 1`.
- Wake rails do not wobble along their length; no boat-to-boat wake interaction.
- A faint dotted hatch survives on a few hull panels at 4× zoom where non-planar
  quads break by more than 43° along their triangulation diagonal.

## Process note

Two bugs in the **harness** invalidated earlier verification and are worth
recording, because both produced confident-but-wrong conclusions:

1. `simulate()` awaited between batches while the rAF loop kept stepping the
   simulation, so identical inputs produced different state. Every "verified"
   claim before that fix rested on a non-deterministic harness.
2. The scripted player drove `steer: 0` at full throttle — 1.2 km off course
   inside a minute. That, not a game bug, produced "WRONG WAY in every frame" and
   "the AI scattered out of shot". Three separate agents reported a heading bug
   that did not exist; a numeric probe (`heading · tangent = 1.00` for all four
   boats) disproved it in one run.

The lesson: screenshots are good at catching art defects and bad at diagnosing
mechanical ones. `harness/probe.mjs` exists because of this and should be the
first tool reached for when behaviour, not appearance, looks wrong.
