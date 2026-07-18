# Multi-Phone Audio Recording Plan (2026-07-18)

One session of ~80–95 minutes. All racket swaps and holdout assignments are
already applied — follow this file top to bottom, no free-text take metadata needed.

**Recorder:** STIGA gh-219 `Audio dataset recorder`
**Phones:** iPhone, Motorola, Huawei — all three record **every** take simultaneously.
**Rackets:** label them A, B, C, D. Racket ID is always entered as `A-red`, `A-black`, `B-red`, … Same side for the whole take.

The recorder stores these as structured metadata:

- **Take:** `T01`, `T02`, …; advances automatically only after a successful Save.
- **Position:** `Left`, `Center`, or `Right`; remembered after each take.
- **Orientation:** `Flat` or `Upright`; remembered after each take.
- **Dataset use:** `Train` or `Final holdout`; remembered after each take.

---

## Prep checklist (5 min)

- [ ] Device alias set once per phone: `CJ-iphone`, `CJ-moto`, `CJ-huawei`
- [ ] Bluetooth OFF on all three phones, no headsets plugged in (built-in mic only)
- [ ] Phones side by side, screens up, at the distance stated per take (distance = phone to bounce point)
- [ ] Speaker ready with two playlists: **Music 1** = percussion-heavy, **Music 2** = TV/speech/pop
- [ ] Rackets labeled A–D
- [ ] Take shows `T01` on all three phones
- [ ] Dataset use is `Train` on all three phones
- [ ] Record one short pilot take, save it, export it, and verify that WAV + JSON open on the computer

### Phone placement schedule

Keep each phone in the same position for a whole block, then rotate. The phones
must point toward the same bounce area.

| Block | iPhone | Motorola | Huawei | Orientation |
|-------|--------|----------|--------|-------------|
| 1     | Left   | Center   | Right  | Flat        |
| 2     | Right  | Left     | Center | Flat        |
| 3     | Center | Right    | Left   | Flat        |
| 4     | Left   | Right    | Center | Flat        |
| 5     | Right  | Center   | Left   | Flat        |
| 6     | Center | Left     | Right  | Upright     |
| 7     | Left   | Center   | Right  | Flat        |

Block 6's upright placement is bonus device variation. It is confounded with
that block's scenarios, so do not use this round to estimate an orientation
effect.

## Rules for every take

1. Start background music/noise **before** pressing record.
2. Stand still through the 2-second baseline; keep the background unchanged during the take.
3. Confirm all three phones show the same Take ID before starting.
4. After all phones show `GO`, wait one second, clap once as a synchronization marker, wait two seconds, then begin the planned activity. The clap is never a bounce label.
5. Positives: bounce to **exactly 30**, then leave three seconds of silence before stopping. Lost count → enter best estimate + write `approx` in notes.
6. Hard negatives: separate individual sounds by 1–2 seconds so review can identify them.
7. Vary bounce tempo slightly between takes (don't metronome everything identically).
8. Save on all three phones before moving to the next take. Saving advances the Take ID automatically.
9. If one phone retries, discards, or misses a save, use its Take stepper to realign it before continuing.

Use `Final holdout` for T04, T07, T12, T14, T19, T23, T29, T32, T34,
T41, and T42. Use `Train` for every other take. This keeps all three
recordings of the same physical event in the same split.

`Final holdout` is the unseen-physical-take test set. Do not use it for model
selection or threshold tuning. Leave-device-out evaluation is separate: derive
device folds from `Train` sessions, train on two phones, and evaluate on the
third phone before opening the final holdout.

Distances: close = 20–30 cm, medium = 40–70 cm, far = over 70 cm.

---

## Block 1 — Counting aloud while bouncing (8 takes, ~13 min)

Recorder fields: scenario `Racket bounce + speaking/counting`, noise level `Quiet`,
noise source `Speech / counting`, count `30`.
**Count aloud 1–30 continuously while bouncing.** Room otherwise quiet.

| Take | Distance | Racket |
|------|----------|--------|
| T01  | close    | A-red  |
| T02  | close    | B-red  |
| T03  | close    | C-black|
| T04  | close    | D-black|
| T05  | medium   | A-black|
| T06  | medium   | C-red  |
| T07  | far      | B-black|
| T08  | far      | D-red  |

## Block 2 — Loud music + bouncing, no talking (8 takes, ~13 min)

Recorder fields: scenario `Racket bounce + loud background sound`, noise level `Loud`,
noise source `Music / TV`, count `30`.
Music loud enough that you'd raise your voice to talk over it.
**T9–T12 use Music 1 (percussion). T13–T16 use Music 2 (TV/speech/pop).**

| Take | Distance | Racket | Music |
|------|----------|--------|-------|
| T09  | close    | A-red  | 1     |
| T10  | close    | C-red  | 1     |
| T11  | close    | D-red  | 1     |
| T12  | far      | A-black| 1     |
| T13  | far      | B-red  | 2     |
| T14  | far      | C-black| 2     |
| T15  | far      | D-black| 2     |
| T16  | medium   | B-black| 2     |

## Block 3 — Counting aloud OVER loud music (3 takes, ~5 min)

Recorder fields: scenario `Racket bounce + speaking/counting`, noise level `Loud`,
noise source `Music / TV`, count `30`, notes: `counting aloud over loud music`.
Count aloud 1–30 with loud music playing — the hardest combined case.

| Take | Distance | Racket |
|------|----------|--------|
| T17  | close    | A-red  |
| T18  | medium   | C-black|
| T19  | far      | B-red  |

## Block 4 — Quiet baselines (8 takes, ~12 min)

Recorder fields: noise level `Quiet`, noise source `None`, count `30`. No talking.

| Take | Scenario                 | Distance | Racket |
|------|--------------------------|----------|--------|
| T20  | Normal racket bounce     | close    | B-black|
| T21  | Normal racket bounce     | medium   | D-red  |
| T22  | Far/soft racket bounce   | far      | A-red  |
| T23  | Far/soft racket bounce   | far      | C-red  |
| T24  | Fast racket bounce       | close    | A-black|
| T25  | Fast racket bounce       | close    | D-black|
| T26  | Slow/high racket bounce  | medium   | B-red  |
| T27  | Slow/high racket bounce  | medium   | C-black|

## Block 5 — Hard negatives (9 takes, ~11 min)

Count `0` on all. Each take ~40 seconds. Racket ID `none` except T35 and T36.
**Zero ball-on-racket bounces anywhere in this block.**

| Take | Scenario                        | Level / Source        | Distance | What to do |
|------|---------------------------------|-----------------------|----------|------------|
| T28  | Floor/table impact, no bounce   | Quiet / None          | close    | table taps, ball dropped on table and floor WITHOUT racket, knuckle knocks |
| T29  | Floor/table impact, no bounce   | Loud / Music-TV (M1)  | close    | same impacts over loud music |
| T30  | Floor/table impact, no bounce   | Quiet / None          | medium   | chair scrapes, footsteps, door close, object drops |
| T31  | Talking/counting, no bounce     | Quiet / Speech        | close    | talk and count aloud 1–30, no bounces |
| T32  | Talking/counting, no bounce     | Loud / Music-TV (M2)  | close    | talk loudly over music, no bounces |
| T33  | Loud music/TV, no bounce        | Loud / Music-TV       | close    | Music 1 only, phones just listen |
| T34  | Loud music/TV, no bounce        | Loud / Music-TV       | close    | Music 2 only, phones just listen |
| T35  | Racket handling, no bounce      | Low / Room            | close    | racket ID `B+C`: pass, spin, re-grip both rackets, tap handles together |
| T36  | Racket drop, no bounce          | Medium / Other        | close    | racket ID `A+B+C+D`: drop each racket three times with 2-second gaps; no ball |

## Block 6 — Final variation block (5 takes, ~10 min)

| Take | Scenario                            | Level / Source   | Distance | Racket | Count | Notes field |
|------|-------------------------------------|------------------|----------|--------|-------|-------------|
| T37  | Racket bounce + background sound    | Medium / Room    | close    | A-black| 30    | `room background` (ventilation/street/kitchen) |
| T38  | Racket bounce + background sound    | Medium / Room    | far      | C-red  | 30    | `room background` |
| T39  | Normal racket bounce                | Quiet / None     | close    | D-red  | 30    | `catch after every 5th bounce` |
| T40  | Racket bounce + speaking/counting   | Low / Speech     | medium   | B-black| 30    | `second person talking` — skip if nobody around |
| T41  | Racket bounce + loud background     | Loud / Music-TV  | medium   | D-black| 30    | Music 1 |

## Block 7 — Supplemental hard-negative coverage (2 takes, ~3 min)

Count `0` on both. Each take ~40 seconds. Use the flat placement in the phone
schedule. T42 is `Final holdout`; T43 is `Train`.

| Take | Scenario                         | Level / Source | Distance | Racket      | What to do |
|------|----------------------------------|----------------|----------|-------------|------------|
| T42  | Racket drop, no bounce           | Medium / Other | close    | `A+B+C+D`   | drop each racket three times with 2-second gaps; no ball |
| T43  | Mixed hard negatives, no bounce  | Medium / Other | close    | `none`      | separated coughs, keys, and cutlery sounds; no ball or racket bounce |

## Wrap-up (5 min)

- [ ] Each phone's session list shows 43 takes, with matching Take IDs
- [ ] Export the dataset zip from **each** phone
- [ ] Copy all three zips to the computer before deleting anything
- [ ] Spot-check one Train JSON and one Final holdout JSON for take, position, orientation, and dataset-use metadata

---

## Racket-side coverage (for verification)

| Side    | Takes                  | Total | Distances covered |
|---------|------------------------|-------|-------------------|
| A-red   | T01, T09, T17, T22     | 4     | close, far        |
| A-black | T05, T12, T24, T37     | 4     | close, medium, far|
| B-red   | T02, T13, T19, T26     | 4     | close, medium, far|
| B-black | T07, T16, T20, T40     | 4     | close, medium, far|
| C-red   | T06, T10, T23, T38     | 4     | close, medium, far|
| C-black | T03, T14, T18, T27     | 4     | close, medium, far|
| D-red   | T08, T11, T21, T39     | 4     | close, medium, far|
| D-black | T04, T15, T25, T41     | 4     | close, medium, far|

Every side appears 4 times, spread over different scenarios, distances, and noise
conditions, so no racket sound is confounded with any single recording condition.

## Why the plan is weighted this way

- **Counting aloud (Blocks 1+3)** and **loud music (Blocks 2+3)** get the most
  takes: these are the two scenarios where live testing (gh-217) fails and where
  no reviewed multi-device data exists yet.
- **Floor/table impacts** get three dedicated takes: only 73 such training rows
  exist in the entire current corpus.
- Quiet baselines are kept small: historically well covered, easy for every model.
- Three phones per take turn 43 takes of effort into 129 sessions, all with
  device identity, noise level, and distance in metadata — ready for
  leave-device-out evaluation after timestamp review.
