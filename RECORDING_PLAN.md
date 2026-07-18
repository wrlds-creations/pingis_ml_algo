# Multi-Phone Audio Recording Plan (2026-07-18)

One session of ~65–75 minutes. All racket swaps and optional-block assignments are
already applied — follow this file top to bottom, no other notes needed.

**Recorder:** STIGA gh-219 `Audio dataset recorder`
**Phones:** iPhone, Motorola, Huawei — all three record **every** take simultaneously.
**Rackets:** label them A, B, C, D. Racket ID is always entered as `A-red`, `A-black`, `B-red`, … Same side for the whole take.

---

## Prep checklist (5 min)

- [ ] Device alias set once per phone: `CJ-iphone`, `CJ-moto`, `CJ-huawei`
- [ ] Bluetooth OFF on all three phones, no headsets plugged in (built-in mic only)
- [ ] Phones side by side, screens up, at the distance stated per take (distance = phone to bounce point)
- [ ] Speaker ready with two playlists: **Music 1** = percussion-heavy, **Music 2** = TV/speech/pop
- [ ] Rackets labeled A–D

## Rules for every take

1. Start background music/noise **before** pressing record.
2. Stand still through the 2-second baseline; keep the background unchanged during the take.
3. Positives: bounce to **exactly 30**, then stop. Lost count → enter best estimate + write `approx` in notes.
4. Vary bounce tempo slightly between takes (don't metronome everything identically).
5. Save on all three phones before moving to the next take.

Distances: close = 20–30 cm, medium = 40–70 cm, far = over 70 cm.

---

## Block 1 — Counting aloud while bouncing (8 takes, ~13 min)

Recorder fields: scenario `Racket bounce + speaking/counting`, noise level `Quiet`,
noise source `Speech / counting`, count `30`.
**Count aloud 1–30 continuously while bouncing.** Room otherwise quiet.

| Take | Distance | Racket |
|------|----------|--------|
| T1   | close    | A-red  |
| T2   | close    | B-red  |
| T3   | close    | C-black|
| T4   | close    | D-black|
| T5   | medium   | A-black|
| T6   | medium   | C-red  |
| T7   | far      | B-black|
| T8   | far      | D-red  |

## Block 2 — Loud music + bouncing, no talking (8 takes, ~13 min)

Recorder fields: scenario `Racket bounce + loud background sound`, noise level `Loud`,
noise source `Music / TV`, count `30`.
Music loud enough that you'd raise your voice to talk over it.
**T9–T12 use Music 1 (percussion). T13–T16 use Music 2 (TV/speech/pop).**

| Take | Distance | Racket | Music |
|------|----------|--------|-------|
| T9   | close    | A-red  | 1     |
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

Count `0` on all. Each take ~40 seconds of continuous activity. Racket ID `none`
except T35. **Zero bounces anywhere in this block.**

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
| T36  | Mixed hard negatives, no bounce | Medium / Other        | close    | claps, coughs, keys on table, cutlery, placing a phone down |

## Block 6 — Optional extras (5 takes, ~10 min, do if energy remains)

| Take | Scenario                            | Level / Source   | Distance | Racket | Count | Notes field |
|------|-------------------------------------|------------------|----------|--------|-------|-------------|
| T37  | Racket bounce + background sound    | Medium / Room    | close    | A-black| 30    | `room background` (ventilation/street/kitchen) |
| T38  | Racket bounce + background sound    | Medium / Room    | far      | C-red  | 30    | `room background` |
| T39  | Normal racket bounce                | Quiet / None     | close    | D-red  | 30    | `catch after every 5th bounce` |
| T40  | Racket bounce + speaking/counting   | Low / Speech     | medium   | B-black| 30    | `second person talking` — skip if nobody around |
| T41  | Racket bounce + loud background     | Loud / Music-TV  | medium   | D-black| 30    | Music 1 |

## Wrap-up (5 min)

- [ ] Each phone's session list shows 36 takes (41 with optional block)
- [ ] Export the dataset zip from **each** phone
- [ ] Copy all three zips to the computer before deleting anything

---

## Racket-side coverage (for verification, with optional block done)

| Side    | Takes                  | Total | Distances covered |
|---------|------------------------|-------|-------------------|
| A-red   | T1, T9, T17, T22       | 4     | close, far        |
| A-black | T5, T12, T24, T37      | 4     | close, medium, far|
| B-red   | T2, T13, T19, T26      | 4     | close, medium, far|
| B-black | T7, T16, T20, T40      | 4     | close, medium, far|
| C-red   | T6, T10, T23, T38      | 4     | close, medium, far|
| C-black | T3, T14, T18, T27      | 4     | close, medium, far|
| D-red   | T8, T11, T21, T39      | 4     | close, medium, far|
| D-black | T4, T15, T25, T41      | 4     | close, medium, far|

Every side appears 4 times, spread over different scenarios, distances, and noise
conditions, so no racket sound is confounded with any single recording condition.
Without the optional block, five sides have 3 appearances instead of 4 — still fine.

## Why the plan is weighted this way

- **Counting aloud (Blocks 1+3)** and **loud music (Blocks 2+3)** get the most
  takes: these are the two scenarios where live testing (gh-217) fails and where
  no reviewed multi-device data exists yet.
- **Floor/table impacts** get three dedicated takes: only 73 such training rows
  exist in the entire current corpus.
- Quiet baselines are kept small: historically well covered, easy for every model.
- Three phones per take turn ~40 takes of effort into ~120 sessions, all with
  device identity, noise level, and distance in metadata — ready for
  leave-device-out evaluation after timestamp review.
