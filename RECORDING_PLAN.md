# Multi-Phone Audio Recording Plan (2026-07-23)

One official round takes about 80-95 minutes. All three phones record every take
simultaneously, producing **43 recordings per phone / 129 recordings total**.

The binary objective is only:

- `racket_bounce`: a ball contacts the racket.
- `not_racket`: no ball contacts any racket anywhere in the take.

Follow T01-T43 in order. Do not improvise new official takes inside the round.

## Recorded round and racket identities

The data corresponding to this plan is recording round `CJ-20260723-01`. Its
local corrected metadata is under `data/rounds/CJ-20260723-01/`.

| Racket ID | Physical racket |
|---|---|
| A | `CJ-Long handle Racket` |
| B | `CJ-short handle Racket` |
| C | `Stiga hex Racket - star club 68-061` |
| D | `Stiga Racket - royal 3 start` |

## What the recorder must capture

Use the STIGA QA `Audio dataset recorder` in **Round plan** mode. Set these once
before the official round:

- The same **Round ID** `CJ-20260723-01` on all phones.
- A different **Device alias** on each phone: `CJ-iphone`, `CJ-moto`, and
  `CJ-huawei`.
- **Take** `T01` on all phones.

For each take, the phone operator only needs to confirm:

- the same Take ID is shown on all three phones;
- the target is `Racket bounce` or `Hard negative`; and
- an exception note only if something departed from this plan.

The recorder automatically stores the raw WAV, device/audio provenance, Round
ID, Take ID, device alias, and target. A successful Save advances the Take ID.
Retry, Discard, and a failed save do not advance it.

Do **not** enter scenario, phone position, orientation, dataset split, racket,
distance, background, or expected count on each phone. Those fields are joined
from this plan during post-processing.

## Prep checklist (5 min)

- [ ] Bluetooth is OFF on all phones and no headset is connected.
- [ ] Device aliases are set once and the same official Round ID is on all phones.
- [ ] Phones are side by side, screens up, and **flat for the whole round**.
- [ ] The stated distance is measured from the phones to the bounce/sound point.
- [ ] A separate playback device and speaker are ready with the downloaded
      **Music 1** and **Music 2** files below.
- [ ] The NIOSH Sound Level Meter app is ready on the iPhone for preflight
      calibration only.
- [ ] Rackets are labeled A-D; red/black means the side facing the ball.
- [ ] A separate pilot has passed; the official round shows T01 on every phone.

### Fixed local music

Use these downloaded YouTube Audio Library files. Play the local MP3 files, not
a streamed YouTube video, so an advertisement, network pause, or notification
cannot enter a take.

| ID | Track | Role | Fixed excerpt |
|---|---|---|---|
| Music 1 | `Beat Your Competition - Vibe Tracks.mp3` | percussion-heavy/transient | `01:00-02:00` |
| Music 2 | `Spring In My Step - Silent Partner.mp3` | melodic/pop contrast | `00:10-01:10` |

Use the same playback device, speaker/output path, speaker position, and volume
for every music take. Do not use any of the three recorder phones as the music
source. Put the playback device in Do Not Disturb and use offline playback;
airplane mode is preferred when it does not disable the chosen local speaker.

Cue the file to the listed start time and start it about two seconds before
pressing Record. Play the 60-second excerpt once, with repeat, shuffle,
crossfade, and sound enhancement disabled. Do not loop it: a loop boundary can
become an artificial negative event. Keep the music running until all phones
have stopped. If a take will not finish before the excerpt ends, retry it rather
than recording the end fade or silence.

### Loudness calibration

Use the free **NIOSH Sound Level Meter** iOS app:
<https://www.cdc.gov/niosh/noise/about/app.html>.

Use these settings:

- Calibration offset: `0.0 dB` unless a real calibrated reference meter is
  available. Do not offset the app merely to match another phone.
- Standard: `NIOSH`.
- Threshold: `80 dB`.
- Exchange rate: `3 dB`.
- Time weighting: `Slow`.
- Frequency weighting: `A`.
- Read **LAeq** over a fresh 15-20 second measurement, not the changing
  instantaneous number. Reset the measurement before each check.

The Standard, Threshold, and Exchange rate settings mainly affect dose/TWA. This
plan uses the LAeq reading as a repeatable sound-field estimate.

| Sound condition | Takes | Where to place the meter | Target |
|---|---|---|---:|
| Music 1 or Music 2 | T09-T19, T29, T32-T34, T41 | center of the three-phone recording cluster, microphone at phone height | `75 +/- 3 dBA LAeq` |
| Normal clear counting | T01-T08, T31 | exactly 1.0 m from the person's mouth, at mouth height | `63 +/- 3 dBA LAeq` |
| Raised voice over music | T17-T19, T32 | exactly 1.0 m from the person's mouth, at mouth height | `73 +/- 3 dBA LAeq` |

Calibrate each voice style by counting 1-30 at the intended pace. During the
official takes, preserve that voice effort; do not speak louder for a far take
or quieter for a close take. T40 deliberately remains natural low background
speech rather than raised calibrated speech.

For music, place the iPhone at the center of the planned recorder cluster,
measure the selected excerpt, adjust the playback volume until it is in range,
then close the NIOSH app and return the iPhone to the STIGA recorder. Keep that
one volume setting for both tracks unless the two source files require one
documented preflight correction to reach the same target.

The three phones may still save different digital waveform levels from the same
physical sound field. In particular, do not turn the music up only for the
iPhone because its recording appears quieter. That device/microphone gain
difference is valid cross-device evidence and is handled during
post-processing, not by giving each phone a different acoustic source. Do not
raise the music above the `78 dBA LAeq` upper tolerance to compensate for one
phone.

### Pilot without consuming T01

Use a separate Round ID, for example `CJ-20260723-PILOT`, and Take `T00`. Record,
save, export, and open one WAV + JSON from every phone. Then set the official
Round ID and reset every phone to T01. The pilot is not part of the 43 official
takes and must not enter training or holdout data.

### Phone placement schedule

Keep each phone in the same physical position for a whole block, then rotate it
as shown. All phones stay flat and point toward the same action area. Position
is not entered on the phone; post-processing derives it from device alias,
block, and this table.

| Block | iPhone | Motorola | Huawei |
|---|---|---|---|
| 1 | Left | Center | Right |
| 2 | Right | Left | Center |
| 3 | Center | Right | Left |
| 4 | Left | Right | Center |
| 5 | Right | Center | Left |
| 6 | Center | Left | Right |
| 7 | Left | Center | Right |

## Rules for every take

1. Start the planned background before pressing Record.
2. Stay still and keep the background unchanged through the recorder's initial
   noise-floor measurement.
3. Confirm all three phones show the same Round ID and Take ID.
4. After all phones show `GO`, wait one second, clap once, wait two seconds, and
   begin the planned action. The clap is a sync marker, not a racket bounce.
5. For positives, make **exactly 30 ball-on-racket contacts**, then leave three
   seconds of silence before stopping. If the count is uncertain, add the
   exception note `approx count` and continue; exact timestamps are reviewed
   later.
6. For hard negatives, separate individual sounds by 1-2 seconds. There must be
   **zero ball-on-racket contacts** in the entire take.
7. Vary bounce tempo slightly between positive takes.
8. Save on all three phones before continuing. Confirm that every phone advanced
   to the same next Take ID.
9. If a phone retries, discards, fails to save, or drifts, use its Take control
   to realign it before the next action.
10. If an unplanned ball-on-racket contact occurs in a hard-negative take, retry
    it. Do not describe a contaminated take as fully negative.

Distances: close = 20-30 cm, medium = 40-70 cm, far = over 70 cm.

---

## Block 1 - Counting aloud while bouncing (8 takes, about 13 min)

Recorder target: `Racket bounce`.

Count aloud 1-30 continuously using the normal clear voice calibrated to
`63 +/- 3 dBA LAeq` at 1 m. The room is otherwise quiet.

| Take | Action | Racket | Distance | Background |
|---|---|---|---|---|
| T01 | Count aloud while bouncing | A-red | close | spoken counting |
| T02 | Count aloud while bouncing | B-red | close | spoken counting |
| T03 | Count aloud while bouncing | C-black | close | spoken counting |
| T04 | Count aloud while bouncing | D-black | close | spoken counting |
| T05 | Count aloud while bouncing | A-black | medium | spoken counting |
| T06 | Count aloud while bouncing | C-red | medium | spoken counting |
| T07 | Count aloud while bouncing | B-black | far | spoken counting |
| T08 | Count aloud while bouncing | D-red | far | spoken counting |

## Block 2 - Loud music + bouncing, no talking (8 takes, about 13 min)

Recorder target: `Racket bounce`.

Use the fixed `75 +/- 3 dBA LAeq` music setting. T09-T12 use the Music 1
excerpt; T13-T16 use the Music 2 excerpt.

| Take | Action | Racket | Distance | Background |
|---|---|---|---|---|
| T09 | Bounce, no talking | A-red | close | loud Music 1 |
| T10 | Bounce, no talking | C-red | close | loud Music 1 |
| T11 | Bounce, no talking | D-red | close | loud Music 1 |
| T12 | Bounce, no talking | A-black | far | loud Music 1 |
| T13 | Bounce, no talking | B-red | far | loud Music 2 |
| T14 | Bounce, no talking | C-black | far | loud Music 2 |
| T15 | Bounce, no talking | D-black | far | loud Music 2 |
| T16 | Bounce, no talking | B-black | medium | loud Music 2 |

## Block 3 - Counting aloud over loud music (3 takes, about 5 min)

Recorder target: `Racket bounce`.

Count aloud 1-30 with the raised voice calibrated to `73 +/- 3 dBA LAeq` at
1 m while the fixed `75 +/- 3 dBA LAeq` Music 2 excerpt plays. This is the
hardest combined positive condition.

| Take | Action | Racket | Distance | Background |
|---|---|---|---|---|
| T17 | Count aloud while bouncing | A-red | close | loud Music 2 + speech |
| T18 | Count aloud while bouncing | C-black | medium | loud Music 2 + speech |
| T19 | Count aloud while bouncing | B-red | far | loud Music 2 + speech |

## Block 4 - Quiet positive baselines (8 takes, about 12 min)

Recorder target: `Racket bounce`. Do not talk.

| Take | Action | Racket | Distance | Background |
|---|---|---|---|---|
| T20 | Normal racket bounce | B-black | close | quiet |
| T21 | Normal racket bounce | D-red | medium | quiet |
| T22 | Far/soft racket bounce | A-red | far | quiet |
| T23 | Far/soft racket bounce | C-red | far | quiet |
| T24 | Fast racket bounce | A-black | close | quiet |
| T25 | Fast racket bounce | D-black | close | quiet |
| T26 | Slow/high racket bounce | B-red | medium | quiet |
| T27 | Slow/high racket bounce | C-black | medium | quiet |

## Block 5 - Hard negatives (9 takes, about 11 min)

Change the recorder target to `Hard negative`. Each take is about 40 seconds and
contains zero ball-on-racket contacts.

| Take | Action | Racket | Distance | Background |
|---|---|---|---|---|
| T28 | Table taps, ball dropped on table/floor without racket, knuckle knocks | none | close | quiet |
| T29 | Repeat T28 surface impacts | none | close | loud Music 1 |
| T30 | Chair scrapes, footsteps, door close, separated object drops | none | medium | quiet |
| T31 | Talk and count aloud 1-30 | none | close | speech only |
| T32 | Talk loudly over music | none | close | loud Music 2 + speech |
| T33 | Phones listen; no performed impacts | none | close | loud Music 1 |
| T34 | Phones listen; no performed impacts | none | close | loud Music 2 |
| T35 | Pass, spin, and re-grip rackets; tap handles together | B+C | close | low room noise |
| T36 | Drop each racket three times with two-second gaps; no ball | A+B+C+D | close | medium room noise |

T30 is intentionally a **mixed household-impact** take, not a floor/table-impact
take. This prevents its post-processing scenario from contradicting the action.
T29, T33, and T34 use the fixed music level. T31 uses the normal clear counting
voice. T32 uses the fixed Music 2 level plus the raised voice.

## Block 6 - Final positive variations (5 takes, about 10 min)

Change the recorder target back to `Racket bounce`. All five takes are required
and all phones remain flat.

| Take | Action | Racket | Distance | Background |
|---|---|---|---|---|
| T37 | Normal bounce | A-black | close | medium room noise: ventilation/street/kitchen |
| T38 | Normal bounce | C-red | far | medium room noise: ventilation/street/kitchen |
| T39 | Catch after every fifth bounce, then resume | D-red | close | quiet |
| T40 | Bounce while a second person talks; if alone, play recorded speech from the speaker | B-black | medium | low speech |
| T41 | Normal bounce | D-black | medium | loud Music 1 |

T39 supplies catch/after-sound variation. T40 is not optional; recorded speech
is the fallback when no second person is available. T41 uses the fixed Music 1
excerpt and level.

## Block 7 - Supplemental hard negatives (2 takes, about 3 min)

Change the recorder target to `Hard negative`. Each take is about 40 seconds and
contains zero ball-on-racket contacts.

| Take | Action | Racket | Distance | Background |
|---|---|---|---|---|
| T42 | Drop each racket three times with two-second gaps; no ball | A+B+C+D | close | medium room noise |
| T43 | Separated coughs, finger snaps, keys, glass/cutlery sounds, and small object set-downs | none | close | medium room noise |

## Wrap-up (5 min)

- [ ] Filter/check the official Round ID: every phone has T01-T43 exactly once.
- [ ] Every take has the intended `Racket bounce` or `Hard negative` target.
- [ ] Export the dataset ZIP from each phone.
- [ ] Copy all three ZIPs to the computer before deleting anything.
- [ ] Spot-check one positive and one hard-negative WAV + JSON per phone.
- [ ] Keep the pilot export separate from the official round.

---

## Post-processing manifest - not entered during recording

Create one manifest row for every official phone recording. Join recorder output
to this plan by the composite key:

`round_id + take_id + device_alias`

The manifest, not the phone operator, supplies:

- `physical_event_group_id = round_id + take_id` so the three recordings of one
  physical event can never cross train/evaluation boundaries;
- block, planned action, scenario ID, racket ID/side, distance band, background
  level/source, and expected count (`30` for positives, `0` for negatives);
- fixed music filename/track, excerpt start/end, playback target, sound-meter
  app/settings, and the round-level measured LAeq values;
- voice reference style and target for the calibrated counting/talking takes;
- phone position derived from the placement schedule and device alias;
- dataset use, using the fixed split below;
- reviewed event timestamps, actual event count, sync offset, and the excluded
  sync-clap region;
- quality flags such as approximate count, contamination, clipping, interruption,
  route mismatch, missing device, or timing drift.

Expected count is a collection check, **not** timestamp ground truth. Positive
timestamps must be reviewed. A hard-negative take may be treated as negative
only after confirming that it contains no ball-on-racket contact. Exclude the
sync-clap region from both training and final evaluation so the repeated clap
cannot become a class or split shortcut.

### Fixed final holdout assignment

Final holdout takes are:

`T04, T07, T12, T14, T19, T23, T29, T32, T34, T41, T42`

All other takes are Train. Apply this after import to the entire
`physical_event_group_id`; never split the three phones from one take. The final
holdout stays closed during model selection and threshold tuning.

Leave-device-out evaluation is separate and uses only Train groups: train on two
phones and evaluate on the third before opening the final holdout.

### Canonical scenario corrections

- T30: `mixed_household_impacts`, not `floor_table_impact`.
- T39: `racket_bounce_catch_after_sound`.
- T40: `racket_bounce_speech`, using live or recorded speech.
- T43: `mixed_sharp_hard_negatives`.

## Racket-side coverage verification

| Side | Takes | Total | Distances covered |
|---|---|---:|---|
| A-red | T01, T09, T17, T22 | 4 | close, far |
| A-black | T05, T12, T24, T37 | 4 | close, medium, far |
| B-red | T02, T13, T19, T26 | 4 | close, medium, far |
| B-black | T07, T16, T20, T40 | 4 | close, medium, far |
| C-red | T06, T10, T23, T38 | 4 | close, medium, far |
| C-black | T03, T14, T18, T27 | 4 | close, medium, far |
| D-red | T08, T11, T21, T39 | 4 | close, medium, far |
| D-black | T04, T15, T25, T41 | 4 | close, medium, far |

Every racket side appears four times across different distances and background
conditions. Blocks 5 and 7 cover surface impacts, household/object impacts,
speech, music, racket handling/drop, and sharp mixed transients. That is enough
physical coverage for this round's binary objective without adding more operator
metadata or removing any of the 43 takes.
