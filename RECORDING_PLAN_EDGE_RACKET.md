# Dual Residual Edge-Racket Recording Plan and Actual Session

This targeted round adds the missing racket-edge sound coverage needed to
retrain `dual_residual` without weakening its current noise resistance.

**Recording status:** completed on 2026-07-27 as Round
`CJ-20260727-01`, Takes T01-T04, on iPhone, Motorola, and Huawei. This
document now records both the intended sub-block sequence and the actual
capture-time deviations so processing does not depend on chat history.

The observed failure is specific: the HF gate detected all three edge contacts,
but Dual classified two of them as `table_bounce` (`ct=92%` and `ct=62%`,
both with `cr=1%`). This plan therefore targets classifier training, not gate or
threshold tuning.

## Objective

- Add precise edge-rubber contacts as `racket_bounce` positives.
- Cover every A-D racket, red/black rubber face, and top/left/right/throat edge
  zone.
- Add matched ball-on-table contacts as `not_racket` / `table_bounce`
  negatives from the same room and phone placement.
- Make capture practical by recording deterministic sub-blocks inside four
  continuous physical recordings rather than saving 46 separate takes.
- Train the primary new Dual candidate from scratch on the complete original
  Train set plus this new Train round.
- Keep the original sealed holdout closed. Treat this round's T03-T04 as a
  same-day diagnostic set, not a replacement sealed holdout.

New center-rubber control recordings are not required in this round. Center
coverage and center regression evaluation remain in the original Train and
sealed-holdout datasets.

Do not fine-tune on the new edge recordings alone.

## Round structure

The actual session used one Round ID and four sequential Take IDs:

| Dataset use | Actual Round ID | Saved takes | Physical recordings | Phone recordings |
|---|---|---:|---:|---:|
| Train | `CJ-20260727-01` | T01-T02 | 2 | 6 |
| Same-day diagnostic | `CJ-20260727-01` | T03-T04 | 2 | 6 |
| Recorded total | - | T01-T04 | 4 | 12 |

The exported JSON normalizes the Round ID to lowercase
`cj-20260727-01`. Keep the raw value unchanged. The stable recording identity
is:

`round_id + take_id + device_alias`

All three phones recorded every physical take simultaneously. The intended and
actual mapping is:

| Take | Intended content | Actual placement | Raw target metadata | Processing use |
|---|---|---|---|---|
| T01 | 32 edge-racket sub-blocks | P1 | Huawei `bounce`; iPhone/Motorola incorrectly `hard_negative` | Train; override all three to `racket_bounce` |
| T02 | 20 table contacts | P2 | all `hard_negative` | Train hard negative |
| T03 | 9 fresh edge-racket sub-blocks | P1, after moving the phones back to the T01 area | all `bounce` | same-day edge diagnostic |
| T04 | 10 fresh table contacts | P2, after moving the phones back to the T02 area | all `hard_negative` | same-day table diagnostic |

P1 and P2 are different physical phone/action placements. Racket content was
recorded only at P1 and table content only at P2 in this round, so placement is
correlated with class. T03 and T04 also recreate the same-day placement used by
their Train counterpart. For those reasons, T03-T04 must not be presented as
an independent sealed edge holdout or used as the sole promotion gate. The
unchanged original sealed holdout remains authoritative.

The current QA recorder writes the native PCM stream continuously until Stop.
At 22.05 kHz mono PCM16, a 10-15 minute WAV is approximately 27-40 MB per
phone and is well within the recorder's file limit. Check that every phone has
at least 200 MB free before starting.

## Definition of an edge contact

An edge contact means:

- the ball contacts the rubber approximately **5-20 mm inside** the outer
  rubber boundary;
- the contact is clearly outside the central half of the racket face; and
- the ball does **not** contact the exposed blade, rim, frame, handle, table,
  floor, or another object.

Zones are named relative to a racket held with its handle pointing toward the
person:

- `edge_top`: opposite the handle;
- `edge_left`: left side;
- `edge_right`: right side; and
- `edge_throat`: bottom edge near the handle.

If a sub-block contains a clear center, rim, frame, table, or floor contact,
follow the redo procedure below. Do not relabel an uncertain contact as clean
edge training data.

## Recorder setup

Use the STIGA QA `Audio dataset recorder` in **Round plan** mode.

The actual session used:

- Round ID `CJ-20260727-01` on every phone;
- device aliases `CJ-iphone`, `CJ-moto`, and `CJ-huawei`;
- sequential Take IDs T01, T02, T03, and T04; and
- targets intended as `Racket bounce`, `Hard negative`, `Racket bounce`, and
  `Hard negative`.

The operator only confirms the shared Round ID, Take ID, binary target, and an
exception note. Sub-block, zone, racket, face, expected count, reviewed
timestamps, and excluded intervals are added during post-processing.

T01 was accidentally saved as `hard_negative` on iPhone and Motorola. Huawei
was correctly saved as `bounce`. This mistake changes metadata only; it does
not change native PCM capture. Do not edit the raw JSON. Apply the correction
in the derived manifest.

### Phone placement

The phones were flat and screens up in this left-to-right order:

| Position | Device |
|---|---|
| Left | iPhone |
| Center | Motorola |
| Right | Huawei |

- Put the phones side by side at the same height.
- Keep the intended ball-contact point exactly **50 cm** from the center
  Motorola phone.
- Point the microphone ends toward the action area when the device design makes
  that possible without tilting the phone.
- Do not rotate or reposition the phones while a physical recording is active.
- Bluetooth must be off and no headset may be connected.

Actual placement sequence:

1. T01 used placement P1 for racket contacts.
2. The phones moved to placement P2 for T02 table contacts.
3. The phones moved back to the T01 area, recorded as P1, for T03 racket
   contacts.
4. The phones moved back to the T02 area, recorded as P2, for T04 table
   contacts.

The exact centimeter-level replacement offset was not measured. Preserve P1
and P2 as categorical placement metadata and do not claim that T02/T04 are
position-matched to T01/T03.

## Rackets

- Use the same physical rackets labeled A-D in the original
  `CJ-20260723-01` round.
- `A-red` means Racket A with its red rubber facing the ball; `A-black` means
  the black rubber faces the ball. Apply the same rule to B-D.
- Follow the exact sequence below. Do not substitute a different face based on
  the earlier live test.

| Racket ID | Physical racket |
|---|---|
| A | `CJ-Long handle Racket` |
| B | `CJ-short handle Racket` |
| C | `Stiga hex Racket - star club 68-061` |
| D | `Stiga Racket - royal 3 start` |

Hold the racket approximately horizontal with the planned rubber face upward.
These are controlled vertical bounces, not normal strokes.

## Fixed ball height

Use only the Normal height for this complete round:

- racket contacts: keep the approximate highest point after each contact at
  `30 cm` above the rubber;
- table contacts: release the ball from `30 cm`, allow exactly one table
  impact, catch it, wait 1-2 seconds, and repeat.

An approximate variation of `+/- 5 cm` is acceptable. Set one visual reference
before starting the round and do not change it between sub-blocks.

## Pilot without consuming official T01

Use a separate Round ID such as `CJ-EDGE-PILOT-YYYYMMDD` and Take `T00`.
The pilot is excluded from all training and evaluation, so it may contain:

1. three valid edge-rubber contacts at Normal height (`30 cm`);
2. three center-rubber contacts at Normal height (`30 cm`); and
3. three individual ball-on-table drops from `30 cm`, catching after the first
   impact.

Record the pilot continuously on all three phones with one synchronization clap.
Export and listen to one WAV from every phone. Confirm the files are not
clipped, silent, routed through Bluetooth, or dominated by handling noise.
Then reset to the official Train Round ID and Take T01.

No pilot files are present in the delivered `CJ-20260727-01` folder, which is
correct for the official-data export. This document does not assert whether a
pilot was recorded and exported elsewhere.

## Continuous recording procedure

Use this procedure for every official physical recording:

1. Keep the room quiet except for the spoken sub-block slates and the holdout
   speech sub-block.
2. Stay still during the recorder's initial noise-floor measurement.
3. Confirm all phones show the same Round ID, Take ID, and binary target.
4. Start all three phones. After every phone shows `GO`, wait one second, clap
   once, and wait two seconds. This is the only synchronization clap in that
   physical recording.
5. Set up the first listed sub-block completely.
6. Clearly say its short slate, for example `A red top`. Wait two seconds.
7. Make exactly **10 valid contacts** at a controlled tempo. Count silently.
8. After contact 10, wait two seconds before moving the racket or ball.
9. Change to the next listed sub-block. The transition may take as long as
   needed. When ready, say the next slate, wait two seconds, and make its 10
   contacts.
10. Do not clap between sub-blocks.
11. After the final sub-block, leave three seconds of silence and stop all
    phones.
12. Save on all phones and confirm that all three show the same saved Round ID,
    Take ID, duration, and target.

Spoken slates are navigation markers, not training examples. Post-processing
must exclude the initial clap, every slate, all transition/handling intervals,
and the silence before the first valid contact.

### Redo without restarting the long recording

If the wrong zone, face, racket, count, or contact occurs:

1. stop bouncing;
2. wait two seconds;
3. say `redo` followed by the full slate, for example
   `redo C black throat`;
4. reset the condition;
5. wait two seconds; and
6. repeat all 10 contacts for that sub-block.

Add a take-level exception note before saving. Post-processing keeps the final
complete attempt and excludes the failed attempt and redo speech.

If any phone stops, changes audio route, or shows an error, stop the other
phones. Keep the intact files, restart all three phones using the next unused
Take ID, and say `continuation from` plus the first incomplete sub-block slate.
Only the incomplete and remaining sub-blocks need to be repeated. Record the
continuation Take ID and starting sub-block in the exception notes.

No continuation take was present in the delivered export.

---

## Train takes: `CJ-20260727-01` T01-T02

### T01 - All edge-racket positives

Recorder target: `Racket bounce`.

Record all 32 sub-blocks below inside one continuous physical recording. Finish
all four zones before turning the racket over or changing racket. The first
sub-block is explicitly **A-red top**.

| Sub-block | Spoken slate | Contact zone | Racket/face | Height | Contacts |
|---|---|---|---|---:|---:|
| S01 | `A red top` | `edge_top` | A-red | Normal, 30 cm | 10 |
| S02 | `A red left` | `edge_left` | A-red | Normal, 30 cm | 10 |
| S03 | `A red right` | `edge_right` | A-red | Normal, 30 cm | 10 |
| S04 | `A red throat` | `edge_throat` | A-red | Normal, 30 cm | 10 |
| S05 | `A black top` | `edge_top` | A-black | Normal, 30 cm | 10 |
| S06 | `A black left` | `edge_left` | A-black | Normal, 30 cm | 10 |
| S07 | `A black right` | `edge_right` | A-black | Normal, 30 cm | 10 |
| S08 | `A black throat` | `edge_throat` | A-black | Normal, 30 cm | 10 |
| S09 | `B red top` | `edge_top` | B-red | Normal, 30 cm | 10 |
| S10 | `B red left` | `edge_left` | B-red | Normal, 30 cm | 10 |
| S11 | `B red right` | `edge_right` | B-red | Normal, 30 cm | 10 |
| S12 | `B red throat` | `edge_throat` | B-red | Normal, 30 cm | 10 |
| S13 | `B black top` | `edge_top` | B-black | Normal, 30 cm | 10 |
| S14 | `B black left` | `edge_left` | B-black | Normal, 30 cm | 10 |
| S15 | `B black right` | `edge_right` | B-black | Normal, 30 cm | 10 |
| S16 | `B black throat` | `edge_throat` | B-black | Normal, 30 cm | 10 |
| S17 | `C red top` | `edge_top` | C-red | Normal, 30 cm | 10 |
| S18 | `C red left` | `edge_left` | C-red | Normal, 30 cm | 10 |
| S19 | `C red right` | `edge_right` | C-red | Normal, 30 cm | 10 |
| S20 | `C red throat` | `edge_throat` | C-red | Normal, 30 cm | 10 |
| S21 | `C black top` | `edge_top` | C-black | Normal, 30 cm | 10 |
| S22 | `C black left` | `edge_left` | C-black | Normal, 30 cm | 10 |
| S23 | `C black right` | `edge_right` | C-black | Normal, 30 cm | 10 |
| S24 | `C black throat` | `edge_throat` | C-black | Normal, 30 cm | 10 |
| S25 | `D red top` | `edge_top` | D-red | Normal, 30 cm | 10 |
| S26 | `D red left` | `edge_left` | D-red | Normal, 30 cm | 10 |
| S27 | `D red right` | `edge_right` | D-red | Normal, 30 cm | 10 |
| S28 | `D red throat` | `edge_throat` | D-red | Normal, 30 cm | 10 |
| S29 | `D black top` | `edge_top` | D-black | Normal, 30 cm | 10 |
| S30 | `D black left` | `edge_left` | D-black | Normal, 30 cm | 10 |
| S31 | `D black right` | `edge_right` | D-black | Normal, 30 cm | 10 |
| S32 | `D black throat` | `edge_throat` | D-black | Normal, 30 cm | 10 |

This yields `80` independent physical edge contacts per racket in Train:
`40` on red and `40` on black. Every racket/face/zone cell has 10 physical
contacts. The three synchronized phones provide three device views of each
impact, but they are not counted as independent physical contacts.

Ten impacts per exact racket/face/zone is strong targeted coverage for this
augmentation, not a standalone claim of arbitrary-racket generalization. The
new candidate still trains on the complete original Train set. If per-racket
holdout recall remains unstable, collect a second round in a freshly reset
session rather than adding more contacts to this same recording.

### T02 - Table negatives at placement P2

Recorder target: `Hard negative`.

T02 contains ball-on-table contacts and **zero ball-on-racket contacts**. The
phones moved from T01 placement P1 to table placement P2 before this take.
These are valid hard negatives, but they are not position-matched to the T01
edge positives.

| Sub-block | Spoken slate | Action | Drop height | Contacts |
|---|---|---|---:|---:|
| N01 | `table fixed` | Individual drops at one table position; catch after first impact | Normal, 30 cm | 10 |
| N02 | `table varied` | Individual drops at varied table positions; catch after first impact | Normal, 30 cm | 10 |

---

## Same-day diagnostic takes: `CJ-20260727-01` T03-T04

T03 and T04 were recorded as fresh contacts after moving the phones back to
the nominal T01 and T02 areas. They share the same Round ID, day, room, and
class-specific placement profiles as T01 and T02. Use them for same-day
diagnosis only; do not treat them as an independent sealed holdout.

### T03 - Edge-racket diagnostic positives at placement P1

Recorder target: `Racket bounce`.

All three raw JSON files correctly use target `bounce`. The planned sub-block
sequence was:

| Sub-block | Spoken slate | Action | Racket/face | Height | Contacts | Background |
|---|---|---|---|---:|---:|---|
| H01 | `A red top` | `edge_top` | A-red | Normal, 30 cm | 10 | quiet |
| H02 | `A black throat` | `edge_throat` | A-black | Normal, 30 cm | 10 | quiet |
| H03 | `B red left` | `edge_left` | B-red | Normal, 30 cm | 10 | quiet |
| H04 | `B black right` | `edge_right` | B-black | Normal, 30 cm | 10 | quiet |
| H05 | `C red top` | `edge_top` | C-red | Normal, 30 cm | 10 | quiet |
| H06 | `C black left` | `edge_left` | C-black | Normal, 30 cm | 10 | quiet |
| H07 | `D red right` | `edge_right` | D-red | Normal, 30 cm | 10 | quiet |
| H08 | `D black throat` | `edge_throat` | D-black | Normal, 30 cm | 10 | quiet |
| H09 | `D black top speech` | `edge_top` | D-black | Normal, 30 cm | 10 | low speech |

H01-H08 provide two quiet diagnostic cells per physical racket across both
rubber faces. Together they cover all four zones. H09 is the planned
edge-plus-speech stress segment so D's quiet result is not confounded with
speech.

For H09, a second person should speak naturally during the 10 contacts. Use the
previously established normal-conversation reference of `63 +/- 3 dBA LAeq`
measured at 1 m from the speaker; this is a preflight reference, not something
to remeasure during recording. If alone, use a fixed local spoken recording
from a separate playback device. Do not use one of the recorder phones for
playback.

Center-racket regression remains covered by the unchanged original sealed
holdout.

### T04 - Table-negative diagnostic at placement P2

Recorder target: `Hard negative`.

All three raw JSON files correctly use target `hard_negative`. The planned
sub-block was:

| Sub-block | Spoken slate | Action | Drop height | Contacts | Background |
|---|---|---|---:|---:|---|
| HN01 | `table varied` | Individual drops at varied table positions; catch after first impact | Normal, 30 cm | 10 | quiet |

## Wrap-up

Completed export/integrity checks:

- [x] T01-T04 exist once on iPhone, Motorola, and Huawei: 12 recordings.
- [x] All 12 WAV files are valid 22.05 kHz mono PCM16 and agree with their JSON
  sample counts, durations, and byte sizes.
- [x] Extracted WAV/JSON files match the three ZIP exports byte-for-byte.
- [x] Sample-exact `exerciseStartSample`, `firstSampleAtMs`,
  `exerciseStartedAtMs`, and `lastSampleAtMs` metadata is present.
- [x] T01 iPhone/Motorola raw target mismatch is documented for derived
  correction; raw files remain unchanged.
- [x] T02-T04 target metadata is internally consistent across all phones.
- [ ] Spoken slate boundaries, redos, exact valid contact counts, and contact
  zones still require audio/timestamp review.
- [ ] Do not delete device data until sub-block review and export backup are
  confirmed.

Recorded full-WAV durations were approximately:

| Take | iPhone | Motorola | Huawei |
|---|---:|---:|---:|
| T01 | 421.19 s | 421.63 s | 423.16 s |
| T02 | 51.79 s | 51.83 s | 52.24 s |
| T03 | 165.50 s | 166.21 s | 166.49 s |
| T04 | 28.39 s | 25.64 s | 28.19 s |

Motorola contains a very small number of isolated full-scale samples
(`2`, `53`, `10`, and `11` in T01-T04 respectively; at most two consecutive).
This is negligible for intake and does not justify discarding a take. Preserve
the clipping statistic as a quality flag.

Do not delete device data until all exported ZIP files have passed integrity
checks.

## Post-processing manifest

The recording-level correction layer now exists at:

`data/rounds/CJ-20260727-01/manifest.json`

It contains 12 corrected recording rows and the 44 planned logical sub-blocks.
`manifest.csv` contains the recording rows, while `subblocks.csv` contains the
logical sub-block schedule with reviewed timestamp/count fields still empty.

Join each saved recording by:

`round_id + take_id + device_alias`

Create:

- `recording_group_id = round_id + take_id`;
- `dataset_bucket = train` for T01-T02 and `diagnostic` for T03-T04;
- `placement_id = P1` for T01/T03 and `P2` for T02/T04;
- `subblock_id` from the deterministic sequence or a documented continuation;
- `physical_event_group_id = round_id + take_id + subblock_id + event_index`;
- `binary_label = racket_bounce` for reviewed edge events and `not_racket` for
  reviewed table events;
- internal four-class label `racket_bounce` for reviewed edge events and
  `table_bounce` for reviewed table events;
- `impact_zone`, `racket_id`, `rubber_face`, `ball_height_cm` or
  `table_drop_height_cm`, `background`, `expected_count`, and actual phone
  position;
- timestamps for each spoken slate, sub-block start/end, valid impact, failed
  attempt, and continuation boundary;
- sync offset, drift estimate, and excluded intervals for the initial clap,
  slates, silence, transitions, handling, failed attempts, and interruptions;
  and
- quality flags for off-zone contact, rim/frame contact, uncertain count,
  clipping, interruption, route mismatch, or device drift.

Apply these non-destructive recording-level corrections:

| Take/device | Raw target | Correct target | Action |
|---|---|---|---|
| T01 / iPhone | `hard_negative` | `bounce` | override only in derived manifest |
| T01 / Motorola | `hard_negative` | `bounce` | override only in derived manifest |
| T01 / Huawei | `bounce` | `bounce` | no override |

Normalize device aliases case-insensitively to `CJ-iphone`, `CJ-moto`, and
`CJ-huawei` in the derived manifest while preserving raw JSON values.

Do not automatically mine speech, slates, racket handling, or transition sounds
from a positive T01 recording as either positives or hard negatives. They are
excluded intervals unless separately reviewed and intentionally labeled.

All three device recordings of one physical event must share the same
`physical_event_group_id`. The complete continuous recording and all events
derived from it must remain in its assigned Train or diagnostic bucket; never
randomly move events from one continuous recording into another bucket.

## Training and evaluation contract

The primary candidate must:

1. use the existing `dual_residual` architecture;
2. initialize new weights and train from scratch;
3. train on the original Train split plus
   `CJ-20260727-01` T01-T02 after reviewed manifest correction;
4. exclude T03-T04 and the original final holdout from training;
5. compare against the unchanged current Dual model before any STIGA export;
   and
6. keep all generated checkpoints, features, and reports outside Git.

A warm-start/fine-tuned Dual may be trained as a challenger only when it mixes
the original Train data with the new edge data. Never continue training on edge
data alone.

Report results separately for:

- edge-racket recall;
- center-racket recall;
- table false-positive count;
- speech/noise false-positive count;
- per-racket and per-device precision, recall, F1, and count error; and
- the unchanged original sealed holdout.

Report T03-T04 separately as a same-day, class-position-correlated diagnostic.
Do not merge their metrics into the original sealed-holdout headline. If a
promotion decision needs independent edge-specific evidence, record a new edge
holdout on another day after architecture, weights, and threshold are frozen.
Do not train on that future holdout.

Promote a new QA candidate only if edge recall improves materially without an
unacceptable regression in center recall, table rejection, noise resistance,
or any individual phone.
