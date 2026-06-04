# Test Plan: Audio Review, Playing Retro Audio, and Video Stroke

Date:
Tester:
Device:
App build:
Contact model version:
Round:

## How This File Is Used

1. Record or import one take at a time in `Ljudinsamling`, `Ljud + video ML`, or `Video FH/BH`.
2. Use the reviewed protocol presets: `racket_bounce_fh`, `racket_bounce_bh`, `racket_bounce_mixed`, `table_bounce`, `floor_bounce`, `catch_after_sound`, and `speech_music_noise`.
3. Every training-relevant preset must open review directly after recording.
4. Handle every auto-marker with `Confirm`, label edit, `Ignore`, or `Delete` before saving.
5. Inspelningsvyn ska visa stor kameravy, `3 s` countdown, fast start/stop och ingen vertikal scroll.
6. Review queue i collection-vyn ska bara visa pending takes fran aktuell session; gamla pending filer ska finnas kvar pa disk men inte synas dar.
7. I review, verifiera att header/back/revision-text inte ligger bakom Motorola-notch eller statusbar.
8. Testa dragbar playhead, playback, marker-val och marker-flytt.
9. Fill in each testcase before moving on.
10. Use `Studsdetektor` and `Studs fritt` only with `B0`.
11. `Studs vaxla sida` is a regression check after contact count is acceptable.
12. Do not collect or validate IMU/AirHive flows in current rounds; they are retired from active product scope.

## Round Summary

Vad funkar nu:

Vad ar trasigt nu:

Nasta sak att testa:

## Collector Review Tests

## Ljud + Video ML Staged Review

### AVP-01
Mode: `Ljud + video ML`
Input: Direkt inspelning i appen
Forvantat utfall: Appen sparar en hel WAV + MP4 som en reviewtagning, utan synliga 30s-delar. Review 1 visar vanlig ljudvag och later Love markera `Rackettraff`, `Bordsstuds`, och `Ignorera`.
Faktiskt utfall:
PASS / FAIL:
Kommentar:

### AVP-02
Mode: `Ljud + video ML`
Input: `Importera video` fran Downloads/Drive
Forvantat utfall: Android picker later Love valja en MP4 med ljudspar, appen kopierar videon, extraherar ljud till WAV, och oppnar samma Review 1 som direkt inspelning.
Faktiskt utfall:
PASS / FAIL:
Kommentar:

### AVP-03
Mode: `Ljud + video ML`
Input: Efter sparad ljudreview
Forvantat utfall: `Klar med ljud` sparar ljudmarkers, byter till Review 2, kor pose over hela videon, och visar separata motion markers for `Forehand`, `Backhand`, eller `Oklart`.
Faktiskt utfall:
PASS / FAIL:
Kommentar:

### AVP-04
Mode: `Ljud + video ML`
Input: Slutlig save
Forvantat utfall: JSON sparar ljudmarkers och motion markers som separata rader; motion rows har `event_type: motion` och `source_audio_marker_id` nar de skapats fran en rackettraff.
Faktiskt utfall:
PASS / FAIL:
Kommentar:

## Video Stroke Test

### VS-01
Mode: `Video stroke test`
Kamera: `Front`
Forvantat utfall: En video-only take eller importerad MP4 sparas som `.mp4`, och review-vyn oppnas utan IMU-krav.
Faktiskt utfall:
PASS / FAIL:
Kommentar:

### VS-02
Mode: `Video stroke test`
Kamera: `Front snett framifran`
Forvantat utfall: Det gar att scrubba/spela videon, lagga minst 10 `Forehand` och 10 `Backhand` markers, ta bort en marker, och spara session JSON under `pingis_video_stroke_sessions`.
Faktiskt utfall:
PASS / FAIL:
Kommentar:

### VS-03
Mode: `Video stroke test`
Kamera: `Front`
Forvantat utfall: `Analysera markerade slag` kraschar inte nar `video_stroke_model.json` ar otranad, utan visar `Ingen videomodell exporterad an` per marker.
Faktiskt utfall:
PASS / FAIL:
Kommentar:

### VS-04
Mode: `Video stroke test`
Kamera: `Front`
Forvantat utfall: Efter att `video_stroke_model.json` exporterats visar appen `Forehand`, `Backhand`, eller `Oklart` per marker med confidence.
Faktiskt utfall:
PASS / FAIL:
Kommentar:

### VS-05
Mode: `Video stroke test`
Kamera: `Back/front snett framifran`
Forvantat utfall: `Auto-hitta slagforslag` analyserar hela videon och lagger modellforslag pa tidslinjen utan att markera dem som traningsfacit.
Faktiskt utfall:
PASS / FAIL:
Kommentar:

### VS-06
Mode: `Video stroke test`
Kamera: `Back/front snett framifran`
Forvantat utfall: Det gar att bekrafta ett autoforslag som `FH` eller `BH`, korrigera fel forslag, ta bort forslag, och bara bekraftade markers anvands i nasta preprocess.
Faktiskt utfall:
PASS / FAIL:
Kommentar:

### VS-07
Mode: `Videoinsamling FH/BH`
Kamera: `Back/front snett framifran`
Forvantat utfall: Toggeln `Spela in utan stopp` spelar in en lang kallvideo tills Love stoppar manuellt, native-delar den till cirka `30 s` MP4-reviewdelar, och alla delar gar att spela upp och bladdra mellan utan att blanda markers.
Faktiskt utfall:
PASS / FAIL:
Kommentar:

### VS-08
Mode: `Videoinsamling FH/BH`
Kamera: `Back/front snett framifran`
Forvantat utfall: Lagt visas under `DATA`, inte `TEST MODES`, och review liknar audio-review med toppbar, stor videospelare, progress, `0.25x`, `0.5x`, och `1x` utan att playhead/markers tappar sync.
Faktiskt utfall:
PASS / FAIL:
Kommentar:

### VS-09
Mode: `Videoinsamling FH/BH`
Kamera: `n/a`
Forvantat utfall: `Ateruppta senaste` oppnar den senaste sparade videosessionen med befintliga 30s-delar, och `Importera video` later Love valja en MP4/MOV fran telefonen, kopierar den till video-stroke-sessionen och delar den till reviewbara 30s-delar.
Faktiskt utfall:
PASS / FAIL:
Kommentar:

## Retired Audio + IMU Collector Tests

The AirHive/IMU path is no longer active product scope. Do not run or expand synced sensor test cases unless Love explicitly reopens that direction with a new ticket and decision.

## Playing Review Tests

### PR-01
Mode: `Ljud + video ML`
Scenario: `Playing / racket + bord`
Kalibrering: `n/a`
Forvantat utfall: Review visar bara tre labelval: `Rackettraff forehand`, `Rackettraff backhand`, och `Bordsstuds`. Generisk rackettraff, golv, brus, ignore och other visas inte som Playing-labels.
Faktiskt utfall:
PASS / FAIL:
Kommentar:

### PR-02
Mode: `Ljud + video ML`
Scenario: `Playing / racket + bord`
Kalibrering: `n/a`
Forvantat utfall: Auto-markers visar sakerhet/confidence och filtret `Alla / Medium / Sakra` uppdaterar direkt hur manga auto-detekterade markers som visas utan att manuella eller bekraftade markers forsvinner.
Faktiskt utfall:
PASS / FAIL:
Kommentar:

### PR-03
Mode: `Ljud + video ML`
Scenario: `Playing / racket + bord`
Kalibrering: `n/a`
Forvantat utfall: En lang video startar snabbare vid `1x` playback, syncpanelen faller ihop efter `Synka har`, och `12x/16x` zoom gar att dra med playhead och edge-autoscroll.
Faktiskt utfall:
PASS / FAIL:
Kommentar:

## Collector Review Tests

### CR-01
Mode: `audio_collection`
Preset: `n/a`
Kalibrering: `n/a`
Forvantat utfall: En `30 s` take i `racket_bounce_fh` visar stor kameravy, 3 s countdown, vibration vid start/stopp, auto-stoppar inom ungefaar `30.0-30.5 s`, fryser inte pa slutet, och review-skarmen oppnas direkt.
Faktiskt utfall:
PASS / FAIL:
Kommentar:

### CR-01B
Mode: `audio_collection`
Preset: `n/a`
Kalibrering: `n/a`
Forvantat utfall: Det gar att vaxla mellan `Front` och `Back` fore take. Previewn byter kamera, visar mer av hela bilden utan hard crop, och den valda kameran anvands i den inspelade review-videon.
Faktiskt utfall:
PASS / FAIL:
Kommentar:

### CR-01C
Mode: `audio_collection`
Preset: `n/a`
Kalibrering: `n/a`
Forvantat utfall: Det gar att spela in minst tre takes i rad utan appkrasch eller hang. Efter att review oppnats for en take ska nasta inspelning kunna starta direkt utan fel om att en tidigare recording fortfarande ar aktiv.
Faktiskt utfall:
PASS / FAIL:
Kommentar:

### CR-02
Mode: `audio_collection`
Preset: `n/a`
Kalibrering: `n/a`
Forvantat utfall: Review-skarmen visar `Simple Review UI | attack_start | r11c-overview-fit` och far inte visa gamla guided-review-begrepp.
Faktiskt utfall:
PASS / FAIL:
Kommentar:

### CR-02B
Mode: `audio_collection`
Preset: `n/a`
Kalibrering: `n/a`
Forvantat utfall: Om review-video finns for taket visas en videovy ovanfor tidslinjerna, visar hela bilden begripligt i review, och den foljer playhead vid playback.
Faktiskt utfall:
PASS / FAIL:
Kommentar:

### CR-02C
Mode: `audio_collection`
Preset: `n/a`
Kalibrering: `n/a`
Forvantat utfall: Review-vyn fungerar i portratt med kompakt editor: video och knappar syns samtidigt, och `Detail timeline` + `Overview timeline` syns utan att huvudvyn kraver review-scroll.
Faktiskt utfall:
PASS / FAIL:
Kommentar:

### CR-02D
Mode: `audio_collection`
Preset: `1x`
Kalibrering: `n/a`
Forvantat utfall: Nar `Play from here` kor fran borjan av taket startar videon fran borjan i stallet for mitt i klippet, och videon tar inte slut tydligt fore WAV-ljudet.
Faktiskt utfall:
PASS / FAIL:
Kommentar:

### CR-03
Mode: `audio_collection`
Preset: `n/a`
Kalibrering: `n/a`
Forvantat utfall: Den vita playhead-linjen syns direkt, stannar kvar efter tap i tidslinjen, overview-fonstret foljer playback i stallet for att stanna pa gammal marker, och overview gar att scrubba genom att dra horisontellt.
Faktiskt utfall:
PASS / FAIL:
Kommentar:

### CR-04
Mode: `audio_collection`
Preset: `n/a`
Kalibrering: `n/a`
Forvantat utfall: `Play from here` startar fran vald playhead-position, och ett tryck pa `Pause` racker for att kunna scrubba vidare direkt.
Faktiskt utfall:
PASS / FAIL:
Kommentar:

### CR-05
Mode: `audio_collection`
Preset: `1x`, `0.5x`, `0.25x`
Kalibrering: `n/a`
Forvantat utfall: Bade `Play from here` och `Play marker preview` fungerar i alla tre hastigheterna, och review-videon foljer samma playhead.
Faktiskt utfall:
PASS / FAIL:
Kommentar:

### CR-06
Mode: `audio_collection`
Preset: `n/a`
Kalibrering: `n/a`
Forvantat utfall: `Previous marker`, `Play marker preview` och `Next marker` fungerar pa vald marker utan att hoppa fel.
Faktiskt utfall:
PASS / FAIL:
Kommentar:

### CR-07
Mode: `audio_collection`
Preset: `n/a`
Kalibrering: `n/a`
Forvantat utfall: En vald marker gar att dra grovt i huvudtidslinjen utan att det blir pilligt eller otydligt.
Faktiskt utfall:
PASS / FAIL:
Kommentar:

### CR-08
Mode: `audio_collection`
Preset: `n/a`
Kalibrering: `n/a`
Forvantat utfall: `Detail timeline` later mig finjustera samma marker med mycket mindre hopp an tidigare, `+/-10 ms` finns som snabbknappar, och peak-guide visas separat fran sparad marker.
Faktiskt utfall:
PASS / FAIL:
Kommentar:

### CR-09
Mode: `audio_collection`
Preset: `n/a`
Kalibrering: `n/a`
Forvantat utfall: `Add marker here` skapar en ny marker exakt vid playhead och `Delete marker` fungerar aven pa auto-markorer.
Faktiskt utfall:
PASS / FAIL:
Kommentar:

### CR-09B
Mode: `audio_collection`
Preset: `racket_bounce_fh`, `table_bounce`, or `catch_after_sound`
Kalibrering: `n/a`
Forvantat utfall: `Save take` blockeras tills varje auto-marker ar hanterad med `Confirm`, label edit, `Ignore`, eller `Delete`. `Ignore` far inte raknas som negativ traningsdata.
Faktiskt utfall:
PASS / FAIL:
Kommentar:

### CR-10
Mode: `audio_collection`
Preset: `n/a`
Kalibrering: `n/a`
Forvantat utfall: `Snap to attack` flyttar en marker mot transientens borjan om auto-markorn ligger sent.
Faktiskt utfall:
PASS / FAIL:
Kommentar:

### CR-11
Mode: `audio_collection`
Preset: `n/a`
Kalibrering: `n/a`
Forvantat utfall: `Discard take` tar bort tagningen och den raknas inte som klar i scenariot.
Faktiskt utfall:
PASS / FAIL:
Kommentar:

### CR-12
Mode: `audio_collection`
Preset: `n/a`
Kalibrering: `n/a`
Forvantat utfall: `Review pending` oppnar bara oreviewade takes fran aktuell session. Aldre pending samples ska inte synas i collection-vyn.
Faktiskt utfall:
PASS / FAIL:
Kommentar:

### CR-13
Mode: `audio_collection`
Preset: `n/a`
Kalibrering: `n/a`
Forvantat utfall: Android navbar och statusbar ar dolda i setup, collection, recording, review, live detector och bounce-test.
Faktiskt utfall:
PASS / FAIL:
Kommentar:

## Device Contact Tests

### DC-01
Mode: `live_classification`
Preset: `B0`
Kalibrering: `n/a`
Forvantat utfall: `Studsdetektor` i lugn miljo ska rakna minst `19/20` forehand-studsar.
Faktiskt utfall:
PASS / FAIL:
Kommentar:

### DC-02
Mode: `live_classification`
Preset: `B0`
Kalibrering: `n/a`
Forvantat utfall: `Studsdetektor` i lugn miljo ska rakna minst `19/20` backhand-studsar.
Faktiskt utfall:
PASS / FAIL:
Kommentar:

### DC-03
Mode: `live_classification`
Preset: `B0`
Kalibrering: `n/a`
Forvantat utfall: `Studsdetektor` medan du raknar hogt ska na minst `14/20`.
Faktiskt utfall:
PASS / FAIL:
Kommentar:

### DC-04
Mode: `live_classification`
Preset: `B0`
Kalibrering: `n/a`
Forvantat utfall: `Studsdetektor` med lag musik ska na minst `14/20`.
Faktiskt utfall:
PASS / FAIL:
Kommentar:

## Bounce Tests

### BF-01
Mode: `bounce_free`
Preset: `B0`
Kalibrering:
Forvantat utfall: Lugn miljo, `20` FH-studsar ger minst `19/20` total count och rimlig FH-fordelning.
Faktiskt utfall:
PASS / FAIL:
Kommentar:

### BF-02
Mode: `bounce_free`
Preset: `B0`
Kalibrering:
Forvantat utfall: Lugn miljo, `20` BH-studsar ger minst `19/20` total count och rimlig BH-fordelning.
Faktiskt utfall:
PASS / FAIL:
Kommentar:

### BF-03
Mode: `bounce_free`
Preset: `B0`
Kalibrering:
Forvantat utfall: Rakna hogt medan du studsar. Minst `14/20` total count.
Faktiskt utfall:
PASS / FAIL:
Kommentar:

### BF-04
Mode: `bounce_free`
Preset: `B0`
Kalibrering:
Forvantat utfall: Lag musik medan du studsar. Minst `14/20` total count.
Faktiskt utfall:
PASS / FAIL:
Kommentar:

### BA-01
Mode: `bounce_alternating`
Preset: `B0`
Kalibrering:
Forvantat utfall: `20` medvetna FH/BH-vaxlingar ska inte regressa tydligt mot tidigare beteende.
Faktiskt utfall:
PASS / FAIL:
Kommentar:

## Acceptance For This Iteration

- `AI-01` till `AI-05` ska passera eller ge tydliga, reproducerbara fel i synced collector-flodet.
- `CR-01` till `CR-12`, inklusive `CR-09B`, ska passera eller ge tydliga, reproducerbara UI-fel att fixa.
- `DC-01` och `DC-02` ska passera.
- `DC-03` eller `BF-03` ska na minst `14/20`.
- `DC-04` eller `BF-04` ska na minst `14/20`.
- `BA-01` far inte regressa tydligt.
- Hard negatives: table, floor, and catch-after-sound ska visas som avvisade eller grupperade, inte dubbelraknade.


## Extra checks for current build

| # | Action | Expected result | Result | Notes |
|---|--------|-----------------|--------|-------|
| 1 | In `Studsdetektor`, move `Merge window` and bounce 10 times on racket | Higher merge window should reduce duplicate counts | ? Pass ? Fail | |
| 2 | In `Studsdetektor` and `Studs fritt`, create one racket bounce followed by a catch/stop sound | At most one `group_id` should count; after-sound should show `ignored_duplicate`, `surface_veto`, or another ignored reason | ? Pass ? Fail | |
| 3 | In `Studs fritt`, bounce only on floor | `AUDIO DEBUG` / `CONTACT DEBUG` should show `group_id`, binary decision, surface label, and whether event was counted or vetoed | ? Pass ? Fail | |
| 4 | In all three modes, compare binary + surface debug labels for one event | Debug should show both binary decision and surface (`RACKET/TABLE/FLOOR/NOISE`) | ? Pass ? Fail | |

## What To Send Me After A Test

- Which mode you used: `Studsdetektor`, `Studs fritt`, or `Studs vaxla sida`
- Exact setup: quiet / counting / low music / mid music / floor only
- What you did: for example `10 racket bounces`, `10 floor bounces`, `20 FH bounces`
- Expected vs actual count: for example `expected 10, got 11`
- Merge window value if you changed it
- One or two exact debug rows when something failed
- Short conclusion: `double count`, `floor counted as racket`, `missed contacts`, or `works`
