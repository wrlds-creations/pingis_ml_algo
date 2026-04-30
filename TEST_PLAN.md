# Test Plan: Audio Review, Binary Contact, and Synced Bounce Audio + IMU

Date:
Tester:
Device:
App build:
Contact model version:
Round:

## How This File Is Used

1. Record one take at a time in `Ljud-insamling` or `Studs audio + IMU`.
2. For racket and noise scenarios, review the take directly after recording.
3. I review, verifiera att skarmen visar `Simple Review UI | attack_start | r11c-overview-fit`.
4. Testa dragbar playhead, playback, marker-val och marker-flytt.
5. Fill in each testcase before moving on.
6. Use `Studsdetektor` and `Studs fritt` only with `B0`.
7. `Studs vaxla sida` is a regression check after contact count is acceptable.
8. In `Studs audio + IMU`, keep roughly `0.5-1.0 s` between bounces. Do not collect fast double contacts in this round.

## Round Summary

Vad funkar nu:

Vad ar trasigt nu:

Nasta sak att testa:

## Collector Review Tests

## Synced Audio + IMU Collector Tests

### AI-01
Mode: `bounce_audio_imu_collection`
Preset: `n/a`
Kalibrering: `table_only`
Forvantat utfall: Mode startar via kalibrering, visar samma scenarios som `Ljud-insamling`, och blockerar inte record om sensorn ar korrekt ansluten.
Faktiskt utfall:
PASS / FAIL:
Kommentar:

### AI-02
Mode: `bounce_audio_imu_collection`
Preset: `n/a`
Kalibrering: `table_only`
Forvantat utfall: En `30 s` take i `racket_quiet` auto-stoppar inom ungefaar `30.0-30.5 s`, fryser inte pa slutet, och review-skarmen oppnas direkt, precis som i audio-only-laget.
Faktiskt utfall:
PASS / FAIL:
Kommentar:

### AI-03
Mode: `bounce_audio_imu_collection`
Preset: `n/a`
Kalibrering: `table_only`
Forvantat utfall: Collectorn visar tydligt att den kor synced IMU och sample rate ligger rimligt nara `50 Hz` under inspelning.
Faktiskt utfall:
PASS / FAIL:
Kommentar:

### AI-04
Mode: `bounce_audio_imu_collection`
Preset: `n/a`
Kalibrering: `table_only`
Forvantat utfall: Efter sparad take innehaller session-JSON både vanlig audio-eventdata och `imu_recording` med samples, start/slut-tid och sample count.
Faktiskt utfall:
PASS / FAIL:
Kommentar:

### AI-05
Mode: `bounce_audio_imu_collection`
Preset: `n/a`
Kalibrering: `table_only`
Forvantat utfall: Review-flodet fungerar oforandrat pa synced takes: save, discard och `Review next pending` beter sig som i audio-only-laget.
Faktiskt utfall:
PASS / FAIL:
Kommentar:

## Collector Review Tests

### CR-01
Mode: `audio_collection`
Preset: `n/a`
Kalibrering: `n/a`
Forvantat utfall: En `30 s` take i `racket_quiet` auto-stoppar inom ungefaar `30.0-30.5 s`, fryser inte pa slutet, och review-skarmen oppnas direkt.
Faktiskt utfall:
PASS / FAIL:
Kommentar:

### CR-01B
Mode: `audio_collection`
Preset: `n/a`
Kalibrering: `n/a`
Forvantat utfall: Det gar att vaxla mellan `Front camera` och `Back camera` fore take. Previewn byter kamera, visar mer av hela bilden utan hard crop, och den valda kameran anvands i den inspelade review-videon.
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
Forvantat utfall: `Review next pending` oppnar en aldre oreviewad take fran ko-listan.
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
- `CR-01` till `CR-12` ska passera eller ge tydliga, reproducerbara UI-fel att fixa.
- `DC-01` och `DC-02` ska passera.
- `DC-03` eller `BF-03` ska na minst `14/20`.
- `DC-04` eller `BF-04` ska na minst `14/20`.
- `BA-01` far inte regressa tydligt.


## Extra checks for current build

| # | Action | Expected result | Result | Notes |
|---|--------|-----------------|--------|-------|
| 1 | In `Studsdetektor`, move `Merge window` and bounce 10 times on racket | Higher merge window should reduce duplicate counts | ? Pass ? Fail | |
| 2 | In `Studs fritt`, bounce only on floor | `AUDIO DEBUG` / `CONTACT DEBUG` should make it obvious if event was counted or vetoed | ? Pass ? Fail | |
| 3 | In all three modes, compare binary + surface debug labels for one event | Debug should show both binary decision and surface (`RACKET/TABLE/FLOOR/NOISE`) | ? Pass ? Fail | |

## What To Send Me After A Test

- Which mode you used: `Studsdetektor`, `Studs fritt`, or `Studs vaxla sida`
- Exact setup: quiet / counting / low music / mid music / floor only
- What you did: for example `10 racket bounces`, `10 floor bounces`, `20 FH bounces`
- Expected vs actual count: for example `expected 10, got 11`
- Merge window value if you changed it
- One or two exact debug rows when something failed
- Short conclusion: `double count`, `floor counted as racket`, `missed contacts`, or `works`
