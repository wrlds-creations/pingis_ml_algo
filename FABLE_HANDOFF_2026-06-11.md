# Överlämning: Fable-arbetet 2026-06-10/11 → stiga-app-v2

Karta för en agent (eller människa) som ska portera de nya modellerna och
komponenterna till en annan app. Allt ligger på branchen
**`claude/audio-noise-robust-racket-bounce`** i detta repo (55 commits, main orörd).

## TL;DR — vad som finns

| Förmåga | Modell (JSON i appen) | Runtime (TS) | Verifiering |
|---|---|---|---|
| 1. Brusrobust racketstuds (ljud, realtid) | `apps/collector/src/models/fable_audio_model.json` (HistGB, 4 klasser) | `nrFeatures.ts` + `hgbRuntime.ts` + `fableEngine.ts` | Musik 0,58→0,96, prat 0,83→0,94 (orörd testdata); livetestad |
| 2. Studshöjd | — (ren fysik) | `fableEngine.ts` → `bounceHeightMeters()` | h = g·(Δt/2)²/2, gap 250–1500 ms |
| 3. Retrospektiv videoanalys | återanvänder 1 + 4 | `skills/.../noise_robust/analyze_video_retro.py` (PC-pipeline) | tidslinje + slagräkning med rimlighetslogik |
| 4. FH/BH-slag (video) | `apps/collector/src/models/video_stroke_model.json` (RF v2, 46 features) | `videoStrokeFeatures.ts` | 0,71→0,98 cross-session; bitexakt TS↔Python-paritet |
| 5. FH/BH-studs LIVE | `apps/collector/src/models/bounce_side_model.json` (ExtraTrees, 65 features) | `bounceSideInference.ts` | 100 % på beslutade studsar, 4 livepass i rad (alternationsfacit) |

## Filer att kopiera, per förmåga

### 1. Ljuddetektorn (kärnan — krävs för 2, 3, 5)

**TS (domänoberoende, inga RN-beroenden utom modellimport):**
- `apps/collector/src/nrFeatures.ts` — 83 features ur 300 ms PCM-klipp @ 22050 Hz (62 bas + 21 brusrobusta `nr_`). Portad scipy-exakt (Hann 512/128, sosfiltfilt, PCEN). Ändra INTE konstanter — bitexakt paritet mot träningen.
- `apps/collector/src/hgbRuntime.ts` — HistGradientBoosting flat-tree-inferens (iteration-major, softmax).
- `apps/collector/src/fableEngine.ts` — `FableCounter`: fönsterlogik (samma-studs/eko/grupp), inaktualitetsspärr, adaptiv konfidens, studshöjd. Läs filens kommentarer — varje konstant har en motivering från livetester.
- `apps/collector/src/models/fable_audio_model.json` (2,5 MB)

**Native (Android, Kotlin):**
- `AudioStreamModule.kt` — mikrofonström 22050 Hz mono + adaptiv onset-gate med **bandpass 1,5–7 kHz** (biquad-kaskad). Viktigt API: `setGateConfig('bandpass', false, absMin)` måste anropas **EFTER** `startStreaming` (start återställer till default).
- Eventet `onBounceDetected` levererar `audio_b64` (300 ms PCM) + `native_debug.onset_time_ms`/`rms` — `FableCounter.process()` tar dessa rakt av.

**Referensskärm:** `apps/collector/src/FableLiveScreen.tsx` (räknare + höjd + bakgrundsläge).

**Beslutskedjan (ordning spelar roll, se fableEngine-kommentarerna):**
stale-spärr → fönsterkontroller (FÖRE feature-extraktion, sparar ~165 ms/kandidat på Hermes) → features → HGB → racket_bounce + adaptiv konfidens (0,65 tyst / 0,9 vid bakgrund ≥ −42 dB).

### 5. Live FH/BH-studs (ljud + frontkamera)

- `apps/collector/src/bounceSideInference.ts` — grid-färgfeatures (8×8 HSV-statistik) + flat-tree-RF. 64×64 RGB-crop in.
- `apps/collector/src/models/bounce_side_model.json`
- `BounceSideLiveModule.kt` — CameraX + **16-frames ringbuffert**; `captureCrop(targetTimeMs)` väljer framen närmast träffögonblicket (kritiskt: bilden "nu" är 200–300 ms för sen), MediaPipe-pose → handledsankrad crop.
- `apps/collector/android/app/src/main/assets/pose_landmarker_lite.task` (MediaPipe-modellen, 5,8 MB) + beroenden i `app/build.gradle` (CameraX, MediaPipe tasks-vision).
- **Referensskärm:** `apps/collector/src/BounceSideLiveScreen.tsx` — notera räknarkonfigen `new FableCounter({ loudBgDb: -36, loudConfidence: 0.85 })`: tätt självstudsande höjer uppmätt bakgrund, default-tröskeln gav missar (se commit 37023a5).
- Forehandfärg-toggle: modellen förutsäger FÄRGSIDA (röd/svart tränat som forehand=röd); spelarens mappning görs i UI.
- Sidokonfidens < 0,6 → "Osäker" i stället för gissning.

### 4. Video FH/BH (efterhandsanalys)

- `apps/collector/src/videoStrokeFeatures.ts` — v2-features (46 st, tidsbinnade 4×300 ms över −700/+500 ms, axelbredds-normerade, spegelinvarianta) + `detectRacketHandedness()` (auto-detektering av racketarm ur handledsrörelse — profilens hand räcker INTE, se commit-historik).
- `apps/collector/src/models/video_stroke_model.json`
- `VideoPoseModule.kt` — sekventiell MediaCodec-avkodning (10–50× snabbare än getFrameAtTime), fysisk rotationsbakning, `extractPoseInWindows()` (pose bara runt ljudankare).
- **Referensskärm:** `VideoOnlyStrokeCollectionScreen.tsx` — ljudankrad analys: `findAudioPeaks()` → pose-fönster −800/+600 ms → slag-villkor P(FH)+P(BH) ≥ 0,40 → dedupe 600 ms.

### 2 + 3. Studshöjd och retrospektiv tidslinje

- Höjd: `bounceHeightMeters()` i `fableEngine.ts` (används i FableLiveScreen).
- Retrospektiv PC-pipeline: `skills/pingis-audio-classification/scripts/noise_robust/analyze_video_retro.py` (tidslinje + alternationsbaserad slagräkning + rimlighetslogik för avbrutna dueller).

## Träning/regenerering (Python, PC)

- Ljud: `skills/pingis-audio-classification/scripts/noise_robust/` — `NR_SPEC.md` och `RESULTS.md` är dokumentationen. Export: `export_fable_hgb_model_json.py` (**full float64 — 8 decimaler räcker INTE**, avrundning flippar trösklar).
- Studssida: `skills/pingis-stroke-detection/scripts/classify_bounce_side.py` (`--train`, `--export-app`). Tränar på granskade sessioner + app-crops + live-dumpar (`data/video/raw/live_sidedebug/*.json`, alternationsfacit, se `live_debug_rows()`).
- Slag: `skills/pingis-stroke-detection/scripts/train_video_stroke_v2.py` (`--export-app`).
- Paritetstester (kör efter varje export): `check_fable_ts_parity.js`, `check_bounce_side_ts_parity.js`, `check_stroke_v2_ts_parity.js` — kräver 0 argmax-avvikelser.

## Fallgropar (alla kostade oss en device-iteration)

1. `setGateConfig` efter `startStreaming`, inte före.
2. Modell-JSON med avrundade trösklar ⇒ fel klass nära beslutsgränser. Exportera full precision.
3. Kamerabilden måste tas vid `onset_time_ms` (ringbuffert), inte vid JS-callback.
4. Spelarens profilhand ≠ racketarm i videon — använd `detectRacketHandedness`.
5. ML Kit (telefon) och MediaPipe (PC) har olika z-semantik — z-features är EXKLUDERADE ur app-modellen; håll dem ute.
6. Bulk-godkända markörer (`bulk_confirmed`) är inte träningssanning.
7. Hermes saknar JIT — håll feature-extraktion gles (sparse mel-ranges); fönsterkontroller före features.

## Färdig prompt till stiga-app-v2

> Läs `FABLE_HANDOFF_2026-06-11.md` på branchen
> `claude/audio-noise-robust-racket-bounce` i repot
> `C:\Users\lovea\Desktop\dev\STIGA SPORTS\pingis_ml_algo` och portera
> [förmåga X] till den här appen. Kopiera modell-JSON + TS-runtimes
> oförändrade, följ referensskärmen för kopplingen, respektera
> fallgropslistan, och kör paritetstesterna efteråt.
