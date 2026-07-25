# HF-Gated Four-Class CNN Evaluation

This package implements the offline experiment tracked by GitHub issue `#4`.
It compares three bounce-counting paths without changing the STIGA runtime:

1. deployed Fable candidate gate, classifier, and timing;
2. adaptive high-frequency (HF) candidate gate followed by the existing Fable
   classifier and timing;
3. the same HF candidate gate followed by an experimental four-class CNN and
   the existing count timing/dedupe contract.

The HF gate only proposes event timestamps. Classifiers receive a centered
clip from the original full-band PCM, not high-pass-filtered audio.

## Data Safety Rules

- A reviewed event timestamp can label the nearest candidate as
  `racket_bounce`.
- Other candidates near a reviewed positive are `ambiguous`, not negative.
- Hard negatives come only from explicit negative sessions or reviewed-complete
  regions where the absence of an event is known.
- A session-level expected count is an evaluation target, not a source of event
  timestamps or candidate labels.
- Duplicate legacy copies are collapsed first by resolved audio path and then
  by `(source, session_id)`. The copy with richer reviewed labels is retained.
- Route and recording provenance are retained in the manifest. Unknown or
  non-built-in routes must not silently become trusted cross-device evidence.

The CNN classes are:

- `racket_bounce`
- `table_bounce`
- `floor_or_other_impact`
- `voice_music_or_noise`

Clips are 400 ms long with 100 ms before and 300 ms after the candidate onset.

## Reproduction

Run these commands from
`skills/pingis-audio-classification/scripts` and replace the data roots as
needed. Generated manifests, feature arrays, checkpoints, and reports belong
under ignored `data/audio/processed/` paths.

```powershell
python -m hf_pcen_cnn.build_candidate_manifest `
  --stiga-root "C:\path\to\stiga_audio_dataset" `
  --legacy-root "motorola=C:\path\to\motorola\audio\raw" `
  --legacy-root "colleague=C:\path\to\colleague\raw" `
  --cutoffs-hz "6000,7000,8000" `
  --onset-ratios "3,4,6" `
  --output-dir "..\..\..\data\audio\processed\hf_pcen_cnn\gate_sweep"
```

After selecting a gate setting, rebuild its manifest alone, then cache the two
front ends and train the ablation:

```powershell
python -m hf_pcen_cnn.build_candidate_manifest `
  --stiga-root "C:\path\to\stiga_audio_dataset" `
  --legacy-root "motorola=C:\path\to\motorola\audio\raw" `
  --legacy-root "colleague=C:\path\to\colleague\raw" `
  --cutoffs-hz "7000" `
  --onset-ratios "4" `
  --output-dir "..\..\..\data\audio\processed\hf_pcen_cnn\selected"

python -m hf_pcen_cnn.build_feature_cache `
  --manifest "..\..\..\data\audio\processed\hf_pcen_cnn\selected\hf_candidate_manifest.csv" `
  --output-dir "..\..\..\data\audio\processed\hf_pcen_cnn\cache"

python -m hf_pcen_cnn.train_ablation `
  --cache-dir "..\..\..\data\audio\processed\hf_pcen_cnn\cache" `
  --output-dir "..\..\..\data\audio\processed\hf_pcen_cnn\ablation"

python -m hf_pcen_cnn.evaluate_counts `
  --ablation-dir "..\..\..\data\audio\processed\hf_pcen_cnn\ablation" `
  --output-dir "..\..\..\data\audio\processed\hf_pcen_cnn\count_eval"

python -m hf_pcen_cnn.evaluate_fable_baselines `
  --stiga-root "C:\path\to\stiga_audio_dataset" `
  --legacy-root "motorola=C:\path\to\motorola\audio\raw" `
  --legacy-root "colleague=C:\path\to\colleague\raw" `
  --split-csv "..\..\..\data\audio\processed\hf_pcen_cnn\ablation\split.csv" `
  --hf-manifest "..\..\..\data\audio\processed\hf_pcen_cnn\selected\hf_candidate_manifest.csv" `
  --model-json "..\..\..\apps\collector\src\models\fable_audio_model.json" `
  --output-dir "..\..\..\data\audio\processed\hf_pcen_cnn\fable_comparison"
```

The final command compares the deployed Fable candidate path against the
selected HF candidate path on the same evaluation sessions.

## Issue #4 Results

The deduplicated corpus contains 180 sessions and 9,520.68 seconds of audio.
Nine copied legacy session identities were removed before evaluation.

### HF Gate Sweep

| Cutoff | Onset ratio | Reviewed-event recall | Candidates/min |
| --- | ---: | ---: | ---: |
| 6 kHz | 3 | 94.75% | 163.67 |
| 6 kHz | 4 | 94.66% | 158.89 |
| 6 kHz | 6 | 93.56% | 147.29 |
| 7 kHz | 3 | 95.01% | 164.33 |
| **7 kHz** | **4** | **94.95%** | **159.39** |
| 7 kHz | 6 | 94.33% | 148.77 |
| 8 kHz | 3 | 94.64% | 165.25 |
| 8 kHz | 4 | 94.60% | 160.85 |
| 8 kHz | 6 | 94.12% | 150.47 |

The selected 7 kHz / ratio 4 setting gives almost the maximum observed recall
while producing fewer candidates than ratio 3. Its selected manifest has
25,292 candidates: 7,401 positives, 1,740 ambiguous candidates, 6,910 trusted
hard negatives, and 9,241 unlabeled candidates.

### CNN Front-End Ablation

| Front end | Validation macro-F1 | Validation racket F1 | Motorola racket F1 |
| --- | ---: | ---: | ---: |
| Log-mel | **0.797** | **0.796** | **0.897** |
| PCEN | 0.757 | 0.717 | 0.585 |

The evaluated PCEN configuration did not improve cross-device robustness.
Log-mel is the stronger experimental CNN front end, but neither CNN is ready
to replace Fable.

### End-to-End Comparison

| Path | Gate recall | Review-complete MAE/session | iPhone count-only MAE/session | Explicit-negative FP |
| --- | ---: | ---: | ---: | ---: |
| Default Fable | 88.07% | 4.14 | **11.60** | 2 |
| HF-gated Fable | **94.94%** | **2.81** | 12.20 | **1** |
| HF + log-mel CNN | 94.95% | 14.09 | 14.60 | 2 |
| HF + PCEN CNN | 94.95% | 12.68 | 9.40 | 1 |

The apparently better aggregate iPhone MAE for PCEN is cancellation between
large per-scenario overcounts and undercounts. It is not evidence of a more
reliable model. For example, it counted loud-background bounces at 17/40 while
overcounting several easier positive scenarios.

### Manually Reviewed Phone Audit (2026-07-17)

A synchronized three-device pack was recorded through the STIGA native audio
path and manually reviewed. It contains nine sessions: normal bounce,
bounce-with-loud-background, and music/TV-only on Huawei, iPhone, and Motorola.
Each positive session has 20 reviewed bounce timestamps, for 120 positives in
total.

The 7 kHz / onset-ratio 3 HF gate matched all 120 reviewed events. Replaying the
deployed Fable classifier and timing logic at confidence 0.85 produced 114 true
counts, 8 unmatched counts, and 6 misses. All six misses came from
bounce-with-loud-background sessions:

- three reviewed bounces had `noise` as the Fable top label;
- three had racket probabilities of approximately 0.55, 0.60, and 0.76;
- the eight false counts had overlapping racket probabilities from 0.67 to
  0.99.

A threshold sweep from 0.05 through 0.95 selected the deployed 0.85 value as
the best F1 operating point on this pack. Lowering the threshold recovers the
three low-confidence bounces but introduces seven additional false counts; it
cannot recover the three top-label `noise` cases. This is a classifier
separation problem, not evidence for another HF-gate or threshold adjustment.

Reproduce the candidate-level audit with:

```powershell
$env:PYTHONPATH='skills/pingis-audio-classification/scripts'
python -m hf_pcen_cnn.audit_review_pack `
  --review-pack data-new/review/20260717_hf_fable_prefill
```

The ignored `analysis/fable_error_audit` directory contains candidate,
session, final-miss, and threshold-sweep CSV evidence. Any next learned model
must train on reviewed HF-gate candidates, include the gate's own false
candidates as hard negatives, and beat this frozen Fable baseline under
leave-device-out evaluation.

### Reviewed-Candidate Device Holdout (2026-07-17)

The next experiment trained the four-class log-mel CNN directly on the
manually reviewed HF-gate candidates and evaluated its racket-vs-not-racket
decision. Reviewed bounces were positives, the gate's own false candidates
were hard negatives, and ambiguous candidates were excluded from training.
Each fold held out one complete reviewed phone while training with the other
two phones plus the deduplicated colleague legacy corpus.

Reproduce the manifest, feature cache, and three device folds with:

```powershell
$env:PYTHONPATH='skills/pingis-audio-classification/scripts'

python -m hf_pcen_cnn.build_review_pack_manifest `
  --review-pack data-new/review/20260717_hf_fable_prefill `
  --model-json apps/collector/src/models/fable_audio_model.json `
  --output data-new/review/20260717_hf_fable_prefill/analysis/review_candidate_manifest.csv

python -m hf_pcen_cnn.build_feature_cache `
  --manifest data-new/review/20260717_hf_fable_prefill/analysis/review_candidate_manifest.csv `
  --output-dir data-new/review/20260717_hf_fable_prefill/analysis/review_candidate_cache

python -m hf_pcen_cnn.train_reviewed_device_lodo `
  --base-cache data/audio/processed/hf_pcen_cnn/issue_4_cache_dedup `
  --review-cache data-new/review/20260717_hf_fable_prefill/analysis/review_candidate_cache `
  --output-dir data-new/review/20260717_hf_fable_prefill/analysis/review_device_lodo `
  --epochs 20 `
  --batch-size 128 `
  --seed 42
```

The candidate manifest contains 349 rows: 120 positives, 221 hard negatives,
and 8 ambiguous candidates. The same Fable timing and dedupe logic was applied
to both classifier outputs.

| Path | TP | FP | FN | Precision | Recall | F1 | MAE/session |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Reviewed-candidate log-mel CNN | 85 | 16 | 35 | 84.16% | 70.83% | 76.92% | 5.00 |
| Frozen Fable | **114** | **8** | **6** | **93.44%** | **95.00%** | **94.21%** | **1.33** |

The CNN lost to Fable on every held-out phone. A second diagnostic selected
the best possible confidence threshold using each held phone's own labels.
That oracle result is an intentionally leaky upper bound, not a deployable
configuration, and it still lost on all three phones:

| Held-out phone | CNN F1 | Oracle CNN F1 | Frozen Fable F1 |
| --- | ---: | ---: | ---: |
| Huawei VOG-L29 | 77.92% | 80.00% | **87.50%** |
| Motorola moto g55 5G | 70.77% | 90.48% | **96.30%** |
| Apple iPhone17,1 | 81.01% | 84.34% | **98.77%** |

This is a strict physical-device holdout for the three reviewed phones. It is
not a universal leave-device-out claim: the older colleague corpus does not
contain sufficiently reliable physical-device identity. The report is saved
under the ignored
`analysis/review_device_lodo/device_lodo_report.json` path.

The candidate-level retrain therefore fails the promotion gate decisively.
The HF gate's own false candidates are useful hard negatives, but this amount
and diversity of reviewed data is not enough for a learned classifier to
generalize better than Fable. Another blind PCEN or threshold rerun is not
justified by these results.

### Expanded Reviewed Round And Frozen Holdout (2026-07-26)

The synchronized `CJ-20260723-01` round expands the reviewed evidence to 43
physical takes recorded on iPhone, Motorola, and Huawei: 129 sessions and
2,865 reviewed bounce timestamps. Model selection used Train sessions only.
The final 33-session holdout remained sealed until the candidate architecture
and thresholds were frozen.

| Path | Counting F1 | Precision | Recall | MAE/session | Candidate macro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `dual_residual` | **0.7226** | **0.8131** | **0.6502** | **5.94** | **0.6065** |
| `pcen_compact` | 0.6629 | 0.6777 | 0.6486 | 10.70 | 0.4846 |
| Frozen Fable | 0.5636 | 0.6988 | 0.4722 | 12.48 | - |

The expanded round changes the conclusion from the smaller July 17 pack:
learned candidates are now useful enough for QA integration. It does not
justify changing the production default before app-side frontend parity,
latency, and physical-device false-positive testing.

Export the frozen QA finalists with:

```powershell
$env:PYTHONPATH='skills/pingis-audio-classification/scripts'
python -m hf_pcen_cnn.export_finalist_onnx `
  --cache-dir data/audio/processed/hf_pcen_cnn/reviewed_round_20260723/cache `
  --final-selection-dir data/audio/processed/hf_pcen_cnn/reviewed_round_20260723/final_selection `
  --output-dir data/audio/processed/hf_pcen_cnn/reviewed_round_20260723/final_selection/deployment_qa `
  --parity-rows 6
```

The ignored output package contains feature-input ONNX graphs, a deployment
contract for the exact gate/frontend/clip/timing configuration, and PCM parity
fixtures. ONNX Runtime parity maximum probability error is below `2.4e-7` for
both finalists.

## Decision

- Keep default Fable as the balanced deployed reference.
- Keep HF-gated Fable as a promising recall-oriented experiment. It improves
  reviewed gate recall by 6.87 percentage points and review-complete count MAE,
  but it changes the candidate distribution and overcounts the new count-only
  iPhone scenarios. It needs classifier calibration and timing/dedupe tuning
  before runtime promotion.
- Do not promote the four-class CNN or the current PCEN front end.
- Do not promote the reviewed-candidate CNN. Frozen Fable remains the
  reference after outperforming it on every held-out reviewed phone, even
  against the CNN's non-deployable per-phone oracle threshold.
- Use `dual_residual` as the primary QA candidate and `pcen_compact` as its
  challenger from the expanded reviewed round. These exports are for runtime
  parity and Android/iOS device comparison only.
- Collect more reviewed, device-diverse floor/other-impact examples. The
  training split contained only 73 floor/other-impact candidates.

The external Motorola set contains only racket and voice/noise labels, while
the new iPhone positive sessions provide expected counts but no event
timestamps. Consequently this experiment is not a complete four-class
leave-device-out validation and must not be described as one.

## Validation

```powershell
$env:PYTHONPATH='skills/pingis-audio-classification/scripts'
python -m unittest discover `
  -s skills/pingis-audio-classification/scripts/hf_pcen_cnn/tests `
  -p "test_*.py" `
  -v
python -m compileall skills/pingis-audio-classification/scripts/hf_pcen_cnn
```
