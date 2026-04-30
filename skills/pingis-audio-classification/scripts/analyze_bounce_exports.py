"""
analyze_bounce_exports.py

Summarize exported bounce sessions from the collector app.

Default input directory:
  C:/Users/lovea/Downloads/pingis_sessions

Example:
  python skills/pingis-audio-classification/scripts/analyze_bounce_exports.py
  python skills/pingis-audio-classification/scripts/analyze_bounce_exports.py --dir C:/path/to/exports
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_EXPORT_DIR = Path.home() / "Downloads" / "pingis_sessions"


def iter_session_files(export_dir: Path):
    for pattern in ("bounce_free_*.json", "bounce_alternating_*.json"):
        yield from sorted(export_dir.glob(pattern))


def summarize_file(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        session = json.load(handle)

    audio_events = session.get("audio_events", [])
    contacts = session.get("bounce_contacts", [])
    preset_id = session.get("preset_id", "unknown")
    mode = session.get("session_meta", {}).get("mode", "unknown")

    audio_labels = Counter(event.get("label", "unknown") for event in audio_events)
    contact_ignored = Counter(
        event.get("ignored_reason", "counted")
        for event in contacts
        if not event.get("counted", False)
    )

    return {
        "file": path.name,
        "mode": mode,
        "preset_id": preset_id,
        "audio_events": len(audio_events),
        "racket_bounce_audio": audio_labels.get("racket_bounce", 0),
        "ignored_low_confidence": contact_ignored.get("low_confidence", 0),
        "ignored_no_bounce_motion": contact_ignored.get("no_bounce_motion", 0),
        "counted_contacts": sum(1 for event in contacts if event.get("counted", False)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", dest="export_dir", default=str(DEFAULT_EXPORT_DIR))
    args = parser.parse_args()

    export_dir = Path(args.export_dir)
    if not export_dir.exists():
        raise SystemExit(f"Export directory not found: {export_dir}")

    summaries = [summarize_file(path) for path in iter_session_files(export_dir)]
    if not summaries:
        raise SystemExit(f"No bounce exports found in: {export_dir}")

    print(f"Export directory: {export_dir}")
    print()
    print("Per file:")
    for summary in summaries:
        print(
            f"- {summary['file']}: mode={summary['mode']} preset={summary['preset_id']}"
            f" audio_events={summary['audio_events']}"
            f" racket_bounce={summary['racket_bounce_audio']}"
            f" low_confidence={summary['ignored_low_confidence']}"
            f" no_bounce_motion={summary['ignored_no_bounce_motion']}"
            f" counted={summary['counted_contacts']}"
        )

    by_preset: dict[str, Counter] = defaultdict(Counter)
    for summary in summaries:
        bucket = by_preset[summary["preset_id"]]
        bucket["files"] += 1
        bucket["audio_events"] += summary["audio_events"]
        bucket["racket_bounce_audio"] += summary["racket_bounce_audio"]
        bucket["ignored_low_confidence"] += summary["ignored_low_confidence"]
        bucket["ignored_no_bounce_motion"] += summary["ignored_no_bounce_motion"]
        bucket["counted_contacts"] += summary["counted_contacts"]

    print()
    print("Per preset:")
    for preset_id in sorted(by_preset):
        bucket = by_preset[preset_id]
        print(
            f"- {preset_id}: files={bucket['files']}"
            f" audio_events={bucket['audio_events']}"
            f" racket_bounce={bucket['racket_bounce_audio']}"
            f" low_confidence={bucket['ignored_low_confidence']}"
            f" no_bounce_motion={bucket['ignored_no_bounce_motion']}"
            f" counted={bucket['counted_contacts']}"
        )


if __name__ == "__main__":
    main()
