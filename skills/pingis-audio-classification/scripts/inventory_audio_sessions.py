#!/usr/bin/env python3
"""Build a complete machine-readable inventory of all audio sessions under data/audio/raw.

Scans:
  - top-level session folders ``audio_session_YYYY-MM-DD_NNN`` (group ``main``)
  - ``device_pull/`` (group ``device_pull``)
  - ``archive_m4a/`` (group ``archive_m4a``)

For each session it discovers the session JSON (sibling ``<session_id>.json``
next to the media folder, or any ``*.json`` inside the folder), parses
``session_meta`` and ``events[]``, checks media existence on disk, counts
reviewed markers per class (marker schema evolved over time; every distinct
marker field-set variant is captured), and flags known diagnostic-only
sessions.

Outputs (deterministic):
  - data/audio/processed/audio_inventory_2026_06_10.json
  - data/audio/processed/audio_inventory_2026_06_10.md

Stdlib only. UTF-8 everywhere.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
RAW_ROOT = REPO_ROOT / "data" / "audio" / "raw"
OUT_DIR = REPO_ROOT / "data" / "audio" / "processed"
OUT_JSON = OUT_DIR / "audio_inventory_2026_06_10.json"
OUT_MD = OUT_DIR / "audio_inventory_2026_06_10.md"

INVENTORY_DATE = "2026-06-10"

MEDIA_EXTENSIONS = {".wav", ".mp4", ".m4a"}

# Sessions documented as diagnostic-only. Flagged, never deleted.
# Sources: PROJECT_CONTEXT.md, ITERATION_LOG.md, REPO_CURRENT_STATE.md.
DIAGNOSTIC_SESSIONS = {
    "audio_session_2026-05-11_001": (
        "diagnostic-only: same WAV compared across 9 review configs; "
        "explicitly excluded from training (ITERATION_LOG 2026-05-11)"
    ),
    "audio_session_2026-05-26_002": (
        "diagnostic-only: saved without correction while rejected 05-26 audio "
        "candidate produced hundreds of false racket confirmations; excluded "
        "from audio and video-stroke preprocessing"
    ),
    "audio_session_2026-05-26_003": (
        "diagnostic-only: throwaway same-media comparison test for corrected "
        "candidate; excluded from audio preprocessing"
    ),
    "audio_session_2026-05-26_004": (
        "diagnostic-only: throwaway rollback-baseline comparison test (same "
        "WAV/MP4 hash as 003); excluded from audio/video preprocessing"
    ),
    "audio_session_2026-05-29_001": (
        "diagnostic: pulled only to diagnose failed motion review (0 "
        "video_pose_candidates / 0 motion markers from old hard motion gate); "
        "not video-stroke training truth unless re-reviewed with fixed app"
    ),
    "audio_session_2026-06-04_008": (
        "diagnostic-only: T0032 same-clip audit of T0031 APK versus 139-marker "
        "truth; not training data (REPO_CURRENT_STATE T0032)"
    ),
}

# Fallback chain to resolve a single reviewed class per marker, preferring the
# most specific fields the evolving schema provides.
MARKER_CLASS_FIELDS = (
    "contact_kind",
    "not_racket_kind",
    "class_label",
    "final_label",
    "suggested_label",
    "label",
)


def rel_posix(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def sorted_counter(counter: Counter) -> dict:
    return {key: counter[key] for key in sorted(counter, key=str)}


def marker_class(marker: dict) -> str:
    for field in MARKER_CLASS_FIELDS:
        value = marker.get(field)
        if value:
            return str(value)
    return "(unresolved)"


def load_json(path: Path):
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle), None
    except (OSError, ValueError) as exc:
        return None, f"{type(exc).__name__}: {exc}"


def discover_session_json(group_root: Path, session_dir: Path):
    """Return (primary_json_path_or_None, extra_json_paths, discovery_kind)."""
    candidates = []
    sibling = group_root / f"{session_dir.name}.json"
    if sibling.is_file():
        candidates.append(("sibling", sibling))
    for inner in sorted(session_dir.glob("*.json")):
        candidates.append(("inner", inner))
    if not candidates:
        return None, [], "missing"
    kind, primary = candidates[0]
    extras = [path for _, path in candidates[1:]]
    return primary, extras, kind


class VariantRegistry:
    """Assigns stable ids to distinct marker field-set variants."""

    def __init__(self):
        self.counts = Counter()
        self.sessions = {}

    def record(self, marker: dict, session_id: str) -> tuple:
        fields = tuple(sorted(marker.keys()))
        self.counts[fields] += 1
        self.sessions.setdefault(fields, set()).add(session_id)
        return fields

    def finalize(self):
        ordered = sorted(self.counts)
        ids = {fields: f"variant_{idx + 1:02d}" for idx, fields in enumerate(ordered)}
        rows = [
            {
                "variant_id": ids[fields],
                "fields": list(fields),
                "n_markers": self.counts[fields],
                "n_sessions": len(self.sessions[fields]),
                "sessions": sorted(self.sessions[fields]),
            }
            for fields in ordered
        ]
        return ids, rows


def inventory_event(event: dict, media_dir: Path, registry: VariantRegistry, session_id: str):
    wav_filename = event.get("wav_filename")
    wav_path = media_dir / wav_filename if wav_filename else None
    wav_exists = bool(wav_path and wav_path.is_file())
    media_fallback = None
    if wav_filename and not wav_exists:
        for ext in (".m4a", ".mp4"):
            alt = media_dir / (Path(wav_filename).stem + ext)
            if alt.is_file():
                media_fallback = alt.name
                break

    video_recording = event.get("video_recording") or {}
    video_filename = video_recording.get("video_filename") if isinstance(video_recording, dict) else None
    video_exists = bool(video_filename and (media_dir / video_filename).is_file())

    review = event.get("review") or {}
    markers = review.get("markers") or [] if isinstance(review, dict) else []

    class_counts = Counter()
    review_status_counts = Counter()
    contact_kind_counts = Counter()
    variant_fields = set()
    for marker in markers:
        if not isinstance(marker, dict):
            continue
        variant_fields.add(registry.record(marker, session_id))
        class_counts[marker_class(marker)] += 1
        review_status_counts[str(marker.get("review_status") or "(absent)")] += 1
        kind = marker.get("contact_kind") or marker.get("not_racket_kind")
        if kind:
            contact_kind_counts[str(kind)] += 1

    return {
        "label": event.get("label"),
        "scenario": event.get("scenario"),
        "scenario_id": event.get("scenario_id"),
        "background_condition": event.get("background_condition"),
        "wav_filename": wav_filename,
        "wav_exists": wav_exists,
        "media_fallback": media_fallback,
        "video_filename": video_filename,
        "video_exists": video_exists if video_filename else None,
        "duration_ms": event.get("duration_ms"),
        "n_markers": len(markers),
        "marker_class_counts": sorted_counter(class_counts),
        "marker_review_status_counts": sorted_counter(review_status_counts),
        "marker_contact_kind_counts": sorted_counter(contact_kind_counts),
        "n_model_candidates": len(event.get("model_candidates") or []),
        "n_video_pose_candidates": len(event.get("video_pose_candidates") or []),
    }, class_counts, review_status_counts, contact_kind_counts, variant_fields


def inventory_session(group: str, group_root: Path, session_dir: Path | None,
                      json_path: Path | None, registry: VariantRegistry):
    """Inventory one session. session_dir may be None (orphan JSON without folder)."""
    session_id = session_dir.name if session_dir is not None else (json_path.stem if json_path else "(unknown)")
    media_dir = session_dir if session_dir is not None else (group_root / session_id)

    extra_jsons = []
    discovery = "explicit"
    if session_dir is not None and json_path is None:
        json_path, extra_jsons, discovery = discover_session_json(group_root, session_dir)

    media_files = []
    if session_dir is not None and session_dir.is_dir():
        media_files = sorted(p.name for p in session_dir.iterdir()
                             if p.is_file() and p.suffix.lower() in MEDIA_EXTENSIONS)

    problems = []
    data = None
    if json_path is None:
        problems.append("missing_session_json")
    else:
        data, error = load_json(json_path)
        if error:
            problems.append(f"unparseable_json: {error}")

    duplicate_json_identical = None
    if json_path is not None and extra_jsons:
        primary_bytes = json_path.read_bytes()
        duplicate_json_identical = all(extra.read_bytes() == primary_bytes for extra in extra_jsons)

    session_meta = (data or {}).get("session_meta") or {}
    events = (data or {}).get("events") or []

    event_rows = []
    class_total = Counter()
    review_status_total = Counter()
    contact_kind_total = Counter()
    variant_fields_used = set()
    referenced = set()
    total_duration = 0
    n_markers_total = 0
    n_candidates_total = 0
    n_missing_wav = 0
    n_zero_marker_events = 0

    for event in events:
        row, classes, statuses, kinds, variants = inventory_event(event, media_dir, registry, session_id)
        event_rows.append(row)
        class_total.update(classes)
        review_status_total.update(statuses)
        contact_kind_total.update(kinds)
        variant_fields_used.update(variants)
        if row["wav_filename"]:
            referenced.add(row["wav_filename"])
        if row["video_filename"]:
            referenced.add(row["video_filename"])
        total_duration += row["duration_ms"] or 0
        n_markers_total += row["n_markers"]
        n_candidates_total += row["n_model_candidates"]
        if row["wav_filename"] and not row["wav_exists"]:
            n_missing_wav += 1
        if row["n_markers"] == 0:
            n_zero_marker_events += 1

    unreferenced_media = sorted(set(media_files) - referenced)
    if session_dir is not None and not session_dir.is_dir():
        problems.append("missing_media_folder")
    if n_missing_wav:
        problems.append(f"missing_wav_files:{n_missing_wav}")

    diagnostic_note = DIAGNOSTIC_SESSIONS.get(session_id)

    return {
        "session_id": session_id,
        "group": group,
        "json_path": rel_posix(json_path) if json_path else None,
        "json_discovery": discovery,
        "extra_json_paths": [rel_posix(p) for p in extra_jsons],
        "duplicate_json_identical": duplicate_json_identical,
        "media_dir": rel_posix(session_dir) if session_dir is not None else None,
        "session_meta": {
            "recorder_name": session_meta.get("recorder_name"),
            "player_name": session_meta.get("player_name"),
            "session_date": session_meta.get("session_date"),
            "app_version": session_meta.get("app_version"),
            "collection_mode": session_meta.get("collection_mode"),
            "recording_mode": session_meta.get("recording_mode"),
            "collection_type": session_meta.get("collection_type"),
            "scenarios": session_meta.get("scenarios"),
        },
        "is_diagnostic": diagnostic_note is not None,
        "diagnostic_note": diagnostic_note,
        "problems": problems,
        "n_events": len(events),
        "events": event_rows,
        "totals": {
            "total_duration_ms": total_duration,
            "n_markers": n_markers_total,
            "n_model_candidates": n_candidates_total,
            "n_events_missing_wav": n_missing_wav,
            "n_zero_marker_events": n_zero_marker_events,
            "marker_class_counts": sorted_counter(class_total),
            "marker_review_status_counts": sorted_counter(review_status_total),
            "marker_contact_kind_counts": sorted_counter(contact_kind_total),
        },
        "media_files": media_files,
        "unreferenced_media": unreferenced_media,
        "_variant_fields": variant_fields_used,  # stripped before writing
    }


def scan_group(group: str, group_root: Path, registry: VariantRegistry):
    sessions = []
    if not group_root.is_dir():
        return sessions
    session_dirs = sorted(p for p in group_root.iterdir()
                          if p.is_dir() and p.name.startswith("audio_session_"))
    claimed_jsons = set()
    for session_dir in session_dirs:
        record = inventory_session(group, group_root, session_dir, None, registry)
        if record["json_path"]:
            claimed_jsons.add(REPO_ROOT / record["json_path"])
        for extra in record["extra_json_paths"]:
            claimed_jsons.add(REPO_ROOT / extra)
        sessions.append(record)
    # Orphan JSONs (no matching media folder).
    for orphan in sorted(group_root.glob("*.json")):
        if orphan in claimed_jsons:
            continue
        record = inventory_session(group, group_root, None, orphan, registry)
        record["problems"].append("orphan_json_no_media_folder")
        sessions.append(record)
    sessions.sort(key=lambda rec: rec["session_id"])
    return sessions


def build_scenario_summary(sessions):
    rows = {}
    for record in sessions:
        for event in record["events"]:
            key = event["scenario_id"] or "(none)"
            row = rows.setdefault(key, {
                "sessions": set(), "events": 0, "duration_ms": 0,
                "markers": Counter(), "n_markers": 0,
            })
            row["sessions"].add(record["session_id"])
            row["events"] += 1
            row["duration_ms"] += event["duration_ms"] or 0
            row["markers"].update(event["marker_class_counts"])
            row["n_markers"] += event["n_markers"]
    return rows


def format_counter_inline(counts: dict, limit: int | None = None) -> str:
    items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    if limit is not None and len(items) > limit:
        shown = items[:limit]
        rest = sum(count for _, count in items[limit:])
        return ", ".join(f"{name}={count}" for name, count in shown) + f", other={rest}"
    return ", ".join(f"{name}={count}" for name, count in items) or "-"


def write_markdown(payload, variant_rows):
    sessions = payload["sessions"]
    agg = payload["aggregates"]
    lines = []
    add = lines.append
    add(f"# Audio session inventory ({INVENTORY_DATE})")
    add("")
    add(f"Source: `data/audio/raw/` — generated by `skills/pingis-audio-classification/scripts/inventory_audio_sessions.py`.")
    add("")
    add("## Totals")
    add("")
    add(f"- Sessions: **{agg['n_sessions']}** "
        f"(main: {agg['n_sessions_by_group'].get('main', 0)}, "
        f"device_pull: {agg['n_sessions_by_group'].get('device_pull', 0)}, "
        f"archive_m4a: {agg['n_sessions_by_group'].get('archive_m4a', 0)})")
    add(f"- Events: **{agg['n_events']}**, total recorded duration: "
        f"{agg['total_duration_ms'] / 1000.0:.1f} s")
    add(f"- Reviewed markers: **{agg['n_markers']}**, model candidates: {agg['n_model_candidates']}")
    add(f"- Diagnostic-flagged sessions: {agg['n_diagnostic_sessions']}")
    add(f"- Marker schema variants observed: {len(variant_rows)}")
    add("")

    add("## Per scenario_id")
    add("")
    add("| scenario_id | sessions | events | duration_s | markers | marker classes |")
    add("|---|---:|---:|---:|---:|---|")
    scenario_rows = build_scenario_summary(sessions)
    for key in sorted(scenario_rows):
        row = scenario_rows[key]
        add(f"| {key} | {len(row['sessions'])} | {row['events']} | "
            f"{row['duration_ms'] / 1000.0:.1f} | {row['n_markers']} | "
            f"{format_counter_inline(dict(row['markers']), limit=6)} |")
    add("")

    add("## background_condition distribution (events)")
    add("")
    add("| background_condition | events |")
    add("|---|---:|")
    for key in sorted(agg["background_condition_counts"]):
        add(f"| {key} | {agg['background_condition_counts'][key]} |")
    add("")

    add("## Global marker class counts (reviewed truth)")
    add("")
    add("| marker class | count |")
    add("|---|---:|")
    for key in sorted(agg["marker_class_counts"]):
        add(f"| {key} | {agg['marker_class_counts'][key]} |")
    add("")
    add("Review status: " + format_counter_inline(agg["marker_review_status_counts"]))
    add("")

    add("## Sessions per group")
    add("")
    add("| session_id | group | events | duration_s | markers | candidates | flags |")
    add("|---|---|---:|---:|---:|---:|---|")
    for record in sorted(sessions, key=lambda r: (r["group"], r["session_id"])):
        flags = []
        if record["is_diagnostic"]:
            flags.append("DIAGNOSTIC")
        flags.extend(record["problems"])
        if record["totals"]["n_markers"] == 0:
            flags.append("zero_markers")
        add(f"| {record['session_id']} | {record['group']} | {record['n_events']} | "
            f"{record['totals']['total_duration_ms'] / 1000.0:.1f} | "
            f"{record['totals']['n_markers']} | {record['totals']['n_model_candidates']} | "
            f"{'; '.join(flags) or '-'} |")
    add("")

    add("## Problem sessions")
    add("")
    problem_records = [r for r in sessions
                       if r["problems"] or r["is_diagnostic"] or r["totals"]["n_markers"] == 0]
    if not problem_records:
        add("None.")
    for record in problem_records:
        notes = []
        if record["is_diagnostic"]:
            notes.append(f"diagnostic: {record['diagnostic_note']}")
        notes.extend(record["problems"])
        if record["totals"]["n_markers"] == 0:
            notes.append("zero reviewed markers")
        add(f"- `{record['session_id']}` ({record['group']}): " + "; ".join(notes))
    add("")

    add("## Marker schema variants")
    add("")
    add("Field sets observed across all `review.markers[]` entries (schema evolved over time).")
    add("")
    add("| variant | markers | sessions | fields |")
    add("|---|---:|---:|---|")
    for row in variant_rows:
        add(f"| {row['variant_id']} | {row['n_markers']} | {row['n_sessions']} | "
            f"{', '.join(row['fields'])} |")
    add("")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main():
    registry = VariantRegistry()
    groups = {
        "main": RAW_ROOT,
        "device_pull": RAW_ROOT / "device_pull",
        "archive_m4a": RAW_ROOT / "archive_m4a",
    }
    all_sessions = []
    for group, root in groups.items():
        all_sessions.extend(scan_group(group, root, registry))

    # Anything at top level that is neither a session folder/JSON nor a known group.
    known = {"device_pull", "archive_m4a"}
    other_entries = sorted(
        p.name for p in RAW_ROOT.iterdir()
        if not p.name.startswith("audio_session_") and p.name not in known
    )

    variant_ids, variant_rows = registry.finalize()
    for record in all_sessions:
        fields_used = record.pop("_variant_fields")
        record["marker_schema_variants"] = sorted(variant_ids[f] for f in fields_used)

    scenario_counts = Counter()
    background_counts = Counter()
    class_counts = Counter()
    status_counts = Counter()
    contact_kind_counts = Counter()
    n_events = 0
    total_duration = 0
    n_markers = 0
    n_candidates = 0
    group_counts = Counter()
    for record in all_sessions:
        group_counts[record["group"]] += 1
        n_events += record["n_events"]
        total_duration += record["totals"]["total_duration_ms"]
        n_markers += record["totals"]["n_markers"]
        n_candidates += record["totals"]["n_model_candidates"]
        class_counts.update(record["totals"]["marker_class_counts"])
        status_counts.update(record["totals"]["marker_review_status_counts"])
        contact_kind_counts.update(record["totals"]["marker_contact_kind_counts"])
        for event in record["events"]:
            scenario_counts[event["scenario_id"] or "(none)"] += 1
            background_counts[event["background_condition"] or "(none)"] += 1

    payload = {
        "inventory_date": INVENTORY_DATE,
        "raw_root": rel_posix(RAW_ROOT),
        "groups": {g: sorted(r["session_id"] for r in all_sessions if r["group"] == g)
                   for g in ("main", "device_pull", "archive_m4a")},
        "other_top_level_entries": other_entries,
        "aggregates": {
            "n_sessions": len(all_sessions),
            "n_sessions_by_group": sorted_counter(group_counts),
            "n_events": n_events,
            "total_duration_ms": total_duration,
            "n_markers": n_markers,
            "n_model_candidates": n_candidates,
            "n_diagnostic_sessions": sum(1 for r in all_sessions if r["is_diagnostic"]),
            "scenario_id_event_counts": sorted_counter(scenario_counts),
            "background_condition_counts": sorted_counter(background_counts),
            "marker_class_counts": sorted_counter(class_counts),
            "marker_review_status_counts": sorted_counter(status_counts),
            "marker_contact_kind_counts": sorted_counter(contact_kind_counts),
        },
        "marker_schema_variants": variant_rows,
        "sessions": all_sessions,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_JSON.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    write_markdown(payload, variant_rows)

    print(f"sessions={len(all_sessions)} events={n_events} markers={n_markers}")
    print(f"wrote {rel_posix(OUT_JSON)}")
    print(f"wrote {rel_posix(OUT_MD)}")


if __name__ == "__main__":
    main()
