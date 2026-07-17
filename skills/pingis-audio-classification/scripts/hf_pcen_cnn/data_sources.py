"""Adapters for reviewed legacy sessions and STIGA native-recorder exports."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .candidate_labels import ReviewedEvent, canonical_four_class


NEGATIVE_SCENARIO_TOKENS = (
    "noise",
    "talking",
    "speech",
    "music",
    "handling",
    "drop",
    "clap",
    "cough",
    "negative",
)


@dataclass(frozen=True)
class AudioSession:
    session_id: str
    audio_path: Path
    source: str
    device_id: str
    scenario_id: str
    reviewed_events: tuple[ReviewedEvent, ...] = ()
    review_complete: bool = False
    explicit_negative: bool = False
    expected_bounce_count: int | None = None
    input_route: str = ""
    route_class: str = "unknown"
    metadata: dict = field(default_factory=dict)


def classify_input_route(route: str, audio_source: str = "") -> str:
    value = f"{route} {audio_source}".strip().lower()
    if any(token in value for token in ("bluetooth", "headset", "usb", "telephony", "hfp", "wired")):
        return "external"
    if any(token in value for token in ("builtin", "built-in", "microphonebuiltin", "default_mic")):
        return "built_in"
    return "unknown"


def _read_json(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _looks_explicit_negative(*values: object) -> bool:
    text = " ".join(str(value or "").lower() for value in values)
    return any(token in text for token in NEGATIVE_SCENARIO_TOKENS)


def _legacy_event_is_explicit_negative(event_label: str, scenario_id: str) -> bool:
    """Treat the event target as authoritative over noisy-scene wording."""
    canonical = canonical_four_class(event_label)
    if canonical in {"racket_bounce", "table_bounce", "floor_other_impact"}:
        return False
    if canonical == "voice_music_noise":
        return True
    return _looks_explicit_negative(event_label, scenario_id)


def discover_stiga_sessions(root: Path) -> list[AudioSession]:
    sessions: list[AudioSession] = []
    for json_path in sorted(root.rglob("session.json")):
        payload = _read_json(json_path)
        if payload is None or payload.get("type") != "stiga_audio_dataset_session":
            continue
        capture = payload.get("capture") or {}
        device = payload.get("device") or {}
        audio_path = json_path.with_name("audio.wav")
        if not audio_path.is_file():
            continue
        session_id = str(payload.get("sessionId") or json_path.parent.name)
        expected = payload.get("expectedBounceCount")
        expected_count = int(expected) if isinstance(expected, (int, float)) else None
        target = str(payload.get("target") or "")
        scenario_id = str(payload.get("scenarioId") or "")
        explicit_negative = target.lower() in {"negative", "noise"} or expected_count == 0
        if expected_count is None and target.lower() not in {"bounce", "positive"}:
            explicit_negative = explicit_negative or _looks_explicit_negative(
                scenario_id, payload.get("scenarioLabel")
            )
        route = str(capture.get("inputRoute") or "")
        audio_source = str(capture.get("audioSource") or capture.get("source") or "")
        install_id = str(device.get("installId") or "").strip()
        model = str(device.get("model") or "unknown_device")
        sessions.append(
            AudioSession(
                session_id=session_id,
                audio_path=audio_path,
                source="stiga_native_recorder",
                device_id=install_id or model,
                scenario_id=scenario_id,
                reviewed_events=(),
                review_complete=explicit_negative,
                explicit_negative=explicit_negative,
                expected_bounce_count=expected_count,
                input_route=route,
                route_class=classify_input_route(route, audio_source),
                metadata=payload,
            )
        )
    return sessions


def _audio_index(root: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for suffix in ("*.wav", "*.m4a"):
        for path in root.rglob(suffix):
            index.setdefault(path.name.lower(), []).append(path)
    return index


def _resolve_event_audio(json_path: Path, wav_filename: str, index: dict[str, list[Path]]) -> Path | None:
    direct = (
        json_path.parent / json_path.stem / wav_filename,
        json_path.parent / wav_filename,
    )
    for path in direct:
        if path.is_file():
            return path
    choices = index.get(Path(wav_filename).name.lower(), [])
    if not choices:
        return None
    session_matches = [path for path in choices if json_path.stem.lower() in {part.lower() for part in path.parts}]
    if len(session_matches) == 1:
        return session_matches[0]
    return choices[0] if len(choices) == 1 else None


def _marker_event(marker: dict, event_label: str, scenario_id: str) -> ReviewedEvent | None:
    status = str(marker.get("review_status") or "confirmed").lower()
    final_label = str(marker.get("final_label") or "").lower()
    class_label = str(
        marker.get("class_label")
        or marker.get("contact_kind")
        or marker.get("not_racket_kind")
        or marker.get("surface_label")
        or final_label
        or event_label
    )
    canonical = canonical_four_class(class_label)
    trainable = status not in {"ignored", "rejected", "unreviewed", "diagnostic"} and final_label != "ignore"
    if canonical is None and final_label == "racket_contact":
        canonical = "racket_bounce"
    if canonical is None and final_label == "not_racket_contact":
        canonical = canonical_four_class(event_label) or canonical_four_class(scenario_id) or "voice_music_noise"
    if canonical is None and not trainable:
        canonical = "voice_music_noise"
    timestamp = marker.get("timestamp_ms")
    if timestamp is None:
        return None
    return ReviewedEvent(float(timestamp), canonical or class_label, trainable and canonical is not None)


def discover_legacy_sessions(root: Path, *, device_hint: str, source_name: str) -> list[AudioSession]:
    index = _audio_index(root)
    sessions: list[AudioSession] = []
    seen_audio: set[Path] = set()
    for json_path in sorted(root.rglob("*.json")):
        payload = _read_json(json_path)
        events = payload.get("events") if payload else None
        if not isinstance(events, list):
            continue
        session_meta = payload.get("session_meta") or {}
        session_id = str(session_meta.get("session_id") or json_path.stem)
        for event_index, event in enumerate(events):
            if not isinstance(event, dict):
                continue
            wav_filename = str(event.get("wav_filename") or "")
            if not wav_filename:
                continue
            audio_path = _resolve_event_audio(json_path, wav_filename, index)
            if audio_path is None or audio_path in seen_audio:
                continue
            seen_audio.add(audio_path)
            event_label = str(event.get("label") or "")
            scenario_id = str(event.get("scenario_id") or event_label)
            review = event.get("review") or {}
            raw_markers = review.get("markers") or []
            reviewed = tuple(
                marker_event
                for marker in raw_markers
                if isinstance(marker, dict)
                for marker_event in [_marker_event(marker, event_label, scenario_id)]
                if marker_event is not None
            )
            explicit_negative = _legacy_event_is_explicit_negative(event_label, scenario_id)
            review_complete = bool(reviewed) and bool(review.get("completed_at") or not review.get("required", True))
            if explicit_negative and not reviewed:
                review_complete = True
            sessions.append(
                AudioSession(
                    session_id=f"{session_id}:{event_index:03d}:{Path(wav_filename).stem}",
                    audio_path=audio_path,
                    source=source_name,
                    device_id=device_hint,
                    scenario_id=scenario_id,
                    reviewed_events=reviewed,
                    review_complete=review_complete,
                    explicit_negative=explicit_negative,
                    input_route="legacy_unknown",
                    route_class="legacy_unknown",
                    metadata={"session_meta": session_meta, "event": event, "json_path": str(json_path)},
                )
            )
    return sessions


def deduplicate_sessions(sessions: Iterable[AudioSession]) -> list[AudioSession]:
    by_audio: dict[Path, AudioSession] = {}
    for session in sessions:
        key = session.audio_path.resolve()
        previous = by_audio.get(key)
        if previous is None or len(session.reviewed_events) > len(previous.reviewed_events):
            by_audio[key] = session

    by_identity: dict[tuple[str, str], AudioSession] = {}
    for session in by_audio.values():
        key = (session.source, session.session_id)
        previous = by_identity.get(key)
        candidate_rank = (len(session.reviewed_events), -len(session.audio_path.parts))
        previous_rank = (
            (len(previous.reviewed_events), -len(previous.audio_path.parts))
            if previous is not None
            else None
        )
        if previous is None or candidate_rank > previous_rank:
            by_identity[key] = session

    return sorted(
        by_identity.values(),
        key=lambda item: (item.source, item.session_id, str(item.audio_path)),
    )
