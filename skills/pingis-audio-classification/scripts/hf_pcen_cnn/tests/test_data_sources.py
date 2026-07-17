from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from hf_pcen_cnn.data_sources import (
    AudioSession,
    _legacy_event_is_explicit_negative,
    classify_input_route,
    deduplicate_sessions,
    discover_stiga_sessions,
)


class DataSourceTests(unittest.TestCase):
    def test_stiga_expected_count_is_not_fabricated_into_timestamps(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            session_dir = Path(temp_dir) / "session"
            session_dir.mkdir()
            sf.write(session_dir / "audio.wav", np.zeros(22_050, dtype=np.float32), 22_050)
            payload = {
                "type": "stiga_audio_dataset_session",
                "schemaVersion": 2,
                "sessionId": "stiga-1",
                "target": "bounce",
                "scenarioId": "normal_bounce",
                "expectedBounceCount": 30,
                "capture": {
                    "inputRoute": "MicrophoneBuiltIn:iPhone Microphone",
                    "audioSource": "ios_avaudioengine_measurement",
                },
                "device": {"installId": "phone-a", "model": "iPhone17,1"},
            }
            (session_dir / "session.json").write_text(json.dumps(payload), encoding="utf-8")

            sessions = discover_stiga_sessions(Path(temp_dir))

            self.assertEqual(len(sessions), 1)
            self.assertEqual(sessions[0].expected_bounce_count, 30)
            self.assertEqual(sessions[0].reviewed_events, ())
            self.assertFalse(sessions[0].review_complete)
            self.assertEqual(sessions[0].route_class, "built_in")

    def test_external_routes_are_detected(self) -> None:
        self.assertEqual(classify_input_route("BluetoothHFP:Headset"), "external")
        self.assertEqual(classify_input_route("MicrophoneBuiltIn:iPhone Microphone"), "built_in")

    def test_positive_bounce_with_noise_words_is_not_negative(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            session_dir = Path(temp_dir) / "session"
            session_dir.mkdir()
            sf.write(session_dir / "audio.wav", np.zeros(22_050, dtype=np.float32), 22_050)
            payload = {
                "type": "stiga_audio_dataset_session",
                "sessionId": "stiga-noisy-positive",
                "target": "bounce",
                "scenarioId": "bounce_speaking_loud_noise",
                "scenarioLabel": "Racket bounce with loud music",
                "expectedBounceCount": 30,
            }
            (session_dir / "session.json").write_text(json.dumps(payload), encoding="utf-8")

            session = discover_stiga_sessions(Path(temp_dir))[0]

            self.assertFalse(session.explicit_negative)
            self.assertFalse(session.review_complete)

    def test_legacy_racket_target_overrides_noisy_scenario(self) -> None:
        self.assertFalse(_legacy_event_is_explicit_negative("racket_bounce", "racket_noisy_speech"))
        self.assertTrue(_legacy_event_is_explicit_negative("talking", "talking_only"))

    def test_duplicate_session_identity_prefers_richer_then_shorter_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            direct_path = root / "direct.wav"
            nested_path = root / "copied" / "nested.wav"
            nested_path.parent.mkdir(parents=True)
            direct_path.touch()
            nested_path.touch()

            direct = AudioSession(
                session_id="duplicate-session:000:event",
                audio_path=direct_path,
                source="legacy:colleague",
                device_id="unknown",
                scenario_id="racket_bounce",
            )
            nested = AudioSession(
                session_id=direct.session_id,
                audio_path=nested_path,
                source=direct.source,
                device_id=direct.device_id,
                scenario_id=direct.scenario_id,
            )

            result = deduplicate_sessions([nested, direct])

            self.assertEqual(result, [direct])


if __name__ == "__main__":
    unittest.main()
