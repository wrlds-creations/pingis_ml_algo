from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from hf_pcen_cnn.build_edge_round_manifest import resolve_source_wav


class EdgeRoundManifestTests(unittest.TestCase):
    def test_resolve_source_wav_strips_portable_data_prefix(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio = root / "CJ-round" / "device" / "session" / "audio.wav"
            audio.parent.mkdir(parents=True)
            audio.write_bytes(b"RIFF")

            resolved = resolve_source_wav(
                root,
                "data/CJ-round/device/session/audio.wav",
            )

            self.assertEqual(resolved, audio.resolve())

    def test_resolve_source_wav_rejects_missing_audio(self) -> None:
        with TemporaryDirectory() as temp_dir:
            with self.assertRaises(FileNotFoundError):
                resolve_source_wav(
                    Path(temp_dir),
                    "data/missing/audio.wav",
                )


if __name__ == "__main__":
    unittest.main()
