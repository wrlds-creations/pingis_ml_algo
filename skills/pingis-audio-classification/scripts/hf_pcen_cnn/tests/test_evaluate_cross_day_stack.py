from __future__ import annotations

import unittest

import pandas as pd

from hf_pcen_cnn.evaluate_cross_day_stack import _device_contract


class DeviceContractTests(unittest.TestCase):
    def test_uses_stable_hardware_identity(self) -> None:
        metadata = pd.DataFrame(
            [
                {
                    "device_id": "round-a-install",
                    "device_manufacturer": "Apple",
                    "device_model": "iPhone17,1",
                    "device_platform": "ios",
                },
                {
                    "device_id": "round-a-install",
                    "device_manufacturer": "Apple",
                    "device_model": "iPhone17,1",
                    "device_platform": "ios",
                },
                {
                    "device_id": "round-b-install",
                    "device_manufacturer": "motorola",
                    "device_model": "moto g55 5G",
                    "device_platform": "android",
                },
            ]
        )

        contract = _device_contract(metadata)

        self.assertEqual(
            contract[("apple", "iphone17,1", "ios")],
            "round-a-install",
        )
        self.assertEqual(
            contract[("motorola", "moto g55 5g", "android")],
            "round-b-install",
        )


if __name__ == "__main__":
    unittest.main()
