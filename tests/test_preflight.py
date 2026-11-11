import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from safecompress_aya.preflight import audit


class PreflightTest(unittest.TestCase):
    def test_writes_immutable_hashed_run(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = root / "experiment.yaml"
            config.write_text("model_id: CohereLabs/tiny-aya-global\n")
            with patch("safecompress_aya.preflight.command_result", return_value={"exit_code": 0}), \
                 patch("safecompress_aya.preflight._model_config_cached", return_value=False):
                first = audit(config, root / "runs")
                second = audit(config, root / "runs")
            self.assertNotEqual(first, second)
            report = json.loads((first / "report.json").read_text())
            self.assertFalse(report["quantization_save_reload_generation_verified"])
            for line in (first / "SHA256SUMS").read_text().splitlines():
                digest, filename = line.split("  ")
                self.assertEqual(hashlib.sha256((first / filename).read_bytes()).hexdigest(), digest)


if __name__ == "__main__":
    unittest.main()
