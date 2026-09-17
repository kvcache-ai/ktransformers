import tempfile
import unittest
from pathlib import Path

import yaml

from ci.ci_register import register_cpu_ci
from kt_kernel.cli.commands.config import _parse_value
from kt_kernel.cli.config.settings import Settings


register_cpu_ci(est_time=0.1, suite="default")


class TestConfigParseValue(unittest.TestCase):
    def test_numeric_strings_stay_integers(self):
        for raw, expected in (("0", 0), ("1", 1), ("2", 2), ("30000", 30000)):
            value = _parse_value(raw)
            self.assertIs(type(value), int, raw)
            self.assertEqual(value, expected)

    def test_boolean_words_still_parse_as_booleans(self):
        for raw in ("true", "True", "yes", "on"):
            self.assertIs(_parse_value(raw), True, raw)
        for raw in ("false", "False", "no", "off"):
            self.assertIs(_parse_value(raw), False, raw)

    def test_env_var_set_to_one_is_exported_as_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            config_path = tmp / "config.yaml"
            # Keep every directory Settings creates inside the temp dir.
            config_path.write_text(
                yaml.safe_dump({"paths": {"models": str(tmp / "models"), "cache": str(tmp / "cache")}}),
                encoding="utf-8",
            )

            Settings(config_path=config_path).set("advanced.env.CUDA_VISIBLE_DEVICES", _parse_value("1"))

            self.assertEqual(Settings(config_path=config_path).get_env_vars(), {"CUDA_VISIBLE_DEVICES": "1"})


if __name__ == "__main__":
    unittest.main()
