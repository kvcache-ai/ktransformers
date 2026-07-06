import importlib.util
import socket
from pathlib import Path
import unittest

from ci.ci_register import register_cpu_ci


register_cpu_ci(est_time=0.1, suite="default")


PORT_CHECKER_PATH = Path(__file__).resolve().parents[2] / "python" / "cli" / "utils" / "port_checker.py"
SPEC = importlib.util.spec_from_file_location("port_checker", PORT_CHECKER_PATH)
assert SPEC is not None and SPEC.loader is not None
port_checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(port_checker)


class TestPortChecker(unittest.TestCase):
    def test_bound_port_is_not_available_before_listen(self):
        holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            holder.bind(("127.0.0.1", 0))
            port = holder.getsockname()[1]

            self.assertFalse(port_checker.is_port_available("127.0.0.1", port))
            self.assertEqual(port_checker.find_available_port("127.0.0.1", port, max_attempts=1), (False, port))
        finally:
            holder.close()


if __name__ == "__main__":
    unittest.main()
