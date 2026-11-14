import importlib.util
import socket
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

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

    def _require_ipv6(self, host):
        try:
            with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as sock:
                sock.bind((host, 0))
        except OSError as exc:
            self.skipTest(f"IPv6 binding is unavailable: {exc}")

    def test_ipv6_loopback_port_is_available(self):
        self._require_ipv6("::1")
        self.assertTrue(port_checker.is_port_available("::1", 0))

    def test_ipv6_wildcard_port_is_available(self):
        self._require_ipv6("::")
        self.assertTrue(port_checker.is_port_available("::", 0))

    def test_ipv6_occupied_port_and_release(self):
        self._require_ipv6("::1")
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as holder:
            holder.bind(("::1", 0))
            port = holder.getsockname()[1]
            self.assertFalse(port_checker.is_port_available("::1", port))
            self.assertEqual(port_checker.find_available_port("::1", port, max_attempts=1), (False, port))
            holder.listen()
            self.assertFalse(port_checker.is_port_available("::1", port))
        self.assertTrue(port_checker.is_port_available("::1", port))
        self.assertEqual(port_checker.find_available_port("::1", port, max_attempts=1), (True, port))

    def test_ipv6_socket_creation_failure_is_unavailable(self):
        with patch.object(port_checker.socket, "socket", side_effect=OSError("IPv6 unavailable")):
            self.assertFalse(port_checker.is_port_available("::1", 12345))

    def test_ipv6_windows_bind_check_does_not_use_reuseaddr(self):
        sock = MagicMock()
        sock.__enter__.return_value = sock
        with patch.object(port_checker.sys, "platform", "win32"):
            with patch.object(port_checker.socket, "socket", return_value=sock) as create_socket:
                self.assertTrue(port_checker.is_port_available("::1", 12345))
        create_socket.assert_called_once_with(socket.AF_INET6, socket.SOCK_STREAM)
        sock.setsockopt.assert_not_called()
        sock.bind.assert_called_once_with(("::1", 12345))

    def test_non_windows_bind_check_uses_reuseaddr(self):
        sock = MagicMock()
        sock.__enter__.return_value = sock

        with patch.object(port_checker.sys, "platform", "linux"):
            with patch.object(port_checker.socket, "socket", return_value=sock):
                self.assertTrue(port_checker.is_port_available("127.0.0.1", 12345))

        sock.setsockopt.assert_called_once_with(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind.assert_called_once_with(("127.0.0.1", 12345))


if __name__ == "__main__":
    unittest.main()
