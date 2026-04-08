"""
Port availability checking utilities.
"""

import socket
from typing import Tuple


def is_port_available(host: str, port: int) -> bool:
    """Check if a port is available on the given host.

    Args:
        host: Host address (e.g., "0.0.0.0", "127.0.0.1")
        port: Port number to check

    Returns:
        True if port is available, False if occupied
    """
    try:
        # Try to bind to the port
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)

        # Use SO_REUSEADDR to allow binding to recently closed ports
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        # Try to bind
        result = sock.connect_ex((host if host != "0.0.0.0" else "127.0.0.1", port))
        sock.close()

        # If connect_ex returns 0, port is occupied
        # If it returns error (non-zero), port is available
        return result != 0

    except Exception:
        # If any error occurs, assume port is not available
        return False


def find_available_port(host: str, start_port: int, max_attempts: int = 100) -> Tuple[bool, int]:
    """Find an available port starting from start_port.

    Args:
        host: Host address
        start_port: Starting port number to check
        max_attempts: Maximum number of ports to try

    Returns:
        Tuple of (found, port_number)
        - found: True if an available port was found
        - port_number: The available port number (or start_port if not found)
    """
    for port in range(start_port, start_port + max_attempts):
        if is_port_available(host, port):
            return True, port

    return False, start_port
