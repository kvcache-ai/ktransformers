"""HMAC-authenticated pickle serialization for ZMQ RPC.

Mitigates CVE-class pickle deserialization attacks on the scheduler RPC
channel by requiring a shared HMAC-SHA256 signature on every message.
The secret is generated once per server lifetime and passed to clients
via the (local, temporary) config file.

See: https://github.com/kvcache-ai/ktransformers/issues/2042
"""

import hashlib
import hmac
import os
import pickle
import struct

# 32 bytes of cryptographic randomness
HMAC_KEY_SIZE = 32

# Wire format: [4-byte big-endian signature length][signature][payload]
_SIG_LEN_FMT = "!I"
_SIG_LEN_SIZE = struct.calcsize(_SIG_LEN_FMT)


def generate_hmac_key() -> bytes:
    """Generate a random HMAC key for a server session."""
    return os.urandom(HMAC_KEY_SIZE)


def _sign(key: bytes, payload: bytes) -> bytes:
    if not key:
        raise ValueError("HMAC key is empty or None")
    return hmac.new(key, payload, hashlib.sha256).digest()


def dumps_signed(obj, key: bytes) -> bytes:
    """Serialize *obj* with pickle and prepend an HMAC-SHA256 signature."""
    if key is None:
        raise ValueError("HMAC key is not set — cannot send unauthenticated RPC messages")
    payload = pickle.dumps(obj)
    sig = _sign(key, payload)
    return struct.pack(_SIG_LEN_FMT, len(sig)) + sig + payload


def loads_signed(data: bytes, key: bytes):
    """Verify the HMAC-SHA256 signature and deserialize.

    Raises ``ValueError`` if the signature is missing, truncated, or invalid.
    """
    if key is None:
        raise ValueError("HMAC key is not set — cannot verify unauthenticated RPC messages")
    if len(data) < _SIG_LEN_SIZE:
        raise ValueError("Message too short: missing signature header")

    (sig_len,) = struct.unpack_from(_SIG_LEN_FMT, data)
    header_end = _SIG_LEN_SIZE + sig_len

    if len(data) < header_end:
        raise ValueError("Message truncated: incomplete signature")

    sig = data[_SIG_LEN_SIZE:header_end]
    payload = data[header_end:]

    expected = _sign(key, payload)
    if not hmac.compare_digest(sig, expected):
        raise ValueError("HMAC verification failed: message may have been tampered with")

    return pickle.loads(payload)
