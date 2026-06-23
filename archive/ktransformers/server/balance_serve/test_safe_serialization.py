"""Tests for HMAC-authenticated pickle serialization."""

import pytest
import sys
import os

# Ensure the archive package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from ktransformers.server.balance_serve.safe_serialization import (
    dumps_signed,
    generate_hmac_key,
    loads_signed,
)


def test_roundtrip():
    key = generate_hmac_key()
    obj = {"method": "add_query", "params": {"id": 42}}
    data = dumps_signed(obj, key)
    assert loads_signed(data, key) == obj


def test_wrong_key_rejected():
    key1 = generate_hmac_key()
    key2 = generate_hmac_key()
    data = dumps_signed({"ok": True}, key1)
    with pytest.raises(ValueError, match="HMAC verification failed"):
        loads_signed(data, key2)


def test_tampered_payload_rejected():
    key = generate_hmac_key()
    data = bytearray(dumps_signed({"ok": True}, key))
    # Flip a byte in the payload (after the signature)
    data[-1] ^= 0xFF
    with pytest.raises(ValueError, match="HMAC verification failed"):
        loads_signed(bytes(data), key)


def test_truncated_message_rejected():
    key = generate_hmac_key()
    with pytest.raises(ValueError, match="too short"):
        loads_signed(b"\x00", key)


def test_incomplete_signature_rejected():
    key = generate_hmac_key()
    # Header says 32-byte sig but only 4 bytes follow
    data = b"\x00\x00\x00\x20" + b"\x00" * 4
    with pytest.raises(ValueError, match="truncated"):
        loads_signed(data, key)


def test_complex_objects():
    key = generate_hmac_key()
    obj = {
        "status": "ok",
        "batch_todo": [1, 2, 3],
        "nested": {"a": [None, True, 3.14]},
    }
    assert loads_signed(dumps_signed(obj, key), key) == obj


def test_key_length():
    key = generate_hmac_key()
    assert len(key) == 32


def test_none_key_rejected_on_dumps():
    with pytest.raises(ValueError, match="HMAC key is not set"):
        dumps_signed({"ok": True}, None)


def test_none_key_rejected_on_loads():
    key = generate_hmac_key()
    data = dumps_signed({"ok": True}, key)
    with pytest.raises(ValueError, match="HMAC key is not set"):
        loads_signed(data, None)
