"""The evidence observer must safely sample large frozen tensors."""

import pytest
import torch

from test.run_dsv4_acceptance import AcceptanceAudit


@pytest.mark.parametrize("size", [0, 1, 63, 64, 65, 2**24 + 4])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_frozen_sample_indices_remain_in_bounds(size, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    parameter = torch.zeros(size, dtype=torch.uint8, device=device)
    count = min(64, size)
    positions = [i * (size - 1) // max(count - 1, 1) for i in range(count)]
    expected = torch.arange(count, dtype=torch.uint8)
    if positions:
        parameter[positions] = expected.to(device)
    actual = AcceptanceAudit._sample(parameter)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
