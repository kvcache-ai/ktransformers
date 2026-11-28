"""Extract the two measured windows from the indexed, packed Alpaca dataset."""

import argparse
import json
import struct
from pathlib import Path

import numpy as np


def windows(prefix):
    index = prefix.with_suffix(".idx").read_bytes()
    assert index[:9] == b"MMIDIDX\0\0" and index[17] == 4
    count = struct.unpack_from("<Q", index, 18)[0]
    sizes = np.frombuffer(index, dtype="<i4", count=count, offset=34)
    offsets = np.frombuffer(index, dtype="<i8", count=count, offset=34 + count * 4) // 4
    data = np.memmap(prefix.with_suffix(".bin"), dtype="<i4", mode="r")
    return [data[offsets[i] : offsets[i] + sizes[i]] for i in (677, 1340)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packed_directory", type=Path)
    args = parser.parse_args()
    keys = ("labels", "input_ids", "attention_mask")
    columns = [
        windows(args.packed_directory / f"alpaca_packed_{k}_document") for k in keys
    ]
    batch = []
    for values in zip(*columns):
        labels = values[0]
        start = max(
            0, min(len(labels) - 1024, int(np.flatnonzero(labels != -100)[0]) - 256)
        )
        record = {k: v[start : start + 1024].tolist() for k, v in zip(keys, values)}
        assert (
            len(record["input_ids"]) == 1024
            and sum(x != -100 for x in record["labels"][1:]) == 768
        )
        batch.append(record)
    Path("batch.json").write_text(json.dumps(batch))
