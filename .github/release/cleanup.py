"""Standard-library-only cleanup, including when build dependency installation fails."""

import argparse
import os
import shutil
from pathlib import Path


def cleanup(directory, parent):
    directory, parent = Path(directory), Path(parent).resolve()
    if directory.is_symlink():
        raise ValueError("Refuse symlink cleanup")
    resolved = directory.resolve(strict=True)
    if resolved.parent != parent or not resolved.name.startswith("kt-four-main."):
        raise ValueError("Refuse broad cleanup")
    if (resolved / ".kt-release-owned").read_text().strip() != os.environ[
        "GITHUB_RUN_ID"
    ]:
        raise ValueError("Build directory is not owned by this run")
    shutil.rmtree(resolved)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    args = parser.parse_args()
    cleanup(args.directory, args.parent)
