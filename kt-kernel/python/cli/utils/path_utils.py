from pathlib import Path


def find_existing_parent(path: str) -> str:
    """Return the nearest existing parent for a path without assuming POSIX roots."""
    parent = Path(path).parent
    while not parent.exists():
        next_parent = parent.parent
        if next_parent == parent:
            break
        parent = next_parent
    return str(parent)
