#!/usr/bin/env sh
# Install git hooks from .githooks into .git/hooks by creating symlinks (or copying if symlink fails).

set -eu

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GIT_DIR="$REPO_ROOT/.git"
HOOKS_SRC="$REPO_ROOT/.githooks"
HOOKS_DEST="$GIT_DIR/hooks"

if [ ! -d "$GIT_DIR" ]; then
  echo "Not a git repository (no .git directory) at $REPO_ROOT" >&2
  exit 1
fi

if [ ! -d "$HOOKS_SRC" ]; then
  echo "No .githooks directory found at $HOOKS_SRC" >&2
  exit 1
fi

echo "Installing git hooks from $HOOKS_SRC to $HOOKS_DEST"

# Ensure all source hook files are executable so that even if copied (not symlinked) they run.
for src_hook in "$HOOKS_SRC"/*; do
  [ -f "$src_hook" ] || continue
  if [ ! -x "$src_hook" ]; then
    chmod +x "$src_hook" || true
  fi
done

for hook in "$HOOKS_SRC"/*; do
  [ -e "$hook" ] || continue
  name=$(basename "$hook")
  dest="$HOOKS_DEST/$name"

  # Remove existing hook if it's our symlink or a file
  if [ -L "$dest" ] || [ -f "$dest" ]; then
    rm -f "$dest"
  fi

  # Try symlink first
  if ln -s "$hook" "$dest" 2>/dev/null; then
    echo "linked $name"
  else
    # Fall back to copying and preserve executable bit
    cp "$hook" "$dest"
    chmod +x "$dest"
    echo "copied $name"
  fi
done

echo "Done. Hooks installed."
