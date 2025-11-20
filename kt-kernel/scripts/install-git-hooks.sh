#!/usr/bin/env sh
# Install git hooks from kt-kernel/.githooks into the monorepo's .git/hooks by
# creating symlinks (or copying if symlink fails).

set -eu

# This script lives in kt-kernel/scripts/, so REPO_ROOT = kt-kernel
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HOOKS_SRC="$REPO_ROOT/.githooks"

# Detect the top-level Git worktree (the monorepo root: ktransformers)
GIT_TOP="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [ -z "$GIT_TOP" ] || [ ! -d "$GIT_TOP/.git" ]; then
  echo "[install-git-hooks] Not inside a git worktree; skipping hooks installation." >&2
  exit 0
fi

GIT_DIR="$GIT_TOP/.git"
HOOKS_DEST="$GIT_DIR/hooks"

if [ ! -d "$HOOKS_SRC" ]; then
  echo "[install-git-hooks] No .githooks directory found at $HOOKS_SRC" >&2
  exit 1
fi

echo "[install-git-hooks] Installing git hooks from $HOOKS_SRC to $HOOKS_DEST (repo: $GIT_TOP)"

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

echo "[install-git-hooks] Done. Hooks installed."
