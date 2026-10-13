#!/usr/bin/env bash
set -euo pipefail

# Release helper script for CLIAgentDispatch
# Conforms to Steven T. Pelech's Engineering Archetype standards.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

if [ $# -lt 1 ]; then
    echo "Usage: $0 [patch|minor|major|<specific-version>]"
    echo "Example: $0 minor"
    echo "Example: $0 0.2.0"
    exit 1
fi

BUMP_TYPE="$1"

# Get current version from pyproject.toml
CURRENT_VER=$(grep -E '^version = ' pyproject.toml | head -n 1 | sed -E 's/version = "([^"]+)"/\1/')
echo "Current version: $CURRENT_VER"

IFS='.' read -r MAJOR MINOR PATCH <<< "$CURRENT_VER"

case "$BUMP_TYPE" in
    patch)
        NEW_VER="$MAJOR.$MINOR.$((PATCH + 1))"
        ;;
    minor)
        NEW_VER="$MAJOR.$((MINOR + 1)).0"
        ;;
    major)
        NEW_VER="$((MAJOR + 1)).0.0"
        ;;
    *)
        # Specific version provided
        NEW_VER="$BUMP_TYPE"
        ;;
esac

echo "Target release version: $NEW_VER"

# 1. Update pyproject.toml
sed -i -E "s/^version = \"[^\"]+\"/version = \"$NEW_VER\"/" pyproject.toml

# 2. Update __init__.py
sed -i -E "s/__version__ = \"[^\"]+\"/__version__ = \"$NEW_VER\"/" src/cli_agent_dispatch/__init__.py

# 3. Update config.py
sed -i -E "s/app_version: str = \"[^\"]+\"/app_version: str = \"$NEW_VER\"/" src/cli_agent_dispatch/config.py

# 4. Update mcp/server.py
sed -i -E "s/version=\"[^\"]+\"/version=\"$NEW_VER\"/" src/cli_agent_dispatch/mcp/server.py

# 5. Update README.md title
sed -i -E "s/# CLIAgentDispatch.*/# CLIAgentDispatch (v$NEW_VER)/" README.md

echo "Updated manifests to v$NEW_VER."

# Run verify_release.py
python3 scripts/verify_release.py

# Run tests
echo "Running pytest suite..."
uv run pytest

# Git commit and tag
echo ""
echo "Creating commit and tag for v$NEW_VER..."
git add pyproject.toml src/cli_agent_dispatch/__init__.py src/cli_agent_dispatch/config.py src/cli_agent_dispatch/mcp/server.py README.md
git commit -m "chore(release): bump version to v$NEW_VER"
git tag -a "v$NEW_VER" -m "Release v$NEW_VER"

echo ""
echo "🎉 Release v$NEW_VER successfully created locally!"
echo "To publish and trigger the GitHub Actions release workflow, run:"
echo "   git push origin HEAD"
echo "   git push origin v$NEW_VER"
