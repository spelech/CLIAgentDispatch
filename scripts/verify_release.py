#!/usr/bin/env python3
"""
Release and Version Verification Engine for CLIAgentDispatch.
Validates version consistency across manifests, markdown relative links, and packaging readiness.
Conforms to Steven T. Pelech's Engineering Archetype standards.
"""

import argparse
import re
import sys
from pathlib import Path


def check_markdown_links(root_dir: Path) -> bool:
    print("🔍 Checking markdown relative links...")
    has_errors = False
    excluded_parts = {
        "node_modules",
        ".node_modules_root",
        "venv",
        ".venv",
        "dist",
        "htmlcov",
        ".pytest_cache",
        "coverage",
        ".git",
    }

    checked_count = 0
    for md_file in root_dir.glob("**/*.md"):
        if any(part in excluded_parts for part in md_file.parts):
            continue

        try:
            content = md_file.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            print(f"⚠️  Could not read {md_file.relative_to(root_dir)}: {e}")
            continue

        # Match markdown links [text](target)
        links = re.findall(r"\[([^\]]+)\]\(([^)]+)\)", content)
        for text, link in links:
            link = link.strip()
            # Ignore absolute URLs, mailto, in-page anchors, and file:// URIs
            if (
                link.startswith("http://")
                or link.startswith("https://")
                or link.startswith("#")
                or link.startswith("mailto:")
                or link.startswith("file://")
            ):
                continue

            # Strip query params and in-page anchor
            target_path = link.split("?")[0].split("#")[0]
            if not target_path:
                continue

            # Normal relative filesystem path
            resolved = (md_file.parent / target_path).resolve()
            if not resolved.exists():
                alt_md = md_file.parent / f"{target_path}.md"
                alt_idx = md_file.parent / target_path / "index.md"
                if alt_md.exists() or alt_idx.exists():
                    checked_count += 1
                    continue

                print(
                    f"❌ Broken relative link in {md_file.relative_to(root_dir)}: [{text}]({link})"
                )
                has_errors = True
            else:
                checked_count += 1

    if not has_errors:
        print(f"✅ Verified {checked_count} markdown relative links successfully.")
    return not has_errors


def check_version_sync(root_dir: Path) -> tuple[bool, str]:
    print("🔍 Checking version consistency across project manifests...")
    versions: dict[str, str] = {}

    # 1. pyproject.toml
    pyproject = root_dir / "pyproject.toml"
    if pyproject.exists():
        match = re.search(
            r'version\s*=\s*["\']([^"\']+)["\']', pyproject.read_text(encoding="utf-8")
        )
        if match:
            versions["pyproject.toml"] = match.group(1).strip()

    # 2. src/cli_agent_dispatch/__init__.py
    init_py = root_dir / "src" / "cli_agent_dispatch" / "__init__.py"
    if init_py.exists():
        match = re.search(
            r'__version__\s*=\s*["\']([^"\']+)["\']', init_py.read_text(encoding="utf-8")
        )
        if match:
            versions["__init__.py"] = match.group(1).strip()

    # 3. src/cli_agent_dispatch/config.py
    config_py = root_dir / "src" / "cli_agent_dispatch" / "config.py"
    if config_py.exists():
        match = re.search(
            r'app_version:\s*str\s*=\s*["\']([^"\']+)["\']', config_py.read_text(encoding="utf-8")
        )
        if match:
            versions["config.py"] = match.group(1).strip()

    # 4. src/cli_agent_dispatch/mcp/server.py
    server_py = root_dir / "src" / "cli_agent_dispatch" / "mcp" / "server.py"
    if server_py.exists():
        match = re.search(
            r'version\s*=\s*["\']([^"\']+)["\']', server_py.read_text(encoding="utf-8")
        )
        if match:
            versions["mcp/server.py"] = match.group(1).strip()

    # 5. README.md
    readme = root_dir / "README.md"
    if readme.exists():
        match = re.search(
            r"#\s+CLIAgentDispatch\s+\(v([^)]+)\)", readme.read_text(encoding="utf-8")
        )
        if match:
            versions["README.md"] = match.group(1).strip()

    for manifest, ver in versions.items():
        print(f"   • {manifest:26}: {ver}")

    unique_versions = set(versions.values())
    if len(unique_versions) > 1:
        print(f"❌ Version mismatch detected across manifests: {unique_versions}")
        return False, ""

    if not unique_versions:
        print("❌ No version declarations found in required manifests.")
        return False, ""

    version = next(iter(unique_versions))
    print(f"✅ Version consistency check passed (v{version} across {len(versions)} manifests).")
    return True, version


def main() -> int:
    parser = argparse.ArgumentParser(description="Release Verification Engine")
    parser.add_argument("--skip-links", action="store_true", help="Skip markdown link checking")
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parent.parent

    links_ok = True if args.skip_links else check_markdown_links(root_dir)
    versions_ok, version = check_version_sync(root_dir)

    success = links_ok and versions_ok
    if not success:
        print("\n❌ Release verification FAILED. Correct issues before cutting a release.")
        return 1

    print(f"\n🎉 Release verification PASSED. Ready for release v{version}!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
