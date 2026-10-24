"""`web/scripts/vercel-ignore.sh`: which commits Vercel builds.

A mistake here either builds a preview URL nobody guards or stops the page from deploying at
all, and neither is visible until a release. The script runs under /bin/sh, as Vercel runs it.
"""

import subprocess
from pathlib import Path

WEB = Path(__file__).resolve().parents[2] / "web"
IGNORE = WEB / "scripts" / "vercel-ignore.sh"
SHELL = "/bin/sh"


def _ignore(env: dict[str, str]) -> int:
    completed = subprocess.run(  # noqa: S603 (a fixed script path, no shell)
        [SHELL, str(IGNORE)],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", **env},
        timeout=30,
        check=False,
    )
    return completed.returncode


def test_a_release_commit_on_production_builds() -> None:
    assert (
        _ignore(
            {
                "VERCEL_ENV": "production",
                "VERCEL_GIT_COMMIT_MESSAGE": "chore(main): release paperwatch 0.8.0",
            }
        )
        == 1
    )


def test_a_feature_commit_on_production_skips() -> None:
    assert (
        _ignore({"VERCEL_ENV": "production", "VERCEL_GIT_COMMIT_MESSAGE": "feat: add the sheet"})
        == 0
    )


def test_a_preview_never_builds() -> None:
    # Even a release-shaped message in a preview must not produce an unprotected URL.
    assert (
        _ignore(
            {
                "VERCEL_ENV": "preview",
                "VERCEL_GIT_COMMIT_MESSAGE": "chore(main): release paperwatch 0.8.0",
            }
        )
        == 0
    )
    assert _ignore({"VERCEL_GIT_COMMIT_MESSAGE": "chore(main): release 1.0.0"}) == 0
