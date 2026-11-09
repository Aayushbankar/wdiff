from __future__ import annotations

import shutil
import subprocess
import sys

import pytest

import wdiff


class TestPackaging:
    def test_version_attribute(self):
        assert hasattr(wdiff, "__version__")
        assert wdiff.__version__ == "0.3.0"

    def test_cli_entrypoint_executable(self):
        """Verify wdiff executable runs via subprocess."""
        exe = shutil.which("wdiff")
        if not exe:
            pytest.skip("wdiff not installed in current PATH")

        proc = subprocess.run([exe, "--version"], capture_output=True, text=True, check=True)
        assert f"wdiff {wdiff.__version__}" in proc.stdout

    def test_module_execution(self):
        """Verify python -m wdiff.cli --version works."""
        proc = subprocess.run(
            [sys.executable, "-m", "wdiff.cli", "--version"],
            capture_output=True,
            text=True,
            check=True,
        )
        assert f"wdiff {wdiff.__version__}" in proc.stdout
