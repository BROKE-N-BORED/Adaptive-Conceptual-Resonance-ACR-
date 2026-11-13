"""Installer and launcher tests (v0.8.1): the desktop and menu
icon must actually start the app.

v0.8.1 is a launch fix. ACR.desktop used to find run.sh through
%k, the desktop file's own path -- but install.sh copies the
.desktop to ~/.local/share/applications and ~/Desktop, where
run.sh is not, so clicking the icon did nothing (and
Terminal=false hid the failure). The fix bakes the app dir into
the Exec line at install, the way the Icon line already did.
These tests pin that, and simulate a real installed launch --
the test that would have caught the bug.

Install-ACR.desktop keeps its %k: it sits in the extracted
folder next to install.sh, so %k is correct there. That is
verified here too, not assumed.

Run with the Qt test env: LD_LIBRARY_PATH to the extracted libs,
QT_QPA_PLATFORM=offscreen, TMPDIR=/tmp, default HOME.
"""

import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# install.sh's substitution, verbatim: the app dir replaces the
# __APP_DIR__ placeholder everywhere it appears in ACR.desktop.
SED_CMD = "sed \"s|__APP_DIR__|$DIR|g\" ACR.desktop"


def _read_desktop(path):
    """Read a .desktop file into a dict of its key/value lines."""
    values = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if "=" in line and not line.startswith("["):
                key, value = line.split("=", 1)
                values[key] = value
    return values


def _substitute(app_dir, out_path):
    """Run install.sh's exact substitution on ACR.desktop."""
    proc = subprocess.run(
        ["bash", "-c", SED_CMD],
        cwd=ROOT, capture_output=True, text=True,
        env=dict(os.environ, DIR=app_dir), timeout=60)
    if proc.returncode != 0:
        raise AssertionError("substitution failed: " + proc.stderr)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(proc.stdout)
    return out_path


def _exec_argv(exec_value, desktop_path=None):
    """Turn a .desktop Exec value into an argv list, the way a
    desktop environment would: split the line honouring its
    quotes FIRST, then replace a %k token with the desktop
    file's own path. The order matters: expanding %k before
    splitting would break a spaced path into pieces, while the
    desktop keeps it as one argument."""
    argv = shlex.split(exec_value)
    if desktop_path is not None:
        argv = [desktop_path if arg == "%k" else arg
                for arg in argv]
    return argv


class LauncherShapeTests(unittest.TestCase):
    def test_acr_desktop_exec_bakes_the_app_dir(self):
        # The Exec line must carry the __APP_DIR__ placeholder
        # (install.sh bakes the real dir in) and no %k: %k
        # resolves to the .desktop file's own folder, which after
        # install is ~/.local/share/applications or ~/Desktop --
        # where run.sh is not. That was the v0.8.1 bug.
        values = _read_desktop(os.path.join(ROOT, "ACR.desktop"))
        self.assertIn("__APP_DIR__", values["Exec"])
        self.assertNotIn("%k", values["Exec"])
        self.assertNotIn("%", values["Exec"])

    def test_install_desktop_keeps_percent_k(self):
        # Install-ACR.desktop sits in the extracted folder next
        # to install.sh, so %k is correct there -- and it is the
        # only way the installer finds its own folder before
        # anything is installed.
        values = _read_desktop(
            os.path.join(ROOT, "Install-ACR.desktop"))
        self.assertIn("%k", values["Exec"])
        self.assertIn("install.sh", values["Exec"])


class SubstitutionTests(unittest.TestCase):
    def test_substituted_exec_and_icon_resolve(self):
        # Run install.sh's real substitution with a SPACED app
        # path: every placeholder must be replaced, and the Exec
        # and Icon lines must point at real files.
        tmp = tempfile.mkdtemp()
        try:
            app_dir = os.path.join(tmp, "ACR Test App")
            os.makedirs(os.path.join(app_dir, "icons"))
            shutil.copy(os.path.join(ROOT, "run.sh"),
                        os.path.join(app_dir, "run.sh"))
            shutil.copy(os.path.join(ROOT, "icons", "acr-48.png"),
                        os.path.join(app_dir, "icons", "acr-48.png"))
            out = os.path.join(tmp, "ACR.desktop")
            _substitute(app_dir, out)
            with open(out, encoding="utf-8") as fh:
                text = fh.read()
            self.assertNotIn("__APP_DIR__", text)
            values = _read_desktop(out)
            argv = _exec_argv(values["Exec"])
            self.assertEqual(argv[0], "bash")
            self.assertEqual(argv[1],
                             os.path.join(app_dir, "run.sh"))
            self.assertTrue(os.path.isfile(argv[1]))
            self.assertTrue(os.path.isfile(values["Icon"]))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class InstallDesktopTests(unittest.TestCase):
    def test_install_desktop_runs_install_sh_beside_it(self):
        # Install-ACR.desktop's %k must resolve to install.sh in
        # the same folder -- proven by running its real Exec line
        # against a stub install.sh and a spaced folder name,
        # from an unrelated working directory.
        tmp = tempfile.mkdtemp()
        try:
            folder = os.path.join(tmp, "ACR Extracted Folder")
            os.makedirs(folder)
            desktop = os.path.join(folder, "Install-ACR.desktop")
            shutil.copy(os.path.join(ROOT, "Install-ACR.desktop"),
                        desktop)
            stub = os.path.join(folder, "install.sh")
            with open(stub, "w", encoding="utf-8") as fh:
                fh.write("#!/bin/sh\n"
                         "DIR=\"$(cd \"$(dirname \"$0\")\" "
                         "&& pwd)\"\n"
                         "touch \"$DIR/installed.marker\"\n")
            os.chmod(stub, 0o755)
            values = _read_desktop(desktop)
            argv = _exec_argv(values["Exec"], desktop)
            proc = subprocess.run(
                argv, cwd=tmp, capture_output=True, text=True,
                timeout=60)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertTrue(os.path.exists(
                os.path.join(folder, "installed.marker")),
                "install.sh next to the .desktop was not run")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class LaunchSimulationTests(unittest.TestCase):
    def test_installed_icon_launches_the_app(self):
        # The test that would have caught the v0.8.1 bug: build
        # a fake install the way install.sh does (app dir with a
        # SPACE in its name, substituted .desktop placed in a
        # DIFFERENT folder), run its Exec from an arbitrary
        # working directory, and prove the app actually starts:
        # the process stays alive and writes its startup line to
        # the log.
        tmp = tempfile.mkdtemp()
        proc = None
        try:
            app_dir = os.path.join(tmp, "ACR Test App")
            shutil.copytree(
                ROOT, app_dir,
                ignore=shutil.ignore_patterns(
                    "__pycache__", "*.zip", "logs"))
            # Fake .venv: a WRAPPER script, not a symlink -- a
            # symlinked python loses the venv's site-packages.
            # Whatever python runs this suite is the one with the
            # interface library, so wrap sys.executable.
            venv_bin = os.path.join(app_dir, ".venv", "bin")
            os.makedirs(venv_bin)
            wrapper = os.path.join(venv_bin, "python")
            with open(wrapper, "w", encoding="utf-8") as fh:
                fh.write("#!/bin/sh\n"
                         "exec \"%s\" \"$@\"\n" % sys.executable)
            os.chmod(wrapper, 0o755)
            # install.sh puts the substituted .desktop somewhere
            # else entirely -- that is the whole point of the
            # bug, so the simulation must do the same.
            apps_dir = os.path.join(
                tmp, ".local", "share", "applications")
            os.makedirs(apps_dir)
            installed = os.path.join(apps_dir, "ACR.desktop")
            _substitute(app_dir, installed)
            values = _read_desktop(installed)
            argv = _exec_argv(values["Exec"])
            env = dict(os.environ)
            env["QT_QPA_PLATFORM"] = "offscreen"
            proc = subprocess.Popen(
                argv, cwd=apps_dir, env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE)
            log_path = os.path.join(app_dir, "logs", "acr.log")
            deadline = time.time() + 30
            started = False
            while time.time() < deadline:
                if proc.poll() is not None:
                    break
                if os.path.exists(log_path):
                    with open(log_path, encoding="utf-8",
                              errors="replace") as fh:
                        if "started" in fh.read():
                            started = True
                            break
                time.sleep(0.25)
            self.assertIsNone(
                proc.poll(),
                "the app exited instead of staying open")
            self.assertTrue(
                started,
                "the app never wrote its startup line to the log")
            # It stays open: give it a moment, then stop it.
            time.sleep(1)
            self.assertIsNone(proc.poll())
        finally:
            if proc is not None:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait(timeout=10)
                if proc.stdout:
                    proc.stdout.close()
                if proc.stderr:
                    proc.stderr.close()
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
