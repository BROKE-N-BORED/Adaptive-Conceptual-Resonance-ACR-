"""ACR v0.12 tests: silent update check + ask-first auto-update.

The check, the verified download, staging, the launcher apply,
the restart-now updater script, and the GUI flows (manual check,
silent startup check, update offer, download with cancel, the
restart question Yes and No). No real network is touched: the
check and download tests use a stub urlopen or a local fake
release server, in the fake-service shape of
tests/test_local_encoder.py.

Run with the Qt test env: LD_LIBRARY_PATH to the extracted libs,
QT_QPA_PLATFORM=offscreen, TMPDIR=/tmp, default HOME.
"""

import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import zipfile
from contextlib import redirect_stderr
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock
from urllib.error import URLError

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

from PySide6.QtCore import QProcess, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication, QMessageBox, QPushButton)

import acr_app  # noqa: E402
import acr_launch  # noqa: E402
import acr_updates  # noqa: E402
from acr_app import MainWindow  # noqa: E402
from acr_updates import ACRUpdateError, UpdateCancelled  # noqa: E402

APP = None
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def setUpModule():
    global APP
    APP = QApplication.instance()
    if APP is None:
        APP = QApplication(sys.argv)


# ------------------------------------------------------------ stubs


class _StubResponse:
    """A urlopen answer: bytes, a Content-Length header, and the
    read-until-empty shape the download loop relies on."""

    def __init__(self, body, headers=None):
        self._body = body
        self._pos = 0
        self.headers = headers or {}

    def read(self, size=-1):
        if size is None or size < 0:
            chunk = self._body[self._pos:]
            self._pos = len(self._body)
            return chunk
        chunk = self._body[self._pos:self._pos + size]
        self._pos += len(chunk)
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _json_bytes(payload):
    return json.dumps(payload).encode("utf-8")


def _release_payload(tag, zip_url, sha_url):
    assets = []
    if zip_url is not None:
        assets.append({"name": "acr-%s.zip" % tag,
                       "browser_download_url": zip_url})
    if sha_url is not None:
        assets.append({"name": "acr-%s.zip.sha256" % tag,
                       "browser_download_url": sha_url})
    return {"tag_name": tag,
            "html_url": "http://fake/releases/tag/%s" % tag,
            "assets": assets}


def _stub_urlopen(mapping):
    """A urlopen stub: mapping is {substring: body-or-exception}.
    The first substring found in the URL wins. Bodies are bytes
    (a Content-Length header is set from them)."""
    def _open(url, timeout=None):
        for needle, body in mapping.items():
            if needle in url:
                if isinstance(body, Exception):
                    raise body
                return _StubResponse(
                    body, {"Content-Length": str(len(body))})
        raise URLError("no stub for %s" % url)
    return _open


def _fake_zip_bytes(files):
    """A release-shaped zip in memory: files is
    {arcname: (content bytes, mode or None)}."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, (content, mode) in files.items():
            info = zipfile.ZipInfo(name)
            if mode is not None:
                info.external_attr = mode << 16
            zf.writestr(info, content)
    return buf.getvalue()


def _fake_update_files(app_text=b"NEW APP\n"):
    return {
        "acr/acr_app.py": (app_text, 0o100644),
        "acr/acr_updates.py": (b"NEW MODULE\n", 0o100644),
        "acr/run.sh": (b"#!/usr/bin/env bash\n# new run\n",
                       0o100755),
        "acr/icons/acr.svg": (b"<svg/>\n", 0o100644),
    }


def _sha256_text(data):
    return hashlib.sha256(data).hexdigest()


def _write(path, content, mode=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(content)
    if mode is not None:
        os.chmod(path, mode)


def _read(path):
    with open(path, "rb") as fh:
        return fh.read()


def _fake_install_dir(tmp, with_run_sh=True):
    """A stand-in installed tree: app, launcher, venv, logs."""
    install = os.path.join(tmp, "install")
    _write(os.path.join(install, "acr_app.py"), b"OLD APP\n")
    if with_run_sh:
        _write(os.path.join(install, "run.sh"),
               b"#!/usr/bin/env bash\n# old run\n", 0o755)
    _write(os.path.join(install, ".venv", "bin", "python"),
           b"OLD VENV\n")
    _write(os.path.join(install, "logs", "acr.log"), b"OLD LOG\n")
    return install


# ------------------------------------------------------- version parse


class VersionParseTests(unittest.TestCase):
    def test_parse_plain_and_v_prefixed(self):
        self.assertEqual(acr_updates.parse_version("0.12"),
                         (0, 12))
        self.assertEqual(acr_updates.parse_version("v0.12"),
                         (0, 12))
        self.assertEqual(acr_updates.parse_version("1.2.3"),
                         (1, 2, 3))

    def test_parse_rejects_non_versions(self):
        for bad in ("", "v", "abc", "v0.12-beta", "0.12a",
                    "1..2", None, 12):
            self.assertIsNone(acr_updates.parse_version(bad), bad)

    def test_is_newer(self):
        self.assertTrue(acr_updates.is_newer("0.13", "0.12"))
        self.assertTrue(acr_updates.is_newer("v0.13", "0.12"))
        self.assertFalse(acr_updates.is_newer("0.12", "0.12"))
        self.assertFalse(acr_updates.is_newer("0.11", "0.12"))
        self.assertTrue(acr_updates.is_newer("1.0", "0.99"))

    def test_shorter_tuples_pad_with_zeros(self):
        self.assertFalse(acr_updates.is_newer("0.12.0", "0.12"))
        self.assertFalse(acr_updates.is_newer("0.12", "0.12.0"))
        self.assertTrue(acr_updates.is_newer("0.12.1", "0.12"))

    def test_unparsable_is_never_newer(self):
        self.assertFalse(acr_updates.is_newer("v0.13-beta",
                                              "0.12"))
        self.assertFalse(acr_updates.is_newer("0.13", "dev"))
        # A dev tree newer than the latest release reads current.
        self.assertFalse(acr_updates.is_newer("0.12", "0.99"))


# ------------------------------------------------------------ the check


class CheckTests(unittest.TestCase):
    def _check(self, payload, current="0.12"):
        stub = _stub_urlopen(
            {"latest": _json_bytes(payload)})
        return acr_updates.check_for_update(
            current, urlopen=stub)

    def test_newer_release(self):
        result = self._check(_release_payload(
            "v0.13", "http://fake/acr-v0.13.zip",
            "http://fake/acr-v0.13.zip.sha256"))
        self.assertEqual(result["state"], "newer")
        self.assertEqual(result["latest"], "0.13")
        self.assertEqual(result["zip_url"],
                         "http://fake/acr-v0.13.zip")
        self.assertEqual(result["sha_url"],
                         "http://fake/acr-v0.13.zip.sha256")

    def test_current_release(self):
        result = self._check(_release_payload(
            "v0.12", "http://fake/acr-v0.12.zip",
            "http://fake/acr-v0.12.zip.sha256"))
        self.assertEqual(result["state"], "current")
        self.assertEqual(result["latest"], "0.12")

    def test_dev_tree_newer_reads_current(self):
        result = self._check(_release_payload(
            "v0.13", "http://fake/acr-v0.13.zip",
            "http://fake/acr-v0.13.zip.sha256"),
            current="0.99")
        self.assertEqual(result["state"], "current")

    def test_unknown_on_network_failure(self):
        stub = _stub_urlopen(
            {"latest": URLError("network is down")})
        result = acr_updates.check_for_update(
            "0.12", urlopen=stub)
        self.assertEqual(result["state"], "unknown")
        self.assertIn("GitHub", result["reason"])

    def test_unknown_on_bad_json(self):
        stub = _stub_urlopen({"latest": b"not json {"})
        result = acr_updates.check_for_update(
            "0.12", urlopen=stub)
        self.assertEqual(result["state"], "unknown")

    def test_unknown_on_missing_tag(self):
        result = self._check({"assets": []})
        self.assertEqual(result["state"], "unknown")
        self.assertIn("version", result["reason"])

    def test_unknown_on_bad_tag(self):
        result = self._check(_release_payload(
            "v0.13-beta", "http://fake/x.zip",
            "http://fake/x.zip.sha256"))
        self.assertEqual(result["state"], "unknown")

    def test_unknown_on_no_download_file(self):
        result = self._check({"tag_name": "v0.13",
                              "assets": []})
        self.assertEqual(result["state"], "unknown")
        self.assertIn("download", result["reason"])

    def test_missing_sidecar_is_newer_not_unknown(self):
        # The check reports the release; the download step is
        # what refuses a release with no checksum file.
        result = self._check(_release_payload(
            "v0.13", "http://fake/acr-v0.13.zip", None))
        self.assertEqual(result["state"], "newer")
        self.assertIsNone(result["sha_url"])


# ------------------------------------------------- fake release server


class _FakeReleaseHandler(BaseHTTPRequestHandler):
    """A fake GitHub releases endpoint. Class attrs are the
    script: tag, zip bytes, sidecar text, whether the sidecar
    asset is listed."""

    tag = "v0.13"
    zip_bytes = b""
    sidecar = ""
    include_sidecar = True
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _base(self):
        return "http://127.0.0.1:%d" % self.server.server_address[1]

    def _send(self, body, ctype):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/releases/latest":
            assets = [{"name": "acr-v0.13.zip",
                       "browser_download_url":
                           self._base() + "/acr-v0.13.zip"}]
            if self.include_sidecar:
                assets.append(
                    {"name": "acr-v0.13.zip.sha256",
                     "browser_download_url":
                         self._base() + "/acr-v0.13.zip.sha256"})
            payload = {"tag_name": self.tag,
                       "html_url": self._base() + "/releases",
                       "assets": assets}
            self._send(_json_bytes(payload), "application/json")
        elif self.path == "/acr-v0.13.zip":
            self._send(self.zip_bytes, "application/zip")
        elif self.path == "/acr-v0.13.zip.sha256":
            self._send(self.sidecar.encode("utf-8"), "text/plain")
        else:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()


class FakeReleaseServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(
            ("127.0.0.1", 0), _FakeReleaseHandler)
        cls.base = "http://127.0.0.1:%d" % (
            cls.server.server_address[1])
        cls.thread = threading.Thread(
            target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=10)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.zip_bytes = _fake_zip_bytes(_fake_update_files())
        _FakeReleaseHandler.tag = "v0.13"
        _FakeReleaseHandler.zip_bytes = self.zip_bytes
        _FakeReleaseHandler.sidecar = "%s  acr-v0.13.zip\n" % (
            _sha256_text(self.zip_bytes))
        _FakeReleaseHandler.include_sidecar = True

    def tearDown(self):
        self.tmp.cleanup()

    def _latest_url(self):
        return self.base + "/releases/latest"

    def test_check_newer_via_server(self):
        result = acr_updates.check_for_update(
            "0.12", url=self._latest_url())
        self.assertEqual(result["state"], "newer")
        self.assertEqual(result["latest"], "0.13")
        self.assertTrue(result["zip_url"].startswith(self.base))
        self.assertTrue(result["sha_url"].startswith(self.base))

    def test_check_current_via_server(self):
        _FakeReleaseHandler.tag = "v0.12"
        result = acr_updates.check_for_update(
            "0.12", url=self._latest_url())
        self.assertEqual(result["state"], "current")

    def test_download_verifies(self):
        result = acr_updates.check_for_update(
            "0.12", url=self._latest_url())
        out = acr_updates.download_update(
            result, self.tmp.name)
        self.assertEqual(out["version"], "0.13")
        self.assertEqual(out["sha256"],
                         _sha256_text(self.zip_bytes))
        self.assertEqual(_read(out["zip_path"]), self.zip_bytes)

    def test_progress_reports_bytes(self):
        result = acr_updates.check_for_update(
            "0.12", url=self._latest_url())
        seen = []
        acr_updates.download_update(
            result, self.tmp.name,
            progress_cb=lambda d, t: seen.append((d, t)))
        self.assertTrue(seen)
        self.assertEqual(seen[-1][0], len(self.zip_bytes))
        self.assertEqual(seen[-1][1], len(self.zip_bytes))

    def test_tampered_zip_refused(self):
        _FakeReleaseHandler.zip_bytes = b"tampered bytes"
        result = acr_updates.check_for_update(
            "0.12", url=self._latest_url())
        with self.assertRaises(ACRUpdateError) as ctx:
            acr_updates.download_update(result, self.tmp.name)
        self.assertIn("checksum", str(ctx.exception))
        self.assertEqual(os.listdir(self.tmp.name), [])

    def test_missing_sidecar_refused(self):
        _FakeReleaseHandler.include_sidecar = False
        result = acr_updates.check_for_update(
            "0.12", url=self._latest_url())
        self.assertEqual(result["state"], "newer")
        with self.assertRaises(ACRUpdateError) as ctx:
            acr_updates.download_update(result, self.tmp.name)
        self.assertIn("checksum", str(ctx.exception))
        self.assertEqual(os.listdir(self.tmp.name), [])

    def test_cancelled_download_throws_partial_away(self):
        result = acr_updates.check_for_update(
            "0.12", url=self._latest_url())
        with self.assertRaises(UpdateCancelled):
            acr_updates.download_update(
                result, self.tmp.name, cancel=lambda: True)
        self.assertEqual(os.listdir(self.tmp.name), [])


# ------------------------------------------------ staging and apply


class StageApplyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.install = _fake_install_dir(self.tmp.name)
        self.staging = os.path.join(self.tmp.name, "staging")

    def tearDown(self):
        self.tmp.cleanup()

    def _stage(self, files=None):
        zip_bytes = _fake_zip_bytes(
            files if files is not None else _fake_update_files())
        zip_path = os.path.join(self.tmp.name, "acr-v0.13.zip")
        _write(zip_path, zip_bytes)
        return acr_updates.stage_update(
            zip_path, "0.13", _sha256_text(zip_bytes),
            staging=self.staging)

    def test_apply_swaps_files_and_preserves_venv_and_logs(self):
        self._stage()
        ok, version = acr_updates.apply_staged_update(
            self.install, self.staging)
        self.assertTrue(ok, version)
        self.assertEqual(version, "0.13")
        self.assertEqual(
            _read(os.path.join(self.install, "acr_app.py")),
            b"NEW APP\n")
        self.assertEqual(
            _read(os.path.join(self.install, "acr_updates.py")),
            b"NEW MODULE\n")
        self.assertEqual(
            _read(os.path.join(self.install, "icons", "acr.svg")),
            b"<svg/>\n")
        run_sh = os.path.join(self.install, "run.sh")
        self.assertEqual(_read(run_sh),
                         b"#!/usr/bin/env bash\n# new run\n")
        # The launcher stays executable after the swap.
        self.assertTrue(os.stat(run_sh).st_mode & 0o111)
        # The venv and the logs are untouched.
        self.assertEqual(
            _read(os.path.join(self.install, ".venv", "bin",
                               "python")),
            b"OLD VENV\n")
        self.assertEqual(
            _read(os.path.join(self.install, "logs", "acr.log")),
            b"OLD LOG\n")
        # Staging is gone once applied.
        self.assertFalse(os.path.exists(self.staging))

    def test_corrupt_staging_is_dropped_and_old_files_kept(self):
        self._stage()
        staged_zip = os.path.join(self.staging, "acr-v0.13.zip")
        _write(staged_zip, b"garbage, not the zip")
        ok, reason = acr_updates.apply_staged_update(
            self.install, self.staging)
        self.assertFalse(ok)
        self.assertIn("checksum", reason)
        self.assertFalse(os.path.exists(self.staging))
        self.assertEqual(
            _read(os.path.join(self.install, "acr_app.py")),
            b"OLD APP\n")

    def test_no_staging_is_not_an_error(self):
        ok, reason = acr_updates.apply_staged_update(
            self.install, self.staging)
        self.assertFalse(ok)
        self.assertIn("no update", reason)
        self.assertEqual(
            _read(os.path.join(self.install, "acr_app.py")),
            b"OLD APP\n")

    def test_damaged_manifest_is_dropped(self):
        os.makedirs(self.staging)
        _write(os.path.join(self.staging, "manifest.json"),
               b"{not json")
        ok, reason = acr_updates.apply_staged_update(
            self.install, self.staging)
        self.assertFalse(ok)
        self.assertIn("damaged", reason)
        self.assertFalse(os.path.exists(self.staging))

    def test_unsafe_entries_are_skipped_not_written(self):
        files = {
            "acr/../../evil.txt": (b"EVIL\n", None),
            "/etc/evil.txt": (b"EVIL\n", None),
            "acr/acr_app.py": (b"NEW APP\n", 0o100644),
        }
        self._stage(files)
        ok, version = acr_updates.apply_staged_update(
            self.install, self.staging)
        self.assertTrue(ok, version)
        self.assertEqual(
            _read(os.path.join(self.install, "acr_app.py")),
            b"NEW APP\n")
        self.assertFalse(os.path.exists(
            os.path.join(self.tmp.name, "evil.txt")))
        self.assertFalse(os.path.exists("/tmp/evil.txt"))

    def test_preserved_tops_are_never_written(self):
        files = {
            "acr/.venv/evil.py": (b"EVIL\n", None),
            "acr/logs/evil.log": (b"EVIL\n", None),
            "acr/acr_app.py": (b"NEW APP\n", 0o100644),
        }
        self._stage(files)
        ok, _version = acr_updates.apply_staged_update(
            self.install, self.staging)
        self.assertTrue(ok)
        self.assertFalse(os.path.exists(
            os.path.join(self.install, ".venv", "evil.py")))
        self.assertFalse(os.path.exists(
            os.path.join(self.install, "logs", "evil.log")))

    def test_staging_clears_an_older_staging(self):
        first = self._stage()
        self.assertTrue(os.path.exists(
            os.path.join(first, "acr-v0.13.zip")))
        zip_bytes = _fake_zip_bytes(_fake_update_files(b"V14\n"))
        zip_path = os.path.join(self.tmp.name, "acr-v0.14.zip")
        _write(zip_path, zip_bytes)
        second = acr_updates.stage_update(
            zip_path, "0.14", _sha256_text(zip_bytes),
            staging=self.staging)
        self.assertEqual(first, second)
        with open(os.path.join(second, "manifest.json"),
                  encoding="utf-8") as fh:
            manifest = json.load(fh)
        self.assertEqual(manifest["version"], "0.14")
        self.assertFalse(os.path.exists(
            os.path.join(second, "acr-v0.13.zip")))

    def test_staging_dir_respects_xdg_data_home(self):
        with mock.patch.dict(os.environ,
                             {"XDG_DATA_HOME": "/xdg"},
                             clear=True):
            self.assertEqual(acr_updates.staging_dir(),
                             "/xdg/ACR/update-pending")
        with mock.patch.dict(os.environ,
                             {"HOME": "/home/test"},
                             clear=True):
            self.assertEqual(
                acr_updates.staging_dir(),
                "/home/test/.local/share/ACR/update-pending")


# ------------------------------------------------------- launcher apply


class LauncherApplyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.install = _fake_install_dir(self.tmp.name)
        self.staging = os.path.join(self.tmp.name, "staging")

    def tearDown(self):
        self.tmp.cleanup()

    def _stage(self):
        zip_bytes = _fake_zip_bytes(_fake_update_files())
        zip_path = os.path.join(self.tmp.name, "acr-v0.13.zip")
        _write(zip_path, zip_bytes)
        return acr_updates.stage_update(
            zip_path, "0.13", _sha256_text(zip_bytes),
            staging=self.staging)

    def test_launcher_applies_a_staged_update(self):
        self._stage()
        with mock.patch.object(acr_updates, "staging_dir",
                               return_value=self.staging):
            applied = acr_launch.maybe_apply_staged_update(
                self.install)
        self.assertTrue(applied)
        self.assertEqual(
            _read(os.path.join(self.install, "acr_app.py")),
            b"NEW APP\n")
        self.assertFalse(os.path.exists(self.staging))

    def test_launcher_with_no_staging_does_nothing(self):
        with mock.patch.object(acr_updates, "staging_dir",
                               return_value=self.staging):
            applied = acr_launch.maybe_apply_staged_update(
                self.install)
        self.assertFalse(applied)
        self.assertEqual(
            _read(os.path.join(self.install, "acr_app.py")),
            b"OLD APP\n")

    def test_launcher_drops_corrupt_staging(self):
        self._stage()
        _write(os.path.join(self.staging, "acr-v0.13.zip"),
               b"garbage, not the zip")
        with mock.patch.object(acr_updates, "staging_dir",
                               return_value=self.staging):
            with redirect_stderr(io.StringIO()):
                applied = acr_launch.maybe_apply_staged_update(
                    self.install)
        self.assertFalse(applied)
        self.assertFalse(os.path.exists(self.staging))
        self.assertEqual(
            _read(os.path.join(self.install, "acr_app.py")),
            b"OLD APP\n")


# ------------------------------------------------------ updater script


class UpdaterScriptTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.install = os.path.join(self.tmp.name, "install")
        os.makedirs(self.install)
        # The updater script imports acr_updates from the install
        # dir: the fake tree carries the real module, like a real
        # v0.12 install does.
        shutil.copy2(os.path.join(ROOT, "acr_updates.py"),
                     os.path.join(self.install, "acr_updates.py"))
        self.staging = os.path.join(self.tmp.name, "staging")
        self.marker = os.path.join(self.tmp.name, "relaunched")
        self._sleepers = []

    def tearDown(self):
        for proc in self._sleepers:
            try:
                proc.terminate()
            except Exception:
                pass
        self.tmp.cleanup()

    def _dead_pid(self):
        proc = subprocess.Popen([sys.executable, "-c", "pass"])
        proc.wait()
        return proc.pid

    def _stage(self, files):
        zip_bytes = _fake_zip_bytes(files)
        zip_path = os.path.join(self.tmp.name, "acr-v0.13.zip")
        _write(zip_path, zip_bytes)
        return acr_updates.stage_update(
            zip_path, "0.13", _sha256_text(zip_bytes),
            staging=self.staging)

    def _run_script(self, pid, timeout=30):
        script = acr_updates.write_updater_script(self.staging)
        return subprocess.run(
            [sys.executable, script, str(pid), self.install,
             self.staging],
            capture_output=True, text=True, timeout=timeout)

    def test_script_applies_and_relaunches(self):
        # The new run.sh writes the marker; the old one writes a
        # different one, so the marker proves the NEW tree ran.
        old_marker = os.path.join(self.tmp.name, "old-ran")
        _write(os.path.join(self.install, "run.sh"),
               ("#!/usr/bin/env bash\necho old > %s\n"
                % old_marker).encode("utf-8"), 0o755)
        _write(os.path.join(self.install, "acr_app.py"),
               b"OLD APP\n")
        files = dict(_fake_update_files())
        files["acr/run.sh"] = (
            ("#!/usr/bin/env bash\necho new > %s\n"
             % self.marker).encode("utf-8"), 0o100755)
        self._stage(files)
        proc = self._run_script(self._dead_pid())
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertTrue(os.path.exists(self.marker))
        self.assertFalse(os.path.exists(old_marker))
        self.assertEqual(
            _read(os.path.join(self.install, "acr_app.py")),
            b"NEW APP\n")
        self.assertFalse(os.path.exists(self.staging))

    def test_script_waits_for_a_live_app(self):
        # The wait is 30 s in the real script; the test copy
        # waits half a second, so this stays fast.
        _write(os.path.join(self.install, "run.sh"),
               b"#!/usr/bin/env bash\n", 0o755)
        _write(os.path.join(self.install, "acr_app.py"),
               b"OLD APP\n")
        self._stage(_fake_update_files())
        script = acr_updates.write_updater_script(self.staging)
        text = _read(script).decode("utf-8").replace(
            "WAIT_TIMEOUT_S = 30.0", "WAIT_TIMEOUT_S = 0.5")
        _write(script, text.encode("utf-8"))
        sleeper = subprocess.Popen(["sleep", "20"])
        self._sleepers.append(sleeper)
        proc = subprocess.run(
            [sys.executable, script, str(sleeper.pid),
             self.install, self.staging],
            capture_output=True, text=True, timeout=30)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        # Nothing applied, nothing relaunched, staging kept for
        # the next launch.
        self.assertTrue(os.path.exists(self.staging))
        self.assertEqual(
            _read(os.path.join(self.install, "acr_app.py")),
            b"OLD APP\n")
        self.assertFalse(os.path.exists(self.marker))

    def test_script_falls_back_without_run_sh(self):
        # No run.sh: the script execs the app directly.
        _write(os.path.join(self.install, "acr_app.py"),
               b"OLD APP\n")
        files = dict(_fake_update_files())
        files["acr/acr_app.py"] = (
            ("import os\nopen(%r, 'w').write('ran')\n"
             % self.marker).encode("utf-8"), 0o100644)
        del files["acr/run.sh"]
        self._stage(files)
        proc = self._run_script(self._dead_pid())
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertTrue(os.path.exists(self.marker))
        self.assertFalse(os.path.exists(self.staging))


# ------------------------------------------------------------------ GUI


class WindowCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.window = MainWindow(log_dir=self.tmp.name)
        self.window.show()
        # The update seams point at the temp dir, never at the
        # real home or the real tree.
        self.window._update_staging_dir = os.path.join(
            self.tmp.name, "staging")
        self.window._update_install_dir = _fake_install_dir(
            self.tmp.name)

    def tearDown(self):
        self.window.close()
        self.window.deleteLater()
        self.tmp.cleanup()

    def _wait_update_idle(self, timeout=30.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.window._update_thread is None:
                return True
            QTest.qWait(50)
        return False

    def _release_stub(self, tag="v0.13", with_sidecar=True):
        """A urlopen stub serving a release plus its zip and
        sidecar, all from memory."""
        zip_bytes = _fake_zip_bytes(_fake_update_files())
        sidecar = "%s  acr-%s.zip\n" % (
            _sha256_text(zip_bytes), tag)
        payload = _release_payload(
            tag, "http://fake/acr-%s.zip" % tag,
            "http://fake/acr-%s.zip.sha256" % tag
            if with_sidecar else None)
        return _stub_urlopen({
            "latest": _json_bytes(payload),
            ".zip.sha256": sidecar.encode("utf-8"),
            ".zip": zip_bytes,
        })


class HelpMenuTests(WindowCase):
    def test_update_actions_are_in_the_help_menu(self):
        self.assertEqual(
            self.window.action_check_updates.text(),
            "Check for updates")
        self.assertEqual(
            self.window.action_update_on_startup.text(),
            "Check for updates on startup")
        self.assertTrue(
            self.window.action_update_on_startup.isCheckable())

    def test_startup_toggle_defaults_on(self):
        from PySide6.QtCore import QSettings
        settings = QSettings("BrokeNBored", "ACR")
        settings.remove("check_updates_on_startup")
        self.addCleanup(
            settings.remove, "check_updates_on_startup")
        window2 = MainWindow(log_dir=self.tmp.name)
        try:
            self.assertTrue(
                window2.action_update_on_startup.isChecked())
            self.assertTrue(acr_app.update_check_on_startup())
        finally:
            window2.close()
            window2.deleteLater()

    def test_startup_toggle_persists(self):
        from PySide6.QtCore import QSettings
        settings = QSettings("BrokeNBored", "ACR")
        self.addCleanup(
            settings.remove, "check_updates_on_startup")
        self.window.action_update_on_startup.setChecked(False)
        QTest.qWait(50)
        self.assertFalse(acr_app.update_check_on_startup())
        self.window.action_update_on_startup.setChecked(True)
        QTest.qWait(50)
        self.assertTrue(acr_app.update_check_on_startup())


class ManualCheckTests(WindowCase):
    def test_manual_check_current_tells_the_user(self):
        self.window._update_urlopen = self._release_stub("v0.12")
        seen = []
        with mock.patch.object(
                QMessageBox, "information",
                side_effect=lambda *a: seen.append(a[2])):
            self.window.on_check_updates()
            self.assertTrue(self._wait_update_idle())
        self.assertEqual(len(seen), 1)
        self.assertIn("up to date", seen[0])
        self.assertIn("up to date",
                      self.window.status_label.text())

    def test_manual_check_unknown_says_why(self):
        self.window._update_urlopen = _stub_urlopen(
            {"latest": URLError("network is down")})
        seen = []
        with mock.patch.object(
                QMessageBox, "warning",
                side_effect=lambda *a: seen.append(a[2])):
            self.window.on_check_updates()
            self.assertTrue(self._wait_update_idle())
        self.assertEqual(len(seen), 1)
        self.assertIn("GitHub", seen[0])

    def test_manual_check_newer_and_later_downloads_nothing(self):
        self.window._update_urlopen = self._release_stub("v0.13")
        asked = []
        with mock.patch.object(
                QMessageBox, "question",
                side_effect=lambda *a: asked.append(a[2])
                or QMessageBox.No):
            self.window.on_check_updates()
            self.assertTrue(self._wait_update_idle())
        self.assertEqual(len(asked), 1)
        self.assertIn("v0.13", asked[0])
        self.assertIn("Download and install it now", asked[0])
        self.assertFalse(os.path.exists(
            self.window._update_staging_dir))
        self.assertIsNone(self.window._update_thread)

    def test_manual_check_twice_at_once_is_one_check(self):
        self.window._update_urlopen = self._release_stub("v0.12")
        with mock.patch.object(QMessageBox, "information"):
            self.assertTrue(
                self.window._start_update_check(manual=True))
            self.assertFalse(
                self.window._start_update_check(manual=True))
            self.assertTrue(self._wait_update_idle())


class DownloadFlowTests(WindowCase):
    def test_full_flow_restart_later_leaves_staging(self):
        self.window._update_urlopen = self._release_stub("v0.13")
        answers = [QMessageBox.Yes, QMessageBox.No]
        asked = []
        with mock.patch.object(
                QMessageBox, "question",
                side_effect=lambda *a: asked.append(a[2])
                or answers.pop(0)):
            self.window.on_check_updates()
            self.assertTrue(self._wait_update_idle())
            QTest.qWait(200)
        # The offer, then the restart question.
        self.assertEqual(len(asked), 2)
        self.assertIn("v0.13", asked[0])
        self.assertIn("Restart now to complete the update",
                      asked[1])
        staging = self.window._update_staging_dir
        self.assertTrue(os.path.exists(
            os.path.join(staging, "acr-v0.13.zip")))
        with open(os.path.join(staging, "manifest.json"),
                  encoding="utf-8") as fh:
            manifest = json.load(fh)
        self.assertEqual(manifest["version"], "0.13")
        self.assertIn("next time ACR starts",
                      self.window.status_label.text())
        # Nothing swapped yet: the restart is what applies it.
        self.assertEqual(
            _read(os.path.join(self.window._update_install_dir,
                               "acr_app.py")),
            b"OLD APP\n")

    def test_restart_yes_writes_script_starts_it_and_quits(self):
        self.window._update_urlopen = self._release_stub("v0.13")
        answers = [QMessageBox.Yes, QMessageBox.Yes]
        with mock.patch.object(
                QMessageBox, "question",
                side_effect=lambda *a: answers.pop(0)), \
                mock.patch.object(
                    QProcess, "startDetached") as detached, \
                mock.patch.object(
                    QApplication, "quit") as quit_mock:
            self.window.on_check_updates()
            self.assertTrue(self._wait_update_idle())
            QTest.qWait(200)
        self.assertEqual(detached.call_count, 1)
        program, args = detached.call_args[0]
        self.assertEqual(program, sys.executable)
        self.assertTrue(args[0].endswith("update.py"))
        self.assertTrue(os.path.exists(args[0]))
        self.assertEqual(args[1], str(os.getpid()))
        self.assertEqual(args[2],
                         self.window._update_install_dir)
        self.assertEqual(args[3],
                         self.window._update_staging_dir)
        self.assertEqual(quit_mock.call_count, 1)

    def test_download_refused_on_bad_checksum(self):
        zip_bytes = _fake_zip_bytes(_fake_update_files())
        # The sidecar claims a different file's hash.
        sidecar = "%s  acr-v0.13.zip\n" % _sha256_text(b"other")
        payload = _release_payload(
            "v0.13", "http://fake/acr-v0.13.zip",
            "http://fake/acr-v0.13.zip.sha256")
        self.window._update_urlopen = _stub_urlopen({
            "latest": _json_bytes(payload),
            ".zip.sha256": sidecar.encode("utf-8"),
            ".zip": zip_bytes,
        })
        warned = []
        with mock.patch.object(
                QMessageBox, "question",
                return_value=QMessageBox.Yes), \
                mock.patch.object(
                    QMessageBox, "warning",
                    side_effect=lambda *a: warned.append(a[2])):
            self.window.on_check_updates()
            self.assertTrue(self._wait_update_idle())
            QTest.qWait(200)
        self.assertEqual(len(warned), 1)
        self.assertIn("checksum", warned[0])
        self.assertFalse(os.path.exists(
            self.window._update_staging_dir))

    def test_download_cancel_button_stops_the_download(self):
        class _SlowResponse(_StubResponse):
            def read(self, size=-1):
                if self._pos == 0:
                    time.sleep(2.0)
                return super().read(size)

        zip_bytes = _fake_zip_bytes(_fake_update_files())
        sidecar = "%s  acr-v0.13.zip\n" % _sha256_text(zip_bytes)
        payload = _release_payload(
            "v0.13", "http://fake/acr-v0.13.zip",
            "http://fake/acr-v0.13.zip.sha256")

        def _slow_open(url, timeout=None):
            if "latest" in url:
                return _StubResponse(_json_bytes(payload))
            if url.endswith(".sha256"):
                return _StubResponse(sidecar.encode("utf-8"))
            return _SlowResponse(
                zip_bytes, {"Content-Length": str(len(zip_bytes))})

        self.window._update_urlopen = _slow_open
        # Belt and braces: if the cancel missed, the restart
        # path must not fire for real in this test.
        with mock.patch.object(
                QMessageBox, "question",
                return_value=QMessageBox.Yes), \
                mock.patch.object(
                    QProcess, "startDetached") as detached, \
                mock.patch.object(
                    QApplication, "quit") as quit_mock:
            self.window.on_check_updates()
            deadline = time.time() + 10.0
            dialog = None
            while time.time() < deadline:
                dialog = self.window._update_dialog
                if dialog is not None and dialog.isVisible():
                    break
                QTest.qWait(50)
            self.assertIsNotNone(dialog)
            self.assertTrue(dialog.isVisible())
            cancel = dialog.findChild(QPushButton)
            self.assertIsNotNone(cancel)
            QTest.mouseClick(cancel, Qt.LeftButton)
            self.assertTrue(self._wait_update_idle())
            QTest.qWait(200)
        detached.assert_not_called()
        quit_mock.assert_not_called()
        self.assertIn("cancelled",
                      self.window.status_label.text())
        self.assertFalse(os.path.exists(
            self.window._update_staging_dir))


class StartupCheckTests(WindowCase):
    def test_startup_check_silent_when_current(self):
        self.window._update_urlopen = self._release_stub("v0.12")
        with mock.patch.object(
                QMessageBox, "information") as info, \
                mock.patch.object(
                    QMessageBox, "warning") as warn, \
                mock.patch.object(
                    QMessageBox, "question") as ask:
            self.window.maybe_start_update_check()
            self.assertTrue(self._wait_update_idle())
        info.assert_not_called()
        warn.assert_not_called()
        ask.assert_not_called()
        self.assertEqual(self.window.status_label.text(),
                         "Ready.")

    def test_startup_check_silent_when_unknown(self):
        self.window._update_urlopen = _stub_urlopen(
            {"latest": URLError("network is down")})
        with mock.patch.object(
                QMessageBox, "information") as info, \
                mock.patch.object(
                    QMessageBox, "warning") as warn, \
                mock.patch.object(
                    QMessageBox, "question") as ask:
            self.window.maybe_start_update_check()
            self.assertTrue(self._wait_update_idle())
        info.assert_not_called()
        warn.assert_not_called()
        ask.assert_not_called()
        self.assertEqual(self.window.status_label.text(),
                         "Ready.")

    def test_startup_check_runs_once_per_launch(self):
        self.window._update_urlopen = self._release_stub("v0.12")
        with mock.patch.object(QMessageBox, "information"):
            self.window.maybe_start_update_check()
            self.assertTrue(self._wait_update_idle())
            # A second call starts nothing: the flag is set.
            self.window.maybe_start_update_check()
            QTest.qWait(100)
            self.assertIsNone(self.window._update_thread)

    def test_startup_check_off_starts_nothing(self):
        from PySide6.QtCore import QSettings
        settings = QSettings("BrokeNBored", "ACR")
        settings.setValue("check_updates_on_startup", False)
        self.addCleanup(
            settings.remove, "check_updates_on_startup")
        window2 = MainWindow(log_dir=self.tmp.name)
        try:
            window2._update_urlopen = self._release_stub("v0.13")
            with mock.patch.object(QMessageBox, "question") as ask:
                window2.maybe_start_update_check()
                QTest.qWait(200)
            ask.assert_not_called()
            self.assertIsNone(window2._update_thread)
        finally:
            window2.close()
            window2.deleteLater()


# ------------------------------------------------------------ the build


class BuildTests(unittest.TestCase):
    def test_zip_list_includes_update_files(self):
        import build_zip
        self.assertIn("acr_updates.py", build_zip.PLAIN_FILES)
        self.assertIn("tests/test_v12.py", build_zip.PLAIN_FILES)

    def test_build_writes_a_matching_sidecar(self):
        tmp = tempfile.mkdtemp()
        try:
            clone = os.path.join(tmp, "acr-clone")
            shutil.copytree(
                ROOT, clone,
                ignore=shutil.ignore_patterns(
                    "__pycache__", "*.zip", "*.sha256", "logs"))
            proc = subprocess.run(
                [sys.executable, "build_zip.py"],
                cwd=clone, capture_output=True, text=True,
                timeout=180)
            self.assertEqual(proc.returncode, 0, proc.stdout
                             + proc.stderr)
            zip_path = os.path.join(clone, "acr-v0.12.zip")
            sidecar_path = zip_path + ".sha256"
            self.assertTrue(os.path.exists(sidecar_path))
            with open(sidecar_path, encoding="utf-8") as fh:
                sidecar = fh.read()
            digest = hashlib.sha256()
            with open(zip_path, "rb") as fh:
                digest.update(fh.read())
            self.assertEqual(
                sidecar, "%s  acr-v0.12.zip\n" % digest.hexdigest())
            with zipfile.ZipFile(zip_path) as zf:
                names = zf.namelist()
            self.assertIn("acr/acr_updates.py", names)
            self.assertIn("acr/tests/test_v12.py", names)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class StaticTests(unittest.TestCase):
    def test_acr_updates_has_no_qt_import(self):
        # The launcher and the updater script import it without
        # the interface library; it must stay stdlib-only.
        with open(os.path.join(ROOT, "acr_updates.py"),
                  encoding="utf-8") as fh:
            text = fh.read()
        self.assertNotIn("PySide6", text)
        self.assertNotIn("PyQt", text)


if __name__ == "__main__":
    unittest.main()
