"""v0.9 design-rules tests: menu bar, How to Use window, single
instance, busy light, worker-threaded local calls, guided model
picker, visible-text sweep, RAM-adaptive snapshots, geometry
memory, and the feel layer (dark theme, bento results, resonance
moment, completion moment, press dip, reduced motion)."""

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import uuid
from http.server import ThreadingHTTPServer
from unittest import mock

sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtCore import QEasingCurve, QSettings, Qt, QTimer  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox,  # noqa: E402
                               QFrame, QLabel, QMessageBox,
                               QPushButton, QTableWidget, QTextBrowser,
                               QTextEdit, QWidget)

from acr_app import (ACCENT_GOLD, BUSY_OFF, DARK_STYLESHEET,  # noqa: E402
                     FLAG_TINT, MainWindow, apply_theme,
                     reduced_motion)
from acr_engine import Engine, default_snapshot_limit  # noqa: E402
from local_encoder import model_display_text  # noqa: E402
from tests.test_analyze import OFF, THEME, FakeSlowHandler  # noqa: E402
from tests.test_local_encoder import FakeLocalHandler  # noqa: E402

APP = QApplication.instance()
if APP is None:
    APP = QApplication(sys.argv)


def _clear_settings():
    QSettings("BrokeNBored", "ACR").clear()


def _pump_until(condition, timeout=10.0, step=0.05):
    """Pump events until condition() is true or the timeout hits.
    Uses time.sleep + processEvents, NOT QTest.qWait: in this
    PySide6 build qWait starves worker threads of the GIL."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if condition():
            return True
        time.sleep(step)
        QApplication.processEvents()
    return condition()


class WindowCase(unittest.TestCase):
    def setUp(self):
        _clear_settings()
        self.tmp = tempfile.TemporaryDirectory()
        self.window = MainWindow(log_dir=self.tmp.name)
        self.window.show()

    def tearDown(self):
        self.window.close()
        self.window.deleteLater()
        self.tmp.cleanup()
        _clear_settings()

    def _step_text(self, text):
        self.window.input_edit.setPlainText(text)
        QTest.mouseClick(self.window.step_button, Qt.LeftButton)

    def _run_batch(self, paths, source_note="fixture.txt",
                   timeout=25.0):
        self.window.start_batch(paths, source_note)
        return _pump_until(
            lambda: self.window._batch_thread is None,
            timeout=timeout)

    def _theme_fixture(self, name="fixture.txt"):
        path = os.path.join(self.tmp.name, name)
        with open(path, "w", encoding="utf-8") as fh:
            for i in range(16):
                fh.write(THEME[i % 4] + "\n")
            for line in OFF:
                fh.write(line + "\n")
        return path


# ---------------------------------------------------------------- menu


class MenuTests(WindowCase):
    def test_menu_bar_has_file_view_help(self):
        names = [action.text()
                 for action in self.window.menuBar().actions()]
        self.assertEqual(names, ["File", "View", "Help"])

    def test_file_menu_actions(self):
        file_menu = self.window.menuBar().actions()[0].menu()
        names = [action.text() for action in file_menu.actions()
                 if not action.isSeparator()]
        self.assertEqual(names, [
            "Save session", "Load session", "Export trace JSON",
            "Export trace CSV", "Export session",
            "Generate debug report", "Reset", "Quit"])

    def test_view_menu_opens_every_window(self):
        pairs = [
            (self.window.action_show_controls,
             self.window.controls_window),
            (self.window.action_show_trace, self.window.trace_window),
            (self.window.action_show_views, self.window.views_window),
            (self.window.action_show_analyze,
             self.window.analyze_window),
        ]
        for action, sub in pairs:
            action.trigger()
            QTest.qWait(50)
            self.assertTrue(sub.isVisible(),
                            "%s did not open" % sub.windowTitle())
            sub.close()
            QTest.qWait(50)
            self.assertFalse(sub.isVisible())

    def test_help_menu_actions(self):
        help_menu = self.window.menuBar().actions()[2].menu()
        names = [action.text() for action in help_menu.actions()]
        self.assertEqual(names, ["How to Use", "About"])
        self.window.action_how_to_use.trigger()
        QTest.qWait(50)
        self.assertTrue(self.window.howto_window.isVisible())


# ------------------------------------------------------------ how to use


class HowToTests(WindowCase):
    SECTIONS = ["What ACR is for", "How to use it",
                "Reading the numbers", "The windows",
                "Analyze a file", "STELLA data packs",
                "Synthesis head", "Sessions and controls"]

    def test_window_opens_with_all_sections(self):
        self.window.show_howto_window()
        QTest.qWait(50)
        howto = self.window.howto_window
        self.assertTrue(howto.isVisible())
        self.assertEqual(howto.sections, self.SECTIONS)
        self.assertEqual(howto.index_list.count(), len(self.SECTIONS))

    def test_content_matches_the_guide_file(self):
        guide_path = os.path.join(
            os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))),
            "HOW-TO-USE.md")
        with open(guide_path, encoding="utf-8") as fh:
            guide = fh.read()
        self.window.show_howto_window()
        QTest.qWait(50)
        text = self.window.howto_window.browser.toPlainText()
        self.assertIn("What ACR is for", text)
        self.assertIn("Analyze a file", text)
        self.assertIn("STELLA data packs", text)
        # The window reads the file: a body line from the file
        # is in the window.
        body = [line for line in guide.splitlines()
                if line and not line.startswith("#")][0]
        self.assertIn(body.strip(), text)

    def test_window_reopens_after_close(self):
        self.window.show_howto_window()
        QTest.qWait(50)
        howto = self.window.howto_window
        self.assertTrue(howto.isVisible())
        howto.close()
        QTest.qWait(50)
        self.assertFalse(howto.isVisible())
        self.window.show_howto_window()
        QTest.qWait(50)
        self.assertTrue(howto.isVisible())
        self.assertEqual(howto.sections, self.SECTIONS)

    def test_clicking_a_section_jumps_to_it(self):
        # Regression: _on_section_picked used cursor.Start, which
        # does not exist on the instance in this PySide6 build
        # (QTextCursor.Start is the class enum), so picking a
        # section raised AttributeError and jumped nowhere.
        self.window.show_howto_window()
        QTest.qWait(50)
        howto = self.window.howto_window
        row = self.SECTIONS.index("Analyze a file")
        howto.index_list.setCurrentRow(row)
        QTest.qWait(50)
        # find() selects the match: the cursor sits on the
        # heading the user picked.
        self.assertEqual(
            howto.browser.textCursor().selectedText(),
            "Analyze a file")

    def test_guide_file_ships_in_the_zip(self):
        import build_zip
        self.assertIn("HOW-TO-USE.md", build_zip.PLAIN_FILES)
        guide_path = os.path.join(
            os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))),
            "HOW-TO-USE.md")
        self.assertTrue(os.path.exists(guide_path))


# -------------------------------------------------------- single instance


class SingleInstanceTests(unittest.TestCase):
    """Two REAL processes: the second exits 0 fast and the first
    raises; a stale lock is taken over after kill -9."""

    def setUp(self):
        self.build_dir = os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))
        self.app_path = os.path.join(self.build_dir, "acr_app.py")
        self.log_path = os.path.join(self.build_dir, "logs",
                                     "acr.log")
        self.procs = []
        self.names = []

    def tearDown(self):
        for proc in self.procs:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
        for name in self.names:
            lock = os.path.join(tempfile.gettempdir(),
                                "acr-%s.lock" % name)
            try:
                os.remove(lock)
            except OSError:
                pass

    def _launch(self, name):
        env = dict(os.environ)
        env["ACR_INSTANCE_NAME"] = name
        env["QT_QPA_PLATFORM"] = "offscreen"
        env["TMPDIR"] = "/tmp"
        proc = subprocess.Popen(
            [sys.executable, self.app_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            env=env, cwd=self.build_dir)
        self.procs.append(proc)
        self.names.append(name)
        return proc

    def _wait_for_lock(self, name, timeout=20.0):
        lock = os.path.join(tempfile.gettempdir(),
                            "acr-%s.lock" % name)
        deadline = time.time() + timeout
        while time.time() < deadline:
            if os.path.exists(lock):
                return True
            time.sleep(0.1)
        return os.path.exists(lock)

    def _log_size(self):
        try:
            return os.path.getsize(self.log_path)
        except OSError:
            return 0

    def _log_since(self, offset):
        try:
            with open(self.log_path, encoding="utf-8",
                      errors="replace") as fh:
                fh.seek(offset)
                return fh.read()
        except OSError:
            return ""

    def test_second_launch_exits_zero_and_first_raises(self):
        name = "test-" + uuid.uuid4().hex[:12]
        first = self._launch(name)
        self.assertTrue(self._wait_for_lock(name))
        time.sleep(0.5)
        offset = self._log_size()
        second = self._launch(name)
        self.assertEqual(second.wait(timeout=15), 0)
        self.assertIsNone(first.poll())
        # No QApplication in this test process: plain sleep loop.
        found = False
        deadline = time.time() + 5.0
        while time.time() < deadline:
            if "single-instance raise requested" in \
                    self._log_since(offset):
                found = True
                break
            time.sleep(0.1)
        self.assertTrue(
            found,
            "the first instance never logged the raise request")

    def test_stale_lock_takeover_after_kill(self):
        name = "test-" + uuid.uuid4().hex[:12]
        first = self._launch(name)
        self.assertTrue(self._wait_for_lock(name))
        first.kill()
        first.wait(timeout=5)
        # The stale window is 1000 ms; wait past it.
        time.sleep(1.5)
        second = self._launch(name)
        # A duplicate exits 0 in well under 2 s. A takeover
        # stays alive: it holds the lock and serves raises.
        time.sleep(2.0)
        self.assertIsNone(second.poll())
        third = self._launch(name)
        self.assertEqual(third.wait(timeout=15), 0)


# ------------------------------------------------ busy light + threads


class SlowHandler(FakeSlowHandler):
    """The slow fake service, slower: a 0.6 s embed leaves time
    to watch the busy light and the event loop."""
    embed_delay = 0.6


class LocalWindowCase(unittest.TestCase):
    def setUp(self):
        _clear_settings()
        self.tmp = tempfile.TemporaryDirectory()
        self.server = ThreadingHTTPServer(
            ("127.0.0.1", 0), SlowHandler)
        self.thread = threading.Thread(
            target=self.server.serve_forever, daemon=True)
        self.thread.start()
        base_url = ("http://127.0.0.1:%d"
                    % self.server.server_address[1])
        self.window = MainWindow(log_dir=self.tmp.name,
                                 local_base_url=base_url)
        self.window.show()

    def tearDown(self):
        self.window.close()
        self.window.deleteLater()
        self.server.shutdown()
        self.server.server_close()
        self.tmp.cleanup()
        _clear_settings()

    def _wait_idle(self, timeout=25.0):
        return _pump_until(
            lambda: (self.window._local_thread is None
                     and self.window._batch_thread is None),
            timeout=timeout)

    def _go_local(self):
        # v0.10: the offer and the activation run on worker
        # threads. Wait for the list to fill, then for the
        # activation to land.
        self.window.encoder_combo.setCurrentIndex(1)
        combo = self.window.model_combo
        # Wait for the list AND the thread to clear: the thread
        # reference outlives the fill by a pump cycle, and a
        # pick in that gap is refused.
        self.assertTrue(_pump_until(
            lambda: (combo.count() > 0
                     and self.window._local_thread is None),
            timeout=25.0),
            "model offer did not land")
        combo.setCurrentIndex(
            combo.findData("nomic-embed-text:latest"))
        # The encoder flips in the finished handler, before
        # the thread reference clears: wait for both, or the
        # next step is refused.
        self.assertTrue(_pump_until(
            lambda: (self.window.encoder_kind == "local"
                     and self.window._local_thread is None),
            timeout=25.0), "model did not activate")


class BusyLightTests(LocalWindowCase):
    def test_busy_light_lit_during_local_call(self):
        self._go_local()
        self.window.input_edit.setPlainText(
            "quantum particles stay connected")
        QTest.mouseClick(self.window.step_button, Qt.LeftButton)
        lit = _pump_until(
            lambda: ACCENT_GOLD
            in self.window.busy_light.styleSheet(),
            timeout=5.0)
        self.assertTrue(lit, "the busy light never lit")
        self.assertTrue(self._wait_idle())
        self.assertIn(BUSY_OFF,
                      self.window.busy_light.styleSheet())
        self.assertEqual(self.window.step_label.text(), "Step: 1")

    def test_local_call_runs_in_a_worker_thread(self):
        self._go_local()
        fired_while_busy = []
        timer = QTimer()
        timer.setSingleShot(True)
        timer.timeout.connect(
            lambda: fired_while_busy.append(
                self.window._local_thread is not None))
        self.window.input_edit.setPlainText(
            "quantum particles stay connected")
        QTest.mouseClick(self.window.step_button, Qt.LeftButton)
        timer.start(100)
        self.assertTrue(self._wait_idle())
        self.assertEqual(fired_while_busy, [True])

    def test_cancel_local_step(self):
        self._go_local()
        self.window.input_edit.setPlainText(
            "quantum particles stay connected")
        QTest.mouseClick(self.window.step_button, Qt.LeftButton)
        started = _pump_until(
            lambda: self.window._local_thread is not None,
            timeout=5.0)
        self.assertTrue(started)
        self.window.on_cancel_local()
        self.assertTrue(self._wait_idle())
        self.assertEqual(self.window.engine.step_count, 0)
        self.assertEqual(self.window.status_label.text(),
                         "Cancelled.")
        self.assertIn(BUSY_OFF,
                      self.window.busy_light.styleSheet())


# --------------------------------------------------------- model picker


class RichHandler(FakeLocalHandler):
    """A fake service whose /api/tags reports sizes and
    parameter counts, including a model too big for this PC."""
    models = ["nomic-embed-text:latest", "huge-model:70b"]

    def do_GET(self):
        if self.path == "/api/tags":
            self._send_json({"models": [
                {"name": "nomic-embed-text:latest",
                 "size": 274302686,
                 "details": {"parameter_size": "137M"}},
                {"name": "huge-model:70b",
                 "size": 41000000000,
                 "details": {"parameter_size": "70.6B"}},
            ]})
        else:
            self._send_json({"error": "not found"}, status=404)


class ModelPickerTests(unittest.TestCase):
    def setUp(self):
        _clear_settings()
        self.tmp = tempfile.TemporaryDirectory()
        self.server = ThreadingHTTPServer(
            ("127.0.0.1", 0), RichHandler)
        self.thread = threading.Thread(
            target=self.server.serve_forever, daemon=True)
        self.thread.start()
        base_url = ("http://127.0.0.1:%d"
                    % self.server.server_address[1])
        self.window = MainWindow(log_dir=self.tmp.name,
                                 local_base_url=base_url)
        self.window.show()

    def tearDown(self):
        self.window.close()
        self.window.deleteLater()
        self.server.shutdown()
        self.server.server_close()
        self.tmp.cleanup()
        _clear_settings()

    def test_picker_starts_unselected_with_fit_info(self):
        self.window.encoder_combo.setCurrentIndex(1)
        combo = self.window.model_combo
        # v0.10: the offer runs on a worker thread.
        self.assertTrue(_pump_until(
            lambda: combo.count() == 2, timeout=25.0),
            "model offer did not land")
        self.assertEqual(combo.count(), 2)
        self.assertEqual(combo.currentIndex(), -1)
        self.assertEqual(combo.placeholderText(), "Pick a model")
        embed_row = combo.findData("nomic-embed-text:latest")
        huge_row = combo.findData("huge-model:70b")
        self.assertGreaterEqual(embed_row, 0)
        self.assertGreaterEqual(huge_row, 0)
        embed_line = combo.itemText(embed_row)
        huge_line = combo.itemText(huge_row)
        self.assertIn("embedding model", embed_line)
        self.assertIn("137M parameters", embed_line)
        self.assertIn("261 MB", embed_line)
        self.assertIn("fits this PC", embed_line)
        self.assertIn("text model", huge_line)
        self.assertIn("70.6B parameters", huge_line)
        self.assertIn("38.2 GB", huge_line)
        self.assertIn("too big for this PC", huge_line)
        # The guided lines carry no em-dash or bare ampersand.
        for line in (embed_line, huge_line):
            self.assertNotIn("\u2014", line)
            self.assertNotIn("&", line)

    def test_model_display_text_names_only(self):
        # A service that reports names only still gets a useful
        # line: name and kind, no invented numbers.
        line = model_display_text({"name": "plain:1b"}, 0)
        self.assertEqual(line, "plain:1b (text model)")


# --------------------------------------------------- visible-text sweep


class VisibleTextTests(WindowCase):
    def _collect_texts(self):
        texts = []
        windows = [self.window, self.window.controls_window,
                   self.window.trace_window,
                   self.window.views_window,
                   self.window.howto_window,
                   self.window.analyze_window]
        for window in windows:
            texts.append((window.windowTitle(), "window title"))
            for widget in window.findChildren(QWidget):
                if isinstance(widget, QLabel):
                    texts.append((widget.text(), "label"))
                elif isinstance(widget, QPushButton):
                    texts.append((widget.text(), "button"))
                elif isinstance(widget, QCheckBox):
                    texts.append((widget.text(), "checkbox"))
                elif isinstance(widget, QComboBox):
                    texts.append((widget.placeholderText(),
                                  "combo placeholder"))
                    for i in range(widget.count()):
                        texts.append((widget.itemText(i),
                                      "combo item"))
                elif isinstance(widget, QTextBrowser):
                    texts.append((widget.toPlainText(),
                                  "how-to browser"))
                elif isinstance(widget, QTextEdit):
                    texts.append((widget.toPlainText(),
                                  "text edit"))
            if window is self.window:
                for action in window.menuBar().actions():
                    texts.append((action.text(), "menu"))
                    menu = action.menu()
                    if menu is not None:
                        for item in menu.actions():
                            texts.append((item.text(), "menu item"))
            table = self.window.trace_table
            for col in range(table.columnCount()):
                header = table.horizontalHeaderItem(col)
                if header is not None:
                    texts.append((header.text(), "table header"))
        return texts

    def test_no_em_dash_or_bare_ampersand_in_visible_text(self):
        offenders = []
        for text, where in self._collect_texts():
            if "\u2014" in text:
                offenders.append("%s has an em-dash: %r"
                                 % (where, text))
            if "&" in text:
                offenders.append("%s has a bare &: %r"
                                 % (where, text))
        self.assertEqual(offenders, [])

    def test_dialog_texts_have_no_em_dash_or_ampersand(self):
        captured = []
        with mock.patch.object(
                QMessageBox, "about",
                side_effect=lambda *args: captured.append(args[2])):
            self.window.on_about()
        with mock.patch.object(
                QMessageBox, "question",
                return_value=QMessageBox.No,
                side_effect=lambda *args: captured.append(args[2])
                or QMessageBox.No):
            self.window.on_reset()
        self._step_text("quantum particles stay connected")
        self.window.trace_table.setCurrentCell(0, 0)
        with mock.patch.object(
                QMessageBox, "question",
                return_value=QMessageBox.No,
                side_effect=lambda *args: captured.append(args[2])
                or QMessageBox.No):
            self.window.on_revert()
        self.assertEqual(len(captured), 3)
        for text in captured:
            self.assertNotIn("\u2014", text)
            self.assertNotIn("&", text)


# ------------------------------------------------------- snapshot limits


class SnapshotLimitTests(unittest.TestCase):
    def _meminfo(self, kib):
        tmp = tempfile.NamedTemporaryFile(
            "w", delete=False, suffix=".txt")
        tmp.write("MemTotal:       %d kB\n" % kib)
        tmp.write("MemFree:        123456 kB\n")
        tmp.close()
        self.addCleanup(os.remove, tmp.name)
        return tmp.name

    def test_limit_adapts_to_ram(self):
        self.assertEqual(default_snapshot_limit(
            self._meminfo(8 * 1024 * 1024)), 200)
        self.assertEqual(default_snapshot_limit(
            self._meminfo(4 * 1024 * 1024)), 100)
        self.assertEqual(default_snapshot_limit(
            self._meminfo(16 * 1024 * 1024)), 400)
        self.assertEqual(default_snapshot_limit(
            self._meminfo(1 * 1024 * 1024)), 50)
        self.assertEqual(default_snapshot_limit(
            self._meminfo(64 * 1024 * 1024)), 1000)

    def test_limit_falls_back_when_meminfo_is_missing(self):
        self.assertEqual(default_snapshot_limit(
            "/nonexistent/meminfo"), 200)
        tmp = tempfile.NamedTemporaryFile(
            "w", delete=False, suffix=".txt")
        tmp.write("nothing useful here\n")
        tmp.close()
        self.addCleanup(os.remove, tmp.name)
        self.assertEqual(default_snapshot_limit(tmp.name), 200)

    def test_explicit_limit_still_wins(self):
        engine = Engine(snapshot_limit=3)
        self.assertEqual(engine.snapshot_limit, 3)

    def test_default_limit_matches_this_machine(self):
        engine = Engine()
        self.assertEqual(engine.snapshot_limit,
                         default_snapshot_limit())


# ------------------------------------------------------------- geometry


class SubWindowGeometryTests(unittest.TestCase):
    def setUp(self):
        _clear_settings()
        self.tmps = []
        self.windows = []

    def tearDown(self):
        for window in self.windows:
            window.close()
            window.deleteLater()
        for tmp in self.tmps:
            tmp.cleanup()
        _clear_settings()

    def _make_window(self):
        tmp = tempfile.TemporaryDirectory()
        self.tmps.append(tmp)
        window = MainWindow(log_dir=tmp.name)
        self.windows.append(window)
        window.show()
        return window

    def _wait_open_anim(self, sub):
        _pump_until(lambda: sub._open_anim is None, timeout=2.0)

    def test_controls_window_remembers_height(self):
        first = self._make_window()
        controls = first.controls_window
        controls.resize(800, 650)
        first.show_controls_window()
        self._wait_open_anim(controls)
        controls.close()
        QApplication.processEvents()
        saved = QSettings("BrokeNBored", "ACR").value(
            "controls_geometry")
        self.assertIsNotNone(saved)
        second = self._make_window()
        self.assertEqual(second.controls_window.height(), 650)

    def test_views_window_remembers_height(self):
        first = self._make_window()
        views = first.views_window
        views.resize(800, 620)
        first.show_views_window()
        self._wait_open_anim(views)
        views.close()
        QApplication.processEvents()
        saved = QSettings("BrokeNBored", "ACR").value(
            "views_geometry")
        self.assertIsNotNone(saved)
        second = self._make_window()
        self.assertEqual(second.views_window.height(), 620)


# ---------------------------------------------------------------- theme


class ThemeTests(WindowCase):
    def setUp(self):
        super().setUp()
        self._previous = QApplication.instance().styleSheet()
        apply_theme(QApplication.instance())

    def tearDown(self):
        QApplication.instance().setStyleSheet(self._previous)
        super().tearDown()

    def test_app_stylesheet_carries_the_identity(self):
        sheet = QApplication.instance().styleSheet()
        self.assertEqual(sheet, DARK_STYLESHEET)
        self.assertIn("#c9a227", sheet)
        self.assertIn("#1d1d23", sheet)
        self.assertIn("QFrame#tile", sheet)
        self.assertIn("QPushButton#primaryButton", sheet)

    def test_step_and_go_are_primary_buttons(self):
        self.assertEqual(self.window.step_button.objectName(),
                         "primaryButton")
        self.assertEqual(
            self.window.analyze_window.go_button.objectName(),
            "primaryButton")

    def test_subwindows_carry_the_theme_object_name(self):
        for sub in (self.window.controls_window,
                    self.window.trace_window,
                    self.window.views_window,
                    self.window.howto_window,
                    self.window.analyze_window):
            self.assertEqual(sub.objectName(), "subwindow")

    def test_analyze_tiles_carry_the_tile_object_name(self):
        tiles = [child for child in
                 self.window.analyze_window.findChildren(QFrame)
                 if child.objectName() == "tile"]
        self.assertEqual(len(tiles), 6)


# ---------------------------------------------------------------- bento


class BentoTests(WindowCase):
    def test_hero_tile_exists_before_any_run(self):
        analyze = self.window.analyze_window
        self.assertEqual(analyze.hero_title.text(),
                         "Nothing analyzed yet.")
        self.assertIn("Pick a file",
                      analyze.hero_theme.text())

    def test_hero_tile_fills_from_a_run(self):
        path = self._theme_fixture()
        self.assertTrue(self._run_batch([path]))
        analyze = self.window.analyze_window
        self.assertEqual(analyze.hero_title.text(),
                         "1 theme found in 24 steps in this "
                         "session")
        self.assertIn("Strongest theme",
                      analyze.hero_theme.text())
        self.assertIn("quantum particles stay connected",
                      analyze.hero_theme.text())
        self.assertIn("step 15", analyze.hero_theme.text())

    def test_tiles_hold_the_right_findings(self):
        path = self._theme_fixture()
        self.assertTrue(self._run_batch([path]))
        analyze = self.window.analyze_window
        self.assertEqual(analyze.themes_list.count(), 1)
        self.assertGreater(analyze.core_list.count(), 0)
        self.assertGreater(analyze.outlier_list.count(), 0)
        self.assertEqual(analyze.shift_list.count(), 1)
        self.assertEqual(len(analyze.chart.values), 24)
        # Every finding list lives inside a tile.
        for widget in (analyze.themes_list, analyze.core_list,
                       analyze.outlier_list, analyze.shift_list,
                       analyze.chart):
            parent = widget.parent()
            while parent is not None and not (
                    isinstance(parent, QFrame)
                    and parent.objectName() == "tile"):
                parent = parent.parent()
            self.assertIsNotNone(
                parent, "%s is not inside a tile" % widget)

    def test_hero_with_no_themes(self):
        path = os.path.join(self.tmp.name, "mixed.txt")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("quantum particles stay connected\n")
            fh.write("the toaster needs a new heating element\n")
        self.assertTrue(self._run_batch([path], "mixed.txt"))
        analyze = self.window.analyze_window
        self.assertEqual(analyze.hero_title.text(),
                         "0 themes found in 2 steps in this "
                         "session")
        self.assertIn("No resonance spikes",
                      analyze.hero_theme.text())


# ---------------------------------------------------------- flag moment


class FlagMomentTests(WindowCase):
    def test_flag_moment_fires_on_a_flagged_step(self):
        for i in range(15):
            self._step_text(THEME[i % 4])
        self.assertEqual(self.window._known_flag_steps, {15})
        chart = self.window.norm_chart
        self.assertIsNotNone(chart._pulse_anim)
        self.assertEqual(chart._pulse_anim.duration(), 300)
        self.assertEqual(chart._pulse_index, 14)
        self.assertIsNotNone(self.window._glow_anim)
        self.assertEqual(self.window._glow_anim.duration(), 300)
        self.assertIsNotNone(self.window._flag_anim)
        self.assertEqual(self.window._flag_anim.duration(), 250)
        # The glow is mid-flight: the readout carries a color.
        # Pump until a frame lands instead of a fixed wait: under
        # sandbox load a fixed qWait can pass without the
        # animation timer firing once.
        glowed = _pump_until(
            lambda: "color:"
            in self.window.resonance_label.styleSheet(),
            timeout=3.0)
        self.assertTrue(glowed, "the resonance glow never ran")
        # Everything settles, then still.
        settled = _pump_until(
            lambda: chart._pulse_anim is None
            and self.window._glow_anim is None
            and self.window._flag_anim is None,
            timeout=3.0)
        self.assertTrue(settled, "the flag moment never settled")
        self.assertIsNone(chart._pulse_anim)
        self.assertIsNone(self.window._glow_anim)
        self.assertIsNone(self.window._flag_anim)
        self.assertEqual(
            self.window.resonance_label.styleSheet(), "")
        # The flagged row rests at the calm tint; the other rows
        # carry no background at all.
        row = self.window._trace_row_for_step(15)
        item = self.window.trace_table.item(row, 0)
        self.assertEqual(item.background().color().name(),
                         FLAG_TINT.name())
        plain = self.window.trace_table.item(0, 0)
        self.assertEqual(plain.background().style(), Qt.NoBrush)

    def test_flag_fires_once_per_flag(self):
        for i in range(16):
            self._step_text(THEME[i % 4])
        _pump_until(
            lambda: self.window.norm_chart._pulse_anim is None
            and self.window._glow_anim is None
            and self.window._flag_anim is None,
            timeout=3.0)
        self.assertEqual(self.window._known_flag_steps, {15})
        # A refresh with no new flag starts no animation.
        self.window._refresh_all()
        self.assertIsNone(self.window.norm_chart._pulse_anim)
        self.assertIsNone(self.window._glow_anim)
        self.assertIsNone(self.window._flag_anim)


# ------------------------------------------------------ completion moment


class CompletionMomentTests(WindowCase):
    def test_completion_mark_on_batch_finish(self):
        path = self._theme_fixture()
        self.assertTrue(self._run_batch([path]))
        mark = self.window.completion_mark
        self.assertTrue(mark.isVisible())
        # Pump until the checkmark finishes drawing: a fixed
        # qWait can pass without an animation frame under load.
        drew = _pump_until(lambda: mark._progress == 1.0,
                           timeout=3.0)
        self.assertTrue(drew, "the checkmark never drew")
        self.assertEqual(mark._progress, 1.0)

    def test_completion_mark_on_pack_write(self):
        path = self._theme_fixture()
        self.assertTrue(self._run_batch([path]))
        self.window.completion_mark.clear()
        dest = os.path.join(self.tmp.name, "packs")
        # v0.10 (R11): export previews first, then Write saves
        # the packs. The mark lands on the write, not the
        # preview.
        self.window.analyze_window.on_export_stella_packs()
        self.assertFalse(self.window.completion_mark.isVisible())
        with mock.patch(
                "acr_app.QFileDialog.getExistingDirectory",
                return_value=dest):
            self.window.analyze_window.on_write_stella_packs()
        self.assertTrue(self.window.completion_mark.isVisible())

    def test_completion_mark_clears_on_next_step(self):
        path = self._theme_fixture()
        self.assertTrue(self._run_batch([path]))
        self.assertTrue(self.window.completion_mark.isVisible())
        self._step_text("quantum particles stay connected")
        self.assertFalse(self.window.completion_mark.isVisible())


# ------------------------------------------------------------ press dip


class PressDipTests(WindowCase):
    def test_step_button_dips_on_press(self):
        QTest.mousePress(self.window.step_button, Qt.LeftButton)
        anim = getattr(self.window.step_button, "_dip_anim", None)
        self.assertIsNotNone(anim)
        self.assertEqual(anim.duration(), 150)
        self.assertEqual(anim.easingCurve().type(),
                         QEasingCurve.OutBack)
        QTest.mouseRelease(self.window.step_button,
                           Qt.LeftButton)
        _pump_until(
            lambda: self.window.step_button.graphicsEffect()
            is None,
            timeout=3.0)
        self.assertIsNone(
            self.window.step_button.graphicsEffect())

    def test_go_button_dips_on_press(self):
        go = self.window.analyze_window.go_button
        QTest.mousePress(go, Qt.LeftButton)
        anim = getattr(go, "_dip_anim", None)
        self.assertIsNotNone(anim)
        self.assertEqual(anim.duration(), 150)
        self.assertEqual(anim.easingCurve().type(),
                         QEasingCurve.OutBack)
        QTest.mouseRelease(go, Qt.LeftButton)
        _pump_until(lambda: go.graphicsEffect() is None,
                    timeout=3.0)
        self.assertIsNone(go.graphicsEffect())

    def test_no_dip_under_reduced_motion(self):
        with mock.patch.dict(
                os.environ, {"ACR_REDUCED_MOTION": "1"}):
            self.assertTrue(reduced_motion())
            QTest.mousePress(self.window.step_button,
                             Qt.LeftButton)
            self.assertIsNone(getattr(
                self.window.step_button, "_dip_anim", None))
            self.assertIsNone(
                self.window.step_button.graphicsEffect())
            QTest.mouseRelease(self.window.step_button,
                               Qt.LeftButton)


# --------------------------------------------------------- reduced motion


class ReducedMotionTests(WindowCase):
    def test_flag_state_lands_without_motion(self):
        with mock.patch.dict(
                os.environ, {"ACR_REDUCED_MOTION": "1"}):
            for i in range(16):
                self._step_text(THEME[i % 4])
            self.assertEqual(self.window._known_flag_steps, {15})
            self.assertIsNone(
                self.window.norm_chart._pulse_anim)
            self.assertIsNone(self.window._glow_anim)
            self.assertIsNone(self.window._flag_anim)
            row = self.window._trace_row_for_step(15)
            item = self.window.trace_table.item(row, 0)
            self.assertEqual(item.background().color().name(),
                             FLAG_TINT.name())

    def test_completion_mark_appears_without_motion(self):
        with mock.patch.dict(
                os.environ, {"ACR_REDUCED_MOTION": "1"}):
            path = self._theme_fixture()
            self.assertTrue(self._run_batch([path]))
            mark = self.window.completion_mark
            self.assertTrue(mark.isVisible())
            self.assertEqual(mark._progress, 1.0)
            self.assertIsNone(mark._anim)

    def test_window_opens_without_spring(self):
        with mock.patch.dict(
                os.environ, {"ACR_REDUCED_MOTION": "1"}):
            self.window.show_controls_window()
            QTest.qWait(100)
            self.assertIsNone(
                self.window.controls_window._open_anim)
            self.assertTrue(
                self.window.controls_window.isVisible())


# ------------------------------------------------------------- calm tech


class CalmTechTests(WindowCase):
    def test_no_dialogs_on_routine_events(self):
        calls = []
        patches = [
            mock.patch.object(
                QMessageBox, "information",
                side_effect=lambda *a: calls.append("info")),
            mock.patch.object(
                QMessageBox, "question",
                side_effect=lambda *a: calls.append("question")
                or QMessageBox.No),
            mock.patch.object(
                QMessageBox, "about",
                side_effect=lambda *a: calls.append("about")),
            mock.patch.object(
                QMessageBox, "warning",
                side_effect=lambda *a: calls.append("warning")),
            mock.patch.object(
                QMessageBox, "critical",
                side_effect=lambda *a: calls.append("critical")),
        ]
        for patch in patches:
            patch.start()
        self.addCleanup(lambda: [p.stop() for p in patches])
        for i in range(16):
            self._step_text(THEME[i % 4])
        path = self._theme_fixture()
        self.assertTrue(self._run_batch([path]))
        dest = os.path.join(self.tmp.name, "packs")
        with mock.patch(
                "acr_app.QFileDialog.getExistingDirectory",
                return_value=dest):
            self.window.analyze_window.on_export_stella_packs()
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
