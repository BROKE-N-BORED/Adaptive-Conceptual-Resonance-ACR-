"""Tests for ACR v0.10: the audit redesign, tasks R1-R15.

Each task gets its own test class, in the same order as the
task list. Where the audit measured a number (the 4.21 s dead
window, the 15.34 s uninterruptible start), the test proves the
fix against the same shape of slow service.
"""

import json
import math
import os
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

from PySide6.QtCore import QPoint, QPointF, QEvent, Qt, QSettings, QTimer
from PySide6.QtGui import QMouseEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox, QTextEdit

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import acr_app  # noqa: E402
from acr_app import (  # noqa: E402
    ACCENT_GOLD, BIG_TEXT_STEP, CHART_BG, CHART_SPIKE,
    DARK_STYLESHEET, EXAMPLE_SENTENCES, GLOW_GOLD,
    HEAT_HISTORY_LIMIT, MainWindow, big_text_enabled,
    reduced_motion)
from tests.test_analyze import OFF, THEME  # noqa: E402
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


def _contrast_ratio(foreground, background):
    """WCAG contrast ratio between two QColors."""
    def luminance(color):
        def channel(value):
            value = value / 255.0
            if value <= 0.03928:
                return value / 12.92
            return ((value + 0.055) / 1.055) ** 2.4
        return (0.2126 * channel(color.red())
                + 0.7152 * channel(color.green())
                + 0.0722 * channel(color.blue()))
    light = max(luminance(foreground), luminance(background))
    dark = min(luminance(foreground), luminance(background))
    return (light + 0.05) / (dark + 0.05)


class SlowTagsHandler(BaseHTTPRequestHandler):
    """A fake service with a slow /api/tags: the offer path
    (probe, then list) hits it twice, so a 2 s tags delay is a
    ~4 s offer, the shape of the audit's 4.21 s dead window."""
    tags_delay = 2.0
    embed_delay = 0.0

    def log_message(self, _format, *_args):
        pass

    def do_GET(self):
        if self.path == "/api/tags":
            time.sleep(self.tags_delay)
            body = json.dumps({"models": [
                {"name": "nomic-embed-text:latest"},
                {"name": "other:1b"}]}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        if self.path == "/api/embed":
            time.sleep(self.embed_delay)
            body = json.dumps(
                {"embeddings": [[0.1] * 16]}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()


class SlowEmbedHandler(FakeLocalHandler):
    """The standard fake service with a slow embedding: the
    activation probe takes 2 s, the shape of the audit's
    3.12 s probe."""
    embed_delay = 2.0


def _start_server(handler):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever,
                              daemon=True)
    thread.start()
    return server, "http://127.0.0.1:%d" % server.server_address[1]


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

    def _step_theme(self, count):
        for i in range(count):
            self._step_text(THEME[i % 4])

    def _wait_batch(self, timeout=60.0):
        return _pump_until(
            lambda: self.window._batch_thread is None,
            timeout=timeout)

    def _run_batch(self, paths, source_note="fixture.txt",
                   timeout=25.0):
        self.window.start_batch(paths, source_note)
        return self._wait_batch(timeout=timeout)

    def _theme_fixture(self, name="fixture.txt"):
        path = os.path.join(self.tmp.name, name)
        with open(path, "w", encoding="utf-8") as fh:
            for i in range(16):
                fh.write(THEME[i % 4] + "\n")
            for line in OFF:
                fh.write(line + "\n")
        return path


class LocalWindowCase(WindowCase):
    """A window pointed at a fake local service."""
    handler = FakeLocalHandler

    def setUp(self):
        _clear_settings()
        self.tmp = tempfile.TemporaryDirectory()
        self.server, url = _start_server(self.handler)
        self.window = MainWindow(log_dir=self.tmp.name,
                                 local_base_url=url)
        self.window.show()

    def tearDown(self):
        self.window.close()
        self.window.deleteLater()
        self.server.shutdown()
        self.server.server_close()
        self.tmp.cleanup()
        _clear_settings()

    def _wait_local_idle(self, timeout=30.0):
        return _pump_until(
            lambda: self.window._local_thread is None,
            timeout=timeout)


# ---------------------------------------------------------------- version


class VersionTests(unittest.TestCase):
    def test_version_is_0_10(self):
        self.assertEqual(acr_app.VERSION, "0.10")


# --------------------------------------------------------------------- R1


class OfferThreadTests(LocalWindowCase):
    """R1: the model-list fetch runs off the UI thread. The
    audit measured a 4.21 s dead window (a 100 ms timer fired 0
    times); the same slow service must now leave the window
    responsive."""
    handler = SlowTagsHandler

    def test_offer_keeps_the_window_responsive(self):
        ticks = []
        timer = QTimer()
        timer.timeout.connect(lambda: ticks.append(1))
        timer.start(100)
        try:
            self.window.encoder_combo.setCurrentIndex(1)
            self.assertTrue(_pump_until(
                lambda: self.window._local_thread is not None,
                timeout=10.0), "the offer did not start")
            self.assertIn(ACCENT_GOLD,
                          self.window.busy_light.styleSheet())
            self.assertIn("Checking the local model service",
                          self.window.status_label.text())
            self.assertTrue(self._wait_local_idle(),
                            "the offer did not land")
        finally:
            timer.stop()
        # ~4 s of work at a 100 ms tick: dozens of ticks. The
        # audit's frozen window fired 0.
        self.assertGreater(len(ticks), 5,
                           "the 100 ms timer froze during the "
                           "model-list fetch")
        self.assertEqual(self.window.model_combo.count(), 2)


class ActivateThreadTests(LocalWindowCase):
    """R1: the activation probe runs off the UI thread. The
    audit measured a 3.12 s dead window on a slow first
    embedding."""
    handler = SlowEmbedHandler

    def test_activation_keeps_the_window_responsive(self):
        self.window.encoder_combo.setCurrentIndex(1)
        # The list fills in the worker's finished handler, but
        # the thread reference clears only once the thread has
        # stopped, a pump cycle later: picking before it clears
        # is refused ("Wait for the local model call to
        # finish."). Wait for both.
        self.assertTrue(_pump_until(
            lambda: (self.window.model_combo.count() > 0
                     and self.window._local_thread is None),
            timeout=25.0), "the offer did not land")
        ticks = []
        timer = QTimer()
        timer.timeout.connect(lambda: ticks.append(1))
        timer.start(100)
        try:
            combo = self.window.model_combo
            combo.setCurrentIndex(
                combo.findData("nomic-embed-text:latest"))
            self.assertTrue(_pump_until(
                lambda: self.window._local_thread is not None,
                timeout=10.0), "the activation did not start")
            self.assertIn(ACCENT_GOLD,
                          self.window.busy_light.styleSheet())
            self.assertTrue(self._wait_local_idle(),
                            "the activation did not land")
        finally:
            timer.stop()
        self.assertGreater(len(ticks), 5,
                           "the 100 ms timer froze during the "
                           "activation probe")
        self.assertEqual(self.window.encoder_kind, "local")
        self.assertEqual(self.window.local_model,
                         "nomic-embed-text:latest")


class StartServiceThreadTests(WindowCase):
    """R1: starting the service runs off the UI thread. The
    audit measured a 15.34 s uninterruptible start; the same
    wait must now leave the window responsive and be
    cancellable."""

    def _dead_url(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0),
                                     BaseHTTPRequestHandler)
        port = server.server_address[1]
        server.server_close()
        return "http://127.0.0.1:%d" % port

    def test_start_keeps_the_window_responsive(self):
        url = self._dead_url()
        window = MainWindow(log_dir=self.tmp.name,
                            local_base_url=url)
        window.show()
        try:
            probe = mock.Mock(side_effect=[False, False, False,
                                           True])
            with mock.patch.object(window.client, "probe",
                                   probe), \
                    mock.patch("acr_app.shutil.which",
                               return_value="/usr/bin/ollama"), \
                    mock.patch(
                        "acr_app.QProcess.startDetached",
                        return_value=(True, 123)):
                ticks = []
                timer = QTimer()
                timer.timeout.connect(lambda: ticks.append(1))
                timer.start(100)
                try:
                    QTest.mouseClick(window.start_button,
                                     Qt.LeftButton)
                    self.assertTrue(_pump_until(
                        lambda: window._local_thread is not None,
                        timeout=10.0), "the start did not begin")
                    self.assertIn(
                        ACCENT_GOLD,
                        window.busy_light.styleSheet())
                    self.assertTrue(_pump_until(
                        lambda: window._local_thread is None,
                        timeout=30.0), "the start did not land")
                finally:
                    timer.stop()
            self.assertGreater(
                len(ticks), 5,
                "the 100 ms timer froze while starting the "
                "service")
            self.assertIn("is running", window.status_label.text())
        finally:
            window.close()
            window.deleteLater()

    def test_start_cancel_abandons_the_wait(self):
        url = self._dead_url()
        window = MainWindow(log_dir=self.tmp.name,
                            local_base_url=url)
        window.show()
        try:
            probe = mock.Mock(return_value=False)
            with mock.patch.object(window.client, "probe",
                                   probe), \
                    mock.patch("acr_app.shutil.which",
                               return_value="/usr/bin/ollama"), \
                    mock.patch(
                        "acr_app.QProcess.startDetached",
                        return_value=(True, 123)):
                QTest.mouseClick(window.start_button,
                                 Qt.LeftButton)
                self.assertTrue(_pump_until(
                    lambda: window._local_thread is not None,
                    timeout=10.0), "the start did not begin")
                QTest.qWait(300)
                window.on_cancel_local()
                started = time.monotonic()
                # The uninterruptible wait was 15 s. A cancelled
                # start must be done in a few seconds.
                self.assertTrue(_pump_until(
                    lambda: window._local_thread is None,
                    timeout=10.0), "the cancelled start hung")
                self.assertLess(time.monotonic() - started, 5.0,
                                "cancel did not abandon the "
                                "service start")
            self.assertIn("Cancelled", window.status_label.text())
        finally:
            window.close()
            window.deleteLater()


# --------------------------------------------------------------------- R2


class KeyboardTests(WindowCase):
    """R2: keyboard navigation. A focused control must LOOK
    focused, Tab must follow visual order, the input must have
    the focus at launch, and every menu action (except About,
    which has no standard shortcut) must have a shortcut."""

    def test_focus_rules_exist_in_the_stylesheet(self):
        for selector in ("QPushButton:focus",
                         "QPlainTextEdit:focus",
                         "QComboBox:focus",
                         "QListWidget:focus",
                         "QTableWidget:focus",
                         "QTextBrowser:focus",
                         "QCheckBox:focus",
                         "QSlider:focus::groove:horizontal"):
            self.assertIn(selector, DARK_STYLESHEET, selector)
        # The ring is the gold glow colour, visible on the dark
        # surfaces.
        self.assertIn("QPushButton:focus { border: 2px solid %s; }"
                      % GLOW_GOLD, DARK_STYLESHEET)

    def test_input_focused_at_launch(self):
        self.assertTrue(_pump_until(
            lambda: QApplication.focusWidget()
            is self.window.input_edit,
            timeout=5.0),
            "the input did not have the focus at launch")

    def test_tab_order_follows_visual_order(self):
        window = self.window
        window.input_edit.setFocus()
        QTest.qWait(50)
        self.assertIs(QApplication.focusWidget(),
                      window.input_edit)
        QTest.keyClick(window.input_edit, Qt.Key_Tab)
        self.assertIs(QApplication.focusWidget(),
                      window.step_button)
        QTest.keyClick(window.step_button, Qt.Key_Tab)
        self.assertIs(QApplication.focusWidget(),
                      window.analyze_button)
        QTest.keyClick(window.analyze_button, Qt.Key_Tab)
        self.assertIs(QApplication.focusWidget(),
                      window.example_button)

    def test_shortcuts_on_menu_actions(self):
        expected = {
            "action_save_session": "Ctrl+S",
            "action_load_session": "Ctrl+O",
            "action_export_json": "Ctrl+J",
            "action_export_csv": "Ctrl+Shift+J",
            "action_export_session": "Ctrl+E",
            "action_debug_report": "Ctrl+D",
            "action_reset": "Ctrl+R",
            "action_quit": "Ctrl+Q",
            "action_show_controls": "Ctrl+1",
            "action_show_trace": "Ctrl+2",
            "action_show_views": "Ctrl+3",
            "action_show_analyze": "Ctrl+4",
            "action_reduced_motion": "Ctrl+M",
            "action_big_text": "Ctrl+T",
            "action_how_to_use": "F1",
        }
        for name, keys in expected.items():
            action = getattr(self.window, name)
            self.assertEqual(action.shortcut().toString(), keys,
                             name)
        # About has no standard shortcut; it stays unbound.
        self.assertEqual(
            self.window.action_about.shortcut().toString(), "")


# --------------------------------------------------------------------- R3


class ReadableEntriesTests(WindowCase):
    """R3: entry text is readable everywhere. Finding items
    carry the full text as a tooltip and open it on
    double-click; a trace row shows its full input below the
    table."""

    def test_finding_items_carry_full_text_tooltips(self):
        path = self._theme_fixture()
        self.assertTrue(self._run_batch([path]))
        analyze = self.window.analyze_window
        item = analyze.themes_list.item(0)
        self.assertIn("quantum particles stay connected",
                      item.toolTip())
        for listing in (analyze.core_list, analyze.outlier_list,
                        analyze.shift_list):
            if listing.count():
                self.assertTrue(listing.item(0).toolTip(),
                                "a finding has no tooltip")

    def test_double_click_shows_the_full_entry(self):
        path = self._theme_fixture()
        self.assertTrue(self._run_batch([path]))
        analyze = self.window.analyze_window
        item = analyze.themes_list.item(0)
        # Offscreen Qt never synthesizes a double-click, so the
        # handler is called directly: it is the same code the
        # double-click runs.
        analyze._on_finding_double_clicked(item)
        dialog = analyze._entry_dialog
        self.assertIsNotNone(dialog)
        self.assertTrue(dialog.isVisible())
        view = dialog.findChild(QTextEdit)
        self.assertIsNotNone(view)
        self.assertIn("quantum particles stay connected",
                      view.toPlainText())
        dialog.close()

    def test_trace_row_shows_full_input(self):
        long_text = ("quantum entanglement links particles "
                     "across a very long distance "
                     "and back again and again and again")
        self.assertGreater(len(long_text), 80,
                           "the test line must exceed the "
                           "80-char preview")
        self._step_text(long_text)
        self.window.trace_table.setCurrentCell(0, 0)
        self.assertIn(long_text, self.window.trace_detail.text())
        cell = self.window.trace_table.item(0, 1)
        self.assertEqual(cell.toolTip(), long_text)


# --------------------------------------------------------------------- R4


class ExampleTests(WindowCase):
    """R4: teach-by-playing first run. Try an example steps
    the four theme sentences as real steps, shows only while
    the session is empty, and goes away once the user steps
    anything."""

    def test_button_shows_only_on_empty_session(self):
        self.assertTrue(self.window.example_button.isVisible())
        self.assertTrue(self.window.example_label.isVisible())
        self.assertIn("four sentences",
                      self.window.example_label.text())
        self._step_text("quantum particles stay connected")
        self.assertFalse(self.window.example_button.isVisible())
        self.assertFalse(self.window.example_label.isVisible())

    def test_example_steps_four_real_steps(self):
        QTest.mouseClick(self.window.example_button,
                         Qt.LeftButton)
        self.assertEqual(self.window.engine.step_count, 4)
        inputs = [entry["input"]
                  for entry in self.window.engine.trace]
        self.assertEqual(inputs, list(EXAMPLE_SENTENCES))
        self.assertFalse(self.window.example_button.isVisible())
        self.assertIn("Example done",
                      self.window.status_label.text())
        # Real steps: the trace log and the chart hold them.
        self.assertEqual(self.window.trace_table.rowCount(), 4)
        self.assertEqual(len(self.window.norm_chart.values), 4)

    def test_example_returns_after_reset(self):
        QTest.mouseClick(self.window.example_button,
                         Qt.LeftButton)
        self.assertFalse(self.window.example_button.isVisible())
        self.window.do_reset()
        self.assertTrue(self.window.example_button.isVisible())
        self.assertTrue(self.window.example_label.isVisible())


# --------------------------------------------------------------------- R5


class LiveResultsTests(WindowCase):
    """R5: the Analyze chart and hero tile update as steps
    land, not only at the end."""

    def test_show_live_updates_chart_and_hero(self):
        self._step_theme(6)
        analyze = self.window.analyze_window
        analyze.show_live(list(self.window.engine.trace), 6, 24)
        self.assertEqual(len(analyze.chart.values), 6)
        self.assertIn("Analyzing...", analyze.hero_title.text())
        self.assertIn("so far", analyze.hero_title.text())
        self.assertIn("line 6 of 24 in this run",
                      analyze.hero_theme.text())

    def test_chart_fills_mid_run(self):
        # 280 built-in steps take seconds: plenty of time to
        # look while the run is still going.
        path = os.path.join(self.tmp.name, "big.txt")
        with open(path, "w", encoding="utf-8") as fh:
            for i in range(280):
                fh.write(THEME[i % 4] + "\n")
        analyze = self.window.analyze_window
        self.window.start_batch([path], "big.txt")
        seen = _pump_until(
            lambda: (self.window._batch_thread is not None
                     and len(analyze.chart.values) > 0),
            timeout=30.0)
        self.assertTrue(seen, "the chart stayed empty mid-run")
        self.assertIn("Analyzing...", analyze.hero_title.text())
        self.assertTrue(self._wait_batch(timeout=60.0))
        self.assertEqual(self.window.engine.step_count, 280)


# --------------------------------------------------------------------- R6


class ChartScrubTests(WindowCase):
    """R6: the chart scrubs. Hover shows a step's values,
    click jumps the heatmap and the trace selection to that
    step."""

    def _chart_x(self, chart, index):
        width = chart.width()
        count = len(chart.values)
        return 48 + int(round(index * (width - 58)
                              / float(count - 1)))

    def test_click_jumps_heatmap_and_trace(self):
        self._step_theme(16)
        self.window._on_chart_step_clicked(14)
        self.assertEqual(self.window.heat_spin.value(), 15)
        self.assertEqual(self.window.trace_table.currentRow(), 14)

    def test_chart_click_emits_the_step(self):
        self._step_theme(16)
        self.window.show_views_window()
        chart = self.window.norm_chart
        self.assertTrue(_pump_until(lambda: chart.width() > 100,
                                    timeout=5.0))
        seen = []
        chart.stepClicked.connect(seen.append)
        x = self._chart_x(chart, 14)
        QTest.mouseClick(chart, Qt.LeftButton,
                         pos=QPoint(x, chart.height() // 2))
        self.assertEqual(seen, [14])
        self.assertEqual(self.window.heat_spin.value(), 15)
        self.assertEqual(self.window.trace_table.currentRow(), 14)

    def test_hover_shows_step_values(self):
        self._step_theme(16)
        self.window.show_views_window()
        chart = self.window.norm_chart
        self.assertTrue(_pump_until(lambda: chart.width() > 100,
                                    timeout=5.0))
        x = self._chart_x(chart, 14)
        # Offscreen Qt never delivers QTest.mouseMove to the
        # widget (probed: 0 mouseMoveEvent calls), so the move
        # event is dispatched directly: it is the same handler
        # a real mouse move runs.
        event = QMouseEvent(
            QEvent.Type.MouseMove, QPointF(x, chart.height() // 2),
            QPointF(chart.mapToGlobal(
                QPoint(x, chart.height() // 2))),
            Qt.MouseButton.NoButton, Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier)
        QApplication.sendEvent(chart, event)
        self.assertIn("Step 15", chart.toolTip())
        self.assertIn("W norm", chart.toolTip())
        self.assertIn("new-layer candidate", chart.toolTip())


# --------------------------------------------------------------------- R7


class FlagMarkerTests(WindowCase):
    """R7: a persistent resonance flag on Main. The marker
    stays after the 300 ms moment passes, until the next flag
    or a reset."""

    def test_marker_persists_after_the_moment(self):
        self.assertFalse(self.window.flag_marker.isVisible())
        self._step_theme(16)
        self.assertTrue(_pump_until(
            lambda: (self.window._flag_anim is None
                     and self.window._glow_anim is None),
            timeout=10.0), "the flag moment did not settle")
        marker = self.window.flag_marker
        self.assertTrue(marker.isVisible())
        self.assertIn("step 15", marker.toolTip())
        self.window.do_reset()
        self.assertFalse(marker.isVisible())


# --------------------------------------------------------------------- R8


class AlphaLabelTests(WindowCase):
    """R8: a live alpha value next to the slider in the
    Controls window, the same effective rate Main shows."""

    def test_controls_alpha_label_tracks_the_slider(self):
        self.window.show_controls_window()
        self.window.alpha_slider.setValue(25)
        self.assertEqual(self.window.alpha_value_label.text(),
                         "Alpha: 0.25")
        self.assertEqual(self.window.alpha_label.text(),
                         "Alpha: 0.25")

    def test_controls_label_shows_the_effective_rate(self):
        # Fast-adapt doubles the base at step 0: the Controls
        # label must agree with Main, not with the slider.
        self.window.schedule_combo.setCurrentIndex(1)
        self.window.alpha_slider.setValue(30)
        self.assertEqual(self.window.alpha_value_label.text(),
                         "Alpha: 0.60")
        self.assertEqual(self.window.alpha_label.text(),
                         "Alpha: 0.60")


# --------------------------------------------------------------------- R9


class AccessibilityTests(WindowCase):
    """R9: accessibility in the menu. Reduced motion and big
    text persist in settings, and the chart legend contrast
    passes AA (the audit measured 4.16:1, below the 4.5:1
    bar)."""

    def test_reduced_motion_toggle_persists(self):
        action = self.window.action_reduced_motion
        self.assertFalse(action.isChecked())
        action.setChecked(True)
        self.assertTrue(reduced_motion())
        self.assertTrue(QSettings("BrokeNBored", "ACR").value(
            "reduced_motion", False, type=bool))
        other = MainWindow(log_dir=self.tmp.name)
        try:
            self.assertTrue(
                other.action_reduced_motion.isChecked())
        finally:
            other.close()
            other.deleteLater()
        action.setChecked(False)
        self.assertFalse(reduced_motion())

    def test_big_text_toggle_persists_and_resizes(self):
        app = QApplication.instance()
        before = app.font().pointSize()
        action = self.window.action_big_text
        self.assertFalse(action.isChecked())
        try:
            action.setChecked(True)
            self.assertEqual(app.font().pointSize(),
                             before + BIG_TEXT_STEP)
            self.assertTrue(big_text_enabled())
            other = MainWindow(log_dir=self.tmp.name)
            try:
                self.assertTrue(other.action_big_text.isChecked())
            finally:
                other.close()
                other.deleteLater()
        finally:
            action.setChecked(False)
        self.assertEqual(app.font().pointSize(), before)
        self.assertFalse(big_text_enabled())

    def test_legend_contrast_passes_aa(self):
        ratio = _contrast_ratio(CHART_SPIKE, CHART_BG)
        self.assertGreaterEqual(ratio, 4.5,
                                "legend red on chart "
                                "background: %.2f:1" % ratio)


# -------------------------------------------------------------------- R10


class ProgressiveDisclosureTests(WindowCase):
    """R10: in built-in mode the local-model chrome collapses
    to one line. Nothing is removed: it all returns when
    Local model is picked."""

    def test_built_in_mode_collapses_local_chrome(self):
        window = self.window
        for widget in (window.model_combo, window.model_note,
                       window.switch_note, window.complete_button,
                       window.completion_combo,
                       window.completion_note):
            self.assertFalse(widget.isVisible(), widget)
        # The encoder picker, the Start service button, and the
        # one line stay: Start service is how Local model gets
        # running at all.
        self.assertTrue(window.start_button.isVisible())
        self.assertEqual(window.encoder_label.text(),
                         "Built-in encoder, dim 64.")


class ProgressiveDisclosureLocalTests(LocalWindowCase):
    """R10, the other half: picking Local model brings the
    chrome back, and going back to built-in collapses it
    again."""

    def test_local_mode_brings_it_all_back(self):
        window = self.window
        window.encoder_combo.setCurrentIndex(1)
        # Wait for the list AND the thread to clear: the
        # thread reference outlives the fill by a pump cycle,
        # and a pick in that gap is refused.
        self.assertTrue(_pump_until(
            lambda: (window.model_combo.count() > 0
                     and window._local_thread is None),
            timeout=25.0), "the offer did not land")
        for widget in (window.model_combo, window.model_note,
                       window.switch_note, window.complete_button,
                       window.completion_combo,
                       window.completion_note):
            self.assertTrue(widget.isVisible(), widget)
        self.assertEqual(
            window.encoder_label.text(),
            "Pick a model above to use the local encoder.")
        combo = window.model_combo
        combo.setCurrentIndex(
            combo.findData("nomic-embed-text:latest"))
        # The encoder flips in the finished handler, before
        # the thread reference clears: wait for both, or the
        # switch back to built-in is refused.
        self.assertTrue(_pump_until(
            lambda: (window.encoder_kind == "local"
                     and window._local_thread is None),
            timeout=25.0), "the activation did not land")
        self.assertTrue(window.model_combo.isVisible())
        window.encoder_combo.setCurrentIndex(0)
        self.assertFalse(window.model_combo.isVisible())
        self.assertFalse(window.complete_button.isVisible())
        self.assertEqual(window.encoder_label.text(),
                         "Built-in encoder, dim 64.")


# -------------------------------------------------------------------- R11


class PackPreviewTests(WindowCase):
    """R11: pack export previews before writing, and the
    revert confirm shows the W norm the revert will land
    on."""

    def _run_theme_batch(self):
        path = self._theme_fixture()
        self.assertTrue(self._run_batch([path]))
        self.window.show_analyze_window()
        return self.window.analyze_window

    def test_preview_shows_names_counts_and_rows(self):
        analyze = self._run_theme_batch()
        analyze.on_export_stella_packs()
        self.assertTrue(analyze.pack_preview.isVisible())
        self.assertTrue(analyze.write_packs_button.isVisible())
        text = analyze.pack_preview.toPlainText()
        self.assertIn("Pack:", text)
        self.assertIn("Theme membership", text)
        self.assertIn("Q:", text)
        self.assertIn("A:", text)
        self.assertIn("Preview ready",
                      self.window.status_label.text())
        # The preview commits nothing: no pack folder exists
        # until Write is pressed.
        self.assertFalse(os.path.exists(
            os.path.join(self.tmp.name, "packs")))

    def test_write_after_preview_writes_the_packs(self):
        analyze = self._run_theme_batch()
        analyze.on_export_stella_packs()
        dest = os.path.join(self.tmp.name, "packs")
        with mock.patch(
                "acr_app.QFileDialog.getExistingDirectory",
                return_value=dest):
            analyze.on_write_stella_packs()
        self.assertTrue(os.path.isdir(dest))
        self.assertGreater(len(os.listdir(dest)), 0)
        self.assertIn("Wrote", self.window.status_label.text())
        self.assertFalse(analyze.write_packs_button.isVisible())

    def test_write_after_a_changed_session_says_so(self):
        analyze = self._run_theme_batch()
        analyze.on_export_stella_packs()
        self._step_text("quantum particles stay connected")
        dest = os.path.join(self.tmp.name, "packs")
        with mock.patch(
                "acr_app.QFileDialog.getExistingDirectory",
                return_value=dest):
            analyze.on_write_stella_packs()
        self.assertIn("session changed since the preview",
                      self.window.status_label.text())

    def test_revert_confirm_names_both_norms(self):
        self._step_theme(6)
        target = self.window.engine.trace[3]["new_norm"]
        current = self.window.engine.w_norm()
        self.window.trace_table.setCurrentCell(3, 0)
        with mock.patch.object(
                acr_app.QMessageBox, "question",
                return_value=QMessageBox.No) as asked:
            self.window.on_revert()
        text = asked.call_args[0][2]
        self.assertIn("W norm", text)
        self.assertIn("%.4f" % current, text)
        self.assertIn("%.4f" % target, text)


# -------------------------------------------------------------------- R12


class TraceFollowTests(WindowCase):
    """R12: the trace follows only when the user is already at
    the bottom. Scrolled up to inspect an old row, the place
    is not yanked."""

    def _step_many(self, count=40):
        for i in range(count):
            self._step_text(THEME[i % 4])

    def _scrollbar(self):
        self.window.show_trace_window()
        bar = self.window.trace_table.verticalScrollBar()
        self.assertTrue(_pump_until(lambda: bar.maximum() > 0,
                                    timeout=5.0),
                        "the table never scrolled")
        return bar

    def test_scrolled_up_place_is_kept(self):
        self._step_many()
        bar = self._scrollbar()
        middle = bar.maximum() // 2
        bar.setValue(middle)
        QTest.qWait(50)
        self.assertEqual(bar.value(), middle)
        self._step_text("quantum particles stay connected")
        self.assertEqual(bar.value(), middle)
        self.assertEqual(self.window.trace_table.rowCount(), 41)

    def test_at_bottom_follows(self):
        self._step_many()
        bar = self._scrollbar()
        bar.setValue(bar.maximum())
        QTest.qWait(50)
        self._step_text("quantum particles stay connected")
        self.assertEqual(bar.value(), bar.maximum())


# -------------------------------------------------------------------- R13


class PerformanceTests(WindowCase):
    """R13: long runs are cheaper. Batch progress updates the
    bar at most every ~100 ms, the heat history is bounded,
    and windows open with an opacity fade, not a geometry
    animation."""

    def test_progress_updates_are_throttled(self):
        analyze = self.window.analyze_window
        self.window._on_batch_progress(1, 100, 0.5, 1.6, 0)
        first = analyze.progress_label.text()
        self.assertIn("Line 1 of 100", first)
        # A flood of updates inside the interval: none lands.
        for done in range(2, 50):
            self.window._on_batch_progress(done, 100, 0.5, 1.6,
                                           0)
        self.assertEqual(analyze.progress_label.text(), first)
        self.assertEqual(analyze.progress_bar.value(), 1)
        # The final update always lands, so the bar still
        # reaches 100%.
        self.window._on_batch_progress(100, 100, 0.5, 1.6, 0)
        self.assertIn("Line 100 of 100",
                      analyze.progress_label.text())
        self.assertEqual(analyze.progress_bar.value(), 100)

    def test_heat_history_is_bounded(self):
        self.window._heat_history = [
            (i, [0.1] * 64) for i in range(1, 2501)]
        self.window._trim_heat_history()
        self.assertLessEqual(len(self.window._heat_history),
                             HEAT_HISTORY_LIMIT)
        # The newest steps stay dense.
        self.assertEqual(self.window._heat_history[-1][0], 2500)
        self.assertNotEqual(
            self.window._heat_rows_for(2500), [])
        # An old step was sampled out, not silently kept.
        self.assertEqual(self.window._heat_rows_for(2), [])

    def test_window_open_uses_a_fade_not_geometry(self):
        self.window.show_controls_window()
        anim = self.window.controls_window._open_anim
        self.assertIsNotNone(anim, "no open animation ran")
        self.assertEqual(anim.propertyName(), b"opacity")


# -------------------------------------------------------------------- R14


class ViewMenuTests(WindowCase):
    """R14: the View menu shows which windows are open. The
    four actions are checkable, checked while their window is
    open, updating when windows open or close."""

    def test_view_actions_are_checkable_and_start_unchecked(self):
        for name in ("action_show_controls", "action_show_trace",
                     "action_show_views", "action_show_analyze"):
            action = getattr(self.window, name)
            self.assertTrue(action.isCheckable(), name)
            self.assertFalse(action.isChecked(), name)

    def test_view_action_follows_its_window(self):
        action = self.window.action_show_controls
        action.trigger()
        self.assertTrue(action.isChecked())
        self.assertTrue(self.window.controls_window.isVisible())
        self.window.controls_window.close()
        self.assertFalse(action.isChecked())
        self.assertFalse(
            self.window.controls_window.isVisible())
        # Triggering an open window closes it.
        action.trigger()
        self.assertTrue(self.window.controls_window.isVisible())
        action.trigger()
        self.assertFalse(action.isChecked())
        self.assertFalse(
            self.window.controls_window.isVisible())

    def test_opening_from_a_button_checks_the_action(self):
        self.window.show_analyze_window()
        self.assertTrue(
            self.window.action_show_analyze.isChecked())
        self.window.analyze_window.close()
        self.assertFalse(
            self.window.action_show_analyze.isChecked())


# -------------------------------------------------------------------- R15


class PolishTests(WindowCase):
    """R15: copy and empty-state polish, every nit in audit
    F15."""

    def test_analyze_chart_empty_message_fits_analyze(self):
        self.assertEqual(
            self.window.analyze_window.chart.empty_text,
            "Analyze a file to see the chart.")
        self.assertEqual(self.window.norm_chart.empty_text,
                         "Step some text to see the chart.")

    def test_hero_and_progress_say_what_they_count(self):
        path = self._theme_fixture()
        self.assertTrue(self._run_batch([path]))
        analyze = self.window.analyze_window
        self.assertIn("in this session", analyze.hero_title.text())
        self.assertIn("in this run", analyze.progress_label.text())

    def test_session_load_restores_the_analyze_source(self):
        path = self._theme_fixture()
        self.assertTrue(self._run_batch([path], "fixture.txt"))
        saved = os.path.join(self.tmp.name, "session.json")
        self.window.save_session_to(saved)
        other = MainWindow(log_dir=self.tmp.name)
        other.show()
        try:
            other.load_session_from(saved)
            self.assertIn(
                "fixture.txt",
                other.analyze_window.source_label.text())
        finally:
            other.close()
            other.deleteLater()

    def test_empty_trace_guides_and_disables_revert(self):
        self.window.show_trace_window()
        self.assertTrue(
            self.window.trace_empty_label.isVisible())
        self.assertIn("No steps yet",
                      self.window.trace_empty_label.text())
        self.assertFalse(self.window.revert_button.isEnabled())
        self._step_text("quantum particles stay connected")
        self.assertFalse(
            self.window.trace_empty_label.isVisible())
        self.assertTrue(self.window.revert_button.isEnabled())

    def test_how_to_use_opens_with_no_selection(self):
        self.window.show_howto_window()
        howto = self.window.howto_window
        self.assertTrue(_pump_until(lambda: howto.isVisible(),
                                    timeout=5.0))
        QTest.qWait(200)
        self.assertEqual(howto.index_list.currentRow(), -1)
        self.assertFalse(
            howto.browser.textCursor().hasSelection())

    def test_step_zero_readouts_are_consistent(self):
        # The 1.6404 was the random starting W_t, not a
        # measurement: at step 0 the norm reads "-" like
        # Resonance does.
        self.assertEqual(self.window.norm_label.text(),
                         "W norm: -")
        self.assertEqual(self.window.resonance_label.text(),
                         "Resonance: -")
        self.assertEqual(self.window.spectral_label.text(),
                         "Top singular value: -")
        self._step_text("quantum particles stay connected")
        self.assertNotEqual(self.window.norm_label.text(),
                            "W norm: -")
        self.assertNotEqual(self.window.spectral_label.text(),
                            "Top singular value: -")


if __name__ == "__main__":
    unittest.main()
