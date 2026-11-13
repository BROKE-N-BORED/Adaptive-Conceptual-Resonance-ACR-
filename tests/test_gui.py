"""GUI drives: the window, offscreen, against the real engine.

Run with the Qt test env: LD_LIBRARY_PATH to the extracted libs,
QT_QPA_PLATFORM=offscreen, TMPDIR=/tmp, default HOME.
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from acr_app import MainWindow  # noqa: E402

APP = None


def setUpModule():
    global APP
    APP = QApplication.instance()
    if APP is None:
        APP = QApplication(sys.argv)


class GuiDriveTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.window = MainWindow(log_dir=self.tmp.name)
        self.window.show()

    def tearDown(self):
        self.window.close()
        self.window.deleteLater()
        self.tmp.cleanup()

    def _step_text(self, text):
        self.window.input_edit.setPlainText(text)
        QTest.mouseClick(self.window.step_button, Qt.LeftButton)

    def test_step_updates_readouts_and_trace(self):
        self._step_text("quantum entanglement is strange")
        self.assertEqual(self.window.step_label.text(), "Step: 1")
        self.assertNotEqual(self.window.norm_label.text(), "W norm: -")
        self.assertNotEqual(self.window.resonance_label.text(),
                            "Resonance: -")
        self.assertEqual(self.window.trace_table.rowCount(), 1)
        cell = self.window.trace_table.item(0, 1).text()
        self.assertEqual(cell, "quantum entanglement is strange")
        self.assertEqual(self.window.buffer_list.count(), 1)

    def test_second_step_appends_row(self):
        self._step_text("first sentence here")
        self._step_text("second sentence here")
        self.assertEqual(self.window.step_label.text(), "Step: 2")
        self.assertEqual(self.window.trace_table.rowCount(), 2)

    def test_empty_input_does_nothing(self):
        self._step_text("   ")
        self.assertEqual(self.window.trace_table.rowCount(), 0)
        self.assertEqual(self.window.status_label.text(),
                         "Type something first.")

    def test_freeze_stops_norm_change(self):
        # v0.2 note: two warm-up steps, because the first step logs
        # without an update (no previous state to predict from).
        self._step_text("warm up the weights first")
        self._step_text("warm them up some more")
        before = self.window.norm_label.text()
        self.window.freeze_check.setChecked(True)
        self._step_text("this step is frozen")
        self.assertEqual(self.window.norm_label.text(), before)
        self.assertEqual(self.window.trace_table.rowCount(), 3)

    def test_alpha_slider_updates_label(self):
        self.window.alpha_slider.setValue(25)
        self.assertEqual(self.window.alpha_label.text(), "Alpha: 0.25")

    def test_default_alpha_is_030(self):
        # v0.2: the default is 0.30, the setting the resonance
        # climb was verified at; at 0.10 the climb is much slower.
        self.assertEqual(self.window.alpha_slider.value(), 30)
        self.assertEqual(self.window.alpha_label.text(), "Alpha: 0.30")

    def test_synthesis_shows_neighbours(self):
        self._step_text("quantum entanglement links particles")
        self._step_text("baking bread needs warm dough")
        self._step_text("quantum particles stay connected")
        text = self.window.synthesis_text.toPlainText()
        self.assertIn("Closest past inputs:", text)
        self.assertIn("quantum particles stay connected", text)

    def test_export_writes_json(self):
        self._step_text("first sentence here")
        self._step_text("second sentence here")
        path = os.path.join(self.tmp.name, "trace.json")
        self.window.export_trace_to(path)
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        self.assertEqual(len(data), 2)
        self.assertEqual(data[0]["input"], "first sentence here")

    def test_debug_report_written(self):
        self._step_text("something to log")
        path = self.window.generate_debug_report()
        self.assertTrue(os.path.exists(path))
        with open(path, encoding="utf-8") as fh:
            report = fh.read()
        self.assertIn("DEBUG REPORT", report)
        self.assertIn("step 1", report)
        self.assertIn("Debug report:", self.window.status_label.text())

    def test_reset_clears(self):
        self._step_text("first sentence here")
        self.window.do_reset()
        self.assertEqual(self.window.step_label.text(), "Step: 0")
        self.assertEqual(self.window.trace_table.rowCount(), 0)
        self.assertEqual(self.window.buffer_list.count(), 0)
        self.assertEqual(self.window.resonance_label.text(),
                         "Resonance: -")

    def test_window_size(self):
        self.assertEqual(self.window.width(), 1100)
        self.assertEqual(self.window.height(), 680)


if __name__ == "__main__":
    unittest.main()
