"""Resonance view tests: chart, heatmap, layer view (v0.3).

Run with the Qt test env: LD_LIBRARY_PATH to the extracted libs,
QT_QPA_PLATFORM=offscreen, TMPDIR=/tmp, default HOME.
"""

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
from acr_engine import DIM  # noqa: E402

APP = None

THEME = ["quantum entanglement links particles",
         "entangled particles share a state",
         "quantum particles stay connected",
         "entanglement links distant particles"]


def setUpModule():
    global APP
    APP = QApplication.instance()
    if APP is None:
        APP = QApplication(sys.argv)


class VizTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.window = MainWindow(log_dir=self.tmp.name)
        self.window.show()
        self.window.tabs.setCurrentIndex(1)

    def tearDown(self):
        self.window.close()
        self.window.deleteLater()
        self.tmp.cleanup()

    def _step_text(self, text):
        self.window.input_edit.setPlainText(text)
        QTest.mouseClick(self.window.step_button, Qt.LeftButton)

    def _step_theme(self, count, alpha=30):
        self.window.alpha_slider.setValue(alpha)
        for i in range(count):
            self._step_text(THEME[i % len(THEME)])

    def test_chart_updates_after_steps(self):
        self._step_text("first sentence here")
        self._step_text("second sentence here")
        self._step_text("third sentence here")
        chart = self.window.norm_chart
        self.assertEqual(len(chart.values), 3)
        self.assertEqual(len(chart.flags), 3)
        self.assertEqual(chart.values[-1],
                         self.window.engine.trace[-1]["new_norm"])
        self.assertEqual(chart.values[0],
                         self.window.engine.trace[0]["new_norm"])

    def test_chart_flags_mirror_trace(self):
        # v0.3 note: under the v0.2 engine the first flag lands at
        # step 15 (the v0.1 engine flagged at step 11), so the
        # flag-bearing tests step 16 theme sentences.
        self._step_theme(16)
        chart = self.window.norm_chart
        trace = self.window.engine.trace
        self.assertEqual(chart.flags,
                         [entry["new_layer"] for entry in trace])
        self.assertTrue(any(chart.flags))

    def test_chart_paints_without_crash(self):
        self._step_theme(5)
        pixmap = self.window.norm_chart.grab()
        self.assertFalse(pixmap.isNull())
        self.assertGreater(pixmap.width(), 0)
        self.assertGreater(pixmap.height(), 0)

    def test_heatmap_shows_latest_step(self):
        self._step_text("first sentence here")
        self._step_text("second sentence here")
        heatmap = self.window.heatmap
        self.assertEqual(heatmap.step_id, 2)
        self.assertEqual(len(heatmap.values), DIM)
        self.assertGreater(max(heatmap.values), 0.0)
        self.assertEqual(self.window.heat_spin.value(), 2)
        self.assertIn("Step 2",
                      self.window.heat_summary.text())
        self.assertIn("most moved rows",
                      self.window.heat_summary.text())

    def test_heatmap_step_selector_looks_back(self):
        self._step_text("first sentence here")
        self._step_text("second sentence here")
        self._step_text("third sentence here")
        first_rows = self.window._heat_rows_for(1)
        self.window.heat_spin.setValue(1)
        heatmap = self.window.heatmap
        self.assertEqual(heatmap.step_id, 1)
        self.assertEqual(heatmap.values, first_rows)
        self.assertIn("Step 1",
                      self.window.heat_summary.text())
        # Picking the latest step again re-enables follow mode.
        self.window.heat_spin.setValue(3)
        self.assertEqual(self.window.heatmap.step_id, 3)
        self._step_text("fourth sentence here")
        self.assertEqual(self.window.heatmap.step_id, 4)
        self.assertEqual(self.window.heat_spin.value(), 4)

    def test_heatmap_pinned_step_stays_put(self):
        self._step_text("first sentence here")
        self._step_text("second sentence here")
        self.window.heat_spin.setValue(1)
        self._step_text("third sentence here")
        self.assertEqual(self.window.heatmap.step_id, 1)
        self.assertEqual(self.window.heat_spin.value(), 1)

    def test_heatmap_frozen_step_is_zero(self):
        self._step_text("warm up the weights first")
        self.window.freeze_check.setChecked(True)
        self._step_text("this step is frozen")
        heatmap = self.window.heatmap
        self.assertEqual(heatmap.step_id, 2)
        self.assertEqual(max(heatmap.values), 0.0)
        self.assertIn("did not move",
                      self.window.heat_summary.text())

    def test_heatmap_first_step_has_no_previous_state(self):
        # v0.3: under the v0.2 engine the first step logs without
        # an update (no previous state to predict from), so its
        # heatmap is empty and the summary says so — it is not a
        # frozen step.
        self._step_text("first sentence here")
        heatmap = self.window.heatmap
        self.assertEqual(heatmap.step_id, 1)
        self.assertEqual(max(heatmap.values), 0.0)
        self.assertIn("no previous step",
                      self.window.heat_summary.text())

    def test_heatmap_paints_without_crash(self):
        self._step_text("first sentence here")
        pixmap = self.window.heatmap.grab()
        self.assertFalse(pixmap.isNull())
        self.assertGreater(pixmap.width(), 0)

    def test_layer_view_lists_candidate(self):
        self._step_theme(16)
        self.assertGreaterEqual(self.window.layer_list.count(), 1)
        first = self.window.layer_list.item(0).text()
        self.assertIn("Step 15", first)
        self.assertIn("quantum particles stay connected", first)
        self.assertIn("resonance", first)

    def test_layer_click_shows_trace_entries(self):
        self._step_theme(16)
        self.window.layer_list.setCurrentRow(0)
        detail = self.window.layer_detail.toPlainText()
        self.assertIn("Step 15 (spike)", detail)
        self.assertIn("quantum particles stay connected", detail)
        self.assertIn("W norm:", detail)
        self.assertIn("Resonance:", detail)
        # Context entries around the spike are shown too.
        self.assertIn("Step 14 (before)", detail)
        self.assertIn("Step 16 (after)", detail)

    def test_layer_view_empty_before_spikes(self):
        self._step_text("one lonely sentence")
        self.assertEqual(self.window.layer_list.count(), 0)
        self.assertIn("No resonance spikes yet",
                      self.window.layer_detail.toPlainText())

    def test_reset_clears_views(self):
        self._step_theme(16)
        self.assertGreater(len(self.window.norm_chart.values), 0)
        self.assertGreater(self.window.layer_list.count(), 0)
        self.window.do_reset()
        self.assertEqual(self.window.norm_chart.values, [])
        self.assertEqual(self.window.norm_chart.flags, [])
        self.assertEqual(self.window.heatmap.values, [])
        self.assertEqual(self.window.heatmap.step_id, 0)
        self.assertEqual(self.window.layer_list.count(), 0)
        self.assertEqual(self.window.heat_summary.text(),
                         "No steps yet.")
        self.assertFalse(self.window.heat_spin.isEnabled())

    def test_views_tab_exists(self):
        self.assertEqual(self.window.tabs.count(), 2)
        self.assertEqual(self.window.tabs.tabText(0), "Trace log")
        self.assertEqual(self.window.tabs.tabText(1),
                         "Resonance views")


if __name__ == "__main__":
    unittest.main()
