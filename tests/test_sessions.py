"""Session tests (v0.4): save/load round-trip, trace CSV export,
readable session export, and the Reset confirm.

Engine round-trip tests run without Qt; the window tests run
offscreen against the real engine. Run with the Qt test env:
LD_LIBRARY_PATH to the extracted libs, QT_QPA_PLATFORM=offscreen,
TMPDIR=/tmp, default HOME.
"""

import csv
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

import acr_app  # noqa: E402
from acr_app import TRACE_COLUMNS, MainWindow  # noqa: E402
from acr_engine import DIM, Engine, SessionError  # noqa: E402

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


def _stepped_engine(count=6):
    engine = Engine()
    for i in range(count):
        engine.step(THEME[i % 4], alpha=0.3)
    return engine


class EngineRoundTripTests(unittest.TestCase):
    def test_state_round_trip_is_exact(self):
        engine = _stepped_engine(6)
        # Through JSON, the way a save file carries it.
        saved = json.loads(json.dumps(engine.state_dict()))
        restored = Engine()
        restored.load_state_dict(saved)
        self.assertEqual(restored.state_dict(), engine.state_dict())
        self.assertEqual(restored.w_norm(), engine.w_norm())
        self.assertEqual(restored.trace, engine.trace)
        self.assertEqual(restored.buffer, engine.buffer)
        self.assertEqual(restored.step_count, engine.step_count)
        self.assertEqual(restored.h_prev, engine.h_prev)
        self.assertEqual(restored.prev_resonance,
                         engine.prev_resonance)

    def test_restored_engine_continues_identically(self):
        # The next step predicts from h_prev and flags from
        # prev_resonance, so identical continuation proves both
        # were restored, not just the weights.
        engine = _stepped_engine(6)
        saved = json.loads(json.dumps(engine.state_dict()))
        restored = Engine()
        restored.load_state_dict(saved)
        for i in range(3):
            engine.step(THEME[i % 4], alpha=0.3)
            restored.step(THEME[i % 4], alpha=0.3)
        self.assertEqual(restored.trace, engine.trace)

    def test_empty_session_round_trip(self):
        fresh = Engine()
        saved = json.loads(json.dumps(fresh.state_dict()))
        stepped = _stepped_engine(4)
        stepped.load_state_dict(saved)
        self.assertEqual(stepped.step_count, 0)
        self.assertEqual(stepped.trace, [])
        self.assertEqual(stepped.buffer, [])
        self.assertIsNone(stepped.h_prev)
        self.assertEqual(stepped.w_norm(), fresh.w_norm())

    def test_state_dict_is_a_deep_copy(self):
        engine = _stepped_engine(3)
        state = engine.state_dict()
        state["u"][0][0] += 5.0
        state["trace"][0]["resonance"] = 0.0
        self.assertNotEqual(engine.u[0][0], state["u"][0][0])
        self.assertNotEqual(engine.trace[0]["resonance"], 0.0)

    def test_load_rejects_bad_factor_shape(self):
        engine = _stepped_engine(3)
        state = engine.state_dict()
        state["u"] = state["u"][:10]
        with self.assertRaises(SessionError):
            Engine().load_state_dict(state)
        state = engine.state_dict()
        state["v"][0] = state["v"][0][:2]
        with self.assertRaises(SessionError):
            Engine().load_state_dict(state)

    def test_load_rejects_bad_vector(self):
        engine = _stepped_engine(3)
        state = engine.state_dict()
        state["h_prev"] = [0.0] * (DIM - 1)
        with self.assertRaises(SessionError):
            Engine().load_state_dict(state)

    def test_load_rejects_trace_step_count_mismatch(self):
        engine = _stepped_engine(3)
        state = engine.state_dict()
        state["step_count"] = 99
        with self.assertRaises(SessionError):
            Engine().load_state_dict(state)
        state = engine.state_dict()
        state["trace"] = state["trace"][:2]
        with self.assertRaises(SessionError):
            Engine().load_state_dict(state)

    def test_load_rejects_bad_trace_entry(self):
        engine = _stepped_engine(3)
        state = engine.state_dict()
        del state["trace"][1]["resonance"]
        with self.assertRaises(SessionError):
            Engine().load_state_dict(state)

    def test_load_rejects_bad_buffer(self):
        engine = _stepped_engine(3)
        state = engine.state_dict()
        state["buffer"][0]["vector"] = [0.0] * 3
        with self.assertRaises(SessionError):
            Engine().load_state_dict(state)

    def test_load_rejects_non_dict(self):
        with self.assertRaises(SessionError):
            Engine().load_state_dict(["not", "a", "dict"])


class WindowSessionTests(unittest.TestCase):
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

    def _step_theme(self, count):
        for i in range(count):
            self._step_text(THEME[i % 4])

    def _path(self, name):
        return os.path.join(self.tmp.name, name)

    def test_save_file_shape(self):
        self._step_theme(3)
        path = self._path("session.json")
        self.window.save_session_to(path)
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        self.assertEqual(data["format"], "acr-session")
        self.assertEqual(data["version"], 2)
        self.assertEqual(data["app_version"], "0.5")
        self.assertIn("engine", data)
        self.assertIn("settings", data)
        self.assertIn("heat_history", data)
        self.assertEqual(data["engine"]["step_count"], 3)
        self.assertEqual(len(data["heat_history"]), 3)

    def test_save_load_round_trip_restores_readouts(self):
        self._step_theme(6)
        before = (self.window.step_label.text(),
                  self.window.norm_label.text(),
                  self.window.resonance_label.text(),
                  self.window.trace_table.rowCount(),
                  self.window.buffer_list.count())
        path = self._path("session.json")
        self.window.save_session_to(path)
        self.window.do_reset()
        self.assertEqual(self.window.step_label.text(), "Step: 0")
        self.window.load_session_from(path)
        after = (self.window.step_label.text(),
                 self.window.norm_label.text(),
                 self.window.resonance_label.text(),
                 self.window.trace_table.rowCount(),
                 self.window.buffer_list.count())
        self.assertEqual(after, before)

    def test_load_through_dialog(self):
        self._step_theme(3)
        path = self._path("dialog-load.json")
        self.window.save_session_to(path)
        self.window.do_reset()
        with mock.patch.object(
                acr_app.QFileDialog, "getOpenFileName",
                return_value=(path, "JSON files (*.json)")):
            self.window.on_load_session()
        self.assertEqual(self.window.step_label.text(), "Step: 3")
        self.assertIn("Session loaded from",
                      self.window.status_label.text())

    def test_save_load_restores_settings(self):
        self._step_theme(2)
        self.window.alpha_slider.setValue(25)
        self.window.cap_spin.setValue(2.0)
        self.window.freeze_check.setChecked(True)
        path = self._path("session.json")
        self.window.save_session_to(path)
        self.window.alpha_slider.setValue(10)
        self.window.cap_spin.setValue(1.0)
        self.window.freeze_check.setChecked(False)
        self.window.load_session_from(path)
        self.assertEqual(self.window.alpha_slider.value(), 25)
        self.assertEqual(self.window.alpha_label.text(), "Alpha: 0.25")
        self.assertEqual(self.window.cap_spin.value(), 2.0)
        self.assertTrue(self.window.freeze_check.isChecked())

    def test_save_load_restores_views(self):
        # 16 theme steps flags step 15 under the v0.2 engine, so
        # the chart, the heatmap, and the layer list all have data.
        self._step_theme(16)
        path = self._path("session.json")
        self.window.save_session_to(path)
        self.window.do_reset()
        self.assertEqual(len(self.window.norm_chart.values), 0)
        self.assertEqual(self.window.layer_list.count(), 0)
        self.window.load_session_from(path)
        self.assertEqual(len(self.window.norm_chart.values), 16)
        self.assertEqual(self.window.layer_list.count(), 1)
        self.assertEqual(self.window.layer_list.item(0).text(),
                         "Step 15: quantum particles stay connected "
                         "(resonance 0.482)")
        self.assertEqual(len(self.window.heatmap.values), DIM)
        self.assertIn("Step 16: most moved rows",
                      self.window.heat_summary.text())

    def test_save_load_empty_session(self):
        path = self._path("empty.json")
        self.window.save_session_to(path)
        self._step_theme(2)
        self.window.load_session_from(path)
        self.assertEqual(self.window.step_label.text(), "Step: 0")
        self.assertEqual(self.window.trace_table.rowCount(), 0)
        self.assertEqual(self.window.buffer_list.count(), 0)

    def test_load_rejects_unknown_version(self):
        self._step_theme(2)
        before = self.window.norm_label.text()
        data = self.window.session_dict()
        data["version"] = 99
        path = self._path("future.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh)
        with self.assertRaises(SessionError):
            self.window.load_session_from(path)
        # The live session is untouched.
        self.assertEqual(self.window.step_label.text(), "Step: 2")
        self.assertEqual(self.window.norm_label.text(), before)

    def test_load_rejects_missing_version(self):
        data = self.window.session_dict()
        del data["version"]
        path = self._path("noversion.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh)
        with self.assertRaises(SessionError):
            self.window.load_session_from(path)

    def test_load_rejects_wrong_format(self):
        path = self._path("other.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"format": "something-else", "version": 1}, fh)
        with self.assertRaises(SessionError):
            self.window.load_session_from(path)

    def test_load_rejects_bad_json(self):
        path = self._path("broken.json")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("{this is not json")
        with self.assertRaises(SessionError):
            self.window.load_session_from(path)

    def test_load_rejects_bad_engine_state(self):
        self._step_theme(2)
        data = self.window.session_dict()
        data["engine"]["u"] = data["engine"]["u"][:10]
        path = self._path("badstate.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh)
        with self.assertRaises(SessionError):
            self.window.load_session_from(path)
        self.assertEqual(self.window.step_label.text(), "Step: 2")

    def test_load_bad_file_shows_plain_status(self):
        self._step_theme(2)
        path = self._path("broken.json")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("{this is not json")
        with mock.patch.object(
                acr_app.QFileDialog, "getOpenFileName",
                return_value=(path, "JSON files (*.json)")):
            self.window.on_load_session()
        self.assertTrue(
            self.window.status_label.text().startswith("Load failed"))
        self.assertEqual(self.window.step_label.text(), "Step: 2")

    def test_save_through_dialog(self):
        self._step_theme(2)
        path = self._path("dialog.json")
        with mock.patch.object(
                acr_app.QFileDialog, "getSaveFileName",
                return_value=(path, "JSON files (*.json)")):
            self.window.on_save_session()
        self.assertTrue(os.path.exists(path))
        self.assertIn("Session saved to",
                      self.window.status_label.text())

    def test_export_csv_writes_header_and_rows(self):
        self._step_text("first sentence here")
        self._step_text("second sentence here")
        path = self._path("trace.csv")
        self.window.export_trace_csv_to(path)
        with open(path, encoding="utf-8", newline="") as fh:
            rows = list(csv.reader(fh))
        self.assertEqual(rows[0], TRACE_COLUMNS)
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[1][0], "1")
        self.assertEqual(rows[1][1], "first sentence here")
        self.assertEqual(rows[2][0], "2")
        self.assertEqual(rows[2][1], "second sentence here")

    def test_export_csv_quotes_commas(self):
        self._step_text("hello, world, with commas")
        path = self._path("commas.csv")
        self.window.export_trace_csv_to(path)
        with open(path, encoding="utf-8", newline="") as fh:
            rows = list(csv.reader(fh))
        self.assertEqual(rows[1][1], "hello, world, with commas")

    def test_export_session_text(self):
        self._step_theme(3)
        path = self._path("session.txt")
        self.window.export_session_to(path)
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        self.assertIn("ACR session export", text)
        self.assertIn("Steps: 3", text)
        self.assertIn("W norm:", text)
        self.assertIn("Resonance:", text)
        self.assertIn("Layer candidates:", text)
        # Inputs in time order.
        positions = [text.index(sentence) for sentence in THEME[:3]]
        self.assertEqual(positions, sorted(positions))
        self.assertIn("Step 1: %s" % THEME[0], text)
        self.assertIn("Step 3: %s" % THEME[2], text)

    def test_export_session_text_empty(self):
        path = self._path("empty.txt")
        self.window.export_session_to(path)
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        self.assertIn("Steps: 0", text)
        self.assertIn("No steps yet.", text)

    def test_reset_confirm_yes_clears(self):
        self._step_theme(3)
        with mock.patch.object(
                acr_app.QMessageBox, "question",
                return_value=QMessageBox.Yes) as asked:
            self.window.on_reset()
        self.assertIn("Clear the session and reset W_t?",
                      asked.call_args[0][2])
        self.assertEqual(self.window.step_label.text(), "Step: 0")
        self.assertEqual(self.window.trace_table.rowCount(), 0)
        self.assertEqual(self.window.buffer_list.count(), 0)

    def test_reset_confirm_no_keeps(self):
        self._step_theme(3)
        with mock.patch.object(
                acr_app.QMessageBox, "question",
                return_value=QMessageBox.No):
            self.window.on_reset()
        self.assertEqual(self.window.step_label.text(), "Step: 3")
        self.assertEqual(self.window.trace_table.rowCount(), 3)
        self.assertEqual(self.window.buffer_list.count(), 3)


if __name__ == "__main__":
    unittest.main()
