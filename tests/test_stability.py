"""Stability + control tests (v0.7): freeze modes, schedule presets,
spectral-norm readout + cap flag, and revert.

Engine tests run without Qt; window tests run offscreen with the Qt
test env: LD_LIBRARY_PATH to the extracted libs,
QT_QPA_PLATFORM=offscreen, TMPDIR=/tmp, default HOME.
"""

import csv
import json
import math
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

import acr_app  # noqa: E402
from acr_app import MainWindow  # noqa: E402
from acr_engine import (  # noqa: E402
    RANK, Engine, SessionError, _dominant_direction,
    _materialize_w, schedule_alpha)
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

THEME = ["quantum entanglement links particles",
         "entangled particles share a state",
         "quantum particles stay connected",
         "entanglement links distant particles"]

APP = None


def setUpModule():
    global APP
    APP = QApplication.instance()
    if APP is None:
        APP = QApplication(sys.argv)


def _step_theme(engine, count, **kwargs):
    for i in range(count):
        engine.step(THEME[i % 4], **kwargs)


# ------------------------------------------------------------ schedules


class ScheduleTests(unittest.TestCase):
    def test_steady_is_constant_base(self):
        for n in (1, 2, 10, 50):
            self.assertEqual(schedule_alpha("steady", 0.3, n), 0.3)

    def test_fast_adapt_starts_at_twice_base(self):
        self.assertAlmostEqual(
            schedule_alpha("fast-adapt", 0.3, 1), 0.6)

    def test_fast_adapt_decays_toward_base(self):
        # The excess halves every 3 steps: 2x, 1.5x, 1.25x, 1.125x.
        self.assertAlmostEqual(
            schedule_alpha("fast-adapt", 0.3, 4), 0.45)
        self.assertAlmostEqual(
            schedule_alpha("fast-adapt", 0.3, 7), 0.375)
        self.assertAlmostEqual(
            schedule_alpha("fast-adapt", 0.3, 10), 0.3375)
        late = schedule_alpha("fast-adapt", 0.3, 40)
        self.assertAlmostEqual(late, 0.3, places=3)
        self.assertGreater(late, 0.3)

    def test_gentle_is_one_third_base(self):
        for n in (1, 5, 20):
            self.assertAlmostEqual(
                schedule_alpha("gentle", 0.3, n), 0.1)

    def test_unknown_preset_raises(self):
        with self.assertRaises(ValueError):
            schedule_alpha("turbo", 0.3, 1)


# --------------------------------------------------------- freeze ranks


class FreezeRanksTests(unittest.TestCase):
    def test_frozen_rank_is_exactly_unchanged(self):
        engine = Engine()
        _step_theme(engine, 4, alpha=0.3)
        u0 = [row[0] for row in engine.u]
        v0 = [row[0] for row in engine.v]
        u1 = [row[1] for row in engine.u]
        engine.step(THEME[0], alpha=0.3, frozen_ranks=[0])
        # Frozen: the whole column, U and V, byte for byte. The
        # decay skips frozen components too, or this would fail.
        self.assertEqual([row[0] for row in engine.u], u0)
        self.assertEqual([row[0] for row in engine.v], v0)
        # Unfrozen: the other directions still adapt.
        self.assertNotEqual([row[1] for row in engine.u], u1)

    def test_all_ranks_frozen_is_whole_freeze(self):
        engine = Engine()
        _step_theme(engine, 4, alpha=0.3)
        before = engine.w_norm()
        entry = engine.step(THEME[0], alpha=0.3,
                            frozen_ranks=list(range(RANK)))
        self.assertEqual(entry["old_norm"], entry["new_norm"])
        self.assertEqual(engine.w_norm(), before)

    def test_frozen_ranks_recorded_in_trace(self):
        engine = Engine()
        _step_theme(engine, 2, alpha=0.3)
        entry = engine.step(THEME[0], alpha=0.3,
                            frozen_ranks=[3, 1, 1])
        self.assertEqual(entry["frozen_ranks"], [1, 3])
        self.assertFalse(entry["frozen"])

    def test_whole_freeze_still_wins(self):
        engine = Engine()
        _step_theme(engine, 2, alpha=0.3)
        before = engine.w_norm()
        entry = engine.step(THEME[0], alpha=0.3, frozen=True,
                            frozen_ranks=[0])
        self.assertEqual(engine.w_norm(), before)
        self.assertTrue(entry["frozen"])

    def test_bad_rank_index_raises(self):
        engine = Engine()
        engine.step(THEME[0], alpha=0.3)
        with self.assertRaises(ValueError):
            engine.step(THEME[1], alpha=0.3, frozen_ranks=[RANK])
        with self.assertRaises(ValueError):
            engine.step(THEME[1], alpha=0.3, frozen_ranks=[-1])


# ------------------------------------------------------------- spectral


class SpectralNormTests(unittest.TestCase):
    def test_matches_naive_top_singular_value(self):
        engine = Engine()
        _step_theme(engine, 8, alpha=0.3)
        _vec, lam = _dominant_direction(
            _materialize_w(engine.u, engine.v))
        self.assertAlmostEqual(engine.spectral_norm(),
                               math.sqrt(lam), places=5)

    def test_never_exceeds_frobenius_norm(self):
        engine = Engine()
        for i in range(10):
            engine.step(THEME[i % 4], alpha=0.3)
            self.assertLessEqual(engine.spectral_norm(),
                                 engine.w_norm() + 1e-9)


# -------------------------------------------------------------- cap flag


class CapFlagTests(unittest.TestCase):
    def test_tiny_cap_flags_the_step(self):
        engine = Engine()
        engine.step(THEME[0], alpha=0.3, cap=0.0001)
        entry = engine.step(THEME[1], alpha=0.3, cap=0.0001)
        self.assertTrue(entry["capped"])

    def test_default_cap_does_not_flag_theme_steps(self):
        engine = Engine()
        for i in range(6):
            entry = engine.step(THEME[i % 4], alpha=0.3)
            self.assertFalse(entry["capped"])

    def test_first_step_never_capped(self):
        engine = Engine()
        entry = engine.step(THEME[0], alpha=0.3, cap=0.0001)
        self.assertFalse(entry["capped"])

    def test_frozen_step_never_capped(self):
        engine = Engine()
        _step_theme(engine, 2, alpha=0.3)
        entry = engine.step(THEME[0], alpha=0.3, cap=0.0001,
                            frozen=True)
        self.assertFalse(entry["capped"])


# ---------------------------------------------------------------- revert


class RevertTests(unittest.TestCase):
    def test_revert_restores_exact_snapshot(self):
        engine = Engine()
        _step_theme(engine, 6, alpha=0.3)
        # Capture the factors after step 4 by replaying on a twin.
        twin = Engine()
        _step_theme(twin, 4, alpha=0.3)
        engine.revert_to(4)
        self.assertEqual(engine.u, twin.u)
        self.assertEqual(engine.v, twin.v)

    def test_revert_adds_a_marked_trace_entry(self):
        engine = Engine()
        _step_theme(engine, 6, alpha=0.3)
        entry = engine.revert_to(4)
        self.assertEqual(entry["id"], 7)
        self.assertEqual(entry["input"], "Revert to step 4")
        self.assertEqual(entry["reverted_to"], 4)
        self.assertEqual(entry["new_norm"],
                         engine.trace[3]["new_norm"])
        # Later entries are never deleted: the log keeps all six
        # steps plus the revert.
        self.assertEqual(len(engine.trace), 7)
        self.assertEqual(engine.step_count, 7)

    def test_steps_continue_from_reverted_w(self):
        engine = Engine()
        _step_theme(engine, 6, alpha=0.3)
        engine.revert_to(4)
        entry = engine.step(THEME[0], alpha=0.3)
        # The next step starts from the reverted W_t, not the old
        # one: its old norm is step 4's new norm.
        self.assertEqual(entry["old_norm"],
                         engine.trace[3]["new_norm"])

    def test_revert_to_unknown_step_raises(self):
        engine = Engine()
        _step_theme(engine, 3, alpha=0.3)
        with self.assertRaises(ValueError):
            engine.revert_to(99)
        with self.assertRaises(ValueError):
            engine.revert_to(0)

    def test_snapshots_bounded_and_oldest_ages_out(self):
        engine = Engine(snapshot_limit=3)
        _step_theme(engine, 6, alpha=0.3)
        self.assertEqual(engine.snapshot_step_ids(), [4, 5, 6])
        with self.assertRaises(ValueError) as ctx:
            engine.revert_to(2)
        self.assertIn("aged out", str(ctx.exception))

    def test_reset_clears_snapshots(self):
        engine = Engine()
        _step_theme(engine, 4, alpha=0.3)
        engine.reset()
        self.assertEqual(engine.snapshot_step_ids(), [])
        with self.assertRaises(ValueError):
            engine.revert_to(1)

    def test_load_restarts_history_from_loaded_state(self):
        engine = Engine()
        _step_theme(engine, 6, alpha=0.3)
        state = engine.state_dict()
        loaded = Engine()
        loaded.load_state_dict(state)
        # Only the loaded state is revertable: earlier steps have
        # no snapshots, because snapshots are not saved.
        self.assertEqual(loaded.snapshot_step_ids(), [6])
        with self.assertRaises(ValueError):
            loaded.revert_to(3)
        entry = loaded.revert_to(6)
        self.assertEqual(entry["new_norm"],
                         engine.trace[5]["new_norm"])


class ResonantAfterRevertTests(unittest.TestCase):
    def test_resonant_entries_stay_correct_after_revert(self):
        engine = Engine()
        _step_theme(engine, 16, alpha=0.3)
        before = engine.resonant_entries()
        self.assertGreater(len(before), 0)
        engine.revert_to(10)
        after = engine.resonant_entries()
        # The buffer mapping must not shift onto the revert
        # entry: every resonant entry still points at a real step
        # with a matching resonance.
        for text, _vec, resonance, _flag, step_id in after:
            entry = engine.trace[step_id - 1]
            self.assertEqual(entry["id"], step_id)
            self.assertNotIn("reverted_to", entry)
            self.assertEqual(entry["input"], text)
            self.assertAlmostEqual(entry["resonance"], resonance)
        # And the synthesis head still works on the reverted W_t.
        result = engine.synthesize()
        self.assertIn("analogy", result)
        self.assertIn("counter_examples", result)


class OldTraceLoadTests(unittest.TestCase):
    def test_v06_trace_entries_load_with_defaults(self):
        # A v0.6 save has no frozen_ranks / capped keys: loading
        # fills the defaults instead of rejecting the file.
        engine = Engine()
        _step_theme(engine, 3, alpha=0.3)
        state = engine.state_dict()
        for entry in state["trace"]:
            del entry["frozen_ranks"]
            del entry["capped"]
        loaded = Engine()
        loaded.load_state_dict(state)
        for entry in loaded.trace:
            self.assertEqual(entry["frozen_ranks"], [])
            self.assertFalse(entry["capped"])
        self.assertEqual(loaded.step_count, 3)

    def test_bad_v07_trace_keys_rejected(self):
        engine = Engine()
        _step_theme(engine, 3, alpha=0.3)
        state = engine.state_dict()
        state["trace"][0]["frozen_ranks"] = [RANK]
        with self.assertRaises(SessionError):
            Engine().load_state_dict(state)
        state = engine.state_dict()
        state["trace"][0]["capped"] = "yes"
        with self.assertRaises(SessionError):
            Engine().load_state_dict(state)


# ---------------------------------------------------------------- window


class WindowCase(unittest.TestCase):
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


class FreezeModeWindowTests(WindowCase):
    def test_rank_checks_disabled_unless_rank_mode(self):
        for check in self.window.rank_checks:
            self.assertFalse(check.isEnabled())
        self.window.freeze_combo.setCurrentIndex(1)
        for check in self.window.rank_checks:
            self.assertFalse(check.isEnabled())
        self.window.freeze_combo.setCurrentIndex(2)
        for check in self.window.rank_checks:
            self.assertTrue(check.isEnabled())
        self.assertEqual(len(self.window.rank_checks), RANK)

    def test_rank_freeze_holds_one_direction_in_the_window(self):
        self._step_theme(4)
        self.window.freeze_combo.setCurrentIndex(2)
        self.window.rank_checks[0].setChecked(True)
        u0 = [row[0] for row in self.window.engine.u]
        u1 = [row[1] for row in self.window.engine.u]
        self._step_text(THEME[0])
        self.assertEqual([row[0] for row in self.window.engine.u],
                         u0)
        self.assertNotEqual([row[1] for row in self.window.engine.u],
                            u1)
        entry = self.window.engine.trace[-1]
        self.assertEqual(entry["frozen_ranks"], [0])

    def test_whole_freeze_mode_in_trace(self):
        self._step_theme(2)
        self.window.freeze_combo.setCurrentIndex(1)
        self._step_text(THEME[0])
        self.assertTrue(self.window.engine.trace[-1]["frozen"])


class PresetWindowTests(WindowCase):
    def test_fast_adapt_first_step_is_twice_base(self):
        self.window.schedule_combo.setCurrentIndex(1)
        self.assertEqual(self.window.alpha_label.text(),
                         "Alpha: 0.60")
        self._step_text(THEME[0])
        entry = self.window.engine.trace[-1]
        self.assertAlmostEqual(entry["alpha"], 0.6)
        # The trace table shows the effective alpha, not the
        # slider value.
        cell = self.window.trace_table.item(0, 6).text()
        self.assertEqual(cell, "0.60")

    def test_gentle_is_one_third_base(self):
        self.window.schedule_combo.setCurrentIndex(2)
        self.assertEqual(self.window.alpha_label.text(),
                         "Alpha: 0.10")
        self._step_text(THEME[0])
        self.assertAlmostEqual(
            self.window.engine.trace[-1]["alpha"], 0.1)

    def test_steady_default_is_base(self):
        self.assertEqual(self.window.schedule_combo.currentIndex(),
                         0)
        self.assertEqual(self.window.alpha_label.text(),
                         "Alpha: 0.30")
        self._step_text(THEME[0])
        self.assertAlmostEqual(
            self.window.engine.trace[-1]["alpha"], 0.3)

    def test_slider_moves_the_effective_label(self):
        self.window.schedule_combo.setCurrentIndex(1)
        self.window.alpha_slider.setValue(25)
        # Fast-adapt step 1 is 2x the new base of 0.25.
        self.assertEqual(self.window.alpha_label.text(),
                         "Alpha: 0.50")
        self.window.schedule_combo.setCurrentIndex(0)
        self.assertEqual(self.window.alpha_label.text(),
                         "Alpha: 0.25")


class SpectralWindowTests(WindowCase):
    def test_label_matches_engine(self):
        self._step_theme(4)
        expected = "Top singular value: %.4f" % (
            self.window.engine.spectral_norm())
        self.assertEqual(self.window.spectral_label.text(),
                         expected)

    def test_capped_column_marks_clipped_steps(self):
        # Cap 0.1 clips theme steps at alpha 0.30 (step size
        # about 0.14); the default cap 1.0 never clips them.
        self.window.cap_spin.setValue(0.1)
        self._step_theme(2)
        cell = self.window.trace_table.item(1, 9).text()
        self.assertEqual(cell, "yes")
        self.assertTrue(self.window.engine.trace[1]["capped"])
        self.window.do_reset()
        self.window.cap_spin.setValue(1.0)
        self._step_theme(2)
        cell = self.window.trace_table.item(1, 9).text()
        self.assertEqual(cell, "")
        self.assertFalse(self.window.engine.trace[1]["capped"])

    def test_csv_carries_the_capped_column(self):
        self.window.cap_spin.setValue(0.1)
        self._step_theme(2)
        path = self._path("trace.csv")
        self.window.export_trace_csv_to(path)
        with open(path, encoding="utf-8", newline="") as fh:
            rows = list(csv.reader(fh))
        self.assertEqual(rows[0][-1], "Capped")
        self.assertEqual(rows[2][-1], "yes")


class RevertWindowTests(WindowCase):
    def test_do_revert_restores_norm_and_logs_it(self):
        self._step_theme(6)
        target_norm = self.window.engine.trace[3]["new_norm"]
        self.window.do_revert(4)
        self.assertEqual(self.window.norm_label.text(),
                         "W norm: %.4f" % target_norm)
        self.assertEqual(self.window.step_label.text(), "Step: 7")
        self.assertEqual(self.window.trace_table.rowCount(), 7)
        last = self.window.engine.trace[-1]
        self.assertEqual(last["input"], "Revert to step 4")
        self.assertEqual(last["reverted_to"], 4)
        cell = self.window.trace_table.item(6, 1).text()
        self.assertEqual(cell, "Revert to step 4")
        self.assertEqual(self.window.status_label.text(),
                         "Reverted W_t to step 4.")
        # The heatmap covers the revert entry: W_t moved.
        self.assertEqual(self.window.heatmap.step_id, 7)
        self.assertGreater(max(self.window.heatmap.values), 0.0)

    def test_revert_confirm_yes_reverts(self):
        self._step_theme(6)
        self.window.trace_table.setCurrentCell(3, 0)
        with mock.patch.object(
                acr_app.QMessageBox, "question",
                return_value=QMessageBox.Yes) as asked:
            self.window.on_revert()
        self.assertIn("Revert W_t to the state after step 4?",
                      asked.call_args[0][2])
        self.assertEqual(self.window.step_label.text(), "Step: 7")
        self.assertEqual(
            self.window.engine.trace[-1]["reverted_to"], 4)

    def test_revert_confirm_no_keeps(self):
        self._step_theme(6)
        self.window.trace_table.setCurrentCell(3, 0)
        with mock.patch.object(
                acr_app.QMessageBox, "question",
                return_value=QMessageBox.No):
            self.window.on_revert()
        self.assertEqual(self.window.step_label.text(), "Step: 6")
        self.assertEqual(self.window.trace_table.rowCount(), 6)

    def test_revert_with_no_selection_says_so(self):
        self._step_theme(3)
        self.window.trace_table.setCurrentItem(None)
        self.window.trace_table.clearSelection()
        with mock.patch.object(
                acr_app.QMessageBox, "question",
                return_value=QMessageBox.Yes) as asked:
            self.window.on_revert()
        asked.assert_not_called()
        self.assertEqual(self.window.status_label.text(),
                         "Pick a step in the trace log first.")
        self.assertEqual(self.window.step_label.text(), "Step: 3")

    def test_revert_aged_out_says_so(self):
        self._step_theme(6)
        # Drop one snapshot by hand: the window must handle a
        # missing snapshot with a plain message, not a crash.
        del self.window.engine._snapshots[2]
        self.window.do_revert(2)
        self.assertIn("aged out", self.window.status_label.text())
        self.assertEqual(self.window.step_label.text(), "Step: 6")
        self.assertEqual(self.window.trace_table.rowCount(), 6)

    def test_revert_to_latest_step_reports_no_change(self):
        self._step_theme(4)
        self.window.do_revert(4)
        # W_t was already there: the heatmap says so instead of
        # calling it a frozen step.
        self.assertIn("reverted to step 4",
                      self.window.heat_summary.text())
        self.assertIn("did not change",
                      self.window.heat_summary.text())


class SessionStabilityTests(WindowCase):
    def test_save_load_restores_new_controls(self):
        self._step_theme(2)
        self.window.freeze_combo.setCurrentIndex(2)
        self.window.rank_checks[0].setChecked(True)
        self.window.rank_checks[5].setChecked(True)
        self.window.schedule_combo.setCurrentIndex(1)
        path = self._path("session.json")
        self.window.save_session_to(path)
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        self.assertEqual(data["settings"]["freeze_mode"], "ranks")
        self.assertEqual(data["settings"]["frozen_ranks"], [0, 5])
        self.assertEqual(data["settings"]["preset"], "fast-adapt")
        # Older readers still see the bool: ranks mode is not a
        # whole freeze.
        self.assertFalse(data["settings"]["frozen"])
        self.window.freeze_combo.setCurrentIndex(0)
        self.window.rank_checks[0].setChecked(False)
        self.window.rank_checks[5].setChecked(False)
        self.window.schedule_combo.setCurrentIndex(0)
        self.window.load_session_from(path)
        self.assertEqual(self.window.freeze_combo.currentIndex(), 2)
        self.assertTrue(self.window.rank_checks[0].isChecked())
        self.assertTrue(self.window.rank_checks[5].isChecked())
        self.assertFalse(self.window.rank_checks[1].isChecked())
        self.assertEqual(self.window.schedule_combo.currentIndex(),
                         1)

    def test_old_format_settings_load(self):
        # A v0.6 save: "frozen" bool only, no freeze_mode,
        # frozen_ranks, or preset. It loads as whole-freeze when
        # the bool is true, and steady otherwise.
        self._step_theme(2)
        path = self._path("old.json")
        self.window.save_session_to(path)
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        data["settings"] = {"alpha": 0.25, "cap": 1.0,
                            "frozen": True}
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh)
        self.window.load_session_from(path)
        self.assertEqual(self.window.freeze_combo.currentIndex(), 1)
        self.assertEqual(self.window.schedule_combo.currentIndex(),
                         0)
        data["settings"]["frozen"] = False
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh)
        self.window.load_session_from(path)
        self.assertEqual(self.window.freeze_combo.currentIndex(), 0)

    def test_bad_new_settings_rejected(self):
        self._step_theme(2)
        path = self._path("bad.json")
        self.window.save_session_to(path)
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        data["settings"]["preset"] = "turbo"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh)
        with self.assertRaises(SessionError):
            self.window.load_session_from(path)
        data["settings"]["preset"] = "steady"
        data["settings"]["frozen_ranks"] = [RANK]
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh)
        with self.assertRaises(SessionError):
            self.window.load_session_from(path)


if __name__ == "__main__":
    unittest.main()
