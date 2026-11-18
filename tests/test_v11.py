"""ACR v0.11 tests: the user-audit task list (26 tasks).

The audit ran shipped v0.10 as a user and found 4 blocks, 15
annoys, and 7 nice-to-haves (~/workspace/acr/
USER-AUDIT-2026-11-16.md). These tests pin the fixes, including
regression tests from the audit's own repro fixtures: mixed.txt
(2 themes, 30 lines) and long.txt (2 themes, 312 lines), rebuilt
here from the audit's descriptions so the suite is
self-contained.

Run with the Qt test env: LD_LIBRARY_PATH to the extracted libs,
QT_QPA_PLATFORM=offscreen, TMPDIR=/tmp, default HOME.
"""

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import (QApplication, QMessageBox,  # noqa: E402
                               QTextEdit)

import acr_app  # noqa: E402
import acr_launch  # noqa: E402
from acr_app import (MainWindow, _span_text, _trim_display,  # noqa: E402
                     extract_batch_text, find_themes,
                     summarize_run)
from acr_engine import (COUNTER_EXAMPLE_MAX, Engine,  # noqa: E402
                        RESONANCE_THRESHOLD, RESONANT_FLOOR)
from tests.test_local_encoder import (FakeLocalHandler,  # noqa: E402
                                      FakeServerCase)

APP = None

THEME = ["quantum entanglement links particles",
         "entangled particles share a state",
         "quantum particles stay connected",
         "entanglement links distant particles"]
FISCAL = ["the quarterly budget needs approval",
          "fiscal policy shapes interest rates",
          "the treasury issued new bonds today",
          "inflation rose faster than expected"]
# The audit's off-topic lines (mixed.txt lines 25-30, long.txt
# tail): six distinct sentences, no shared theme.
OFF6 = ["the soup needs more salt before serving",
        "the goalkeeper saved the penalty kick",
        "my bicycle tire went flat on the way home",
        "the meeting starts at nine on tuesday",
        "she watered the tomatoes in the garden",
        "the train leaves the station at dawn"]


def mixed_lines():
    """The audit's mixed.txt: 12 quantum + 12 fiscal + 6
    off-topic. Two clear themes; v0.10 found 0."""
    return ([THEME[i % 4] for i in range(12)]
            + [FISCAL[i % 4] for i in range(12)] + list(OFF6))


def long_lines():
    """The audit's long.txt: fifteen 20-line blocks alternating
    quantum and fiscal, then the 6 off-topic lines twice. Two
    clear themes; v0.10 counted 27 spikes, then 29."""
    lines = []
    for block in range(15):
        sentences = THEME if block % 2 == 0 else FISCAL
        lines.extend(sentences[i % 4] for i in range(20))
    return lines + list(OFF6) * 2


def theme_engine(lines, alpha=0.3):
    engine = Engine()
    for line in lines:
        engine.step(line, alpha=alpha)
    return engine


def setUpModule():
    global APP
    APP = QApplication.instance()
    if APP is None:
        APP = QApplication(sys.argv)


# ------------------------------------------------------- B2/B4 themes


class ThemeDetectionTests(unittest.TestCase):
    def test_canonical_sixteen_is_one_theme(self):
        engine = theme_engine([THEME[i % 4] for i in range(16)])
        summary = summarize_run(engine.trace)
        self.assertEqual(len(summary["themes"]), 1)
        theme = summary["themes"][0]
        # The trigger is the first step that crossed the
        # calibrated 0.4 threshold: step 12, not the old
        # jump-rule flag at step 15.
        self.assertEqual(theme["id"], 12)
        self.assertEqual(theme["input"],
                         "entanglement links distant particles")
        self.assertEqual(theme["spans"], [(1, 16)])
        self.assertAlmostEqual(theme["peak"], 0.522, places=3)

    def test_mixed_fixture_finds_both_themes(self):
        engine = theme_engine(mixed_lines())
        summary = summarize_run(engine.trace)
        self.assertEqual(len(summary["themes"]), 2)
        triggers = sorted(t["id"] for t in summary["themes"])
        self.assertEqual(triggers, [12, 24])
        texts = sorted(t["input"] for t in summary["themes"])
        self.assertEqual(texts, ["entanglement links distant "
                                 "particles",
                                 "inflation rose faster than "
                                 "expected"])

    def test_long_fixture_finds_both_themes(self):
        engine = theme_engine(long_lines())
        summary = summarize_run(engine.trace)
        self.assertEqual(len(summary["themes"]), 2)
        triggers = sorted(t["id"] for t in summary["themes"])
        self.assertEqual(triggers, [12, 40])

    def test_off_topic_only_finds_no_themes(self):
        engine = theme_engine(OFF6)
        self.assertEqual(summarize_run(engine.trace)["themes"],
                         [])

    def test_repeated_off_topic_six_lines_finds_no_themes(self):
        # Six unrelated sentences, each repeated once: the
        # 6-cycle is wider than the segmentation window, so the
        # run splits at the cycle seam (steps 11-12 stand
        # alone, too short to be a theme) and the first stretch
        # repeats too thinly (ratio 1.67) for the weak rule. No
        # theme -- the right answer for unrelated lines.
        engine = theme_engine(OFF6 * 2)
        summary = summarize_run(engine.trace)
        self.assertEqual(summary["themes"], [])

    def test_short_cycle_of_unrelated_lines_reads_weak(self):
        # Documented honest limit: the weak rule fires on
        # sustained repetition, and a SHORT cycle of unrelated
        # lines (4 distinct, fitting the segmentation window)
        # stays one stretch, repeats 3 times, and peaks at
        # 0.184 -- just over the weak floor. The engine cannot
        # tell it from a repeated theme at this scale:
        # repetition is the only signal it has. A file of
        # short-cycle repeated boilerplate will read as a weak
        # theme.
        engine = theme_engine(OFF6[:4] * 3)
        summary = summarize_run(engine.trace)
        self.assertEqual(len(summary["themes"]), 1)
        self.assertAlmostEqual(summary["themes"][0]["peak"],
                               0.184, places=3)

    def test_lone_trailing_line_is_not_a_theme_member(self):
        # A lone off-topic line at the end of a run must not be
        # absorbed into the theme's spans: its pack row would
        # call it central to a theme it has nothing to do with.
        # The short tail stays its own segment, and the theme
        # ends where the words changed.
        engine = theme_engine(
            [THEME[i % 4] for i in range(16)]
            + ["the toaster needs a new heating element"])
        summary = summarize_run(engine.trace)
        self.assertEqual(len(summary["themes"]), 1)
        self.assertEqual(summary["themes"][0]["spans"],
                         [(1, 16)])

    def test_eight_theme_steps_find_no_theme(self):
        # Documented honest limit: 8 steps peak at 0.169, under
        # the 0.18 weak floor. The engine needs about 12 steps
        # of repetition to lock on; lowering the floor to admit
        # this case would leave a 0.013 margin to the off-topic
        # tail, which is threshold-fitting, not detection.
        engine = theme_engine([THEME[i % 4] for i in range(8)])
        self.assertEqual(summarize_run(engine.trace)["themes"],
                         [])

    def test_one_sentence_repeated_finds_no_theme(self):
        # Identical repeats have zero drift, so the drift gate
        # gives alpha 0 and W_t never learns: no resonance, no
        # theme. Correct, not a bug.
        engine = theme_engine([THEME[0]] * 20)
        self.assertEqual(summarize_run(engine.trace)["themes"],
                         [])

    def test_short_second_block_is_not_a_theme(self):
        # 16 theme steps plus 4 fiscal lines: the fiscal block
        # is too short to stand on its own.
        engine = theme_engine([THEME[i % 4] for i in range(16)]
                              + [FISCAL[i % 4] for i in range(4)])
        summary = summarize_run(engine.trace)
        self.assertEqual(len(summary["themes"]), 1)

    def test_theme_then_off_topic_stays_one_theme(self):
        engine = theme_engine([THEME[i % 4] for i in range(16)]
                              + OFF6)
        summary = summarize_run(engine.trace)
        self.assertEqual(len(summary["themes"]), 1)
        self.assertEqual(summary["themes"][0]["spans"], [(1, 16)])

    def test_threshold_is_calibrated(self):
        # The dead 0.75 is gone: the flag threshold is the
        # resonant floor, calibrated to the observed peaks
        # (0.46-0.63).
        self.assertEqual(RESONANCE_THRESHOLD, RESONANT_FLOOR)
        self.assertEqual(RESONANCE_THRESHOLD, 0.4)

    def test_canonical_flags_are_twelve_and_fifteen(self):
        engine = theme_engine([THEME[i % 4] for i in range(16)])
        flags = [e["id"] for e in engine.trace if e["new_layer"]]
        self.assertEqual(flags, [12, 15])
        # The update rule is untouched: the end state is the
        # v0.10 one.
        self.assertAlmostEqual(engine.w_norm(), 1.5938, places=3)
        self.assertAlmostEqual(engine.trace[-1]["resonance"],
                               0.5216, places=3)

    def test_core_and_outliers_are_exclusive(self):
        # A2: no entry in both lists, whatever the run.
        engine = theme_engine(mixed_lines())
        summary = summarize_run(engine.trace)
        core = {e["input"] for e in summary["core"]}
        outliers = {e["input"] for e in summary["outliers"]}
        self.assertFalse(core & outliers)
        engine = theme_engine(long_lines())
        summary = summarize_run(engine.trace)
        core = {e["input"] for e in summary["core"]}
        outliers = {e["input"] for e in summary["outliers"]}
        self.assertFalse(core & outliers)

    def test_fiscal_sentences_belong_to_the_fiscal_theme(self):
        # Span membership, not the resonance floor: F0/F1/F2
        # never score above 0.14, but they are in the fiscal
        # theme's spans, so they belong to it.
        from acr_packs import assign_theme
        engine = theme_engine(mixed_lines())
        summary = summarize_run(engine.trace)
        themes = summary["themes"]
        fiscal = [t for t in themes if t["id"] == 24][0]
        for step_id in (13, 14, 15, 16):
            entry = engine.trace[step_id - 1]
            self.assertIs(assign_theme(entry, themes), fiscal)
        # And the off-topic tail belongs to no theme.
        for step_id in (25, 26, 27, 28, 29, 30):
            entry = engine.trace[step_id - 1]
            self.assertIsNone(assign_theme(entry, themes))


class FrozenFlagTests(unittest.TestCase):
    def test_frozen_step_never_flags(self):
        # A10: drive 14 theme steps (resonance 0.219), then a
        # FROZEN step of the step-15 sentence. Unfrozen, that
        # step jumps to 0.482 (a 0.263 jump, over the 0.25
        # leg) and flags. Frozen, W_t did not move, so no new
        # layer was found -- whatever the resonance jump.
        engine = theme_engine([THEME[i % 4] for i in range(14)])
        entry = engine.step(THEME[14 % 4], alpha=0.3,
                            frozen=True)
        # The frozen resonance is 0.494, not the unfrozen
        # step-15 value 0.482: resonance is measured against
        # W_t after the update, and the frozen step has no
        # update. Either way it jumps 0.27 from 0.219 and
        # crosses 0.4 -- both flag legs would fire unfrozen.
        self.assertAlmostEqual(entry["resonance"], 0.494,
                               places=3)
        self.assertFalse(entry["new_layer"])
        # The control: the same step unfrozen flags.
        control = theme_engine([THEME[i % 4] for i in range(14)])
        entry = control.step(THEME[14 % 4], alpha=0.3)
        self.assertTrue(entry["new_layer"])

    def test_rank_frozen_steps_can_still_flag(self):
        # A partial freeze still adapts, so flagging stays
        # allowed there: the canonical flags survive.
        engine = Engine()
        for i in range(16):
            engine.step(THEME[i % 4], alpha=0.3,
                        frozen_ranks=[0])
        flags = [e["id"] for e in engine.trace if e["new_layer"]]
        self.assertEqual(flags, [12, 15])


class CounterExampleFloorTests(unittest.TestCase):
    def test_pure_theme_has_no_counter_examples(self):
        # A4: nothing in a pure theme run is far from the
        # theme, so the list is empty -- not three theme
        # sentences dressed up as "what this session is not".
        engine = theme_engine([THEME[i % 4] for i in range(16)])
        self.assertEqual(
            engine.synthesize()["counter_examples"], [])

    def test_counter_examples_are_all_below_the_floor(self):
        engine = theme_engine(mixed_lines())
        result = engine.synthesize()["counter_examples"]
        # Non-vacuous: the mixed run has far entries.
        self.assertTrue(result)
        for _text, score in result:
            self.assertLess(score, COUNTER_EXAMPLE_MAX)

    def test_current_input_is_not_its_own_neighbour(self):
        # N5: the neighbours list never holds the current
        # input, at any score.
        engine = theme_engine([THEME[i % 4] for i in range(16)])
        current = engine.buffer[-1]["text"]
        neighbours = engine.synthesize()["neighbours"]
        self.assertTrue(neighbours)
        self.assertNotIn(current,
                         [text for text, _s in neighbours])
        # The new pinned neighbours: 0.721/0.302/0.217 (was
        # 1.000/0.721/0.302).
        scores = [round(s, 3) for _t, s in neighbours]
        self.assertEqual(scores, [0.721, 0.302, 0.217])


class MarkdownTests(unittest.TestCase):
    def test_headers_fences_and_rules_are_skipped(self):
        # A7: markup is not content.
        self.assertIsNone(extract_batch_text("# Title", False,
                                             True))
        self.assertIsNone(extract_batch_text("## Section", False,
                                             True))
        self.assertIsNone(extract_batch_text("```", False, True))
        self.assertIsNone(extract_batch_text("```python", False,
                                             True))
        self.assertIsNone(extract_batch_text("---", False, True))
        self.assertIsNone(extract_batch_text("***", False, True))
        self.assertIsNone(extract_batch_text("___", False, True))

    def test_list_markers_are_stripped_not_stepped(self):
        # The item is content; only the marker is markup.
        self.assertEqual(extract_batch_text(
            "- quantum entanglement links particles", False,
            True), "quantum entanglement links particles")
        self.assertEqual(extract_batch_text(
            "* entangled particles share a state", False, True),
            "entangled particles share a state")
        self.assertEqual(extract_batch_text(
            "1. quantum particles stay connected", False, True),
            "quantum particles stay connected")
        self.assertEqual(extract_batch_text(
            "12. entanglement links distant particles", False,
            True), "entanglement links distant particles")

    def test_plain_lines_and_non_markdown_untouched(self):
        self.assertEqual(extract_batch_text(
            "quantum entanglement links particles", False, True),
            "quantum entanglement links particles")
        # In a .txt file a leading "# " is content, not markup.
        self.assertEqual(extract_batch_text("# Title", False,
                                            False), "# Title")
        self.assertEqual(extract_batch_text("- item", False,
                                            False), "- item")

    def test_trim_display(self):
        # A6: the shared trimmer.
        self.assertEqual(_trim_display("short", 40), "short")
        self.assertEqual(_trim_display("x" * 40, 40), "x" * 40)
        self.assertEqual(_trim_display("x" * 41, 40),
                         "x" * 37 + "...")
        self.assertEqual(_span_text([(1, 16)]), "1-16")
        self.assertEqual(_span_text([(1, 12), (25, 36)]),
                         "1-12 + 1 more")


# ------------------------------------------------------------ window base


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

    def _fixture(self, name, lines):
        path = os.path.join(self.tmp.name, name)
        with open(path, "w", encoding="utf-8") as fh:
            for line in lines:
                fh.write(line + "\n")
        return path

    def _run_batch(self, paths, source_note="fixture.txt",
                   timeout=60.0):
        self.window.start_batch(paths, source_note)
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.window._batch_thread is None:
                return True
            QTest.qWait(50)
        return False


# ---------------------------------------------------------------- B3


class NoThemePackTests(WindowCase):
    def test_no_theme_run_writes_no_packs(self):
        # B3: the audit's no-theme run labelled the file's core
        # entries "Off-topic" and taught STELLA backwards. Now
        # a no-theme run writes no packs at all, and says why.
        path = self._fixture("off.txt", OFF6)
        self.assertTrue(self._run_batch([path], "off.txt"))
        analyze = self.window.analyze_window
        self.assertEqual(analyze.hero_title.text(),
                         "0 themes found in 6 steps in this run")
        packs_dir = os.path.join(self.tmp.name, "packs")
        result = analyze.export_stella_packs_to(packs_dir)
        self.assertEqual(result["packs"], [])
        self.assertIn("No themes were found",
                      result["error"])
        self.assertFalse(os.path.exists(packs_dir))

    def test_no_off_topic_judgment_for_theme_members(self):
        # Mixed run: every judgment row about a quantum or
        # fiscal sentence says "Central to", never "Off-topic".
        path = self._fixture("mixed.txt", mixed_lines())
        self.assertTrue(self._run_batch([path], "mixed.txt"))
        packs_dir = os.path.join(self.tmp.name, "packs")
        result = self.window.analyze_window.\
            export_stella_packs_to(packs_dir)
        self.assertIsNone(result["error"])
        self.assertEqual(len(result["packs"]), 3)
        theme_texts = set(THEME) | set(FISCAL)
        for folder, _rows, _under in result["packs"]:
            with open(os.path.join(folder, "data.jsonl"),
                      encoding="utf-8") as fh:
                for line in fh:
                    row = json.loads(line)
                    if not row["prompt"].startswith(
                            "Is this central"):
                        continue
                    subject = row["prompt"].split("? ", 1)[1]
                    if subject in theme_texts:
                        self.assertTrue(
                            row["completion"].startswith(
                                "Central to"),
                            row)
                    else:
                        self.assertEqual(row["completion"],
                                         "Off-topic")


# ---------------------------------------------------------------- A3


class PerRunTests(WindowCase):
    def test_second_run_shows_only_its_results(self):
        # A3: analyzing a second file shows that file's
        # results, not the whole session's trace mixed in.
        first = self._fixture(
            "first.txt", [THEME[i % 4] for i in range(16)])
        self.assertTrue(self._run_batch([first], "first.txt"))
        analyze = self.window.analyze_window
        self.assertEqual(analyze.hero_title.text(),
                         "1 theme found in 16 steps in this run")
        second = self._fixture(
            "second.txt", [FISCAL[i % 4] for i in range(12)])
        self.assertTrue(self._run_batch([second], "second.txt"))
        self.assertEqual(analyze.hero_title.text(),
                         "1 theme found in 12 steps in this run")
        # The tiles hold the fiscal run: the theme trigger is
        # the fiscal peak step, and no quantum sentence is in
        # the core list.
        self.assertEqual(analyze.themes_list.count(), 1)
        core_texts = [analyze.core_list.item(i).toolTip()
                      for i in range(analyze.core_list.count())]
        self.assertTrue(any("fiscal" in text
                            or "budget" in text
                            or "treasury" in text
                            or "inflation" in text
                            for text in core_texts))
        self.assertFalse(any("quantum" in text
                             or "entanglement" in text
                             for text in core_texts))

    def test_manual_steps_do_not_extend_the_run(self):
        path = self._fixture(
            "theme.txt", [THEME[i % 4] for i in range(16)])
        self.assertTrue(self._run_batch([path], "theme.txt"))
        self._step_text("the toaster needs a new heating element")
        self._step_text("my car battery died this morning")
        analyze = self.window.analyze_window
        # The run is still the batch's 16 steps; the two typed
        # steps are session steps, not run steps.
        self.assertEqual(analyze.hero_title.text(),
                         "1 theme found in 16 steps in this run")
        self.assertEqual(self.window.step_label.text(),
                         "Step: 18")

    def test_packs_come_from_the_run_not_the_session(self):
        first = self._fixture(
            "first.txt", [THEME[i % 4] for i in range(16)])
        self.assertTrue(self._run_batch([first], "first.txt"))
        second = self._fixture(
            "second.txt", [FISCAL[i % 4] for i in range(12)])
        self.assertTrue(self._run_batch([second], "second.txt"))
        packs_dir = os.path.join(self.tmp.name, "packs")
        result = self.window.analyze_window.\
            export_stella_packs_to(packs_dir)
        self.assertIsNone(result["error"])
        # One pack: the fiscal theme. No quantum pack, no
        # off-topic pack (every core entry is in the theme).
        self.assertEqual(len(result["packs"]), 1)
        folder = result["packs"][0][0]
        # The fiscal theme's trigger, after 16 quantum steps,
        # is the peak step: "inflation rose faster than
        # expected" (measured, peak 0.236).
        self.assertIn("inflation-rose-faster-than-expected",
                      os.path.basename(folder))

    def test_hero_counts_real_themes(self):
        # B4: the audit's 2-theme file reported 27, then 29.
        # The hero counts themes now, not spike events.
        path = self._fixture("mixed.txt", mixed_lines())
        self.assertTrue(self._run_batch([path], "mixed.txt"))
        self.assertEqual(
            self.window.analyze_window.hero_title.text(),
            "2 themes found in 30 steps in this run")


# ---------------------------------------------------------------- B1


class AutosaveTests(WindowCase):
    def _autosave_path(self):
        return os.path.join(self.tmp.name, "autosave.json")

    def test_step_writes_autosave_and_marks_dirty(self):
        self.assertFalse(os.path.exists(self._autosave_path()))
        self._step_text("quantum entanglement links particles")
        self.assertTrue(os.path.exists(self._autosave_path()))
        with open(self._autosave_path(),
                  encoding="utf-8") as fh:
            data = json.load(fh)
        self.assertEqual(len(data["engine"]["trace"]), 1)
        self.assertTrue(self.window._dirty)
        self.assertTrue(self.window.dirty_marker.isVisible())
        self.assertIn("Autosaved",
                      self.window.saved_label.text())

    def test_explicit_save_clears_dirty_and_autosave(self):
        self._step_theme(3)
        path = os.path.join(self.tmp.name, "saved.json")
        with mock.patch.object(
                acr_app.QFileDialog, "getSaveFileName",
                return_value=(path, "JSON files (*.json)")):
            self.assertTrue(
                self.window._save_session_interactive())
        self.assertFalse(self.window._dirty)
        self.assertFalse(self.window.dirty_marker.isVisible())
        self.assertFalse(os.path.exists(self._autosave_path()))
        self.assertIn("Saved", self.window.saved_label.text())
        self.assertEqual(self.window._session_path, path)

    def test_ctrl_s_resaves_without_asking(self):
        self._step_theme(2)
        path = os.path.join(self.tmp.name, "saved.json")
        with mock.patch.object(
                acr_app.QFileDialog, "getSaveFileName",
                return_value=(path, "JSON files (*.json)")):
            self.window._save_session_interactive()
        self._step_theme(2)
        # The picker must not open: the path is remembered.
        with mock.patch.object(
                acr_app.QFileDialog, "getSaveFileName",
                side_effect=AssertionError("picker opened")):
            self.window.on_save_session()
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        self.assertEqual(len(data["engine"]["trace"]), 4)
        self.assertFalse(self.window._dirty)

    def test_dirty_quit_keeps_autosave_for_restore(self):
        self._step_theme(3)
        self.window.close()
        self.window.deleteLater()
        QTest.qWait(50)
        # A dirty quit keeps the recovery copy: the next launch
        # offers it back.
        self.assertTrue(os.path.exists(self._autosave_path()))
        self.window = MainWindow(log_dir=self.tmp.name)
        self.window.show()
        with mock.patch("acr_app.QMessageBox.question",
                        return_value=QMessageBox.Yes):
            self.assertTrue(self.window.maybe_offer_restore())
        self.assertEqual(self.window.step_label.text(),
                         "Step: 3")
        # Restored, not saved: dirty, and Ctrl+S asks where
        # (the autosave path must never become the save target).
        self.assertTrue(self.window._dirty)
        self.assertIsNone(self.window._session_path)

    def test_restore_declined_drops_the_autosave(self):
        self._step_theme(2)
        self.window.close()
        self.window.deleteLater()
        QTest.qWait(50)
        self.assertTrue(os.path.exists(self._autosave_path()))
        self.window = MainWindow(log_dir=self.tmp.name)
        self.window.show()
        with mock.patch("acr_app.QMessageBox.question",
                        return_value=QMessageBox.No):
            self.assertFalse(self.window.maybe_offer_restore())
        self.assertFalse(os.path.exists(self._autosave_path()))
        self.assertEqual(self.window.step_label.text(),
                         "Step: 0")

    def test_clean_quit_leaves_no_autosave(self):
        self._step_theme(2)
        path = os.path.join(self.tmp.name, "saved.json")
        with mock.patch.object(
                acr_app.QFileDialog, "getSaveFileName",
                return_value=(path, "JSON files (*.json)")):
            self.window._save_session_interactive()
        self.window.close()
        self.window.deleteLater()
        QTest.qWait(50)
        # Saved and clean: nothing to recover, no copy left.
        self.assertFalse(os.path.exists(self._autosave_path()))
        self.window = MainWindow(log_dir=self.tmp.name)
        self.window.show()

    def test_reset_drops_the_autosave(self):
        self._step_theme(2)
        self.assertTrue(os.path.exists(self._autosave_path()))
        self.window.do_reset()
        # A deliberate wipe drops the recovery copy too.
        self.assertFalse(os.path.exists(self._autosave_path()))
        self.assertFalse(self.window._dirty)
        self.assertIsNone(self.window._session_path)

    def test_corrupt_autosave_is_dropped_not_restored(self):
        with open(self._autosave_path(), "w",
                  encoding="utf-8") as fh:
            fh.write("{this is not json")
        self.assertFalse(self.window.maybe_offer_restore())
        self.assertFalse(os.path.exists(self._autosave_path()))
        self.assertIn("damaged",
                      self.window.status_label.text())

    def test_load_confirm_cancel_keeps_the_session(self):
        self._step_theme(3)
        saved = os.path.join(self.tmp.name, "other.json")
        self.window.save_session_to(saved)
        self._step_theme(2)
        with mock.patch.object(
                acr_app.QFileDialog, "getOpenFileName",
                return_value=(saved, "JSON files (*.json)")), \
                mock.patch("acr_app.QMessageBox.question",
                           return_value=QMessageBox.Cancel):
            self.window.on_load_session()
        self.assertEqual(self.window.step_label.text(),
                         "Step: 5")

    def test_load_confirm_discard_loads(self):
        self._step_theme(3)
        saved = os.path.join(self.tmp.name, "other.json")
        self.window.save_session_to(saved)
        self.window.do_reset()
        self._step_theme(2)
        with mock.patch.object(
                acr_app.QFileDialog, "getOpenFileName",
                return_value=(saved, "JSON files (*.json)")), \
                mock.patch("acr_app.QMessageBox.question",
                           return_value=QMessageBox.Discard):
            self.window.on_load_session()
        self.assertEqual(self.window.step_label.text(),
                         "Step: 3")
        self.assertEqual(self.window._session_path, saved)
        self.assertFalse(self.window._dirty)


# ---------------------------------------------------------------- A1


class InputClearTests(WindowCase):
    def test_input_clears_after_a_step(self):
        # A1: the box clears after each step, so the next
        # sentence starts empty instead of editing the last one.
        self._step_text("quantum entanglement links particles")
        self.assertEqual(self.window.input_edit.toPlainText(),
                         "")
        self._step_text("entangled particles share a state")
        self.assertEqual(self.window.input_edit.toPlainText(),
                         "")

    def test_input_clears_after_enter(self):
        self.window.input_edit.setPlainText(
            "quantum entanglement links particles")
        self.window.input_edit.setFocus()
        QTest.keyClick(self.window.input_edit, Qt.Key_Return)
        self.assertEqual(self.window.trace_table.rowCount(), 1)
        self.assertEqual(self.window.input_edit.toPlainText(),
                         "")


# ---------------------------------------------------------------- A5


class ButtonStateTests(WindowCase):
    def test_go_greyed_out_until_a_file_is_picked(self):
        # A5: Go cannot act with nothing picked, so it is
        # greyed out -- not a status-line refusal after the
        # click.
        analyze = self.window.analyze_window
        self.assertFalse(analyze.go_button.isEnabled())
        path = self._fixture(
            "theme.txt", [THEME[i % 4] for i in range(4)])
        with mock.patch.object(
                acr_app.QFileDialog, "getOpenFileName",
                return_value=(path, "Text files (*.txt)")):
            analyze.on_pick_file()
        self.assertTrue(analyze.go_button.isEnabled())

    def test_complete_greyed_out_without_resonant_inputs(self):
        # Complete needs resonant inputs; with none, it is
        # greyed out even in local mode with a model picked.
        # (The local half of this gate is covered by the
        # updated CompletionPickerTests in test_synthesis.py;
        # here the built-in half: no local encoder at all.)
        self.assertFalse(
            self.window.complete_button.isEnabled())
        self._step_theme(16)
        self.assertFalse(
            self.window.complete_button.isEnabled())


class SlowStepButtonTests(FakeServerCase):
    """A5: Step greys out while a local-model call is running.

    The fake service answers embeddings after a delay, so the
    call is still in flight when the test looks.
    """

    def setUp(self):
        super().setUp()
        FakeLocalHandler.embed_delay = 1.5
        self.tmp = tempfile.TemporaryDirectory()
        self.window = MainWindow(log_dir=self.tmp.name,
                                 local_base_url=self.base_url)
        self.window.show()

    def tearDown(self):
        self.window.close()
        self.window.deleteLater()
        self.tmp.cleanup()
        super().tearDown()

    def _wait_idle(self, timeout=25.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if (self.window._local_thread is None
                    and self.window._batch_thread is None):
                return True
            QTest.qWait(50)
        return False

    def _go_local(self):
        self.window.encoder_combo.setCurrentIndex(1)
        deadline = time.time() + 25.0
        while time.time() < deadline:
            if (self.window.model_combo.count() > 0
                    and self.window._local_thread is None):
                break
            QTest.qWait(50)
        combo = self.window.model_combo
        combo.setCurrentIndex(
            combo.findData("nomic-embed-text:latest"))
        deadline = time.time() + 25.0
        while time.time() < deadline:
            if (self.window.local_model
                    == "nomic-embed-text:latest"
                    and self.window._local_thread is None):
                return
            QTest.qWait(50)
        raise AssertionError("activation did not land")

    def test_step_disabled_during_a_slow_local_call(self):
        self._go_local()
        self.assertTrue(self.window.step_button.isEnabled())
        self.window.input_edit.setPlainText(
            "quantum entanglement links particles")
        QTest.mouseClick(self.window.step_button, Qt.LeftButton)
        # The call is in flight (1.5 s delay): Step is greyed
        # out, and a second click cannot pile on a call.
        self.assertFalse(self.window.step_button.isEnabled())
        QTest.mouseClick(self.window.step_button, Qt.LeftButton)
        self.assertTrue(self._wait_idle(),
                        "the local step did not land")
        self.assertEqual(self.window.step_label.text(),
                         "Step: 1")
        self.assertTrue(self.window.step_button.isEnabled())


# ---------------------------------------------------------------- A6


class ElisionTests(WindowCase):
    def test_long_finding_text_is_cut_with_dots(self):
        # A6: a finding longer than the tile limit shows "...",
        # so a clipped line never reads as a whole sentence.
        # The bicycle line is 42 chars: over the 40-char limit.
        path = self._fixture(
            "theme.txt",
            [THEME[i % 4] for i in range(16)]
            + ["my bicycle tire went flat on the way home"])
        self.assertTrue(self._run_batch([path], "theme.txt"))
        analyze = self.window.analyze_window
        shown = [analyze.core_list.item(i).text()
                 for i in range(analyze.core_list.count())]
        trimmed = [text for text in shown
                   if "bicycle" in text]
        self.assertEqual(len(trimmed), 1)
        self.assertTrue(trimmed[0].endswith("..."), trimmed[0])
        # The full text is still on the tooltip (R3).
        item = analyze.core_list.item(
            shown.index(trimmed[0]))
        self.assertIn("my bicycle tire went flat on the way "
                      "home", item.toolTip())

    def test_finding_lists_elide_instead_of_scrolling(self):
        # A6, caught by eyeballing the drive screenshots: the
        # text trim caps the item, but a medium item (a themes
        # line with its step/peak prefix) still overflowed the
        # narrow tile and grew a horizontal scrollbar, cut off
        # with no "..." in sight. The lists elide at the tile
        # edge instead: no horizontal scrollbar, ever.
        analyze = self.window.analyze_window
        for name in ("themes_list", "core_list",
                     "outlier_list", "shift_list"):
            lst = getattr(analyze, name)
            self.assertEqual(
                lst.horizontalScrollBarPolicy(),
                Qt.ScrollBarAlwaysOff, name)
            self.assertEqual(lst.textElideMode(),
                             Qt.ElideRight, name)


class MarkdownBatchTests(WindowCase):
    def test_markdown_file_steps_content_not_markup(self):
        # A7: headers, fences, and rules are skipped; list
        # items step without their markers.
        lines = ["# A theme file",
                 "",
                 "quantum entanglement links particles",
                 "entangled particles share a state",
                 "- quantum particles stay connected",
                 "1. entanglement links distant particles",
                 "```",
                 "some code here",
                 "```",
                 "---",
                 "quantum entanglement links particles"]
        path = self._fixture("notes.md", lines)
        self.assertTrue(self._run_batch([path], "notes.md"))
        # 6 lines stepped: the 2 plain lines, the 2 list items
        # (markers stripped), the line inside the code fence,
        # and the last line. The fence lines themselves are
        # markup and are skipped; the fenced content is a text
        # line like any other, so it steps.
        self.assertEqual(self.window.step_label.text(),
                         "Step: 6")
        inputs = [entry["input"]
                  for entry in self.window.engine.trace]
        self.assertNotIn("# A theme file", inputs)
        self.assertIn("some code here", inputs)
        self.assertNotIn("- quantum particles stay connected",
                         inputs)
        self.assertIn("quantum particles stay connected", inputs)
        self.assertIn("entanglement links distant particles",
                      inputs)
        self.assertIn("5 lines skipped",
                      self.window.status_label.text())


# ---------------------------------------------------------------- A8


class LaunchTests(unittest.TestCase):
    def test_raise_existing_when_locked(self):
        # A8: with the lock held by a live process, the launcher
        # asks the running instance to raise and reports it --
        # without importing the app. The lock must be held by a
        # DIFFERENT process than the one asking: POSIX locks are
        # per-process, so an in-process check would succeed
        # against its own lock. No server is listening here, so
        # the raise itself fails -- but the launcher still
        # reports the instance as running (the app's own check
        # is the net for that case).
        from PySide6.QtCore import QLockFile
        import tempfile as _tempfile
        name = "acr-v11-launch-test"
        lock_path = os.path.join(_tempfile.gettempdir(),
                                 "acr-%s.lock" % name)
        lock = QLockFile(lock_path)
        lock.setStaleLockTime(1000)
        self.assertTrue(lock.tryLock(0))
        root = os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))
        env = dict(os.environ)
        env["QT_QPA_PLATFORM"] = "offscreen"
        env["ACR_INSTANCE_NAME"] = name
        env["LD_LIBRARY_PATH"] = os.path.expanduser(
            "~/workspace/qt-libs/root/usr/lib/x86_64-linux-gnu")
        try:
            proc = subprocess.run(
                [sys.executable, "-c",
                 "import acr_launch; "
                 "print(acr_launch._try_raise_existing())"],
                cwd=root, env=env, capture_output=True,
                text=True, timeout=60)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(proc.stdout.strip(), "True")
        finally:
            lock.unlock()

    def test_no_raise_when_unlocked(self):
        name = "acr-v11-launch-free"
        with mock.patch.dict(os.environ,
                             {"ACR_INSTANCE_NAME": name}):
            self.assertFalse(acr_launch._try_raise_existing())

    def test_launcher_execs_the_app(self):
        # The unlocked path execs acr_app.py in the same
        # process, passing its arguments through: run it with
        # --smoke-test and the smoke test is what runs.
        root = os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))
        env = dict(os.environ)
        env["QT_QPA_PLATFORM"] = "offscreen"
        env["ACR_INSTANCE_NAME"] = "acr-v11-launch-smoke"
        env["LD_LIBRARY_PATH"] = os.path.expanduser(
            "~/workspace/qt-libs/root/usr/lib/x86_64-linux-gnu")
        proc = subprocess.run(
            [sys.executable,
             os.path.join(root, "acr_launch.py"), "--smoke-test"],
            capture_output=True, text=True, timeout=180,
            env=env, cwd=root)
        self.assertEqual(proc.returncode, 0, proc.stdout
                         + proc.stderr)
        self.assertIn("Smoke test OK", proc.stdout)


# ---------------------------------------------------------------- A9


class SmokeTests(unittest.TestCase):
    def test_smoke_test_flag_passes(self):
        # A9: the install smoke check -- the app starts
        # headless, drives 16 theme steps through the real
        # window, and prints OK.
        root = os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))
        env = dict(os.environ)
        env["QT_QPA_PLATFORM"] = "offscreen"
        env["ACR_INSTANCE_NAME"] = "acr-v11-smoke"
        env["LD_LIBRARY_PATH"] = os.path.expanduser(
            "~/workspace/qt-libs/root/usr/lib/x86_64-linux-gnu")
        proc = subprocess.run(
            [sys.executable,
             os.path.join(root, "acr_app.py"), "--smoke-test"],
            capture_output=True, text=True, timeout=180,
            env=env, cwd=root)
        self.assertEqual(proc.returncode, 0, proc.stdout
                         + proc.stderr)
        self.assertIn("Smoke test OK: 16 steps", proc.stdout)

    def test_install_sh_has_progress_and_smoke_check(self):
        # A9: the installer narrates its phases, shows pip
        # progress, and smoke-checks before writing launchers.
        root = os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))
        with open(os.path.join(root, "install.sh"),
                  encoding="utf-8") as fh:
            text = fh.read()
        self.assertIn("Downloading PySide6", text)
        self.assertIn("Checking the app starts", text)
        self.assertIn("acr_app.py --smoke-test", text)
        self.assertIn("ACR_INSTANCE_NAME=acr-smoke-test", text)
        # The smoke check comes before the launchers are written.
        self.assertLess(text.index("--smoke-test"),
                        text.index("ACR.desktop"))
        # pip shows its progress on the long step.
        self.assertIn("pip install PySide6-Essentials", text)
        self.assertNotIn("pip install --quiet "
                         "PySide6-Essentials", text)


# ----------------------------------------------------------- A11/A12/A13


class AnalyzeSizeTests(WindowCase):
    def test_analyze_window_shrinks_below_750px(self):
        # A11: the Analyze window used to stick at ~750 px
        # tall (minimum hint 663). Its minimum height measures
        # about 580-600 now, and it shrinks when asked -- well
        # under the audit's ~750 bar.
        analyze = self.window.analyze_window
        analyze.show()
        self.assertLess(
            analyze.minimumSizeHint().height(), 650)
        analyze.resize(1000, 400)
        QTest.qWait(100)
        self.assertLess(analyze.height(), 650)


class DebugReportTests(WindowCase):
    def _fake_box_class(self, button):
        """A QMessageBox stand-in: constructing it returns a
        mock dialog whose buttons record themselves, and whose
        clicked button is the chosen one."""
        box = mock.MagicMock()
        buttons = {}

        def _add(text, role=None):
            btn = mock.MagicMock()
            buttons[text] = btn
            return btn

        box.addButton.side_effect = _add
        box.clickedButton.side_effect = (
            lambda: buttons[button])
        box_class = mock.MagicMock(return_value=box)
        return box_class

    def test_open_folder_opens_the_report_folder(self):
        box_class = self._fake_box_class("Open folder")
        with mock.patch("acr_app.QMessageBox",
                        box_class), \
                mock.patch.object(acr_app.QDesktopServices,
                                  "openUrl") as opened:
            self.window.on_debug_report()
        self.assertEqual(opened.call_count, 1)
        url = opened.call_args[0][0]
        self.assertTrue(url.toLocalFile().endswith(
            self.tmp.name))

    def test_save_a_copy_copies_the_report(self):
        box_class = self._fake_box_class("Save a copy")
        dest = os.path.join(self.tmp.name, "copy.txt")
        with mock.patch("acr_app.QMessageBox",
                        box_class), \
                mock.patch.object(
                    acr_app.QFileDialog, "getSaveFileName",
                    return_value=(dest, "Text files (*.txt)")):
            self.window.on_debug_report()
        self.assertTrue(os.path.isfile(dest))
        with open(dest, encoding="utf-8") as fh:
            self.assertIn("DEBUG REPORT", fh.read())


class FlagLabelTests(WindowCase):
    def test_flag_label_names_the_flagged_step(self):
        # A13: a text label next to the flag dot.
        self.assertFalse(self.window.flag_label.isVisible())
        self._step_theme(16)
        self.assertTrue(self.window.flag_label.isVisible())
        self.assertEqual(self.window.flag_label.text(),
                         "Flag at step 15")
        self.window.do_reset()
        self.assertFalse(self.window.flag_label.isVisible())


# ----------------------------------------------------------- A14/A15/N1


class LocalWindowCase(FakeServerCase):
    def setUp(self):
        super().setUp()
        self.tmp = tempfile.TemporaryDirectory()
        self.window = MainWindow(log_dir=self.tmp.name,
                                 local_base_url=self.base_url)
        self.window.show()

    def tearDown(self):
        self.window.close()
        self.window.deleteLater()
        self.tmp.cleanup()
        super().tearDown()

    def _wait_idle(self, timeout=25.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if (self.window._local_thread is None
                    and self.window._batch_thread is None):
                return True
            QTest.qWait(50)
        return False

    def _switch_local(self):
        self.window.encoder_combo.setCurrentIndex(1)
        deadline = time.time() + 25.0
        while time.time() < deadline:
            if (self.window.model_combo.count() > 0
                    and self.window._local_thread is None):
                return
            QTest.qWait(50)
        raise AssertionError("model offer did not land")

    def _pick_model(self, name):
        combo = self.window.model_combo
        index = combo.findData(name)
        self.assertNotEqual(index, -1, "model %r not listed"
                            % name)
        combo.setCurrentIndex(index)
        deadline = time.time() + 25.0
        while time.time() < deadline:
            if (self.window.local_model == name
                    and self.window._local_thread is None):
                return
            QTest.qWait(50)
        raise AssertionError("model %r did not activate" % name)


class TextModelWarningTests(LocalWindowCase):
    def test_text_model_warns_and_no_keeps_builtin(self):
        # A14: picking a text model as the encoder warns
        # upfront. Answering No leaves the built-in encoder in
        # place and the picker unselected.
        self._switch_local()
        combo = self.window.model_combo
        with mock.patch("acr_app.QMessageBox.question",
                        return_value=QMessageBox.No):
            combo.setCurrentIndex(combo.findData("other:1b"))
        self.assertTrue(self._wait_idle())
        self.assertEqual(self.window.encoder_kind, "builtin")
        self.assertIsNone(self.window.local_model)
        self.assertEqual(combo.currentIndex(), -1)

    def test_text_model_yes_activates(self):
        # Answering Yes uses it anyway.
        self._switch_local()
        with mock.patch("acr_app.QMessageBox.question",
                        return_value=QMessageBox.Yes):
            self._pick_model("other:1b")
        self.assertEqual(self.window.local_model, "other:1b")
        self.assertEqual(self.window.encoder_kind, "local")

    def test_embedding_model_warns_never(self):
        # An embedding model activates with no warning at all:
        # the question is never asked.
        self._switch_local()
        with mock.patch("acr_app.QMessageBox.question",
                        side_effect=AssertionError(
                            "no warning expected")):
            self._pick_model("nomic-embed-text:latest")
        self.assertEqual(self.window.local_model,
                         "nomic-embed-text:latest")


class StalePreviewTests(WindowCase):
    def test_preview_goes_stale_when_the_session_moves(self):
        # A15: the pack preview names its step count; once the
        # session moves past it, the preview says it is out of
        # date instead of reading as current.
        path = self._fixture(
            "theme.txt", [THEME[i % 4] for i in range(16)])
        self.assertTrue(self._run_batch([path], "theme.txt"))
        self.window.show_analyze_window()
        analyze = self.window.analyze_window
        analyze.on_export_stella_packs()
        self.assertTrue(analyze.pack_preview.isVisible())
        self.assertFalse(analyze.pack_stale_label.isVisible())
        self._step_text("the toaster needs a new heating element")
        self.assertTrue(analyze.pack_stale_label.isVisible())
        self.assertIn("out of date",
                      analyze.pack_stale_label.text())
        # A fresh preview clears the stale mark.
        analyze.on_export_stella_packs()
        self.assertFalse(analyze.pack_stale_label.isVisible())


class ServiceNoteTests(WindowCase):
    def test_start_service_is_explained(self):
        # N1: first launch never said what the service is. The
        # tooltip and the plain line do now -- and the line is
        # visible in built-in mode, when the local chrome is
        # collapsed.
        self.assertIn("local model service",
                      self.window.start_button.toolTip())
        self.assertTrue(self.window.service_note.isVisible())
        self.assertIn("Start service starts the local model "
                      "service",
                      self.window.service_note.text())
        self.assertIn("keeps running after ACR closes",
                      self.window.service_note.text())


# ------------------------------------------------------------- N2/N3/N4


class ChartHonestyTests(WindowCase):
    def test_chart_notes_name_the_line_and_the_dots(self):
        # N2: the chart says what it shows -- the line is the
        # W_t norm from 0, the dots are resonance events.
        self.window.show_views_window()
        views = self.window.views_window
        notes = [child.text() for child in views.findChildren(
            type(self.window.encoder_label))]
        self.assertTrue(any(
            "The line is the W_t norm, from 0" in text
            for text in notes))
        self.assertTrue(any(
            "red dots are resonance events" in text
            for text in notes))
        analyze = self.window.analyze_window
        notes = [child.text() for child in analyze.findChildren(
            type(self.window.encoder_label))]
        self.assertTrue(any(
            "The line is the W_t norm over the run, from 0"
            in text for text in notes))

    def test_chart_paints_zero_based_without_crash(self):
        # The 0-based axis paints: grab a chart with values
        # near 1.6 and check it renders.
        self._step_theme(16)
        pixmap = self.window.norm_chart.grab()
        self.assertFalse(pixmap.isNull())


class RevertDetailTests(WindowCase):
    def test_revert_row_explains_its_resonance(self):
        # N3: a revert row's resonance is not a step resonance
        # -- the detail line says what it is. do_revert is the
        # no-confirm path (on_revert asks first).
        self._step_theme(6)
        self.window.do_revert(4)
        trace = self.window.engine.trace
        revert_id = trace[-1]["id"]
        self.assertIn("reverted_to", trace[-1])
        row = self.window._trace_row_for_step(revert_id)
        self.window.trace_table.setCurrentCell(row, 0)
        QTest.qWait(50)
        self.assertIn("Resonance here is the last input's match "
                      "against the restored weights",
                      self.window.trace_detail.text())


class SynthesisHeightTests(WindowCase):
    def test_synthesis_box_is_taller(self):
        # N4: the midpoint neighbours were buried below the
        # fold at 170 px; the box is 230 px now.
        self.assertEqual(
            self.window.synthesis_text.maximumHeight(), 230)


# ------------------------------------------------------------- N6/N7


class BufferTruncationTests(WindowCase):
    def test_long_buffer_entries_are_cut_with_dots(self):
        # N6: the latent buffer list truncates like the trace
        # table does, with "..." marking the cut.
        long_text = ("quantum entanglement links particles "
                     "across a very long distance and back")
        self.assertGreater(len(long_text), 60)
        self._step_text(long_text)
        item = self.window.buffer_list.item(0)
        self.assertTrue(item.text().endswith("..."),
                        item.text())
        self.assertLess(len(item.text()), len(long_text))
        # The full 80-char preview stays on the tooltip (R3).
        self.assertIn(long_text[:80], item.toolTip())


class ProgressResetTests(WindowCase):
    def test_cancelled_run_resets_the_progress_bar(self):
        # N7: after a cancelled run the bar goes back to 0,
        # not stuck where the run stopped.
        path = self._fixture(
            "big.txt", [THEME[i % 4] for i in range(280)])
        self.window.start_batch([path], "big.txt")
        QTest.qWait(300)
        self.window.analyze_window.on_cancel()
        deadline = time.time() + 25.0
        while time.time() < deadline:
            if self.window._batch_thread is None:
                break
            QTest.qWait(50)
        self.assertIsNone(self.window._batch_thread)
        self.assertEqual(
            self.window.analyze_window.progress_bar.value(), 0)
        self.assertIn("stopped",
                      self.window.status_label.text())


# -------------------------------------------------------------- version


class VersionTests(unittest.TestCase):
    def test_version_is_0_12_everywhere(self):
        self.assertEqual(acr_app.VERSION, "0.12")
        root = os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))
        with open(os.path.join(root, "README.md"),
                  encoding="utf-8") as fh:
            self.assertIn("v0.12", fh.readline())
        with open(os.path.join(root, "CHANGELOG.md"),
                  encoding="utf-8") as fh:
            self.assertIn("## v0.12", fh.read())
        with open(os.path.join(root, "build_zip.py"),
                  encoding="utf-8") as fh:
            text = fh.read()
        self.assertIn('VERSION = "0.12"', text)
        self.assertIn('ZIP_NAME = "acr-v0.12.zip"', text)


if __name__ == "__main__":
    unittest.main()
