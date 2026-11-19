"""ACR v0.13 tests: the second user-audit fix list.

The audit ran shipped v0.11 as a user (~/workspace/acr/
USER-AUDIT-2026-11-17.md) and found no blocks: 4 annoyances
(A1-A4) and 3 nice-to-haves (N1-N3). These tests pin the
fixes, plus the install-docs follow-up from the no-terminal
rule (release and README instructions lead with the
double-click, never the terminal).

Run with the Qt test env: LD_LIBRARY_PATH to the extracted
libs, QT_QPA_PLATFORM=offscreen, TMPDIR=/tmp, default HOME.
"""

import os
import sys
import tempfile
import time
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

from PySide6.QtCore import (QEvent, QPoint, QPointF,  # noqa: E402
                            Qt)
from PySide6.QtGui import QMouseEvent  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import (QApplication, QMessageBox)  # noqa: E402

import acr_app  # noqa: E402
from acr_app import CHART_SPIKE, MainWindow, NormChart  # noqa: E402

APP = None

THEME = ["quantum entanglement links particles",
         "entangled particles share a state",
         "quantum particles stay connected",
         "entanglement links distant particles"]
FISCAL = ["the quarterly budget needs approval",
          "fiscal policy shapes interest rates",
          "the treasury issued new bonds today",
          "inflation rose faster than expected"]
OFF6 = ["the soup needs more salt before serving",
        "the goalkeeper saved the penalty kick",
        "my bicycle tire went flat on the way home",
        "the meeting starts at nine on tuesday",
        "she watered the tomatoes in the garden",
        "the train leaves the station at dawn"]


def mixed_lines():
    """The audit's mixed.txt: 12 quantum + 12 fiscal + 6
    off-topic lines. Two themes."""
    return ([THEME[i % 4] for i in range(12)]
            + [FISCAL[i % 4] for i in range(12)] + list(OFF6))


def setUpModule():
    global APP
    APP = QApplication.instance()
    if APP is None:
        APP = QApplication(sys.argv)


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


# ---------------------------------------------------------------- A3


class RestoreOfferTests(WindowCase):
    """A3: the restore offer has three choices. Not now keeps
    the autosaved session; only Discard deletes it."""

    def _fake_box_class(self, button):
        """A QMessageBox stand-in (the DebugReportTests
        pattern): the dialog's buttons record themselves by
        text, and the clicked button is the chosen one."""
        box = mock.MagicMock()
        buttons = {}

        def _add(text, role=None):
            btn = mock.MagicMock()
            buttons[text] = btn
            return btn

        box.addButton.side_effect = _add
        box.clickedButton.side_effect = (
            lambda: buttons[button])
        return mock.MagicMock(return_value=box)

    def _autosave_path(self):
        return os.path.join(self.tmp.name, "autosave.json")

    def _seed_and_relaunch(self, steps=3):
        """Step on this window, close it dirty, and build the
        fresh window the next launch would show."""
        self._step_theme(steps)
        self.window.close()
        self.window.deleteLater()
        QTest.qWait(50)
        self.assertTrue(os.path.exists(self._autosave_path()))
        self.window = MainWindow(log_dir=self.tmp.name)
        self.window.show()

    def test_not_now_keeps_the_autosave(self):
        self._seed_and_relaunch()
        with mock.patch("acr_app.QMessageBox",
                        self._fake_box_class("Not now")):
            self.assertFalse(self.window.maybe_offer_restore())
        # The only copy of the session survives "not now", and
        # nothing was restored.
        self.assertTrue(os.path.exists(self._autosave_path()))
        self.assertEqual(self.window.step_label.text(),
                         "Step: 0")
        self.assertIn("Kept the autosaved session",
                      self.window.status_label.text())

    def test_discard_deletes_the_autosave(self):
        self._seed_and_relaunch()
        with mock.patch("acr_app.QMessageBox",
                        self._fake_box_class("Discard")):
            self.assertFalse(self.window.maybe_offer_restore())
        self.assertFalse(os.path.exists(self._autosave_path()))
        self.assertEqual(self.window.step_label.text(),
                         "Step: 0")
        self.assertIn("Discarded the autosaved session",
                      self.window.status_label.text())

    def test_offer_shape_and_default(self):
        # The real dialog: three buttons in order, Restore is
        # the default (Discard never is), the text says only
        # Discard deletes, and closing without a choice counts
        # as Not now.
        captured = []

        def fake_exec(box):
            captured.append(box)
            return 0

        with mock.patch.object(QMessageBox, "exec", fake_exec):
            choice = self.window._restore_offer_choice(
                3, "21:35")
        self.assertEqual(choice, "not now")
        box = captured[0]
        texts = [button.text() for button in box.buttons()]
        self.assertEqual(texts,
                         ["Restore", "Not now", "Discard"])
        self.assertEqual(box.defaultButton().text(), "Restore")
        self.assertIn("Only Discard deletes",
                      box.informativeText())


# ---------------------------------------------------------------- A1


class EmptyRunTests(WindowCase):
    """A1: a run that lands no steps shows its own empty
    result, never the previous run's numbers."""

    def test_empty_run_replaces_the_run_view(self):
        mixed = self._fixture("mixed.txt", mixed_lines())
        self.assertTrue(self._run_batch([mixed], "mixed.txt"))
        analyze = self.window.analyze_window
        self.assertIn("2 themes found in 30 steps in this run",
                      analyze.hero_title.text())
        empty = self._fixture("empty.txt", [])
        self.assertTrue(self._run_batch([empty], "empty.txt"))
        # The hero is the empty run's own result now.
        self.assertEqual(analyze.hero_title.text(),
                         "0 themes found in 0 steps in this run")
        self.assertIn("Nothing in the file became a step",
                      analyze.hero_theme.text())
        self.assertEqual(analyze._summary["steps"], 0)
        self.assertEqual(self.window._run_trace(), [])
        self.assertIn("0 steps in this run",
                      self.window.status_label.text())
        # Exports follow the run: with no steps there is
        # nothing to export.
        self.assertFalse(analyze.export_packs_button.isEnabled())
        self.assertFalse(
            analyze.export_report_button.isEnabled())


# ---------------------------------------------------------------- A2


class WrittenPreviewTests(WindowCase):
    """A2: a written pack preview is marked written, and goes
    out of date on any later change, like a pending one."""

    def _preview(self):
        path = self._fixture(
            "theme.txt", [THEME[i % 4] for i in range(16)])
        self.assertTrue(self._run_batch([path], "theme.txt"))
        self.window.show_analyze_window()
        analyze = self.window.analyze_window
        analyze.on_export_stella_packs()
        self.assertTrue(analyze.pack_preview.isVisible())
        return analyze

    def _write(self, analyze):
        dest = os.path.join(self.tmp.name, "packs")
        with mock.patch(
                "acr_app.QFileDialog.getExistingDirectory",
                return_value=dest):
            analyze.on_write_stella_packs()
        self.assertIn("Wrote", self.window.status_label.text())

    def test_written_preview_is_marked_written(self):
        analyze = self._preview()
        self.assertFalse(analyze.pack_written_label.isVisible())
        self.assertFalse(analyze.pack_stale_label.isVisible())
        self._write(analyze)
        self.assertTrue(analyze.pack_written_label.isVisible())
        self.assertIn("written to disk",
                      analyze.pack_written_label.text())
        self.assertFalse(analyze.pack_stale_label.isVisible())

    def test_written_preview_goes_stale_after_a_later_run(self):
        analyze = self._preview()
        self._write(analyze)
        self.assertTrue(analyze.pack_written_label.isVisible())
        # A later run moves the session: the written preview
        # is out of date now, and says so.
        path = self._fixture(
            "theme2.txt", [THEME[i % 4] for i in range(16)])
        self.assertTrue(self._run_batch([path], "theme2.txt"))
        self.assertTrue(analyze.pack_stale_label.isVisible())
        self.assertIn("out of date",
                      analyze.pack_stale_label.text())
        self.assertFalse(analyze.pack_written_label.isVisible())
        # A fresh preview clears both marks.
        analyze.on_export_stella_packs()
        self.assertFalse(analyze.pack_stale_label.isVisible())
        self.assertFalse(analyze.pack_written_label.isVisible())

    def test_written_preview_goes_stale_after_new_steps(self):
        analyze = self._preview()
        self._write(analyze)
        self._step_text("the toaster needs a new heating "
                        "element")
        self.assertTrue(analyze.pack_stale_label.isVisible())
        self.assertFalse(analyze.pack_written_label.isVisible())


# ---------------------------------------------------------------- A4


class AnalyzeSizeTests(WindowCase):
    """A4: with the pack preview shown, the Analyze window's
    minimum height is at most about 620 px (it was about
    690 px in the audit, 729 px in this test environment).
    Measured now: minimumSizeHint 577 px; the enforced
    minimumSize is 598 px even after several runs (wrapped
    labels grow it a little at the minimum width)."""

    def test_min_height_with_preview_shown(self):
        path = self._fixture(
            "theme.txt", [THEME[i % 4] for i in range(16)])
        self.assertTrue(self._run_batch([path], "theme.txt"))
        self.window.show_analyze_window()
        analyze = self.window.analyze_window
        analyze.on_export_stella_packs()
        self.assertTrue(analyze.pack_preview.isVisible())
        self.assertLessEqual(
            analyze.minimumSizeHint().height(), 620)
        self.assertLessEqual(analyze.minimumSize().height(), 620)
        analyze.resize(1000, 500)
        QTest.qWait(100)
        self.assertLessEqual(analyze.height(), 620)


# ---------------------------------------------------------------- N3


class InstallerPinTests(unittest.TestCase):
    """N3: the installer pins the interface library."""

    def test_pyside6_is_pinned(self):
        root = os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))
        with open(os.path.join(root, "install.sh"),
                  encoding="utf-8") as fh:
            text = fh.read()
        self.assertIn(
            "pip install PySide6-Essentials==6.12.0", text)
        # No unpinned install line remains.
        for line in text.splitlines():
            if "pip install" in line and "PySide6" in line:
                self.assertIn("==6.12.0", line)


# ---------------------------------------------------------------- N1


class ThemeItemTests(WindowCase):
    """N1: theme list items lead with the entry text; the
    step and peak numbers follow in parentheses."""

    def test_theme_item_leads_with_the_entry(self):
        path = self._fixture(
            "theme.txt", [THEME[i % 4] for i in range(16)])
        self.assertTrue(self._run_batch([path], "theme.txt"))
        analyze = self.window.analyze_window
        self.assertEqual(analyze.themes_list.count(), 1)
        item = analyze.themes_list.item(0)
        text = item.text()
        # The trigger of the canonical run is step 12's entry.
        self.assertTrue(text.startswith(
            "entanglement links distant particles"))
        self.assertIn("(step 12, peak 0.522, steps 1-16)",
                      text)
        self.assertEqual(item.data(Qt.UserRole), 12)
        self.assertEqual(item.toolTip(),
                         "entanglement links distant particles")


# ---------------------------------------------------------------- N2


class ChartDensityTests(unittest.TestCase):
    """N2: at 300+ steps the flag dots draw smaller and
    semi-transparent, so flags stay readable as events
    instead of merging into a solid band. Hover and click
    keep working: they are index-based, not pixel-based."""

    def setUp(self):
        self.chart = NormChart()
        self.chart.resize(1000, 200)
        self.chart.show()

    def tearDown(self):
        self.chart.close()
        self.chart.deleteLater()

    def _flag_center(self, index, count):
        # The same geometry paintEvent uses.
        left, width = 48, self.chart.width() - 48 - 10
        x = left + width * index / (count - 1.0)
        height = self.chart.height() - 10 - 22
        y = 10 + height * (1.0 - 1.0 / 1.05)
        return int(x), int(y)

    def test_sparse_flags_are_full_colour(self):
        flags = [False] * 16
        flags[5] = True
        self.chart.set_data([1.0] * 16, flags)
        QTest.qWait(50)
        image = self.chart.grab().toImage()
        x, y = self._flag_center(5, 16)
        self.assertEqual(image.pixelColor(x, y), CHART_SPIKE)

    def test_dense_flags_are_softened(self):
        self.chart.set_data([1.0] * 312, [True] * 312)
        QTest.qWait(50)
        image = self.chart.grab().toImage()
        x, y = self._flag_center(100, 312)
        pixel = image.pixelColor(x, y)
        # Not the solid spike colour any more, but still a
        # red-dominant dot over the line and background.
        self.assertNotEqual(pixel, CHART_SPIKE)
        self.assertGreater(pixel.red(), pixel.green() + 30)
        self.assertGreater(pixel.red(), pixel.blue() + 30)

    def test_hover_and_click_still_work_when_dense(self):
        self.chart.set_data([1.0] * 312, [True] * 312)
        clicked = []
        self.chart.stepClicked.connect(clicked.append)
        x, y = self._flag_center(100, 312)
        # The test_v10 (R6) dispatch pattern: a real QMouseEvent
        # sent through the application (offscreen never
        # synthesizes hover from QTest.mouseMove).
        move = QMouseEvent(
            QEvent.Type.MouseMove, QPointF(x, y),
            QPointF(self.chart.mapToGlobal(QPoint(x, y))),
            Qt.MouseButton.NoButton, Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier)
        QApplication.sendEvent(self.chart, move)
        self.assertIn("Step 101", self.chart.toolTip())
        self.assertIn("new-layer candidate",
                      self.chart.toolTip())
        press = QMouseEvent(
            QEvent.Type.MouseButtonPress, QPointF(x, y),
            QPointF(self.chart.mapToGlobal(QPoint(x, y))),
            Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier)
        QApplication.sendEvent(self.chart, press)
        self.assertEqual(clicked, [100])


# ------------------------------------------------------- install docs


class InstallDocsTests(unittest.TestCase):
    """The no-terminal follow-up: the README's install
    instructions lead with the double-click; the terminal
    commands are a short fallback note."""

    def test_readme_install_leads_with_the_double_click(self):
        root = os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))
        with open(os.path.join(root, "README.md"),
                  encoding="utf-8") as fh:
            text = fh.read()
        section = text.split("## Install", 1)[1].split(
            "## Run", 1)[0]
        self.assertIn("double-click `Install-ACR.desktop`",
                      section)
        self.assertLess(
            section.index("double-click"),
            section.index("bash install.sh"))
        self.assertIn("Fallback", section)


class BuildListTests(unittest.TestCase):
    def test_zip_list_includes_v13_tests(self):
        import build_zip
        self.assertIn("tests/test_v13.py",
                      build_zip.PLAIN_FILES)


if __name__ == "__main__":
    unittest.main()
