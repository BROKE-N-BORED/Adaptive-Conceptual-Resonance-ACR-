"""Final polish tests (v0.8): deduped neighbour lists, chart
legend placement, Enter to step, window icon, geometry memory,
and the build from a fresh clone.

Run with the Qt test env: LD_LIBRARY_PATH to the extracted libs,
QT_QPA_PLATFORM=offscreen, TMPDIR=/tmp, default HOME.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

from PySide6.QtCore import QPoint, QSettings, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from acr_app import MainWindow, NormChart  # noqa: E402
from acr_engine import Engine  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

THEME = (
    "quantum entanglement links particles",
    "quantum particles stay connected",
    "entangled particles share a state",
    "entanglement links distant particles",
)

APP = None


def setUpModule():
    global APP
    APP = QApplication.instance()
    if APP is None:
        APP = QApplication(sys.argv)


def _clear_settings():
    QSettings("BrokeNBored", "ACR").clear()


# ---------------------------------------------------------------- dedupe


class DedupeTests(unittest.TestCase):
    def _theme_engine(self, steps=16):
        engine = Engine()
        for i in range(steps):
            engine.step(THEME[i % 4])
        return engine

    def test_neighbours_show_each_text_once(self):
        result = self._theme_engine().synthesize()
        texts = [text for text, _ in result["neighbours"]]
        self.assertEqual(len(texts), len(set(texts)))
        self.assertLessEqual(len(texts), 3)
        self.assertGreater(len(texts), 0)

    def test_midpoint_shows_each_text_once(self):
        result = self._theme_engine().synthesize()
        texts = [text for text, _ in result["midpoint"]]
        self.assertEqual(len(texts), len(set(texts)))
        self.assertLessEqual(len(texts), 2)

    def test_top_neighbour_still_the_latest_input(self):
        result = self._theme_engine().synthesize()
        top_text, top_score = result["neighbours"][0]
        self.assertEqual(top_text, THEME[15 % 4])
        self.assertAlmostEqual(top_score, 1.0, places=3)

    def test_midpoint_between_two_distinct_neighbours(self):
        # With only two distinct inputs there is nothing left for
        # the midpoint to point at: the top two neighbours are
        # both of them.
        engine = Engine()
        engine.step(THEME[0])
        engine.step(THEME[1])
        engine.step(THEME[0])
        result = engine.synthesize()
        self.assertEqual(result["midpoint"], [])
        # With three distinct inputs the midpoint is non-empty
        # and never repeats the top two neighbours.
        engine.step(THEME[2])
        result = engine.synthesize()
        top_texts = {text for text, _ in result["neighbours"][:2]}
        mid_texts = [text for text, _ in result["midpoint"]]
        self.assertGreater(len(mid_texts), 0)
        for text in mid_texts:
            self.assertNotIn(text, top_texts)

    def test_panel_shows_no_repeated_lines(self):
        # The window-level check: within each neighbour section,
        # no line repeats. Sections may legitimately share a
        # text (a middle-ranked input can be both a neighbour
        # and a counter-example), so the check is per section.
        _clear_settings()
        tmp = tempfile.TemporaryDirectory()
        window = MainWindow(log_dir=tmp.name)
        window.show()
        try:
            for i in range(16):
                window.input_edit.setPlainText(THEME[i % 4])
                QTest.mouseClick(window.step_button, Qt.LeftButton)
            text = window.synthesis_text.toPlainText()
            sections = {"Closest past inputs:": [],
                        "Midpoint neighbours:": []}
            current = None
            for line in text.splitlines():
                if line in sections:
                    current = line
                elif line.startswith("  ") and current is not None:
                    sections[current].append(line.strip())
                elif not line.startswith(" "):
                    current = None
            for name, lines in sections.items():
                self.assertEqual(len(lines), len(set(lines)), name)
                self.assertGreater(len(lines), 0, name)
        finally:
            window.close()
            window.deleteLater()
            tmp.cleanup()
            _clear_settings()


# ---------------------------------------------------------------- legend


class LegendTests(unittest.TestCase):
    def test_legend_sits_bottom_left(self):
        chart = NormChart()
        chart.resize(500, 200)
        # A flagged last step at the top-right: the old legend
        # spot, where the dot used to hide behind the red text.
        chart.set_data([1.60, 1.58, 1.55, 1.62],
                       [False, False, False, True])
        rect = chart.legend_rect()
        self.assertGreater(rect.top(), chart.height() // 2)
        self.assertLess(rect.right(), chart.width() // 2)
        self.assertFalse(
            rect.contains(QPoint(chart.width() - 15, 15)))

    def test_legend_rect_is_sized_to_the_text(self):
        chart = NormChart()
        chart.resize(500, 200)
        rect = chart.legend_rect()
        self.assertGreater(rect.width(), 100)
        self.assertTrue(chart.rect().contains(rect))


# ------------------------------------------------------- enter / icon


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


class EnterToStepTests(WindowCase):
    def test_return_steps_the_loop(self):
        self.window.input_edit.setPlainText(
            "quantum entanglement links particles")
        self.window.input_edit.setFocus()
        QTest.keyClick(self.window.input_edit, Qt.Key_Return)
        self.assertEqual(self.window.trace_table.rowCount(), 1)
        self.assertEqual(self.window.step_label.text(), "Step: 1")

    def test_shift_return_does_not_step(self):
        self.window.input_edit.setPlainText("one line")
        self.window.input_edit.setFocus()
        QTest.keyClick(self.window.input_edit, Qt.Key_Return,
                       Qt.ShiftModifier)
        self.assertEqual(self.window.trace_table.rowCount(), 0)
        self.assertIn("\n",
                      self.window.input_edit.toPlainText())

    def test_return_on_empty_input_shows_note(self):
        self.window.input_edit.setFocus()
        QTest.keyClick(self.window.input_edit, Qt.Key_Return)
        self.assertEqual(self.window.trace_table.rowCount(), 0)
        self.assertEqual(self.window.status_label.text(),
                         "Type something first.")


class WindowIconTests(WindowCase):
    def test_window_icon_is_set(self):
        self.assertFalse(self.window.windowIcon().isNull())

    def test_application_name_matches_launcher(self):
        # ACR.desktop says StartupWMClass=ACR: the app must say
        # the same name, or the desktop groups the running window
        # separately from the launcher icon.
        self.assertEqual(QApplication.applicationName(), "ACR")

    def test_window_title_carries_version(self):
        self.assertIn("0.10", self.window.windowTitle())


# ------------------------------------------------------------- geometry


class GeometryTests(unittest.TestCase):
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

    def test_first_run_is_default_size(self):
        window = self._make_window()
        self.assertEqual(window.width(), 1100)
        self.assertEqual(window.height(), 700)

    def test_close_saves_geometry(self):
        first = self._make_window()
        first.resize(1100, 750)
        QApplication.processEvents()
        first.close()
        QApplication.processEvents()
        saved = QSettings("BrokeNBored", "ACR").value("geometry")
        self.assertIsNotNone(saved)
        self.assertGreater(len(bytes(saved)), 0)

    def test_size_is_remembered_across_runs(self):
        # The height round-trips exactly. The width cannot be
        # pinned here: the offscreen screen is 800 wide, so Qt
        # clamps a restored 1100-wide window to fit it (and the
        # layout then pushes it back to its 918 minimum on show).
        # The pre-show width still proves the restore ran: it is
        # the clamped saved value, not the 1100 default.
        first = self._make_window()
        first.resize(1100, 750)
        QApplication.processEvents()
        first.close()
        QApplication.processEvents()
        tmp = tempfile.TemporaryDirectory()
        self.tmps.append(tmp)
        second = MainWindow(log_dir=tmp.name)
        self.windows.append(second)
        self.assertEqual(second.height(), 750)
        self.assertNotEqual(second.width(), 1100)
        second.show()
        QApplication.processEvents()
        self.assertEqual(second.height(), 750)


# --------------------------------------------------------- shipped files


class ShippedFilesTests(unittest.TestCase):
    def test_changelog_ships_in_the_zip_list(self):
        import build_zip
        self.assertIn("CHANGELOG.md", build_zip.PLAIN_FILES)
        self.assertTrue(os.path.exists(
            os.path.join(ROOT, "CHANGELOG.md")))

    def test_readme_names_the_version(self):
        with open(os.path.join(ROOT, "README.md"),
                  encoding="utf-8") as fh:
            text = fh.read()
        self.assertIn("v0.10", text)
        self.assertIn("10 columns", text)


# ------------------------------------------------------- build from clone


class BuildFromCloneTests(unittest.TestCase):
    def test_build_without_pngs_renders_them(self):
        # A fresh clone holds only icons/acr.svg: the PNGs are
        # zip-only. The build must render them and succeed.
        tmp = tempfile.mkdtemp()
        try:
            clone = os.path.join(tmp, "acr-clone")
            shutil.copytree(
                ROOT, clone,
                ignore=shutil.ignore_patterns(
                    "__pycache__", "*.zip", "logs"))
            for name in ("icons/acr-48.png", "icons/acr-256.png"):
                os.remove(os.path.join(clone, name))
            proc = subprocess.run(
                [sys.executable, "build_zip.py"],
                cwd=clone, capture_output=True, text=True,
                timeout=180)
            self.assertEqual(proc.returncode, 0, proc.stdout
                             + proc.stderr)
            for name in ("icons/acr-48.png", "icons/acr-256.png"):
                path = os.path.join(clone, name)
                self.assertTrue(os.path.exists(path))
                self.assertGreater(os.path.getsize(path), 100)
            zip_path = os.path.join(clone, "acr-v0.10.zip")
            self.assertTrue(os.path.exists(zip_path))
            with zipfile.ZipFile(zip_path) as zf:
                names = zf.namelist()
            self.assertIn("acr/icons/acr-48.png", names)
            self.assertIn("acr/icons/acr-256.png", names)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
