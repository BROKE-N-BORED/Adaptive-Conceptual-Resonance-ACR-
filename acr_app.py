#!/usr/bin/env python3
"""ACR - Adaptive Contextual Resonance, v0.10.

A native desktop window for the ACR adaptive loop at small scale:
type text, step the loop, watch W_t adapt, read the trace log, and
see the resonance views (norm chart, change heatmap, layer
candidates).

What this is: a real, runnable prototype of the blueprint's core
loop (encoder, low-rank W_t updates, resonance detection, trace log).
What it is not: a full self-evolving neural engine. That does not
exist yet, and this app does not claim to be it.

v0.5 adds an optional local-model encoder: when the local model
service is running on this computer, its embeddings can replace
the built-in encoder. The built-in encoder stays the default and
the fallback. Switching encoders starts a fresh W_t, because the
embedding sizes differ.

v0.6 upgrades the synthesis head: analogies and counter-examples
built from the session's own buffer and resonance, plus an
optional completion from another installed local model, labelled
as that model's words.

v0.7 adds stability and control: freeze modes (whole W_t, or
individual rank directions), learning-rate schedule presets
(steady / fast-adapt / gentle), a live top-singular-value readout
next to the stability cap with a trace flag for steps the cap
clipped, and revert: roll W_t back to any recent step from the
trace log.

v0.8 is polish only, no new features: the chart legend moves to
the bottom-left (it used to overlap a red dot at the top-right),
the neighbour lists show each text once, Enter steps the loop
from the input box, the window carries its icon, and the window
remembers its size and position between runs.

v0.8.1 is a launch fix, no new features: the desktop and menu
icon now start the app. It used to look for the program next to
the icon file, which is the wrong folder once installed, so
clicking the icon did nothing.

v0.9 is a UI rebuild to the design rules: a real menu bar, the
crowded single window split into dedicated windows (Main,
Controls, Trace log, Resonance views), a How to Use window that
reads HOW-TO-USE.md, single instance, a busy light, local-model
calls in a worker thread so the window never blocks, a guided
model picker that starts unselected, plain-words explanations
under every field, a restrained identity pass, and a snapshot
limit that adapts to the machine's RAM. The engine is untouched:
the 16-theme numbers are still exact.

v0.10 is the audit redesign: the remaining local-model calls move
off the UI thread, keyboard navigation lands (visible focus,
visual tab order, shortcuts), entry text is readable everywhere,
a first run teaches by playing, results stream live during a
run, the chart scrubs, the flag marker persists, the Controls
window shows the live alpha, the menu carries accessibility
toggles, the main window discloses progressively, pack export
previews before writing, the trace follows only at the bottom,
batch progress is throttled and the heat history bounded, the
View menu shows which windows are open, and the empty states
and copy are polished. The engine is untouched: the 16-theme
numbers are still exact.
"""

import csv
import json
import math
import os
import shutil
import sys
import tempfile
import time

from PySide6.QtCore import (QEasingCurve, QEvent, QLockFile, QObject,
                            QProcess, QPropertyAnimation, QRect,
                            QSettings, Qt, QThread, Signal,
                            QVariantAnimation, Property)
from PySide6.QtGui import (QAction, QColor, QFont, QFontMetrics, QIcon,
                           QPainter, QPen, QTextCursor)
from PySide6.QtNetwork import QLocalServer, QLocalSocket
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QDoubleSpinBox,
    QFileDialog, QFrame, QGraphicsEffect, QGraphicsOpacityEffect,
    QGridLayout, QHBoxLayout, QHeaderView,
    QLabel, QListWidget, QListWidgetItem, QMainWindow, QMessageBox,
    QPlainTextEdit, QProgressBar, QPushButton, QSlider, QSpinBox,
    QTableWidget, QTableWidgetItem, QTextBrowser, QTextEdit,
    QVBoxLayout, QWidget,
)

from acr_engine import (BUFFER_SIZE, DIM, RANK, SCHEDULES,
                        SCHEDULE_STEADY, Engine,
                        SESSION_FORMAT_VERSION, SessionError,
                        mem_total_bytes, schedule_alpha)
from acr_packs import (build_pack_rows, default_packs_dir,
                       export_stella_packs)
from debug_log import DebugLogger, check_ollama
from local_encoder import (
    MSG_NO_MODELS, MSG_SERVICE_DOWN, MSG_SERVICE_NOT_INSTALLED,
    MSG_START_FAILED, LocalModelClient, LocalModelError,
    model_display_text)

APP_NAME = "acr"
VERSION = "0.10"
REPORT_EMAIL = "broke.n.bored.studio@gmail.com"

# v0.9 identity (U15): ACR's own dark look, drawn from the icon
# (dark rounded square, gold dot, blue/gold arcs). Deep dark
# surfaces, gold as the resonance accent, blue as the secondary.
# One stylesheet, applied by main() to the whole app (and by any
# verify harness that builds windows directly).
THEME_BG = "#1d1d23"
THEME_PANEL = "#26262e"
THEME_INPUT = "#2e2e38"
THEME_BORDER = "#3a3a44"
THEME_TEXT = "#e9e9ec"
THEME_DIM = "#a9a9b3"
HEADER_COLOR = "#e3c565"
ACCENT_GOLD = "#c9a227"
ACCENT_GOLD_BRIGHT = "#e3c565"
ACCENT_GOLD_DARK = "#8a6d1a"
ACCENT_BLUE = "#6ea8d8"
GLOW_GOLD = "#ffd75e"
BUSY_OFF = "#4a4a54"

DARK_STYLESHEET = """
QMainWindow { background: #1d1d23; color: #e9e9ec; }
QDialog { background: #1d1d23; color: #e9e9ec; }
QWidget#subwindow { background: #1d1d23; color: #e9e9ec; }
QLabel { color: #e9e9ec; }
QPushButton { background: #33333e; color: #e9e9ec;
    border: 1px solid #4a4a56; border-radius: 6px;
    padding: 6px 14px; }
QPushButton:hover { background: #3d3d49; }
QPushButton:pressed { background: #2a2a32; }
QPushButton:disabled { color: #77777f; background: #26262e;
    border-color: #33333e; }
/* v0.10 keyboard navigation (R2): a focused control must LOOK
   focused. The gold ring is the same gold as the resonance
   accent, so focus reads as part of the identity. */
QPushButton:focus { border: 2px solid #ffd75e; }
QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus,
QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus,
QListWidget:focus, QTableWidget:focus, QTextBrowser:focus {
    border: 2px solid #ffd75e; }
QCheckBox:focus { border: 1px solid #ffd75e; border-radius: 4px; }
QSlider:focus::groove:horizontal { background: #4a4a56; }
QPushButton#primaryButton { background: #c9a227; color: #1d1d23;
    font-weight: bold; border: 1px solid #c9a227; }
QPushButton#primaryButton:hover { background: #d8b73a; }
QPushButton#primaryButton:pressed { background: #a8861d; }
QPushButton#primaryButton:disabled { background: #55553f;
    color: #99998a; border-color: #55553f; }
QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox,
QComboBox, QListWidget, QTableWidget, QTextBrowser {
    background: #2e2e38; color: #e9e9ec;
    border: 1px solid #3a3a44; border-radius: 6px; }
QComboBox::drop-down { border: none; }
QComboBox QAbstractItemView { background: #2e2e38; color: #e9e9ec;
    selection-background-color: #8a6d1a; }
QListWidget::item:selected, QTableWidget::item:selected {
    background: #8a6d1a; color: #ffffff; }
QHeaderView::section { background: #26262e; color: #e9e9ec;
    border: 1px solid #3a3a44; padding: 4px; }
QMenuBar { background: #1d1d23; color: #e9e9ec; }
QMenuBar::item:selected { background: #33333e; }
QMenu { background: #26262e; color: #e9e9ec;
    border: 1px solid #3a3a44; }
QMenu::item:selected { background: #8a6d1a; }
QProgressBar { background: #2e2e38; border: 1px solid #3a3a44;
    border-radius: 6px; text-align: center; color: #e9e9ec; }
QProgressBar::chunk { background: #c9a227; border-radius: 6px; }
QSlider::groove:horizontal { background: #2e2e38; height: 6px;
    border-radius: 3px; }
QSlider::handle:horizontal { background: #c9a227; width: 16px;
    margin: -6px 0; border-radius: 8px; }
QCheckBox::indicator { width: 16px; height: 16px;
    border: 1px solid #4a4a56; border-radius: 4px;
    background: #2e2e38; }
QCheckBox::indicator:checked { background: #c9a227;
    border-color: #c9a227; }
QFrame#tile { background: #26262e; border: 1px solid #33333e;
    border-radius: 14px; }
QScrollBar:vertical { background: #26262e; width: 12px;
    border-radius: 6px; }
QScrollBar::handle:vertical { background: #4a4a56;
    border-radius: 6px; min-height: 20px; }
QScrollBar:horizontal { background: #26262e; height: 12px;
    border-radius: 6px; }
QScrollBar::handle:horizontal { background: #4a4a56;
    border-radius: 6px; min-width: 20px; }
QToolTip { background: #26262e; color: #e9e9ec;
    border: 1px solid #3a3a44; }
"""


def apply_theme(app):
    """Apply ACR's dark stylesheet to the whole app (v0.9, U15).
    main() calls this. A verify harness that builds windows
    directly must call it itself to see the true theme."""
    app.setStyleSheet(DARK_STYLESHEET)


REDUCED_MOTION_ENV = "ACR_REDUCED_MOTION"


def reduced_motion():
    """True when motion should be skipped: the ACR_REDUCED_MOTION
    environment variable says so, or the user saved the choice in
    settings. State changes still land (the flag still marks, the
    checkmark still appears); only the motion is skipped."""
    value = os.environ.get(REDUCED_MOTION_ENV, "").strip().lower()
    if value in ("1", "true", "yes"):
        return True
    if value in ("0", "false", "no"):
        return False
    return QSettings("BrokeNBored", "ACR").value(
        "reduced_motion", False, type=bool)


BIG_TEXT_STEP = 2  # points the big-text preset adds (v0.10, R9)


def big_text_enabled():
    """True when the user saved the big-text preset in settings."""
    return QSettings("BrokeNBored", "ACR").value(
        "big_text", False, type=bool)


def apply_big_text(app, enabled):
    """Apply the big-text preset (v0.10, R9): raise the app font
    two points when enabled, restore it when not. The font is
    separate from the stylesheet, so the identity test still
    holds: the stylesheet string never changes. Idempotent for
    one call per state change: main() calls it once at launch,
    the menu toggle calls it once per flip."""
    font = QFont(app.font())
    size = font.pointSize()
    if size <= 0:
        size = 10
    if enabled:
        size += BIG_TEXT_STEP
    font.setPointSize(size)
    app.setFont(font)


def adjust_big_text(app, enabled):
    """Flip the big-text preset from its current state: add the
    two points when turning on, take them back when turning
    off. The menu toggle uses this; main() uses apply_big_text
    (the font is still the system default at launch)."""
    font = QFont(app.font())
    size = font.pointSize()
    if size <= 0:
        size = 10
    font.setPointSize(size + (BIG_TEXT_STEP if enabled
                              else -BIG_TEXT_STEP))
    app.setFont(font)

# Single-instance names (U27). ACR_INSTANCE_NAME lets tests run a
# private instance without touching a real one.
INSTANCE_ENV = "ACR_INSTANCE_NAME"
INSTANCE_DEFAULT = "acr"

TRACE_COLUMNS = ["Step", "Input", "Delta norm", "W old", "W new",
                 "Resonance", "Alpha", "Confidence", "New layer",
                 "Capped"]

CHART_BG = QColor(0x26, 0x26, 0x2e)
CHART_LINE = QColor(110, 168, 216)
# v0.10 (R9): the spike red is lightened from (224, 92, 80) so
# the legend text passes WCAG AA on the chart background: 5.44:1,
# up from 4.16:1. Still clearly red, against the blue line.
CHART_SPIKE = QColor(240, 120, 110)
LEGEND_TEXT = "red dot = new-layer candidate"
CHART_GRID = QColor(58, 58, 68)
CHART_AXIS = QColor(169, 169, 179)
HEAT_LOW = QColor(0x26, 0x26, 0x2e)
HEAT_HIGH = QColor(201, 162, 39)
# The flagged trace row rests at a calm dark-red tint (the
# chart's red-dot meaning, as a row); the fade starts brighter.
FLAG_TINT = QColor(0x4a, 0x23, 0x20)
FLAG_TINT_BRIGHT = QColor(0x7a, 0x32, 0x2b)

# v0.10 (R13): the heat history is bounded -- one entry per step,
# each holding dim floats, so a 10,000-line run at dim 768 would
# otherwise hold ~240 MB of heatmap data. 1000 entries is ~24 MB
# at dim 768, ~2 MB at dim 64. Past the bound the oldest half is
# thinned (every other entry), so the step picker still spans
# the whole run and a thinned-out step says so (U32).
HEAT_HISTORY_LIMIT = 1000
# v0.10 (R13): batch progress updates the bar and label at most
# this often. The final update always lands.
PROGRESS_UI_INTERVAL = 0.1

# v0.10 (R4): the example run. The four theme sentences, the
# same ones the tests drive: four real steps on one theme.
EXAMPLE_SENTENCES = [
    "quantum entanglement links particles",
    "entangled particles share a state",
    "quantum particles stay connected",
    "entanglement links distant particles",
]


def _row_change_norms(old_u, old_v, new_u, new_v):
    """Per-row Frobenius norm of W_new - W_old, where W = U V^T.

    Row i of W_new - W_old is a combination of the columns of
    V_new and V_old with coefficients (u_new[i], -u_old[i]), so
    its squared norm is a quadratic form in the 2*rank Gram matrix
    of those columns. Same numbers as the naive dim x dim form,
    without forming either matrix: at dim 768 the naive form costs
    seconds per step, this one costs milliseconds. Read-only: it
    never touches the engine's factors.
    """
    dim = len(old_u)
    # Gram of the 2*rank columns [new_v | old_v].
    cols = []
    for k in range(RANK):
        cols.append([new_v[i][k] for i in range(dim)])
    for k in range(RANK):
        cols.append([old_v[i][k] for i in range(dim)])
    size = 2 * RANK
    gram = [[0.0] * size for _ in range(size)]
    for p in range(size):
        for q in range(p, size):
            total = 0.0
            cp = cols[p]
            cq = cols[q]
            for i in range(dim):
                total += cp[i] * cq[i]
            gram[p][q] = total
            gram[q][p] = total
    rows = []
    for i in range(dim):
        coeff = list(new_u[i]) + [-x for x in old_u[i]]
        total = 0.0
        for p in range(size):
            inner = 0.0
            row = gram[p]
            for q in range(size):
                inner += row[q] * coeff[q]
            total += coeff[p] * inner
        # Rounding leaves a zero change a hair off zero (either
        # sign). Real row changes are orders of magnitude above
        # 1e-12, so anything below it is no change at all -- and
        # the heatmap's "did not move" message depends on exact
        # zeros for frozen steps.
        value = math.sqrt(max(0.0, total))
        rows.append(value if value > 1e-12 else 0.0)
    return rows


class LocalServiceManager:
    """Starts the local model service from inside ACR.

    Follows the DUO v0.10 server-manager shape: already running
    (by anyone) counts as success, nothing is spawned twice, and
    start() waits on the probe up to START_TIMEOUT. The service
    runs detached (QProcess.startDetached), so it keeps running
    after ACR closes and ACR never stops it: a service the user
    started another way is never touched either. start() returns
    (ok, message) with a plain-language message, and never raises.
    v0.10 (R1): start() runs on a worker thread, so it no longer
    pumps the event loop from the wait -- pumping from a worker
    thread is not safe, and the window stays responsive on its
    own thread. The wait can be abandoned: stop, when given, is
    a zero-arg callable checked every poll, and a stop returns
    at once with a cancelled message.
    """

    START_TIMEOUT = 15.0
    PROBE_INTERVAL = 0.5

    def __init__(self, client):
        self.client = client

    def start(self, timeout=None, stop=None):
        """Start the service if it is not already answering."""
        if timeout is None:
            timeout = self.START_TIMEOUT
        try:
            if self.client.probe():
                return True, "The local model service is running."
        except Exception:
            pass
        if shutil.which("ollama") is None:
            return False, MSG_SERVICE_NOT_INSTALLED
        try:
            result = QProcess.startDetached("ollama", ["serve"])
        except Exception:
            return False, MSG_START_FAILED
        started = result[0] if isinstance(result, tuple) else result
        if not started:
            return False, MSG_START_FAILED
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if stop is not None and stop():
                return False, "Cancelled."
            try:
                if self.client.probe(timeout=2):
                    return True, "The local model service is running."
            except Exception:
                pass
            time.sleep(self.PROBE_INTERVAL)
        return False, MSG_START_FAILED


class NormChart(QWidget):
    """Line chart of the W_t Frobenius norm, one point per step.

    Steps flagged as new-layer candidates are drawn as red dots so
    resonance spikes stand out. Plain QPainter painting, no extra
    dependencies. `values` and `flags` mirror the trace log.

    v0.10 (R6): the chart scrubs. Hover shows the step's values
    as a tooltip; click (or drag) jumps the heatmap and the trace
    selection to that step, through stepClicked, which carries
    the 0-based index into the values list. The window maps the
    index to a step id through the trace.
    """

    stepClicked = Signal(int)

    def __init__(self, parent=None, empty_text=None):
        super().__init__(parent)
        self.setMinimumHeight(130)
        # v0.10 (R15): the empty message fits where the chart
        # lives. The Views chart says "Step some text"; the
        # Analyze chart says "Analyze a file" -- nobody steps
        # text in Analyze.
        self.empty_text = (empty_text
                           or "Step some text to see the chart.")
        self.values = []
        self.flags = []
        self._pulse_index = None
        self._pulse_scale = 1.0
        self._pulse_anim = None
        self.setMouseTracking(True)

    def set_data(self, values, flags):
        self.values = list(values)
        self.flags = list(flags)
        self.update()

    def _index_at(self, x):
        """The step index nearest a mouse x position, or None
        when the chart is empty. The plot area is the same
        geometry paintEvent uses: left margin 48, right 10."""
        n = len(self.values)
        if n == 0:
            return None
        if n == 1:
            return 0
        left, right = 48, 10
        width = max(1, self.width() - left - right)
        frac = (x - left) / float(width)
        index = int(round(frac * (n - 1)))
        return max(0, min(n - 1, index))

    def mouseMoveEvent(self, event):
        index = self._index_at(event.position().x())
        if index is not None:
            text = ("Step %d: W norm %.4f"
                    % (index + 1, self.values[index]))
            if index < len(self.flags) and self.flags[index]:
                text += " (new-layer candidate)"
            self.setToolTip(text)
            # Drag scrubs: a held left button jumps as the mouse
            # moves, the same as a click.
            if event.buttons() & Qt.LeftButton:
                self.stepClicked.emit(index)
        super().mouseMoveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            index = self._index_at(event.position().x())
            if index is not None:
                self.stepClicked.emit(index)
        super().mousePressEvent(event)

    def _get_pulse_scale(self):
        return self._pulse_scale

    def _set_pulse_scale(self, scale):
        self._pulse_scale = scale
        self.update()

    pulseScale = Property(float, _get_pulse_scale, _set_pulse_scale)

    def pulse_flag(self, index):
        """The resonance moment (v0.9, technique 8): the flagged
        dot lands with a spring settle, 1.8x down to normal over
        300 ms. Under reduced motion the dot just appears:
        set_data already drew it at normal size."""
        if index is None or index >= len(self.values):
            return
        if self._pulse_anim is not None:
            self._pulse_anim.stop()
            self._pulse_anim = None
        self._pulse_index = index
        if reduced_motion():
            self._pulse_scale = 1.0
            self.update()
            return
        anim = QPropertyAnimation(self, b"pulseScale", self)
        anim.setDuration(300)
        anim.setEasingCurve(QEasingCurve.OutBack)
        anim.setStartValue(1.8)
        anim.setEndValue(1.0)
        anim.finished.connect(self._pulse_done)
        self._pulse_anim = anim
        anim.start()

    def _pulse_done(self):
        self._pulse_index = None
        self._pulse_scale = 1.0
        self._pulse_anim = None
        self.update()

    def legend_rect(self):
        """Where the legend sits: bottom-left inside the plot
        area, sized to the text. v0.8: it used to sit top-right,
        where a red dot at the last step could land underneath
        it and hide behind the red text."""
        left, right = 48, 10
        top, bottom = 10, 22
        height = max(1, self.height() - top - bottom)
        font = QFont()
        font.setPointSize(8)
        text_w = QFontMetrics(font).horizontalAdvance(LEGEND_TEXT)
        return QRect(left, top + height - 18, text_w + 8, 16)

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), CHART_BG)
        if not self.values:
            painter.setPen(CHART_AXIS)
            painter.drawText(self.rect(), Qt.AlignCenter,
                             self.empty_text)
            return
        left, right = 48, 10
        top, bottom = 10, 22
        width = max(1, self.width() - left - right)
        height = max(1, self.height() - top - bottom)
        low = min(self.values)
        high = max(self.values)
        if high - low < 1e-9:
            low -= 0.1
            high += 0.1
        pad = (high - low) * 0.1
        low -= pad
        high += pad
        n = len(self.values)

        def point(i, value):
            if n == 1:
                x = left + width / 2.0
            else:
                x = left + width * i / (n - 1.0)
            y = top + height * (1.0 - (value - low) / (high - low))
            return x, y

        painter.setPen(QPen(CHART_GRID))
        for frac in (0.0, 0.5, 1.0):
            y = top + height * frac
            painter.drawLine(left, int(y), left + width, int(y))
        painter.setPen(CHART_AXIS)
        font = QFont()
        font.setPointSize(8)
        painter.setFont(font)
        painter.drawText(2, top + 8, "%.2f" % high)
        painter.drawText(2, top + height, "%.2f" % low)
        painter.drawText(left, self.height() - 6, "step 1")
        painter.drawText(left + width - 40, self.height() - 6,
                         "step %d" % n)
        painter.setPen(QPen(CHART_LINE, 2))
        for i in range(1, n):
            x1, y1 = point(i - 1, self.values[i - 1])
            x2, y2 = point(i, self.values[i])
            painter.drawLine(int(x1), int(y1), int(x2), int(y2))
        for i, value in enumerate(self.values):
            x, y = point(i, value)
            if i < len(self.flags) and self.flags[i]:
                painter.setPen(QPen(CHART_SPIKE, 2))
                painter.setBrush(CHART_SPIKE)
                radius = 4
                if i == self._pulse_index:
                    radius = max(1, int(round(4 * self._pulse_scale)))
                painter.drawEllipse(int(x) - radius, int(y) - radius,
                                    radius * 2, radius * 2)
            else:
                painter.setPen(QPen(CHART_LINE, 2))
                painter.setBrush(CHART_LINE)
                painter.drawEllipse(int(x) - 2, int(y) - 2, 4, 4)
        # Legend, bottom-left, on a patch of the chart background
        # so the line and grid never show through the text.
        rect = self.legend_rect()
        painter.fillRect(rect, CHART_BG)
        painter.setPen(CHART_SPIKE)
        painter.drawText(rect.left() + 4, rect.bottom() - 4,
                         LEGEND_TEXT)


class ChangeHeatmap(QWidget):
    """Per-row change of W_t for one step, one bar per row.

    One bar per row of W_t, longest and darkest for the rows that
    moved most on the step. The three biggest movers get their row
    numbers. `values` holds the per-row norms for `step_id` (64
    with the built-in encoder, more with a local model).
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(130)
        self.values = []
        self.step_id = 0
        self.note = None

    def set_data(self, values, step_id, note=None):
        """Bar values for one step. ``note`` is the message to
        paint when nothing moved: the app supplies it (first
        step, frozen step, or a revert that changed nothing), so
        the painted message always matches the summary line."""
        self.values = list(values)
        self.step_id = step_id
        self.note = note
        self.update()

    def top_rows(self, count=3):
        """Row indexes of the biggest movers, biggest first."""
        order = sorted(range(len(self.values)),
                       key=lambda i: self.values[i], reverse=True)
        return order[:count]

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), CHART_BG)
        if not self.values:
            painter.setPen(CHART_AXIS)
            # A step with no recorded change data (a session
            # loaded without heat history) shows the reason, not
            # the empty-session message.
            message = (self.note if self.note is not None
                       else "Step some text to see the heatmap.")
            painter.drawText(self.rect(), Qt.AlignCenter, message)
            return
        top = set(self.top_rows(3))
        peak = max(self.values)
        if peak <= 0.0:
            painter.setPen(CHART_AXIS)
            if self.note is not None:
                message = self.note
            # v0.3: step 1 never moves W_t (no previous state to
            # predict from), so it is not a frozen step. Fallback
            # for callers that pass no note.
            elif self.step_id == 1:
                message = ("Step 1: W_t did not move (first step "
                           "has no previous state).")
            else:
                message = ("Step %d: W_t did not move (frozen step)."
                           % self.step_id)
            painter.drawText(self.rect(), Qt.AlignCenter, message)
            return
        left = 52
        width = max(1, self.width() - left - 70)
        row_h = self.height() / float(len(self.values))
        font = QFont()
        font.setPointSize(8)
        painter.setFont(font)
        # Each row is only a couple of pixels tall, so labels for
        # adjacent top rows would print on top of each other. Skip
        # a label that lands too close under the last one drawn;
        # the summary line above the heatmap always names all
        # three top rows, so nothing is lost.
        last_label_y = -100
        for i, value in enumerate(self.values):
            frac = value / peak
            red = int(HEAT_LOW.red()
                      + (HEAT_HIGH.red() - HEAT_LOW.red()) * frac)
            green = int(HEAT_LOW.green()
                        + (HEAT_HIGH.green() - HEAT_LOW.green())
                        * frac)
            blue = int(HEAT_LOW.blue()
                       + (HEAT_HIGH.blue() - HEAT_LOW.blue()) * frac)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(red, green, blue))
            y = int(i * row_h)
            bar_w = max(2, int(width * frac))
            painter.fillRect(left, y, bar_w, max(1, int(row_h) - 1),
                             QColor(red, green, blue))
            if i in top and y - last_label_y >= 11:
                last_label_y = y
                painter.setPen(CHART_AXIS)
                painter.drawText(2, y + max(8, int(row_h) - 1),
                                 "row %d" % i)
                painter.drawText(left + bar_w + 4,
                                 y + max(8, int(row_h) - 1),
                                 "%.3f" % value)


class PressDipEffect(QGraphicsEffect):
    """The press dip (v0.9, technique 1): a button scales to 0.97
    while pressed. A paint-time scale around the button's center:
    the layout never moves, only the pixels do."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scale = 1.0

    def _get_scale(self):
        return self._scale

    def _set_scale(self, scale):
        self._scale = scale
        self.update()

    scale = Property(float, _get_scale, _set_scale)

    def draw(self, painter):
        if self._scale == 1.0:
            self.drawSource(painter)
            return
        center = self.sourceBoundingRect().center()
        painter.save()
        painter.translate(center.x(), center.y())
        painter.scale(self._scale, self._scale)
        painter.translate(-center.x(), -center.y())
        self.drawSource(painter)
        painter.restore()


def press_dip(button):
    """Dip a button to 0.97 and back over 150 ms (technique 1).
    User-caused motion, so it springs (OutBack). Reduced motion
    skips it: the press still works, just without the dip.
    Returns the animation, or None when motion is skipped."""
    if reduced_motion():
        return None
    old = getattr(button, "_dip_anim", None)
    if old is not None:
        old.stop()
    effect = PressDipEffect(button)
    button.setGraphicsEffect(effect)
    anim = QPropertyAnimation(effect, b"scale", button)
    anim.setDuration(150)
    anim.setEasingCurve(QEasingCurve.OutBack)
    anim.setKeyValueAt(0.0, 1.0)
    anim.setKeyValueAt(0.5, 0.97)
    anim.setKeyValueAt(1.0, 1.0)

    def _done():
        if button.graphicsEffect() is effect:
            button.setGraphicsEffect(None)

    anim.finished.connect(_done)
    button._dip_anim = anim
    button._dip_effect = effect
    anim.start()
    return anim


class CheckmarkWidget(QWidget):
    """A checkmark that draws itself (v0.9, technique 8): the
    completion moment. Two strokes, drawn progressively over
    200 ms. Under reduced motion it appears fully drawn."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(18, 18)
        self._progress = 0.0
        self._anim = None
        self.hide()

    def _get_progress(self):
        return self._progress

    def _set_progress(self, value):
        self._progress = value
        self.update()

    progress = Property(float, _get_progress, _set_progress)

    def paintEvent(self, _event):
        if self._progress <= 0.0:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(QColor(ACCENT_GOLD), 2.5))
        # The check: (3, 9) down to the bend (7.5, 13.5), then
        # up to (15, 4.5). The first stroke is 35% of the length.
        bend = 0.35
        if self._progress < bend:
            frac = self._progress / bend
            x = 3 + (7.5 - 3) * frac
            y = 9 + (13.5 - 9) * frac
            painter.drawLine(3, 9, int(x), int(y))
        else:
            painter.drawLine(3, 9, 7, 13)
            frac = (self._progress - bend) / (1.0 - bend)
            x = 7.5 + (15 - 7.5) * frac
            y = 13.5 + (4.5 - 13.5) * frac
            painter.drawLine(7, 13, int(x), int(y))

    def draw(self):
        """Show the checkmark, drawing itself over 200 ms
        (instantly, fully drawn, under reduced motion)."""
        if self._anim is not None:
            self._anim.stop()
            self._anim = None
        if reduced_motion():
            self._progress = 1.0
            self.show()
            self.update()
            return
        self._progress = 0.0
        self.show()
        anim = QPropertyAnimation(self, b"progress", self)
        anim.setDuration(200)
        anim.setEasingCurve(QEasingCurve.OutCubic)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.finished.connect(self._draw_done)
        self._anim = anim
        anim.start()

    def _draw_done(self):
        self._anim = None

    def clear(self):
        """Hide the mark: the next run or step starts fresh."""
        if self._anim is not None:
            self._anim.stop()
            self._anim = None
        self._progress = 0.0
        self.hide()


class LocalCallWorker(QObject):
    """One local-model call, off the UI thread (v0.9, U30).

    kind is "embed" or "generate". The call runs in run(), on a
    QThread owned by the window; finished carries the result (a
    vector, or a string) or a LocalModelError. Retry notices come
    back as progress strings: the worker never touches a widget
    itself, so the window stays responsive while the call runs.
    The window honours cancel by ignoring a finished result, not
    by killing the thread: a network call in flight cannot be
    stopped midway, but its answer can be thrown away.
    """

    finished = Signal(object)
    progress = Signal(str)

    def __init__(self, client, kind, model, text):
        super().__init__()
        self.client = client
        self.kind = kind
        self.model = model
        self.text = text

    def _on_retry(self, attempt, delay, kind):
        self.progress.emit(
            "The local model service is slow, retrying "
            "(attempt %d)..." % (attempt + 1))

    def run(self):
        try:
            if self.kind == "embed":
                result = self.client.embed(
                    self.model, self.text, on_retry=self._on_retry)
            else:
                result = self.client.generate(
                    self.model, self.text, on_retry=self._on_retry)
        except LocalModelError as exc:
            self.finished.emit(exc)
            return
        except Exception as exc:  # never strand the window
            self.finished.emit(LocalModelError(str(exc)))
            return
        self.finished.emit(result)


class LocalTaskWorker(QObject):
    """One setup call for the local model service, off the UI
    thread (v0.10, R1 -- the same pattern as LocalCallWorker).

    kind is "offer", "activate", or "start". "offer" probes the
    service and lists the installed models (the picker fill);
    "activate" runs a probe embedding to learn one model's dim;
    "start" starts the service. finished carries the result --
    the model info list, the probe vector, or the service
    start's (ok, message) pair -- or a LocalModelError whose
    message is the plain status text to show. The worker never
    touches a widget itself, so the window stays responsive
    while the call runs. A start can be abandoned mid-wait: the
    stop callable is checked every poll, and a stop returns at
    once.
    """

    finished = Signal(object)
    progress = Signal(str)

    def __init__(self, client, service, kind, model=None,
                 stop=None):
        super().__init__()
        self.client = client
        self.service = service
        self.kind = kind
        self.model = model
        self.stop = stop

    def run(self):
        try:
            if self.kind == "offer":
                if not self.client.probe():
                    self.finished.emit(LocalModelError(
                        MSG_SERVICE_DOWN + " Press Start service, "
                        "or stay on Built-in."))
                    return
                infos = self.client.list_model_info()
                self.finished.emit(infos)
            elif self.kind == "activate":
                probe = self.client.embed(self.model, "hello")
                self.finished.emit(probe)
            elif self.kind == "start":
                self.finished.emit(
                    self.service.start(stop=self.stop))
            else:
                self.finished.emit(LocalModelError(
                    "unknown local task: %s" % self.kind))
        except LocalModelError as exc:
            self.finished.emit(exc)
        except Exception as exc:  # never strand the window
            self.finished.emit(LocalModelError(str(exc)))


# v0.9 Analyze: file types a folder run picks up, and how many
# entries the core and outlier lists hold.
BATCH_EXTENSIONS = (".txt", ".md", ".jsonl")
ANALYSIS_TOP_N = 10
# A theme shift is a step whose resonance falls into the noise
# band (at or under SHIFT_LOW) by at least SHIFT_DROP from the
# previous step, and stays low on the next step (at or under
# SHIFT_STAYS_LOW). Within-theme dips recover at once; a real
# topic change does not. Grounded by probe: a 16-theme run has no
# shifts, a theme switch flags exactly the first off-theme step.
SHIFT_LOW = 0.25
SHIFT_DROP = 0.25
SHIFT_STAYS_LOW = 0.35


def extract_batch_text(line, is_jsonl):
    """The step text for one line of a batch file, or None when
    the line carries no usable text. Plain text lines are used as
    they are. A .jsonl record uses its "text" field, else
    "prompt", else "completion"; a record with none of those, or
    a line that is not JSON at all, is skipped, never guessed."""
    stripped = line.strip()
    if not stripped:
        return None
    if not is_jsonl:
        return stripped
    try:
        record = json.loads(stripped)
    except ValueError:
        return None
    if not isinstance(record, dict):
        return None
    for key in ("text", "prompt", "completion"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def iter_batch_lines(paths, unreadable=None):
    """Yield (text, skipped, path, line number) for every
    candidate line in the given files, in order. text is None
    when the line carries no usable text. The path and line
    number are the step's source locator: the STELLA pack
    manifest records where each entry came from. A file that
    cannot be read is skipped whole and its name appended to
    `unreadable` when one is given (U32: never silently skip)."""
    for path in paths:
        is_jsonl = path.lower().endswith(".jsonl")
        try:
            fh = open(path, "r", encoding="utf-8",
                      errors="replace")
        except OSError:
            if unreadable is not None:
                unreadable.append(os.path.basename(path))
            continue
        with fh:
            for line_no, line in enumerate(fh, start=1):
                text = extract_batch_text(line, is_jsonl)
                yield text, text is None, path, line_no


def collect_batch_files(folder):
    """Text files in a folder for a batch run: the known types,
    in name order, non-recursive."""
    try:
        names = sorted(os.listdir(folder))
    except OSError:
        return []
    return [os.path.join(folder, name) for name in names
            if name.lower().endswith(BATCH_EXTENSIONS)
            and os.path.isfile(os.path.join(folder, name))]


def summarize_run(trace, top_n=ANALYSIS_TOP_N):
    """Themes, core entries, outliers, and theme shifts from a
    trace log (v0.9, Analyze). Pure: no engine, no widgets.
    Revert entries (marked reverted_to) are not steps, so they
    never appear in any list. Core is the highest-resonance
    entries, outliers the lowest, one line per distinct text."""
    steps = [entry for entry in trace if "reverted_to" not in entry]
    themes = [entry for entry in steps if entry.get("new_layer")]
    core = []
    seen = set()
    for entry in sorted(steps, key=lambda e: e["resonance"],
                        reverse=True):
        if entry["input"] in seen:
            continue
        seen.add(entry["input"])
        core.append(entry)
        if len(core) >= top_n:
            break
    outliers = []
    seen = set()
    for entry in sorted(steps, key=lambda e: e["resonance"]):
        if entry["input"] in seen:
            continue
        seen.add(entry["input"])
        outliers.append(entry)
        if len(outliers) >= top_n:
            break
    shifts = []
    for i in range(1, len(steps)):
        prev = steps[i - 1]
        cur = steps[i]
        if (cur["resonance"] <= SHIFT_LOW
                and prev["resonance"] - cur["resonance"]
                >= SHIFT_DROP):
            nxt = steps[i + 1] if i + 1 < len(steps) else None
            if nxt is None or nxt["resonance"] <= SHIFT_STAYS_LOW:
                shifts.append(cur)
    return {"themes": themes, "core": core, "outliers": outliers,
            "shifts": shifts, "steps": len(steps)}


def format_analysis_report(summary, version, text_of,
                           source_note=""):
    """The readable analysis report (v0.9, Analyze): themes, core
    entries, outliers, and theme shifts, in plain words. text_of
    maps a trace entry to its full text (the trace holds an
    80-character preview)."""
    lines = ["ACR analysis report",
             "App version: %s" % version]
    if source_note:
        lines.append("Source: %s" % source_note)
    lines.append("Steps: %d" % summary["steps"])
    lines.append("")
    lines.append("Themes found (%d)" % len(summary["themes"]))
    if not summary["themes"]:
        lines.append("  None yet. Step a few sentences on one "
                     "theme and spikes will show up here.")
    for entry in summary["themes"]:
        lines.append("  Step %d: %s (resonance %.3f)"
                     % (entry["id"], text_of(entry),
                        entry["resonance"]))
    lines.append("")
    lines.append("Core entries (%d)" % len(summary["core"]))
    lines.append("  The most resonant entries: the heart of the "
                 "file.")
    for entry in summary["core"]:
        lines.append("  %.3f  %s" % (entry["resonance"],
                                     text_of(entry)))
    lines.append("")
    lines.append("Outliers (%d)" % len(summary["outliers"]))
    lines.append("  The lowest-resonance entries: the noise and "
                 "off-topic lines.")
    for entry in summary["outliers"]:
        lines.append("  %.3f  %s" % (entry["resonance"],
                                     text_of(entry)))
    lines.append("")
    lines.append("Theme shifts (%d)" % len(summary["shifts"]))
    if not summary["shifts"]:
        lines.append("  None found: no sharp resonance drop that "
                     "stayed low.")
    for entry in summary["shifts"]:
        lines.append("  Step %d: resonance fell to %.3f"
                     % (entry["id"], entry["resonance"]))
    return "\n".join(lines) + "\n"


def theme_window_entries(trace, step_id):
    """The entries around a theme's trigger step: the step before,
    the spike itself, and the step after, in trace order. This is
    the window the Themes tab shows when a theme is clicked."""
    wanted = {step_id - 1, step_id, step_id + 1}
    return [entry for entry in trace
            if entry["id"] in wanted
            and "reverted_to" not in entry]


class BatchWorker(QObject):
    """A batch analysis run (v0.9, Analyze): every line of the
    picked files becomes one Engine.step, in file order, on a
    worker thread (U30). The step call is the same one manual
    stepping uses, with the same parameters, so a batch over a
    file of sentences ends in exactly the state typing them
    would. The worker owns the engine for the run: the window
    reads nothing from the engine until finished lands, and the
    mutating actions refuse while a run is in flight.

    progress carries (done, total, resonance, w_norm, skipped).
    finished carries a result dict: steps, skipped, unreadable
    (file names), heat (step id plus row norms, for the heatmap),
    texts (step id to full text), locators (step id to (file
    name, line number): where each step came from, for the
    STELLA pack manifest), cancelled, error (a LocalModelError
    or None). A local-embed failure stops the run with the
    error: the steps completed so far stay, they are real steps.
    """

    progress = Signal(int, int, float, float, int)
    finished = Signal(object)

    def __init__(self, engine, client, encoder_kind, local_model,
                 paths, preset, base_alpha, frozen, cap,
                 frozen_ranks):
        super().__init__()
        self.engine = engine
        self.client = client
        self.encoder_kind = encoder_kind
        self.local_model = local_model
        self.paths = list(paths)
        self.preset = preset
        self.base_alpha = base_alpha
        self.frozen = frozen
        self.cap = cap
        self.frozen_ranks = list(frozen_ranks)
        self.cancelled = False

    def run(self):
        unreadable = []
        total = 0
        for _text, _skipped, _path, _line in iter_batch_lines(
                self.paths, unreadable):
            total += 1
        done = 0
        skipped = 0
        heat = []
        texts = {}
        locators = {}
        error = None
        resonance = 0.0
        w_norm = self.engine.w_norm()
        for text, is_skipped, path, line_no in iter_batch_lines(
                self.paths):
            if self.cancelled:
                break
            done += 1
            if is_skipped or text is None:
                skipped += 1
                self.progress.emit(done, total, resonance, w_norm,
                                   skipped)
                continue
            h = None
            if self.encoder_kind == "local":
                try:
                    h = self.client.embed(self.local_model, text)
                except LocalModelError as exc:
                    error = exc
                    break
            old_u = [row[:] for row in self.engine.u]
            old_v = [row[:] for row in self.engine.v]
            alpha = schedule_alpha(
                self.preset, self.base_alpha,
                self.engine.step_count + 1)
            entry = self.engine.step(
                text, alpha=alpha, frozen=self.frozen,
                cap=self.cap, h=h,
                frozen_ranks=self.frozen_ranks)
            texts[entry["id"]] = text
            locators[entry["id"]] = (os.path.basename(path),
                                     line_no)
            heat.append((entry["id"], _row_change_norms(
                old_u, old_v, self.engine.u, self.engine.v)))
            resonance = entry["resonance"]
            w_norm = entry["new_norm"]
            self.progress.emit(done, total, resonance, w_norm,
                               skipped)
        self.finished.emit({
            "steps": len(texts),
            "skipped": skipped,
            "unreadable": unreadable,
            "heat": heat,
            "texts": texts,
            "locators": locators,
            "cancelled": self.cancelled,
            "error": error,
        })


def _header_label(text):
    """A section header: bold, in the bright gold (v0.9, U15)."""
    label = QLabel(text)
    label.setStyleSheet("font-weight: bold; color: %s;"
                        % HEADER_COLOR)
    return label


def _note_label(text):
    """A plain-words explanation under a field (v0.9, U20). Dim,
    so it reads as secondary to the control above it."""
    label = QLabel(text)
    label.setWordWrap(True)
    label.setStyleSheet("color: %s;" % THEME_DIM)
    return label


class SubWindow(QWidget):
    """One dedicated window (v0.9, U1): Controls, Trace log,
    Resonance views, How to Use. Closing one hides it instead of
    destroying it, so the View menu can open it again, and each
    remembers its own geometry between runs (U5)."""

    # v0.10 (R14): the View menu's checkable actions follow this,
    # so the menu shows which windows are open.
    visibilityChanged = Signal(bool)

    def __init__(self, title, settings_key, parent=None,
                 default_width=900, default_height=600):
        super().__init__(parent, Qt.Window)
        self.setObjectName("subwindow")
        self.setWindowTitle(title)
        self._settings = QSettings("BrokeNBored", "ACR")
        self._settings_key = settings_key
        self._open_anim = None
        geometry = self._settings.value(settings_key)
        if geometry is None or not self.restoreGeometry(geometry):
            self.resize(default_width, default_height)

    def showEvent(self, event):
        super().showEvent(event)
        self.visibilityChanged.emit(True)
        # Opening a window fades it in (v0.10, R13): the v0.9
        # open animated the window GEOMETRY, a layout property,
        # which the library says never to animate (entries
        # 001/002: animate transform and opacity, never layout).
        # The fade is 150 ms, OutCubic, and the effect is
        # removed when it lands so the window paints normally
        # afterwards. Reduced motion skips it. The main window
        # does not fade: it opens once, at launch, at its saved
        # size.
        if reduced_motion():
            return
        if self._open_anim is not None:
            self._open_anim.stop()
            self._open_anim = None
        effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(effect)
        anim = QPropertyAnimation(effect, b"opacity", self)
        anim.setDuration(150)
        anim.setEasingCurve(QEasingCurve.OutCubic)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.finished.connect(self._open_done)
        self._open_anim = anim
        anim.start()

    def _open_done(self):
        self._open_anim = None
        self.setGraphicsEffect(None)

    def closeEvent(self, event):
        self._settings.setValue(self._settings_key,
                                self.saveGeometry())
        self._settings.sync()
        # Hide, do not destroy: the window reopens from the menu.
        event.ignore()
        self.hide()
        self.visibilityChanged.emit(False)


class HowToUseWindow(SubWindow):
    """The How to Use window (v0.9, U22): an index of sections on
    the left, the guide on the right. It reads HOW-TO-USE.md from
    the app folder, so the file and the window can never drift
    apart. Clicking a section jumps to it in the guide."""

    def __init__(self, parent=None):
        super().__init__("ACR - How to Use", "howto_geometry",
                         parent)
        here = os.path.dirname(os.path.abspath(__file__))
        self.guide_path = os.path.join(here, "HOW-TO-USE.md")
        layout = QHBoxLayout(self)

        self.index_list = QListWidget()
        self.index_list.setMaximumWidth(220)
        self.index_list.currentItemChanged.connect(
            self._on_section_picked)
        layout.addWidget(self.index_list)

        self.browser = QTextBrowser()
        self.browser.setOpenExternalLinks(False)
        layout.addWidget(self.browser, 1)

        self.sections = []
        self._first_show = True
        self._shown_at = None
        self._load_guide()

    def showEvent(self, event):
        super().showEvent(event)
        # v0.10 (R15): on first show the index takes focus and Qt
        # auto-selects row 0, which fires the section jump and
        # leaves the first heading blue-highlighted in the guide.
        # The guide is the reading pane, so it takes the focus
        # instead: the index keeps no current row until the user
        # picks one. Later shows keep the user's pick: the window
        # hides, it is never destroyed.
        if self._first_show:
            self._first_show = False
            self._shown_at = time.monotonic()
            self.browser.setFocus()

    def _on_section_picked(self, current, _previous):
        if current is None:
            return
        # v0.10 (R15) backstop: if the index still auto-selects
        # row 0 right after the first show (the focus landed on
        # the index despite the browser taking it), that pick is
        # not the user's: drop it and clear the row, so a later
        # click on row 0 still fires. Only row 0 within half a
        # second of the first show is dropped: the auto-select
        # always picks the first item, and a real pick of any
        # other row -- or of row 0 after half a second -- is
        # never dropped.
        if (self._shown_at is not None
                and self.index_list.row(current) == 0
                and time.monotonic() - self._shown_at < 0.5):
            self._shown_at = None
            self.index_list.blockSignals(True)
            self.index_list.setCurrentRow(-1)
            self.index_list.clearSelection()
            self.index_list.blockSignals(False)
            return
        # Jump to the section heading in the guide.
        cursor = self.browser.textCursor()
        cursor.movePosition(QTextCursor.Start)
        self.browser.setTextCursor(cursor)
        self.browser.find(current.text())

    def _load_guide(self):
        try:
            with open(self.guide_path, "r",
                      encoding="utf-8") as fh:
                text = fh.read()
        except OSError:
            text = ("# How to Use\n\nThe guide file "
                    "(HOW-TO-USE.md) is missing from the app "
                    "folder.")
        self.browser.setMarkdown(text)
        self.sections = [line[3:].strip()
                         for line in text.splitlines()
                         if line.startswith("## ")]
        self.index_list.clear()
        for section in self.sections:
            self.index_list.addItem(section)


class AnalyzeWindow(SubWindow):
    """The Analyze window (v0.9): point the adaptive loop at a
    whole file or folder and see what is in it. Toy-simple flow
    (U34): pick a file, press Go, read the results. The results
    are display, not controls (U37): the resonance chart over the
    run, the themes found, the core entries, the outliers, and
    the theme shifts. The window holds the pickers, the progress,
    and the results; the run itself lives on the main window,
    which owns the engine."""

    def __init__(self, main):
        super().__init__("ACR - Analyze a file", "analyze_geometry",
                         main, default_width=1000,
                         default_height=700)
        self.main = main
        self._paths = []
        self._source_note = ""
        self._summary = None
        self._trace = []
        # v0.10 (R3): the full-entry dialog, when one is open.
        self._entry_dialog = None
        # v0.10 (R11): the step count the pack preview was built
        # from, so the write can say when the session moved on.
        self._preview_step_count = None
        layout = QVBoxLayout(self)

        layout.addWidget(_header_label("What to analyze"))
        source_row = QHBoxLayout()
        self.source_label = QLabel("No file picked yet.")
        self.source_label.setWordWrap(True)
        source_row.addWidget(self.source_label, 1)
        self.pick_file_button = QPushButton("Pick a file")
        self.pick_file_button.clicked.connect(self.on_pick_file)
        source_row.addWidget(self.pick_file_button)
        self.pick_folder_button = QPushButton("Pick a folder")
        self.pick_folder_button.clicked.connect(self.on_pick_folder)
        source_row.addWidget(self.pick_folder_button)
        layout.addLayout(source_row)
        layout.addWidget(_note_label(
            "Each line becomes one step, in file order. .jsonl "
            "records use their text field. A folder runs every "
            "text file in it, in name order."))

        go_row = QHBoxLayout()
        self.go_button = QPushButton("Go")
        self.go_button.setObjectName("primaryButton")
        self.go_button.pressed.connect(
            lambda: press_dip(self.go_button))
        self.go_button.clicked.connect(self.on_go)
        go_row.addWidget(self.go_button)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.on_cancel)
        self.cancel_button.setEnabled(False)
        go_row.addWidget(self.cancel_button)
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        go_row.addWidget(self.progress_bar, 1)
        layout.addLayout(go_row)
        self.progress_label = QLabel("Nothing analyzed yet.")
        self.progress_label.setWordWrap(True)
        layout.addWidget(self.progress_label)

        # Bento results (v0.9, technique 5): size is the
        # hierarchy. One hero tile for the headline, then the
        # chart, themes, core, outliers, and shifts as tiles
        # with one job each. Rounded corners, uniform 16 px
        # gaps, bigger tile for the more important finding.
        def _tile():
            frame = QFrame()
            frame.setObjectName("tile")
            return frame

        results_grid = QGridLayout()
        results_grid.setSpacing(16)
        layout.addLayout(results_grid, 1)

        self.hero_tile = _tile()
        hero_layout = QVBoxLayout(self.hero_tile)
        hero_layout.setContentsMargins(12, 8, 12, 8)
        hero_layout.addWidget(_header_label("Results"))
        self.hero_title = QLabel("Nothing analyzed yet.")
        self.hero_title.setStyleSheet(
            "font-size: 16px; font-weight: bold;")
        hero_layout.addWidget(self.hero_title)
        self.hero_theme = QLabel(
            "Pick a file, press Go, and the findings land here.")
        self.hero_theme.setWordWrap(True)
        hero_layout.addWidget(self.hero_theme)
        results_grid.addWidget(self.hero_tile, 0, 0, 1, 3)

        chart_tile = _tile()
        chart_layout = QVBoxLayout(chart_tile)
        chart_layout.setContentsMargins(10, 8, 10, 10)
        chart_layout.addWidget(_header_label("Chart"))
        chart_layout.addWidget(_note_label(
            "W_t norm over the whole run. Spikes are resonance "
            "spikes."))
        # v0.10 (R15): the empty message fits Analyze -- nobody
        # steps text here.
        self.chart = NormChart(
            empty_text="Analyze a file to see the chart.")
        # v0.10 (R6): click a point to jump the heatmap and the
        # trace selection to that step, the same as the Views
        # chart.
        self.chart.stepClicked.connect(
            main._on_chart_step_clicked)
        chart_layout.addWidget(self.chart, 1)
        results_grid.addWidget(chart_tile, 1, 0, 1, 2)

        themes_tile = _tile()
        themes_layout = QVBoxLayout(themes_tile)
        themes_layout.setContentsMargins(10, 8, 10, 10)
        themes_layout.addWidget(_header_label("Themes"))
        themes_layout.addWidget(_note_label(
            "Resonance spikes and the entry that triggered each "
            "one."))
        self.themes_list = QListWidget()
        self.themes_list.setUniformItemSizes(True)
        self.themes_list.currentItemChanged.connect(
            self._on_theme_picked)
        # v0.10 (R3): double-click shows the whole entry: the
        # list clips long lines, and the tooltip alone is not
        # enough to read one.
        self.themes_list.itemDoubleClicked.connect(
            self._on_finding_double_clicked)
        themes_layout.addWidget(self.themes_list, 1)
        self.theme_detail = QTextEdit()
        self.theme_detail.setReadOnly(True)
        # Capped: an uncapped QTextEdit's minimum size squeezes
        # the note label above it until its text clips, and
        # starves the list. Floored at three lines: the detail
        # always shows the trigger entry and its neighbours.
        self.theme_detail.setMinimumHeight(55)
        self.theme_detail.setMaximumHeight(80)
        # v0.10 (R2): a read-only display surface: no Tab stop.
        self.theme_detail.setFocusPolicy(Qt.NoFocus)
        themes_layout.addWidget(self.theme_detail, 1)
        results_grid.addWidget(themes_tile, 1, 2, 1, 1)

        core_tile = _tile()
        core_layout = QVBoxLayout(core_tile)
        core_layout.setContentsMargins(10, 8, 10, 10)
        core_layout.addWidget(_header_label("Core entries"))
        core_layout.addWidget(_note_label(
            "Core entries: the most resonant entries overall. "
            "The heart of the file."))
        self.core_list = QListWidget()
        self.core_list.setUniformItemSizes(True)
        self.core_list.itemDoubleClicked.connect(
            self._on_finding_double_clicked)
        core_layout.addWidget(self.core_list, 1)
        results_grid.addWidget(core_tile, 2, 0, 1, 1)

        outlier_tile = _tile()
        outlier_layout = QVBoxLayout(outlier_tile)
        outlier_layout.setContentsMargins(10, 8, 10, 10)
        outlier_layout.addWidget(_header_label("Outliers"))
        outlier_layout.addWidget(_note_label(
            "Outliers: the lowest-resonance entries. The noise "
            "and off-topic lines."))
        self.outlier_list = QListWidget()
        self.outlier_list.setUniformItemSizes(True)
        self.outlier_list.itemDoubleClicked.connect(
            self._on_finding_double_clicked)
        outlier_layout.addWidget(self.outlier_list, 1)
        results_grid.addWidget(outlier_tile, 2, 1, 1, 1)

        shift_tile = _tile()
        shift_layout = QVBoxLayout(shift_tile)
        shift_layout.setContentsMargins(10, 8, 10, 10)
        shift_layout.addWidget(_header_label("Theme shifts"))
        shift_layout.addWidget(_note_label(
            "Theme shifts: the steps where resonance drops "
            "sharply and stays low. Topic boundaries."))
        self.shift_list = QListWidget()
        self.shift_list.setUniformItemSizes(True)
        self.shift_list.itemDoubleClicked.connect(
            self._on_finding_double_clicked)
        shift_layout.addWidget(self.shift_list, 1)
        results_grid.addWidget(shift_tile, 2, 2, 1, 1)

        results_grid.setRowStretch(0, 0)
        results_grid.setRowStretch(1, 3)
        results_grid.setRowStretch(2, 2)
        results_grid.setColumnStretch(0, 1)
        results_grid.setColumnStretch(1, 1)
        results_grid.setColumnStretch(2, 1)

        export_row = QHBoxLayout()
        self.export_text_button = QPushButton(
            "Export curated file")
        self.export_text_button.clicked.connect(
            self.on_export_curated_text)
        self.export_text_button.setEnabled(False)
        export_row.addWidget(self.export_text_button)
        self.export_jsonl_button = QPushButton(
            "Export curated .jsonl")
        self.export_jsonl_button.clicked.connect(
            self.on_export_curated_jsonl)
        self.export_jsonl_button.setEnabled(False)
        export_row.addWidget(self.export_jsonl_button)
        self.export_report_button = QPushButton("Export report")
        self.export_report_button.clicked.connect(
            self.on_export_report)
        self.export_report_button.setEnabled(False)
        export_row.addWidget(self.export_report_button)
        self.export_packs_button = QPushButton(
            "Export STELLA data pack")
        self.export_packs_button.clicked.connect(
            self.on_export_stella_packs)
        self.export_packs_button.setEnabled(False)
        export_row.addWidget(self.export_packs_button)
        export_row.addStretch(1)
        layout.addLayout(export_row)
        layout.addWidget(_note_label(
            "A STELLA data pack teaches a model what ACR found: "
            "themes, matches, analogies, judgments. One folder "
            "per theme."))

        # v0.10 (R11): the pack preview. Export fills this pane
        # with the packs it is about to write -- names, row
        # counts, first rows -- and the Write button appears.
        # Nothing is written until Write is pressed. A pane, not
        # a dialog: export is a routine event, and routine
        # events never pop a dialog (calm technology).
        self.pack_preview_label = _header_label("Pack preview")
        self.pack_preview_label.hide()
        layout.addWidget(self.pack_preview_label)
        self.pack_preview = QTextEdit()
        self.pack_preview.setReadOnly(True)
        self.pack_preview.setFocusPolicy(Qt.NoFocus)
        self.pack_preview.setMaximumHeight(160)
        self.pack_preview.hide()
        layout.addWidget(self.pack_preview)
        self.write_packs_button = QPushButton("Write these packs")
        self.write_packs_button.setObjectName("primaryButton")
        self.write_packs_button.clicked.connect(
            self.on_write_stella_packs)
        self.write_packs_button.hide()
        layout.addWidget(self.write_packs_button)

        # v0.10 (R2): Tab order is the visual order.
        QWidget.setTabOrder(self.pick_file_button,
                            self.pick_folder_button)
        QWidget.setTabOrder(self.pick_folder_button,
                            self.go_button)
        QWidget.setTabOrder(self.go_button, self.cancel_button)
        QWidget.setTabOrder(self.cancel_button, self.themes_list)
        QWidget.setTabOrder(self.themes_list, self.core_list)
        QWidget.setTabOrder(self.core_list, self.outlier_list)
        QWidget.setTabOrder(self.outlier_list, self.shift_list)
        QWidget.setTabOrder(self.shift_list,
                            self.export_text_button)
        QWidget.setTabOrder(self.export_text_button,
                            self.export_jsonl_button)
        QWidget.setTabOrder(self.export_jsonl_button,
                            self.export_report_button)
        QWidget.setTabOrder(self.export_report_button,
                            self.export_packs_button)
        QWidget.setTabOrder(self.export_packs_button,
                            self.write_packs_button)

    # -- picking ------------------------------------------------------
    def on_pick_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Pick a file to analyze", "",
            "Text files (*.txt *.md *.jsonl);;All files (*)")
        if not path:
            return
        self._paths = [path]
        self._source_note = os.path.basename(path)
        self.source_label.setText("File: %s" % path)

    def on_pick_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Pick a folder to analyze", "")
        if not folder:
            return
        paths = collect_batch_files(folder)
        if not paths:
            self.main.status_label.setText(
                "No text files in that folder.")
            return
        self._paths = paths
        self._source_note = ("folder %s (%d files)"
                             % (os.path.basename(folder),
                                len(paths)))
        self.source_label.setText(
            "Folder: %s (%d text files)" % (folder, len(paths)))

    def on_go(self):
        if not self._paths:
            self.main.status_label.setText(
                "Pick a file or a folder first.")
            return
        self.main.start_batch(self._paths, self._source_note)

    def on_cancel(self):
        self.main.on_cancel_batch()

    # -- results ------------------------------------------------------
    def show_results(self, summary, trace):
        """Fill the bento results tiles from a summary and its
        trace. Called on every refresh, so a loaded session
        restores the analysis too: the results are computed from
        the trace, and the trace is what a session saves."""
        self._summary = summary
        self._trace = list(trace)
        self.chart.set_data(
            [entry["new_norm"] for entry in trace],
            [entry["new_layer"] for entry in trace])
        steps = summary["steps"]
        themes = summary["themes"]
        if steps == 0:
            self.hero_title.setText("Nothing analyzed yet.")
            self.hero_theme.setText(
                "Pick a file, press Go, and the findings land "
                "here.")
        else:
            # v0.10 (R15): the hero counts session steps, while
            # the progress label counts the run's steps. Say
            # which, so the two numbers never look like a
            # contradiction.
            self.hero_title.setText(
                "%d theme%s found in %d steps in this session"
                % (len(themes),
                   "" if len(themes) == 1 else "s", steps))
            if themes:
                best = max(themes,
                           key=lambda entry: entry["resonance"])
                self.hero_theme.setText(
                    "Strongest theme: \"%s\" (resonance %.3f, "
                    "step %d)"
                    % (self.main._entry_text(best),
                       best["resonance"], best["id"]))
            else:
                self.hero_theme.setText(
                    "No resonance spikes: nothing in this file "
                    "hung together as a theme.")
        selected = None
        current = self.themes_list.currentItem()
        if current is not None:
            selected = current.data(Qt.UserRole)
        self.themes_list.blockSignals(True)
        self.themes_list.clear()
        restore_row = -1
        for entry in summary["themes"]:
            self.themes_list.addItem(
                "Step %d: %s (resonance %.3f)"
                % (entry["id"], self.main._entry_text(entry),
                   entry["resonance"]))
            item = self.themes_list.item(
                self.themes_list.count() - 1)
            item.setData(Qt.UserRole, entry["id"])
            # v0.10 (R3): the full text, on hover and on
            # double-click: the list clips long lines.
            item.setToolTip(self.main._entry_text(entry))
            if entry["id"] == selected:
                restore_row = self.themes_list.count() - 1
        self.themes_list.blockSignals(False)
        if restore_row >= 0:
            self.themes_list.setCurrentRow(restore_row)
        elif self.themes_list.count() == 0:
            self.theme_detail.setPlainText(
                "No resonance spikes yet.")
        elif current is None:
            self.theme_detail.setPlainText(
                "Click a theme to see its trace entries.")
        self.core_list.clear()
        for entry in summary["core"]:
            self.core_list.addItem(
                "%.3f  %s" % (entry["resonance"],
                              self.main._entry_text(entry)))
            item = self.core_list.item(self.core_list.count() - 1)
            item.setData(Qt.UserRole, entry["id"])
            item.setToolTip(self.main._entry_text(entry))
        self.outlier_list.clear()
        for entry in summary["outliers"]:
            self.outlier_list.addItem(
                "%.3f  %s" % (entry["resonance"],
                              self.main._entry_text(entry)))
            item = self.outlier_list.item(
                self.outlier_list.count() - 1)
            item.setData(Qt.UserRole, entry["id"])
            item.setToolTip(self.main._entry_text(entry))
        self.shift_list.clear()
        by_id = {entry["id"]: entry for entry in trace}
        for entry in summary["shifts"]:
            prev = by_id.get(entry["id"] - 1)
            if prev is not None:
                self.shift_list.addItem(
                    "Step %d: resonance fell from %.3f to %.3f"
                    % (entry["id"], prev["resonance"],
                       entry["resonance"]))
            else:
                self.shift_list.addItem(
                    "Step %d: resonance %.3f"
                    % (entry["id"], entry["resonance"]))
            item = self.shift_list.item(self.shift_list.count() - 1)
            item.setData(Qt.UserRole, entry["id"])
            item.setToolTip(self.main._entry_text(entry))
        has_steps = summary["steps"] > 0
        self.export_text_button.setEnabled(has_steps)
        self.export_jsonl_button.setEnabled(has_steps)
        self.export_report_button.setEnabled(has_steps)
        self.export_packs_button.setEnabled(has_steps)

    def show_live(self, trace, done, total):
        """Mid-run refresh (v0.10, R5): the chart and the hero
        tile update as steps land, so a long run shows its
        findings while it is still going. Only the chart and
        the hero: the finding lists need the finished summary,
        and rebuilding them every progress tick would cost more
        than it shows. Called from the throttled progress
        handler, on the UI thread, with a snapshot of the trace.
        """
        self.chart.set_data(
            [entry["new_norm"] for entry in trace],
            [entry["new_layer"] for entry in trace])
        themes = sum(1 for entry in trace if entry["new_layer"])
        steps = len(trace)
        self.hero_title.setText(
            "Analyzing... %d theme%s so far in %d steps in this "
            "session"
            % (themes, "" if themes == 1 else "s", steps))
        self.hero_theme.setText(
            "Run in progress: line %d of %d in this run."
            % (done, total))

    def _on_finding_double_clicked(self, item):
        """Double-click on a finding shows the whole entry
        (v0.10, R3): the lists clip long lines, and a tooltip
        is not enough to read one."""
        if item is None:
            return
        step_id = item.data(Qt.UserRole)
        for entry in self._trace:
            if entry["id"] == step_id:
                self._show_full_entry(
                    self.main._entry_text(entry),
                    "Step %d" % step_id)
                return

    def _show_full_entry(self, text, title):
        """One entry's full text in a small read-only dialog.
        Modeless: it never blocks the window behind it, and a
        second double-click replaces it."""
        if self._entry_dialog is not None:
            self._entry_dialog.close()
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        dialog.setAttribute(Qt.WA_DeleteOnClose)
        layout = QVBoxLayout(dialog)
        view = QTextEdit()
        view.setReadOnly(True)
        view.setPlainText(text)
        view.setMinimumSize(500, 200)
        layout.addWidget(view)
        close = QPushButton("Close")
        close.clicked.connect(dialog.close)
        layout.addWidget(close)
        dialog.destroyed.connect(self._clear_entry_dialog)
        self._entry_dialog = dialog
        dialog.show()

    def _clear_entry_dialog(self, _obj=None):
        self._entry_dialog = None

    def _on_theme_picked(self, current, _previous):
        if current is None:
            return
        step_id = current.data(Qt.UserRole)
        lines = []
        for entry in self._trace:
            if entry["id"] == step_id - 1:
                lines.append(self.main._format_trace_entry(
                    entry, "Step %d (before)" % entry["id"]))
            elif entry["id"] == step_id:
                lines.append(self.main._format_trace_entry(
                    entry, "Step %d (spike)" % entry["id"]))
            elif entry["id"] == step_id + 1:
                lines.append(self.main._format_trace_entry(
                    entry, "Step %d (after)" % entry["id"]))
        if not lines:
            self.theme_detail.setPlainText(
                "No trace entries for step %d." % step_id)
            return
        self.theme_detail.setPlainText("\n".join(lines))

    # -- exports ------------------------------------------------------
    def _core_texts(self):
        if not self._summary:
            return []
        return [self.main._entry_text(entry)
                for entry in self._summary["core"]]

    def export_curated_text_to(self, path):
        """Write the core entries as plain text, one per line.
        Returns the path. The parent folder is created when it
        is missing, so a folder that vanished between picking
        and writing still gets the export."""
        texts = self._core_texts()
        os.makedirs(os.path.dirname(os.path.abspath(path)),
                    exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            for text in texts:
                fh.write(text + "\n")
        self.main.logger.action(
            "curated file exported to %s" % path)
        return path

    def export_curated_jsonl_to(self, path):
        """Write the core entries as .jsonl, one {"text": ...}
        object per line. Returns the path."""
        texts = self._core_texts()
        os.makedirs(os.path.dirname(os.path.abspath(path)),
                    exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            for text in texts:
                fh.write(json.dumps({"text": text}) + "\n")
        self.main.logger.action(
            "curated .jsonl exported to %s" % path)
        return path

    def export_report_to(self, path):
        """Write the readable analysis report. Returns the path."""
        report = format_analysis_report(
            self._summary, VERSION, self.main._entry_text,
            self.main._batch_source_note)
        os.makedirs(os.path.dirname(os.path.abspath(path)),
                    exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(report)
        self.main.logger.action(
            "analysis report exported to %s" % path)
        return path

    def on_export_curated_text(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export curated file", "acr-curated.txt",
            "Text files (*.txt)")
        if not path:
            return
        try:
            self.export_curated_text_to(path)
        except OSError:
            self.main.logger.exception("curated export failed")
            self.main.status_label.setText(
                "Export failed - could not write.")
            return
        self.main.status_label.setText(
            "Curated file exported to %s" % path)

    def on_export_curated_jsonl(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export curated .jsonl", "acr-curated.jsonl",
            "JSONL files (*.jsonl)")
        if not path:
            return
        try:
            self.export_curated_jsonl_to(path)
        except OSError:
            self.main.logger.exception("curated export failed")
            self.main.status_label.setText(
                "Export failed - could not write.")
            return
        self.main.status_label.setText(
            "Curated .jsonl exported to %s" % path)

    def on_export_report(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export report", "acr-analysis.txt",
            "Text files (*.txt)")
        if not path:
            return
        try:
            self.export_report_to(path)
        except OSError:
            self.main.logger.exception("report export failed")
            self.main.status_label.setText(
                "Export failed - could not write.")
            return
        self.main.status_label.setText(
            "Report exported to %s" % path)

    def export_stella_packs_to(self, dest_dir):
        """Write the STELLA data packs for the current run under
        dest_dir. Returns the export_stella_packs result dict:
        {"packs": [(folder, rows, under_minimum)], "error": None
        or a plain message}. Under-minimum packs are still
        written, with the warning in their README."""
        result = export_stella_packs(
            self._summary, self.main._pack_item,
            self.main._pack_context(), dest_dir,
            self.main._batch_source_note)
        for folder, rows, _under in result["packs"]:
            self.main.logger.action(
                "STELLA data pack exported to %s (%d rows)"
                % (folder, rows))
        return result

    def on_export_stella_packs(self):
        """Preview the packs before writing (v0.10, R11): the
        preview pane shows the pack names, the row counts per
        kind, and the first rows, and the Write button appears.
        Nothing is written until Write is pressed. The plan is
        built from the same summary the write uses, so the
        preview is what the write writes -- unless the session
        changes in between, in which case the write uses the
        newer state and says so in the status line."""
        if self._summary is None or self._summary["steps"] == 0:
            self.main.status_label.setText(
                "Nothing analyzed yet: run an analysis first.")
            return
        packs = build_pack_rows(
            self._summary, self.main._pack_item,
            self.main._pack_context(),
            self.main._batch_source_note)
        if not packs:
            self.main.status_label.setText(
                "No packs to write from this run.")
            return
        self._preview_step_count = self._summary["steps"]
        self.pack_preview.setPlainText(
            self._render_pack_preview(packs))
        self.pack_preview_label.show()
        self.pack_preview.show()
        self.write_packs_button.show()
        total_rows = sum(len(pack["rows"]) for pack in packs)
        self.main.status_label.setText(
            "Preview ready: %d pack%s, %d rows. Press Write "
            "these packs to save them."
            % (len(packs), "s" if len(packs) != 1 else "",
               total_rows))

    def _render_pack_preview(self, packs):
        """The preview text for a pack plan: one block per pack
        -- name, row count, counts per kind, first three rows.
        Rows are shown prompt then completion, trimmed to fit
        the pane."""
        lines = []
        for pack in packs:
            counts = {}
            for _row, entry in pack["rows"]:
                kind = entry["kind"]
                counts[kind] = counts.get(kind, 0) + 1
            lines.append("Pack: %s" % pack["name"])
            lines.append("  %d rows: %s"
                         % (len(pack["rows"]),
                            ", ".join("%d %s" % (counts[kind], kind)
                                      for kind in
                                      ("Theme membership",
                                       "Closest match", "Analogy",
                                       "Resonance judgment")
                                      if counts.get(kind))))
            for row, _entry in pack["rows"][:3]:
                lines.append("  Q: %s"
                             % self._preview_trim(
                                 row["prompt"], 100))
                lines.append("  A: %s"
                             % self._preview_trim(
                                 row["completion"], 100))
            if len(pack["rows"]) > 3:
                lines.append("  ... and %d more rows"
                             % (len(pack["rows"]) - 3))
            lines.append("")
        return "\n".join(lines).rstrip("\n")

    @staticmethod
    def _preview_trim(text, limit):
        if len(text) <= limit:
            return text
        return text[:limit - 3] + "..."

    def on_write_stella_packs(self):
        """Write the previewed packs, after the folder pick.
        The write rebuilds from the current summary: if the
        session changed since the preview, the packs follow the
        newer state, and the status line names the step count
        the preview was built from."""
        folder = QFileDialog.getExistingDirectory(
            self, "Where to save the STELLA data packs",
            default_packs_dir())
        if not folder:
            return
        try:
            result = self.export_stella_packs_to(folder)
        except OSError:
            self.main.logger.exception("pack export failed")
            self.main.status_label.setText(
                "Export failed - could not write.")
            return
        if result["error"] is not None:
            self.main.status_label.setText(result["error"])
            return
        self.write_packs_button.hide()
        count = len(result["packs"])
        under = sum(1 for _f, _n, flag in result["packs"]
                    if flag)
        message = ("Wrote %d STELLA data pack%s to %s."
                   % (count, "s" if count != 1 else "", folder))
        if (self._preview_step_count is not None
                and self._summary is not None
                and self._summary["steps"]
                != self._preview_step_count):
            message += (" The session changed since the preview "
                        "(preview was %d steps, now %d): these "
                        "packs are from the current state."
                        % (self._preview_step_count,
                           self._summary["steps"]))
        self._preview_step_count = None
        if under == 1:
            message += (" 1 pack has fewer than 2 rows: STELLA "
                        "needs 2 or more, add more source "
                        "material before importing.")
        elif under > 1:
            message += (" %d packs have fewer than 2 rows: "
                        "STELLA needs 2 or more, add more source "
                        "material before importing." % under)
        self.main.status_label.setText(message)
        self.main._show_completion_mark()


class MainWindow(QMainWindow):
    def __init__(self, log_dir=None, local_base_url=None):
        super().__init__()
        self.setWindowTitle("ACR v%s - Adaptive Contextual "
                            "Resonance" % VERSION)
        # v0.8: the launcher's StartupWMClass is ACR, so the app
        # says the same name: without it the desktop shows the
        # running window as a separate icon instead of grouping
        # it with the launcher.
        QApplication.instance().setApplicationName("ACR")
        # v0.8: the window carries its icon: the 48px render, or
        # the SVG when the PNGs are missing (as in a fresh clone,
        # where the repo holds only the SVG).
        here = os.path.dirname(os.path.abspath(__file__))
        icon_path = os.path.join(here, "icons", "acr-48.png")
        if not os.path.exists(icon_path):
            icon_path = os.path.join(here, "icons", "acr.svg")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        # v0.8: the window remembers its size and position
        # between runs (saved on close). First run, with no saved
        # geometry, is 1100x700 -- v0.6: 700 tall, up from 680,
        # the completion row under the synthesis panel pushed the
        # layout's minimum height past 680. Still fits his
        # 1366x768 screen.
        self._settings = QSettings("BrokeNBored", "ACR")
        geometry = self._settings.value("geometry")
        if geometry is None or not self.restoreGeometry(geometry):
            self.resize(1100, 700)
        self.engine = Engine()
        # Encoder state: "builtin" (default) or "local". The local
        # encoder needs the local model service; local_model is
        # the model name, local_dim the dim it returns. The engine
        # dim always matches the active encoder.
        self.encoder_kind = "builtin"
        self.local_model = None
        self.client = LocalModelClient(local_base_url)
        self.service = LocalServiceManager(self.client)
        # Heatmap history: one (step id, per-row change norms)
        # entry per step, in step order. The heatmap shows the
        # latest step unless the user picks an older one.
        self._heat_history = []
        self._heat_follow = True
        # Last on-demand completion, shown in the synthesis panel
        # until the session changes (reset, load, encoder switch).
        self._completion_model = None
        self._completion_text = None
        # v0.9: the user picked "Local model" in the encoder
        # picker but has not picked a model yet. The built-in
        # encoder keeps running until a model is chosen: picking
        # a mode is not picking a model, and nothing is selected
        # by default (U6).
        self._local_pending = False
        # Embedding dims the app has actually seen, model name ->
        # dim, from a real first embedding. Shown in the picker
        # once known, never guessed (U39).
        self._known_dims = {}
        # The last model list from the service (info dicts, see
        # parse_tags_info): the pickers rebuild their guided lines
        # from this, so both pickers always agree.
        self._model_infos = []
        # v0.9: one local call at a time, in a worker thread
        # (U30). _local_op is "step" or "complete"; the pending
        # step text waits for its embedding to come back.
        self._local_thread = None
        self._local_worker = None
        self._local_cancelled = False
        self._local_op = None
        self._local_model_name = None
        self._pending_step_text = None
        # Total RAM, for the model picker's fits-this-PC note.
        self._total_ram = mem_total_bytes()
        # v0.9 Analyze: one batch run at a time, in a worker
        # thread. _full_texts maps step id to the full input text:
        # the trace holds an 80-character preview, and the curated
        # export needs the whole line. _locators maps step id to
        # (file name, line number) for batch steps: where each
        # step came from, for the STELLA pack manifest. Typed
        # steps have no locator.
        self._batch_thread = None
        self._batch_worker = None
        self._batch_cancelled = False
        self._batch_source_note = ""
        self._full_texts = {}
        self._locators = {}
        # The resonance moment (v0.9, technique 8): step ids
        # already celebrated, so each flag fires the moment once.
        self._known_flag_steps = set()
        self._glow_anim = None
        self._flag_anim = None
        self._flag_anim_step = None
        # v0.10 (R4): the example run's remaining sentences, when
        # one is in flight on the local encoder (built-in runs
        # at once, so the queue stays empty there).
        self._example_queue = []
        # v0.10 (R13): when the batch progress UI last updated,
        # for the throttle.
        self._last_progress_ui = 0.0

        if log_dir is None:
            log_dir = os.path.join(os.path.dirname(
                os.path.abspath(__file__)), "logs")
        self.logger = DebugLogger(APP_NAME, VERSION, log_dir=log_dir)
        self.logger.register_service("ollama", check_ollama)
        self.logger.register_dependency("PySide6")
        self.logger.set_setting("encoder", "built-in deterministic")
        self.logger.set_setting("dim", DIM)
        self.logger.set_setting("rank", RANK)
        self.logger.set_setting("buffer_size", BUFFER_SIZE)
        self.logger.install_crash_handler()
        self.logger.info("ACR %s started", VERSION)

        self._build_ui()
        # v0.10 (R10): apply the built-in collapsed state at
        # launch: the local-model chrome starts hidden.
        self._set_encoder_ui()
        self._refresh_all()

    def closeEvent(self, event):
        # v0.8: remember the geometry for the next run.
        self._settings.setValue("geometry", self.saveGeometry())
        self._settings.sync()
        # v0.9: the dedicated windows close too (each saves its
        # own geometry as it hides). A local call in flight is
        # left to finish in the background; its answer is ignored
        # because the window is gone.
        self._local_cancelled = True
        self._batch_cancelled = True
        if self._batch_worker is not None:
            try:
                self._batch_worker.cancelled = True
            except RuntimeError:
                pass
        for window in (self.controls_window, self.trace_window,
                       self.views_window, self.howto_window,
                       self.analyze_window):
            window.close()
        super().closeEvent(event)

    def eventFilter(self, watched, event):
        # v0.8: Enter in the input box steps the loop, like the
        # Step button. Shift+Enter still makes a newline.
        if (watched is self.input_edit
                and event.type() == QEvent.KeyPress
                and event.key() in (Qt.Key_Return, Qt.Key_Enter)
                and not event.modifiers() & Qt.ShiftModifier):
            self.on_step()
            return True
        return super().eventFilter(watched, event)

    # -- UI -----------------------------------------------------------
    def _build_ui(self):
        self._build_menu()
        central = QWidget(self)
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(4)

        # Main is toy-simple (U34): encoder, input, Step, the key
        # readouts, the synthesis head, and the status line. The
        # controls live in the Controls window, the trace in the
        # Trace log window, the charts in the Resonance views
        # window: nothing was removed, only reorganised (U1).
        about = QLabel(
            "Small-scale prototype of the adaptive loop. Type text, "
            "press Step or Enter, and W_t adapts. This is not a full "
            "neural engine.")
        about.setWordWrap(True)
        root.addWidget(about)

        # Encoder: built-in (default) or the local model service.
        # Switching encoders starts a fresh W_t: the embedding
        # sizes differ, so nothing carries over. Stated here, not
        # hidden.
        encoder_row = QHBoxLayout()
        encoder_row.addWidget(QLabel("Encoder"))
        self.encoder_combo = QComboBox()
        self.encoder_combo.addItems(["Built-in", "Local model"])
        self.encoder_combo.currentIndexChanged.connect(
            self._on_encoder_changed)
        encoder_row.addWidget(self.encoder_combo)
        self.model_combo = QComboBox()
        self.model_combo.setEnabled(False)
        self.model_combo.setPlaceholderText("Pick a model")
        self.model_combo.currentIndexChanged.connect(
            self._on_model_changed)
        encoder_row.addWidget(self.model_combo, 1)
        self.start_button = QPushButton("Start service")
        self.start_button.clicked.connect(self.on_start_service)
        encoder_row.addWidget(self.start_button)
        root.addLayout(encoder_row)
        self.encoder_label = QLabel("Built-in encoder, dim 64.")
        root.addWidget(self.encoder_label)
        # v0.10 (R10): the local-model notes are named, so the
        # progressive disclosure can hide them in built-in mode.
        self.model_note = _note_label(
            "The model list shows what each model does and whether "
            "it fits this PC. Nothing is picked for you.")
        root.addWidget(self.model_note)
        self.switch_note = QLabel(
            "Switching encoders starts a fresh W_t (the embedding "
            "sizes differ).")
        self.switch_note.setWordWrap(True)
        root.addWidget(self.switch_note)

        root.addWidget(_header_label("Input text"))
        self.input_edit = QPlainTextEdit()
        self.input_edit.setPlaceholderText(
            "Type a sentence, then press Enter or Step...")
        # v0.10 (R2): Tab moves the focus on, like every other
        # control. The input is one sentence; nobody needs a
        # literal tab in it, and the default QPlainTextEdit
        # behaviour swallows Tab for text editing, which broke
        # the visual-order Tab chain at its first hop.
        self.input_edit.setTabChangesFocus(True)
        # v0.6: fixed at two lines. The input is one sentence; the
        # freed height goes to the synthesis panel below, which
        # holds four sections plus the completion.
        self.input_edit.setFixedHeight(56)
        # v0.8: Enter steps the loop; see eventFilter.
        self.input_edit.installEventFilter(self)
        root.addWidget(self.input_edit)

        step_row = QHBoxLayout()
        self.step_button = QPushButton("Step")
        self.step_button.setObjectName("primaryButton")
        self.step_button.pressed.connect(
            lambda: press_dip(self.step_button))
        self.step_button.clicked.connect(self.on_step)
        step_row.addWidget(self.step_button, 1)
        # v0.9 Analyze: the one entry point on the main window.
        # One-at-a-time stepping stays as the explore mode.
        self.analyze_button = QPushButton("Analyze a file")
        self.analyze_button.clicked.connect(self.show_analyze_window)
        step_row.addWidget(self.analyze_button)
        # v0.10 (R4): teach by playing. Offered, not defaulted:
        # visible only while the session is empty, gone once
        # the user steps anything.
        self.example_button = QPushButton("Try an example")
        self.example_button.clicked.connect(self.on_try_example)
        step_row.addWidget(self.example_button)
        root.addLayout(step_row)
        self.example_label = _note_label(
            "New here? Try an example steps four sentences about "
            "quantum entanglement through the loop, so you can "
            "see W_t adapt.")
        root.addWidget(self.example_label)

        readout_row = QHBoxLayout()
        self.step_label = QLabel("Step: 0")
        self.norm_label = QLabel("W norm: -")
        self.resonance_label = QLabel("Resonance: -")
        # v0.10 (R7): the persistent flag marker. The 300 ms
        # moment passes; this gold dot stays, naming the latest
        # flagged step on hover, until the next flag or a reset.
        # A separate label, never the readout's stylesheet: the
        # readout still rests at its normal style after the
        # moment.
        self.flag_marker = QLabel("\u25cf")
        self.flag_marker.setStyleSheet(
            "color: %s;" % ACCENT_GOLD)
        self.flag_marker.hide()
        self.alpha_label = QLabel("Alpha: 0.30")
        readout_row.addWidget(self.step_label)
        readout_row.addWidget(self.norm_label)
        readout_row.addWidget(self.resonance_label)
        readout_row.addWidget(self.flag_marker)
        readout_row.addWidget(self.alpha_label)
        readout_row.addStretch(1)
        root.addLayout(readout_row)

        root.addWidget(_header_label(
            "Synthesis head - synthesis from this session's own "
            "subspace, not general knowledge."))
        self.synthesis_text = QTextEdit()
        self.synthesis_text.setReadOnly(True)
        # A compact panel, not a window-filler: the synthesis head
        # is one readout among several on the main window. Without
        # the cap it takes all the spare height and the completion
        # scroll-to-bottom has nothing to scroll.
        self.synthesis_text.setMaximumHeight(170)
        # v0.10 (R2): a read-only display surface, not a control:
        # it takes no Tab stop.
        self.synthesis_text.setFocusPolicy(Qt.NoFocus)
        root.addWidget(self.synthesis_text, 1)

        # Optional local-model completion: on-demand only, and
        # only while the local encoder is active. The picker lists
        # every installed model with the same guided lines as the
        # encoder picker, and starts unselected (U6): the button
        # enables once a model is picked.
        complete_row = QHBoxLayout()
        self.complete_button = QPushButton("Complete with local model")
        self.complete_button.clicked.connect(self.on_complete)
        self.complete_button.setEnabled(False)
        complete_row.addWidget(self.complete_button)
        self.completion_combo = QComboBox()
        self.completion_combo.setEnabled(False)
        self.completion_combo.setPlaceholderText("Pick a model")
        self.completion_combo.currentIndexChanged.connect(
            self._on_completion_model_changed)
        complete_row.addWidget(self.completion_combo, 1)
        root.addLayout(complete_row)
        self.completion_note = QLabel(
            "Completion needs the Local model encoder.")
        self.completion_note.setWordWrap(True)
        root.addWidget(self.completion_note)

        # Status row: the busy light (U16) plus the status text,
        # and a Cancel button that only appears while a local call
        # is running (U26).
        status_row = QHBoxLayout()
        self.busy_light = QLabel()
        self.busy_light.setFixedSize(12, 12)
        self._set_busy(False)
        status_row.addWidget(self.busy_light)
        self.status_label = QLabel("Ready.")
        self.status_label.setWordWrap(True)
        status_row.addWidget(self.status_label, 1)
        self.completion_mark = CheckmarkWidget()
        status_row.addWidget(self.completion_mark)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.on_cancel_local)
        self.cancel_button.hide()
        status_row.addWidget(self.cancel_button)
        root.addLayout(status_row)

        self._build_controls_window()
        self._build_trace_window()
        self._build_views_window()
        self.howto_window = HowToUseWindow(self)
        self.analyze_window = AnalyzeWindow(self)

        # v0.10 (R14): the View menu's checkable actions follow
        # the windows' visibility, so the menu shows which
        # windows are open.
        self.controls_window.visibilityChanged.connect(
            self.action_show_controls.setChecked)
        self.trace_window.visibilityChanged.connect(
            self.action_show_trace.setChecked)
        self.views_window.visibilityChanged.connect(
            self.action_show_views.setChecked)
        self.analyze_window.visibilityChanged.connect(
            self.action_show_analyze.setChecked)

        # v0.10 (R2): Tab order is the visual order, top to
        # bottom. Hidden widgets are skipped by Qt, so the
        # collapsed local-model chrome (R10) never takes a stop
        # in built-in mode.
        QWidget.setTabOrder(self.encoder_combo, self.model_combo)
        QWidget.setTabOrder(self.model_combo, self.start_button)
        QWidget.setTabOrder(self.start_button, self.input_edit)
        QWidget.setTabOrder(self.input_edit, self.step_button)
        QWidget.setTabOrder(self.step_button, self.analyze_button)
        QWidget.setTabOrder(self.analyze_button,
                            self.example_button)
        QWidget.setTabOrder(self.example_button,
                            self.complete_button)
        QWidget.setTabOrder(self.complete_button,
                            self.completion_combo)
        QWidget.setTabOrder(self.completion_combo,
                            self.cancel_button)
        # v0.10 (R2): the input has the focus at launch, so the
        # first thing the user types goes where it belongs.
        self.input_edit.setFocus()

    def _set_busy(self, busy):
        """The busy light (v0.9, U16): gold while the app is
        working, grey at rest."""
        color = ACCENT_GOLD if busy else BUSY_OFF
        self.busy_light.setStyleSheet(
            "background-color: %s; border-radius: 6px;" % color)

    def _build_menu(self):
        """A real menu bar (v0.9, U38): every function reachable
        from the menu, not only from buttons."""
        # v0.10 (R2): every menu action carries a shortcut, and
        # the shortcuts are listed in HOW-TO-USE.md. About is
        # the one exception: no standard shortcut exists for it.
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("File")
        self.action_save_session = QAction("Save session", self)
        self.action_save_session.setShortcut("Ctrl+S")
        self.action_save_session.triggered.connect(
            self.on_save_session)
        file_menu.addAction(self.action_save_session)
        self.action_load_session = QAction("Load session", self)
        self.action_load_session.setShortcut("Ctrl+O")
        self.action_load_session.triggered.connect(
            self.on_load_session)
        file_menu.addAction(self.action_load_session)
        file_menu.addSeparator()
        self.action_export_json = QAction("Export trace JSON", self)
        self.action_export_json.setShortcut("Ctrl+J")
        self.action_export_json.triggered.connect(self.on_export)
        file_menu.addAction(self.action_export_json)
        self.action_export_csv = QAction("Export trace CSV", self)
        self.action_export_csv.setShortcut("Ctrl+Shift+J")
        self.action_export_csv.triggered.connect(self.on_export_csv)
        file_menu.addAction(self.action_export_csv)
        self.action_export_session = QAction("Export session", self)
        self.action_export_session.setShortcut("Ctrl+E")
        self.action_export_session.triggered.connect(
            self.on_export_session)
        file_menu.addAction(self.action_export_session)
        file_menu.addSeparator()
        self.action_debug_report = QAction(
            "Generate debug report", self)
        self.action_debug_report.setShortcut("Ctrl+D")
        self.action_debug_report.triggered.connect(
            self.on_debug_report)
        file_menu.addAction(self.action_debug_report)
        self.action_reset = QAction("Reset", self)
        self.action_reset.setShortcut("Ctrl+R")
        self.action_reset.triggered.connect(self.on_reset)
        file_menu.addAction(self.action_reset)
        file_menu.addSeparator()
        self.action_quit = QAction("Quit", self)
        self.action_quit.setShortcut("Ctrl+Q")
        self.action_quit.triggered.connect(self.close)
        file_menu.addAction(self.action_quit)

        view_menu = menu_bar.addMenu("View")
        # v0.10 (R14): the window actions are checkable toggles:
        # checked while their window is open, following the
        # windows' visibilityChanged signal (wired in _build_ui,
        # after the windows exist). The show_* methods stay
        # open-only: the Analyze button on Main opens, it never
        # toggles.
        self.action_show_controls = QAction("Controls", self)
        self.action_show_controls.setCheckable(True)
        self.action_show_controls.setShortcut("Ctrl+1")
        self.action_show_controls.triggered.connect(
            self.toggle_controls_window)
        view_menu.addAction(self.action_show_controls)
        self.action_show_trace = QAction("Trace log", self)
        self.action_show_trace.setCheckable(True)
        self.action_show_trace.setShortcut("Ctrl+2")
        self.action_show_trace.triggered.connect(
            self.toggle_trace_window)
        view_menu.addAction(self.action_show_trace)
        self.action_show_views = QAction("Resonance views", self)
        self.action_show_views.setCheckable(True)
        self.action_show_views.setShortcut("Ctrl+3")
        self.action_show_views.triggered.connect(
            self.toggle_views_window)
        view_menu.addAction(self.action_show_views)
        self.action_show_analyze = QAction("Analyze a file", self)
        self.action_show_analyze.setCheckable(True)
        self.action_show_analyze.setShortcut("Ctrl+4")
        self.action_show_analyze.triggered.connect(
            self.toggle_analyze_window)
        view_menu.addAction(self.action_show_analyze)
        view_menu.addSeparator()
        # v0.10 (R9): accessibility in the menu, one tap each.
        # Both persist in settings; the reduced-motion toggle
        # wires to the existing fallback, the big-text preset
        # raises the app font two points.
        self.action_reduced_motion = QAction("Reduced motion",
                                             self)
        self.action_reduced_motion.setCheckable(True)
        self.action_reduced_motion.setChecked(reduced_motion())
        self.action_reduced_motion.setShortcut("Ctrl+M")
        self.action_reduced_motion.toggled.connect(
            self._on_reduced_motion_toggled)
        view_menu.addAction(self.action_reduced_motion)
        self.action_big_text = QAction("Big text", self)
        self.action_big_text.setCheckable(True)
        self.action_big_text.setChecked(big_text_enabled())
        self.action_big_text.setShortcut("Ctrl+T")
        self.action_big_text.toggled.connect(
            self._on_big_text_toggled)
        view_menu.addAction(self.action_big_text)

        help_menu = menu_bar.addMenu("Help")
        self.action_how_to_use = QAction("How to Use", self)
        self.action_how_to_use.setShortcut("F1")
        self.action_how_to_use.triggered.connect(
            self.show_howto_window)
        help_menu.addAction(self.action_how_to_use)
        self.action_about = QAction("About", self)
        self.action_about.triggered.connect(self.on_about)
        help_menu.addAction(self.action_about)

    def toggle_controls_window(self, checked):
        if checked:
            self.show_controls_window()
        else:
            self.controls_window.close()

    def toggle_trace_window(self, checked):
        if checked:
            self.show_trace_window()
        else:
            self.trace_window.close()

    def toggle_views_window(self, checked):
        if checked:
            self.show_views_window()
        else:
            self.views_window.close()

    def toggle_analyze_window(self, checked):
        if checked:
            self.show_analyze_window()
        else:
            self.analyze_window.close()

    def _on_reduced_motion_toggled(self, checked):
        """The reduced-motion menu toggle (v0.10, R9): writes
        the setting the fallback already reads. State changes
        still land; only the motion is skipped."""
        QSettings("BrokeNBored", "ACR").setValue(
            "reduced_motion", checked)

    def _on_big_text_toggled(self, checked):
        """The big-text menu toggle (v0.10, R9): persists the
        preset and adjusts the app font two points, up or back.
        The stylesheet is untouched."""
        QSettings("BrokeNBored", "ACR").setValue(
            "big_text", checked)
        adjust_big_text(QApplication.instance(), checked)

    def show_controls_window(self):
        self.controls_window.show()
        self.controls_window.raise_()
        self.controls_window.activateWindow()

    def show_trace_window(self):
        self.trace_window.show()
        self.trace_window.raise_()
        self.trace_window.activateWindow()

    def show_views_window(self):
        self.views_window.show()
        self.views_window.raise_()
        self.views_window.activateWindow()

    def show_howto_window(self):
        self.howto_window.show()
        self.howto_window.raise_()
        self.howto_window.activateWindow()

    def show_analyze_window(self):
        self.analyze_window.show()
        self.analyze_window.raise_()
        self.analyze_window.activateWindow()

    def on_about(self):
        QMessageBox.about(
            self, "About ACR",
            "ACR v%s - Adaptive Contextual Resonance\n\n"
            "A small-scale working model of an adaptive neural "
            "loop. Type text, step the loop, and watch the weight "
            "matrix adapt to what you type.\n\n"
            "This is not a full self-evolving neural engine."
            % VERSION)

    def _build_controls_window(self):
        """The Controls window (v0.9, U1): everything that tunes
        how W_t adapts. Moved out of the main window whole: the
        widgets keep their names, so the behaviour is unchanged."""
        window = SubWindow("ACR - Controls", "controls_geometry",
                           self, default_width=700,
                           default_height=420)
        layout = QVBoxLayout(window)

        layout.addWidget(_header_label("Learning rate (alpha)"))
        # v0.10 (R8): the live value next to the slider. It shows
        # the effective alpha for the next step, the same number
        # as the main window's readout: under Fast-adapt and
        # Gentle the slider value and the effective rate differ,
        # and two different "Alpha" numbers on screen would lie
        # to one of them.
        alpha_row = QHBoxLayout()
        self.alpha_slider = QSlider(Qt.Horizontal)
        self.alpha_slider.setRange(1, 50)
        self.alpha_slider.setValue(30)
        self.alpha_slider.setTickPosition(QSlider.TicksBelow)
        self.alpha_slider.setTickInterval(10)
        self.alpha_slider.valueChanged.connect(self._on_alpha_changed)
        alpha_row.addWidget(self.alpha_slider, 1)
        self.alpha_value_label = QLabel("Alpha: 0.30")
        alpha_row.addWidget(self.alpha_value_label)
        layout.addLayout(alpha_row)
        layout.addWidget(_note_label(
            "How fast W_t adapts. Higher learns faster but can "
            "overshoot. 0.30 is the setting the resonance climb "
            "was verified at."))

        # v0.7: freeze mode and learning-rate schedule share one
        # row: both are adaptation controls. Freeze Off adapts as
        # usual; Whole W_t logs steps without changing W_t; Rank
        # directions freezes only the checked directions below.
        # The slider sets the base alpha the schedule scales from.
        control_row = QHBoxLayout()
        control_row.addWidget(QLabel("Freeze"))
        self.freeze_combo = QComboBox()
        self.freeze_combo.addItems(
            ["Off", "Whole W_t", "Rank directions"])
        self.freeze_combo.currentIndexChanged.connect(
            self._on_freeze_mode_changed)
        control_row.addWidget(self.freeze_combo)
        control_row.addWidget(QLabel("Schedule"))
        self.schedule_combo = QComboBox()
        self.schedule_combo.addItems(
            ["Steady", "Fast-adapt", "Gentle"])
        self.schedule_combo.currentIndexChanged.connect(
            self._on_schedule_changed)
        control_row.addWidget(self.schedule_combo, 1)
        layout.addLayout(control_row)
        layout.addWidget(_note_label(
            "Freeze Off adapts as usual. Whole W_t logs steps "
            "without changing W_t. Rank directions freezes only "
            "the checked directions below."))
        layout.addWidget(_note_label(
            "Schedule scales the learning rate: Steady keeps it, "
            "Fast-adapt starts at twice the base and settles back, "
            "Gentle runs at one third of the base."))

        # Rank-direction toggles, live only in Rank directions
        # mode. Checked means frozen: that component holds still
        # while the rest adapt.
        rank_row = QHBoxLayout()
        rank_row.addWidget(QLabel("Directions"))
        self.rank_checks = []
        for k in range(RANK):
            check = QCheckBox(str(k + 1))
            check.setEnabled(False)
            self.rank_checks.append(check)
            rank_row.addWidget(check)
        rank_row.addStretch(1)
        layout.addLayout(rank_row)
        layout.addWidget(_note_label(
            "Checked directions hold still while the rest adapt. "
            "Live only in Rank directions mode."))

        cap_row = QHBoxLayout()
        cap_row.addWidget(QLabel("Stability cap"))
        self.cap_spin = QDoubleSpinBox()
        self.cap_spin.setRange(0.1, 5.0)
        self.cap_spin.setSingleStep(0.1)
        self.cap_spin.setValue(1.0)
        cap_row.addWidget(self.cap_spin)
        # v0.7: the top singular value of W_t, live next to the
        # cap: the size of W_t's strongest direction.
        self.spectral_label = QLabel("Top singular value: -")
        cap_row.addWidget(self.spectral_label)
        cap_row.addStretch(1)
        layout.addLayout(cap_row)
        layout.addWidget(_note_label(
            "The cap limits how big one step can be. Steps it "
            "clipped show Capped in the trace log. The top "
            "singular value is the size of W_t's strongest "
            "direction."))
        layout.addStretch(1)
        # v0.10 (R2): Tab order is the visual order.
        QWidget.setTabOrder(self.alpha_slider, self.freeze_combo)
        QWidget.setTabOrder(self.freeze_combo, self.schedule_combo)
        previous = self.schedule_combo
        for check in self.rank_checks:
            QWidget.setTabOrder(previous, check)
            previous = check
        QWidget.setTabOrder(previous, self.cap_spin)
        self.controls_window = window

    def _build_trace_window(self):
        """The Trace log window (v0.9, U1): the trace table, the
        revert button, and the latent state buffer."""
        window = SubWindow("ACR - Trace log", "trace_geometry",
                           self)
        layout = QVBoxLayout(window)
        layout.addWidget(_header_label(
            "Trace log - every weight change"))
        self.trace_table = QTableWidget(0, len(TRACE_COLUMNS))
        self.trace_table.setHorizontalHeaderLabels(TRACE_COLUMNS)
        self.trace_table.setEditTriggers(
            QTableWidget.NoEditTriggers)
        # The Step column already numbers the rows, so the row
        # numbers add nothing. Numeric columns size to their
        # contents; the Input column takes the leftover width so
        # all ten columns fit without scrolling.
        self.trace_table.verticalHeader().setVisible(False)
        header = self.trace_table.horizontalHeader()
        for col in range(len(TRACE_COLUMNS)):
            if col == 1:
                header.setSectionResizeMode(
                    col, QHeaderView.Stretch)
            else:
                header.setSectionResizeMode(
                    col, QHeaderView.ResizeToContents)
        # v0.10 (R3): clicking a row shows the full input below
        # the table -- the Input column holds an 80-character
        # preview, and long lines are unreadable in it.
        self.trace_table.currentItemChanged.connect(
            self._on_trace_row_picked)
        layout.addWidget(self.trace_table, 3)
        self.trace_detail = QLabel("")
        self.trace_detail.setWordWrap(True)
        layout.addWidget(self.trace_detail)
        # v0.10 (R15): the empty window teaches instead of
        # showing two blank boxes.
        self.trace_empty_label = _note_label(
            "No steps yet. Step some text on the main window and "
            "every weight change lands here, one row per step.")
        layout.addWidget(self.trace_empty_label)

        # v0.7: roll W_t back to the state after a selected step.
        revert_row = QHBoxLayout()
        self.revert_button = QPushButton(
            "Revert W_t to selected step")
        self.revert_button.clicked.connect(self.on_revert)
        # v0.10 (R15): Revert is disabled with no steps. It
        # enables in _refresh_trace_table once a step lands.
        self.revert_button.setEnabled(False)
        revert_row.addWidget(self.revert_button)
        revert_row.addStretch(1)
        layout.addLayout(revert_row)
        layout.addWidget(_note_label(
            "Click a step in the table, then Revert, to roll W_t "
            "back to its state after that step. It asks first."))

        layout.addWidget(_header_label(
            "Latent state buffer - last %d inputs, newest first"
            % BUFFER_SIZE))
        self.buffer_list = QListWidget()
        layout.addWidget(self.buffer_list, 1)
        # v0.10 (R2): Tab order is the visual order.
        QWidget.setTabOrder(self.trace_table, self.revert_button)
        QWidget.setTabOrder(self.revert_button, self.buffer_list)
        self.trace_window = window

    def _build_views_window(self):
        """The Resonance views window (v0.9, U1): the norm chart,
        the change heatmap, and the layer candidates."""
        window = SubWindow("ACR - Resonance views",
                           "views_geometry", self)
        layout = QVBoxLayout(window)

        layout.addWidget(_header_label(
            "W_t norm over time - spikes are resonance spikes"))
        self.norm_chart = NormChart()
        # v0.10 (R6): click a point to jump the heatmap and the
        # trace selection to that step.
        self.norm_chart.stepClicked.connect(
            self._on_chart_step_clicked)
        layout.addWidget(self.norm_chart, 2)

        layout.addWidget(_header_label(
            "W_t change per step - which rows moved most"))
        heat_row = QHBoxLayout()
        heat_row.addWidget(QLabel("Step"))
        self.heat_spin = QSpinBox()
        self.heat_spin.setMinimum(1)
        self.heat_spin.setMaximum(1)
        self.heat_spin.setEnabled(False)
        self.heat_spin.valueChanged.connect(self._on_heat_step_changed)
        heat_row.addWidget(self.heat_spin)
        self.heat_summary = QLabel("No steps yet.")
        self.heat_summary.setWordWrap(True)
        heat_row.addWidget(self.heat_summary, 1)
        layout.addLayout(heat_row)
        self.heatmap = ChangeHeatmap()
        layout.addWidget(self.heatmap, 2)

        layout.addWidget(_header_label(
            "Layer candidates - resonance spikes and the input "
            "that triggered them"))
        self.layer_list = QListWidget()
        self.layer_list.setMaximumHeight(90)
        self.layer_list.currentItemChanged.connect(
            self._on_layer_selected)
        layout.addWidget(self.layer_list)
        self.layer_detail = QTextEdit()
        self.layer_detail.setReadOnly(True)
        self.layer_detail.setMaximumHeight(110)
        # v0.10 (R2): a read-only display surface: no Tab stop.
        self.layer_detail.setFocusPolicy(Qt.NoFocus)
        layout.addWidget(self.layer_detail)
        # v0.10 (R2): Tab order is the visual order. The chart
        # takes no focus; the detail pane is display only.
        QWidget.setTabOrder(self.heat_spin, self.layer_list)
        self.views_window = window

    # -- actions ------------------------------------------------------
    def _alpha(self):
        """Base learning rate, from the slider."""
        return self.alpha_slider.value() / 100.0

    def _preset(self):
        """Schedule preset key for the dropdown's current item."""
        return SCHEDULES[self.schedule_combo.currentIndex()]

    def _effective_alpha(self):
        """Learning rate the next step will actually use: the base
        alpha under the current schedule preset."""
        return schedule_alpha(self._preset(), self._alpha(),
                              self.engine.step_count + 1)

    def _freeze_mode(self):
        """Freeze mode key: "off", "whole", or "ranks"."""
        return ("off", "whole",
                "ranks")[self.freeze_combo.currentIndex()]

    def _frozen_ranks(self):
        """Checked rank directions, when the mode is Rank
        directions; otherwise none."""
        if self._freeze_mode() != "ranks":
            return []
        return [k for k, check in enumerate(self.rank_checks)
                if check.isChecked()]

    def _refresh_alpha_label(self):
        # The label shows the effective alpha for the next step,
        # not the slider value: under Fast-adapt and Gentle the
        # two differ, and the trace records the effective one.
        # v0.10 (R8): the Controls window's label shows the same
        # number, next to the slider.
        text = "Alpha: %.2f" % self._effective_alpha()
        self.alpha_label.setText(text)
        self.alpha_value_label.setText(text)

    def _on_alpha_changed(self, _value):
        self._refresh_alpha_label()

    def _on_schedule_changed(self, _index):
        self.logger.action("schedule preset: %s" % self._preset())
        self._refresh_alpha_label()

    def _on_freeze_mode_changed(self, _index):
        mode = self._freeze_mode()
        for check in self.rank_checks:
            check.setEnabled(mode == "ranks")
        self.logger.action("freeze mode: %s" % mode)

    # -- encoder ------------------------------------------------------
    def _model_names(self):
        """Model names in the encoder picker, by stored data (the
        visible lines are guided text, not bare names)."""
        return [self.model_combo.itemData(i)
                for i in range(self.model_combo.count())]

    def _fill_model_combo(self, infos):
        """Fill the encoder picker with guided lines (v0.9, U39):
        what each model does and whether it fits this PC. Starts
        unselected, with a "Pick a model" prompt (U6)."""
        self._model_infos = list(infos)
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        for info in infos:
            known = self._known_dims.get(info["name"])
            self.model_combo.addItem(
                model_display_text(info, self._total_ram, known),
                info["name"])
        self.model_combo.setCurrentIndex(-1)
        self.model_combo.blockSignals(False)

    def _set_encoder_ui(self):
        """Picker, model list, and label to match the encoder state.

        v0.10 (R10): progressive disclosure. In built-in mode
        the local-model chrome collapses: the model combo, the
        two note lines, and the completion row hide, leaving
        the encoder picker, the Start service button, and the
        one line "Built-in encoder, dim 64." Nothing is removed:
        it all returns the moment Local model is picked. The
        Start service button stays visible in both modes: it is
        how the service gets running, which is the step before
        Local model can work at all."""
        self.encoder_combo.blockSignals(True)
        self.encoder_combo.setCurrentIndex(
            1 if (self.encoder_kind == "local"
                  or self._local_pending) else 0)
        self.encoder_combo.blockSignals(False)
        local_visible = (self.encoder_kind == "local"
                         or self._local_pending)
        self.model_combo.setVisible(local_visible)
        self.model_note.setVisible(local_visible)
        self.switch_note.setVisible(local_visible)
        self.complete_button.setVisible(local_visible)
        self.completion_combo.setVisible(local_visible)
        self.completion_note.setVisible(local_visible)
        if self.encoder_kind == "local":
            self.model_combo.setEnabled(True)
            self.encoder_label.setText(
                "Local model: %s, dim %d."
                % (self.local_model, self.engine.dim))
        elif self._local_pending:
            self.model_combo.setEnabled(True)
            self.encoder_label.setText(
                "Pick a model above to use the local encoder.")
        else:
            # No model is running: clear the list so a stale name
            # never sits in the disabled combo.
            self._model_infos = []
            self.model_combo.blockSignals(True)
            self.model_combo.clear()
            self.model_combo.blockSignals(False)
            self.model_combo.setEnabled(False)
            self.encoder_label.setText("Built-in encoder, dim 64.")
        self._refresh_completion_ui()

    def _refresh_completion_ui(self):
        """Completion button, picker, and note to match the encoder
        state. The picker lists every installed model with the
        same guided lines as the encoder picker, and starts
        unselected (U6): the button enables once a model is
        picked. The user's pick is kept when it is still a
        choice. Runs from _set_encoder_ui, which covers every
        encoder change, session load, and failed switch."""
        if self.encoder_kind != "local":
            self.complete_button.setEnabled(False)
            self.completion_combo.blockSignals(True)
            self.completion_combo.clear()
            self.completion_combo.blockSignals(False)
            self.completion_combo.setEnabled(False)
            self.completion_note.setText(
                "Completion needs the Local model encoder.")
            return
        previous = self.completion_combo.currentData()
        self.completion_combo.blockSignals(True)
        self.completion_combo.clear()
        for info in self._model_infos:
            known = self._known_dims.get(info["name"])
            self.completion_combo.addItem(
                model_display_text(info, self._total_ram, known),
                info["name"])
        if previous in [info["name"]
                        for info in self._model_infos]:
            index = self.completion_combo.findData(previous)
            self.completion_combo.setCurrentIndex(index)
        else:
            self.completion_combo.setCurrentIndex(-1)
        self.completion_combo.blockSignals(False)
        if self.completion_combo.count() == 0:
            self.complete_button.setEnabled(False)
            self.completion_combo.setEnabled(False)
            self.completion_note.setText(
                "No model is installed for completion.")
        else:
            self.completion_combo.setEnabled(True)
            self.complete_button.setEnabled(
                self.completion_combo.currentIndex() >= 0)
            self.completion_note.setText(
                "Pick a model to continue the theme from the most "
                "resonant inputs.")

    def _on_completion_model_changed(self, index):
        # The button enables only once a model is picked: nothing
        # is selected by default (U6).
        if self.encoder_kind == "local":
            self.complete_button.setEnabled(index >= 0)

    def _fresh_engine(self, dim):
        """A fresh engine at the given dim, with views cleared."""
        self.engine = Engine(dim=dim)
        self._heat_history = []
        self._heat_follow = True
        self._full_texts = {}
        self._locators = {}
        self._batch_source_note = ""
        self._completion_model = None
        self._completion_text = None
        self._known_flag_steps = set()
        self.logger.set_setting("dim", dim)

    def _on_encoder_changed(self, index):
        if self._refuse_if_batch_running():
            # Put the picker back: the encoder cannot change
            # while the worker owns the engine.
            self._set_encoder_ui()
            return
        if self._local_thread is not None:
            # v0.10 (R1): the encoder cannot change while a
            # local call is in flight: the offer or activation
            # would land on the wrong state.
            self.status_label.setText(
                "Wait for the local model call to finish.")
            self._set_encoder_ui()
            return
        if index == 0:
            if self.encoder_kind == "local":
                self._switch_to_builtin(
                    "Back to the built-in encoder. W_t is fresh.")
            elif self._local_pending:
                # Only peeked at the local picker: nothing changed,
                # so W_t stays as it is.
                self._local_pending = False
                self._set_encoder_ui()
                self.status_label.setText("Built-in encoder.")
            return
        if self.encoder_kind == "local" or self._local_pending:
            return
        self._offer_local_models()

    def _switch_to_builtin(self, message):
        self.encoder_kind = "builtin"
        self.local_model = None
        self._local_pending = False
        self._fresh_engine(DIM)
        self.logger.set_setting("encoder", "built-in deterministic")
        self._set_encoder_ui()
        self._refresh_all()
        self.status_label.setText(message)

    def _offer_local_models(self):
        """Pick the Local model mode: health check, then fill the
        model picker with guided lines. Nothing is activated yet:
        the user picks a model, and that pick activates the local
        encoder. Any failure leaves the built-in encoder in place
        with a plain message. v0.10 (R1): the probe and the list
        run on a worker thread -- with a slow service they took
        4.21 s on the UI thread. The picker area opens at once
        (pending), so the mode switch feels immediate; the list
        fills when the worker lands."""
        self._local_pending = True
        self._set_encoder_ui()
        if not self._start_local_task("offer"):
            self._local_pending = False
            self._set_encoder_ui()

    def _on_offer_finished(self, result):
        """The model-list worker landed, on the UI thread."""
        if isinstance(result, LocalModelError):
            self._local_pending = False
            self._set_encoder_ui()
            self.status_label.setText(str(result))
            return
        if not result:
            self._local_pending = False
            self._set_encoder_ui()
            self.status_label.setText(MSG_NO_MODELS)
            return
        self._local_pending = True
        self._fill_model_combo(result)
        self._set_encoder_ui()
        self.status_label.setText(
            "Pick a model to use the local encoder.")

    def _activate_local_model(self, model):
        """Make a local model the encoder, with a fresh W_t.

        A probe embedding learns the model's dim first: the engine
        is built at the dim the model actually returns. On failure
        the built-in encoder stays in place with a plain message.
        v0.10 (R1): the probe embedding runs on a worker thread --
        with a slow model it took 3.12 s on the UI thread, and a
        60 s timeout times three retries is ~3 min."""
        self._start_local_task("activate", model)

    def _on_activate_finished(self, result):
        """The activation probe landed, on the UI thread."""
        model = self._local_model_name
        if isinstance(result, LocalModelError):
            self._set_encoder_ui()
            # A failed model switch leaves the old model active:
            # put its name back in the list so the list never shows
            # a model that is not running. A failed first pick
            # leaves the picker unselected, so it can be picked
            # again.
            if self.encoder_kind == "local" and self.local_model:
                index = self.model_combo.findData(self.local_model)
                self.model_combo.blockSignals(True)
                self.model_combo.setCurrentIndex(index)
                self.model_combo.blockSignals(False)
            else:
                self.model_combo.blockSignals(True)
                self.model_combo.setCurrentIndex(-1)
                self.model_combo.blockSignals(False)
            self.status_label.setText(str(result))
            return
        probe = result
        self.encoder_kind = "local"
        self.local_model = model
        self._local_pending = False
        self._known_dims[model] = len(probe)
        self._fresh_engine(len(probe))
        self.logger.set_setting("encoder", "local model %s" % model)
        # Rebuild the picker lines so the active model shows its
        # now-known dim, and keep it selected.
        self._fill_model_combo(self._model_infos)
        index = self.model_combo.findData(model)
        self.model_combo.blockSignals(True)
        self.model_combo.setCurrentIndex(index)
        self.model_combo.blockSignals(False)
        self._set_encoder_ui()
        self._refresh_all()
        self.status_label.setText(
            "Local encoder: %s (dim %d). W_t is fresh."
            % (model, len(probe)))

    def _on_model_changed(self, index):
        # Fires while the list is populated too; only a real pick
        # activates. While pending, any pick starts the local
        # encoder; while local, picking another model switches to
        # it (fresh W_t).
        if index < 0:
            return
        name = self.model_combo.itemData(index)
        if not name:
            return
        if self.encoder_kind == "local" and name == self.local_model:
            return
        if self._local_thread is not None:
            # v0.10 (R1): a local call is already running: put
            # the picker back on the running model (or
            # unselected, when only peeking) and wait.
            self.status_label.setText(
                "Wait for the local model call to finish.")
            self.model_combo.blockSignals(True)
            if self.encoder_kind == "local" and self.local_model:
                self.model_combo.setCurrentIndex(
                    self.model_combo.findData(self.local_model))
            else:
                self.model_combo.setCurrentIndex(-1)
            self.model_combo.blockSignals(False)
            return
        if self._local_pending or self.encoder_kind == "local":
            self._activate_local_model(name)

    def on_start_service(self):
        """Start the local model service, off the UI thread
        (v0.10, R1): with a service that never answers, the
        15 s wait used to block the window with no cancel. Now
        the wait runs on a worker thread, the busy light is on,
        and Cancel abandons the wait."""
        self._start_local_task("start")

    def _on_start_finished(self, result):
        """The service start landed, on the UI thread."""
        ok, message = result
        if ok:
            message += " Pick Local model to use it."
        self.status_label.setText(message)

    # -- local calls in a worker thread (v0.9, U30) -------------------
    def _launch_local_worker(self, worker, op, model, status_text):
        """Run one local worker off the UI thread: the shared
        launch for embedding/completion calls (v0.9) and setup
        tasks (v0.10, R1). The busy light goes on, the Step and
        completion buttons go off (one local call at a time),
        and Cancel appears. Returns False when a call is already
        running."""
        if self._local_thread is not None:
            self.status_label.setText(
                "A local model call is already running.")
            return False
        self._local_cancelled = False
        self._local_op = op
        # The model name, kept on the window: the worker object is
        # deleted when its thread winds down, possibly before the
        # queued finished handler runs, so the handler must never
        # read it back off the worker.
        self._local_model_name = model
        self._set_busy(True)
        self.step_button.setEnabled(False)
        self.complete_button.setEnabled(False)
        self.cancel_button.show()
        self.status_label.setText(status_text)
        # The thread is unparented on purpose: if the window
        # closes mid-call, the call finishes in the background
        # and its answer is ignored, instead of Qt destroying a
        # running thread with the window.
        thread = QThread()
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_local_progress)
        worker.finished.connect(self._on_local_finished)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        # The thread reference clears only once the thread has
        # actually stopped: dropping it in the finished handler
        # destroys a still-running QThread.
        thread.finished.connect(self._clear_local_thread)
        self._local_thread = thread
        self._local_worker = worker
        thread.start()
        return True

    def _start_local_call(self, kind, model, text):
        """Run one local-model call off the UI thread. Returns
        False when a call is already running."""
        worker = LocalCallWorker(self.client, kind, model, text)
        op = "step" if kind == "embed" else "complete"
        status = ("Asking the local model..." if kind == "generate"
                  else "Getting the embedding from the local "
                  "model...")
        return self._launch_local_worker(worker, op, model, status)

    def _start_local_task(self, kind, model=None):
        """Run one setup task off the UI thread (v0.10, R1):
        "offer" (list the models), "activate" (probe one
        model's dim), or "start" (start the service). Returns
        False when a call is already running."""
        worker = LocalTaskWorker(
            self.client, self.service, kind, model,
            stop=lambda: self._local_cancelled)
        if kind == "offer":
            status = "Checking the local model service..."
        elif kind == "activate":
            status = "Checking the model %s..." % model
        else:
            status = "Starting the local model service..."
        launched = self._launch_local_worker(
            worker, kind, model, status)
        if launched and kind == "start":
            self.start_button.setEnabled(False)
        return launched

    def _clear_local_thread(self):
        """Drop the local-call thread once it has stopped."""
        self._local_thread = None
        self._local_worker = None

    def _on_local_progress(self, message):
        self.status_label.setText(message)

    def on_cancel_local(self):
        """Cancel the running local call (U26): its answer is
        thrown away when it arrives. The buttons stay off until
        the call returns, so a second call cannot pile on top."""
        if self._local_thread is None:
            return
        self._local_cancelled = True
        self.cancel_button.hide()
        self.status_label.setText("Cancelling, one moment...")

    def _finish_local_call(self):
        """Shared cleanup when a local call lands: light off,
        buttons back, Cancel gone. The thread reference stays
        until the thread has actually stopped (see
        _clear_local_thread)."""
        self._set_busy(False)
        self.step_button.setEnabled(True)
        # v0.10 (R1): the start button is disabled only while a
        # start task runs; re-enabling it here is safe for every
        # op, since it is enabled whenever no start is running.
        self.start_button.setEnabled(True)
        self.cancel_button.hide()
        self._local_op = None
        # The completion button's state depends on the picker.
        self._refresh_completion_ui()

    def _on_local_finished(self, result):
        """A worker call landed, on the UI thread. Cancelled calls
        are thrown away. Errors on a step fall back to the
        built-in encoder; errors on a completion just say so.
        Setup tasks (v0.10, R1) land in their own handlers."""
        op = self._local_op
        cancelled = self._local_cancelled
        text = self._pending_step_text
        model = self._local_model_name
        self._pending_step_text = None
        self._finish_local_call()
        if cancelled:
            # v0.10 (R1/R4): a cancelled offer drops the pending
            # peek; a cancelled activation puts the picker back
            # on the running model (or unselected, when only
            # peeking); a cancelled example run stops where it
            # is. The answer is thrown away in all three cases.
            self._example_queue = []
            if op == "offer":
                self._local_pending = False
                self._set_encoder_ui()
            elif op == "activate":
                if self.encoder_kind == "local" and self.local_model:
                    index = self.model_combo.findData(
                        self.local_model)
                    self.model_combo.blockSignals(True)
                    self.model_combo.setCurrentIndex(index)
                    self.model_combo.blockSignals(False)
                else:
                    self.model_combo.blockSignals(True)
                    self.model_combo.setCurrentIndex(-1)
                    self.model_combo.blockSignals(False)
                self._set_encoder_ui()
            self.status_label.setText("Cancelled.")
            return
        if op == "offer":
            self._on_offer_finished(result)
            return
        if op == "activate":
            self._on_activate_finished(result)
            return
        if op == "start":
            self._on_start_finished(result)
            return
        if isinstance(result, LocalModelError):
            if op == "step":
                self.logger.action(
                    "local encoder failed (%s), fell back to "
                    "built-in" % result)
                message = (
                    "The local model service stopped. Back to the "
                    "built-in encoder with a fresh W_t.")
                self._switch_to_builtin(message)
                self._run_step(text, None, message)
                # v0.10 (R4): an example run in flight keeps
                # going after a fallback: the encoder is
                # built-in now, so the rest run at once.
                self._continue_example()
            else:
                self.logger.action("completion failed: %s" % result)
                self.status_label.setText(str(result))
            return
        if op == "step":
            if len(result) != self.engine.dim:
                self.logger.action(
                    "local model returned dim %d, engine is "
                    "dim %d" % (len(result), self.engine.dim))
                message = (
                    "The local model returned the wrong size. "
                    "Back to the built-in encoder with a fresh "
                    "W_t.")
                self._switch_to_builtin(message)
                self._run_step(text, None, message)
                self._continue_example()
            else:
                self._run_step(text, result, None)
                self._continue_example()
        else:
            self._completion_model = model
            self._completion_text = result.strip()
            self.logger.action("completion from %s" % model)
            self._refresh_synthesis()
            # The completion sits at the bottom of the panel:
            # scroll it into view, so the user sees the words they
            # asked for instead of the top of the panel.
            bar = self.synthesis_text.verticalScrollBar()
            bar.setValue(bar.maximum())
            self.status_label.setText(
                "Completion from %s." % model)

    def _run_step(self, text, h, fallback_message):
        """Run one engine step on the UI thread (fast, always):
        snapshot the factors for the heatmap, step, refresh."""
        # Snapshot the factors before the step so the heatmap can
        # show which rows of W_t moved (read-only copies).
        old_u = [row[:] for row in self.engine.u]
        old_v = [row[:] for row in self.engine.v]
        try:
            entry = self.engine.step(
                text, alpha=self._effective_alpha(),
                frozen=self._freeze_mode() == "whole",
                cap=self.cap_spin.value(), h=h,
                frozen_ranks=self._frozen_ranks())
        except Exception:
            self.logger.exception("step failed")
            self.status_label.setText(
                "Step failed - see the debug report for details.")
            return
        self._heat_history.append(
            (entry["id"], _row_change_norms(
                old_u, old_v, self.engine.u, self.engine.v)))
        self._trim_heat_history()
        self._full_texts[entry["id"]] = text
        self.logger.action(
            "step %d: resonance %.3f, W norm %.3f -> %.3f"
            % (entry["id"], entry["resonance"],
               entry["old_norm"], entry["new_norm"]))
        self._refresh_all()
        note = "Step %d done." % entry["id"]
        if entry["new_layer"]:
            note += " Resonance spike: new-layer candidate."
        if fallback_message:
            # The step ran on the fallback encoder: lead with what
            # happened, so the fallback is never silent.
            note = fallback_message + " " + note
        self.status_label.setText(note)

    def on_step(self):
        text = self.input_edit.toPlainText().strip()
        if not text:
            self.status_label.setText("Type something first.")
            return
        if self._refuse_if_batch_running():
            return
        self._clear_completion_mark()
        if self._local_thread is not None:
            self.status_label.setText(
                "Wait for the local model call to finish.")
            return
        if self.encoder_kind == "local":
            # The embedding comes from the worker thread (U30):
            # the window stays responsive while the model works.
            self._pending_step_text = text
            self._start_local_call("embed", self.local_model, text)
            return
        self._run_step(text, None, None)

    def on_try_example(self):
        """Teach by playing (v0.10, R4): step the four theme
        sentences as real Engine.step calls, through the same
        path as the Step button. Built-in runs at once; local
        steps one sentence per worker call, in order. The
        button only shows while the session is empty, so an
        example run always starts from step 0."""
        if self._refuse_if_batch_running():
            return
        if self._local_thread is not None:
            self.status_label.setText(
                "Wait for the local model call to finish.")
            return
        self._clear_completion_mark()
        if self.encoder_kind == "local":
            self._example_queue = list(EXAMPLE_SENTENCES)
            self._step_next_example()
            return
        for sentence in EXAMPLE_SENTENCES:
            self._run_step(sentence, None, None)
        self.status_label.setText(
            "Example done: 4 steps on one theme. W_t adapted. "
            "See the trace log and the resonance views.")

    def _step_next_example(self):
        """Step the next example sentence on the local encoder,
        or finish the run when the queue is empty."""
        if not self._example_queue:
            self.status_label.setText(
                "Example done: 4 steps on one theme. W_t "
                "adapted. See the trace log and the resonance "
                "views.")
            return
        text = self._example_queue.pop(0)
        self._pending_step_text = text
        self._start_local_call("embed", self.local_model, text)

    def _continue_example(self):
        """Keep an example run going after one step landed: the
        next sentence goes through the worker while the encoder
        is local, and the rest run at once after a fallback to
        built-in. A no-op when no example is in flight."""
        if not self._example_queue:
            return
        if self.encoder_kind == "local":
            self._step_next_example()
        else:
            remaining = self._example_queue
            self._example_queue = []
            for sentence in remaining:
                self._run_step(sentence, None, None)
            self.status_label.setText(
                "Example done: 4 steps on one theme. W_t "
                "adapted. See the trace log and the resonance "
                "views.")

    def _refresh_example(self):
        """The Try an example button and its label show only
        while the session is empty (step count 0) and no batch
        is running: offered, not defaulted (U6)."""
        show = (self.engine.step_count == 0
                and not self._batch_running())
        self.example_button.setVisible(show)
        self.example_label.setVisible(show)

    def on_complete(self):
        """Complete the session's theme with the local model.

        The resonant buffer entries go in as context; the returned
        text shows in the synthesis panel, labelled as the local
        model's words. On-demand only: nothing is generated on
        Step. A failed completion never falls back to the built-in
        encoder: the session is fine, only the completion failed.
        The call runs in a worker thread (U30), so the window
        stays responsive while the model writes.
        """
        if self._refuse_if_batch_running():
            return
        if self.encoder_kind != "local":
            self.status_label.setText(
                "Completion needs the Local model encoder.")
            return
        model = self.completion_combo.currentData()
        if not model:
            self.status_label.setText(
                "Pick a model for completion first.")
            return
        entries = self.engine.resonant_entries(5)
        if not entries:
            self.status_label.setText(
                "No resonant inputs yet to complete from. Step a "
                "few sentences on one theme first.")
            return
        if self._local_thread is not None:
            self.status_label.setText(
                "Wait for the local model call to finish.")
            return
        prompt = ("These are the most resonant inputs from this "
                  "session:\n")
        for entry in entries:
            prompt += "- %s\n" % entry[0]
        prompt += "Continue the theme in one sentence."
        self._start_local_call("generate", model, prompt)

    # -- batch analysis (v0.9, Analyze) -------------------------------
    def _batch_running(self):
        return self._batch_thread is not None

    def _refuse_if_batch_running(self):
        """True when a batch run is in flight: the mutating
        actions say so and do nothing, because the worker owns
        the engine until the run lands."""
        if self._batch_running():
            self.status_label.setText(
                "Wait for the analysis to finish.")
            return True
        return False

    def _entry_text(self, entry):
        """The full input text for a trace entry: the trace holds
        an 80-character preview, and the curated export needs the
        whole line."""
        return self._full_texts.get(entry["id"], entry["input"])

    def _pack_item(self, entry):
        """The manifest item for one trace entry: its source,
        its locator (step number and source line number), and
        its full text. Typed steps have no file or line: their
        source is "typed in this session" and the locator says
        so."""
        locator = self._locators.get(entry["id"])
        if locator is not None:
            source, line_no = locator
            text = ("Step %d, line %d"
                    % (entry["id"], line_no))
        else:
            source = "typed in this session"
            text = "Step %d (typed)" % entry["id"]
        return {"source": source, "locator": text,
                "text": self._entry_text(entry)}

    def _pack_context(self):
        """Everything the pack writer needs from the engine:
        "vectors" (step id to embedding, for the steps still in
        the buffer), "pool" (distinct full text to embedding, the
        neighbour candidates), and "resonant" (the (full text,
        vector) most resonant entries, best first, for the
        analogy's A and B). The buffer holds the last 50 steps in
        step order, so buffer item j is the j-th of the last 50
        real steps: the same mapping Engine.resonant_entries
        uses (revert entries are not steps and are filtered out
        first)."""
        steps = [e for e in self.engine.trace
                 if "reverted_to" not in e]
        count = len(self.engine.buffer)
        vectors = {}
        pool = {}
        for j, item in enumerate(self.engine.buffer):
            entry = steps[len(steps) - count + j]
            vectors[entry["id"]] = item["vector"]
            text = self._entry_text(entry)
            if text not in pool:
                pool[text] = item["vector"]
        resonant = []
        for text, vector, _res, _new, step_id in (
                self.engine.resonant_entries(2)):
            resonant.append((self._full_texts.get(step_id, text),
                             vector))
        return {"vectors": vectors, "pool": pool,
                "resonant": resonant}

    def start_batch(self, paths, source_note=""):
        """Run a batch analysis: every line of the picked files
        becomes one step, in file order, in a worker thread. The
        step parameters are snapshotted from the Controls window
        at the start, so changing a control mid-run never bends
        a run halfway through. Returns False when another
        operation is already running."""
        if (self._local_thread is not None
                or self._batch_thread is not None):
            self.status_label.setText(
                "Wait for the current operation to finish.")
            return False
        if not paths:
            self.status_label.setText("Pick a file or a folder "
                                      "first.")
            return False
        self._batch_cancelled = False
        self._batch_source_note = source_note
        self._clear_completion_mark()
        self._set_busy(True)
        self.step_button.setEnabled(False)
        self.complete_button.setEnabled(False)
        self.analyze_window.go_button.setEnabled(False)
        self.analyze_window.cancel_button.setEnabled(True)
        self.analyze_window.progress_bar.setValue(0)
        self.analyze_window.progress_label.setText("Starting...")
        self.status_label.setText("Analyzing...")
        thread = QThread()
        worker = BatchWorker(
            self.engine, self.client, self.encoder_kind,
            self.local_model, paths, self._preset(), self._alpha(),
            self._freeze_mode() == "whole", self.cap_spin.value(),
            self._frozen_ranks())
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_batch_progress)
        worker.finished.connect(self._on_batch_finished)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        # The thread reference clears only once the thread has
        # actually stopped: dropping it in the finished handler
        # destroys a still-running QThread.
        thread.finished.connect(self._clear_batch_thread)
        self._batch_thread = thread
        self._batch_worker = worker
        self.logger.action("batch analysis started: %s"
                           % (source_note or ", ".join(paths)))
        thread.start()
        return True

    def _on_batch_progress(self, done, total, resonance, w_norm,
                           skipped):
        # v0.10 (R13): the bar and label update at most every
        # ~100 ms, not per line: per-line updates dominated a
        # long run (128 ms/step of UI overhead at 280 lines).
        # The final update always lands, so the bar still
        # reaches 100%.
        now = time.monotonic()
        if (done < total
                and now - self._last_progress_ui
                < PROGRESS_UI_INTERVAL):
            return
        self._last_progress_ui = now
        bar = self.analyze_window.progress_bar
        bar.setMaximum(max(1, total))
        bar.setValue(done)
        steps = done - skipped
        self.analyze_window.progress_label.setText(
            "Line %d of %d (%d steps, %d skipped). Resonance "
            "%.3f, W norm %.3f."
            % (done, total, steps, skipped, resonance, w_norm))
        # v0.10 (R5): the chart and hero update as steps land,
        # not only at the end. A snapshot of the trace: the
        # worker owns the engine until the run lands.
        self.analyze_window.show_live(
            list(self.engine.trace), done, total)

    def _clear_batch_thread(self):
        """Drop the batch thread once it has stopped."""
        self._batch_thread = None
        self._batch_worker = None

    def _on_batch_finished(self, result):
        """A batch run landed, on the UI thread. The heat history
        and full texts join the session, every view refreshes
        (including the analysis), and the buttons come back. A
        cancelled or failed run keeps the steps it completed:
        they are real steps."""
        self._heat_history.extend(result["heat"])
        self._trim_heat_history()
        self._full_texts.update(result["texts"])
        self._locators.update(result["locators"])
        self._set_busy(False)
        self.step_button.setEnabled(True)
        self.analyze_window.go_button.setEnabled(True)
        self.analyze_window.cancel_button.setEnabled(False)
        self._refresh_all()
        self._refresh_completion_ui()
        steps = result["steps"]
        skipped = result["skipped"]
        unreadable = result["unreadable"]
        # v0.10 (R15): the progress label counts the run's
        # steps, while the hero counts the session's. Say
        # which, so the two numbers never look like a
        # contradiction.
        if result["cancelled"]:
            note = ("Analysis stopped at %d steps in this run."
                    % steps)
            self._clear_completion_mark()
        elif result["error"] is not None:
            note = ("The local model service stopped. The "
                    "analysis stopped at %d steps in this run."
                    % steps)
            self._clear_completion_mark()
        else:
            note = ("Analysis done: %d steps in this run." % steps)
            if skipped:
                note += " %d lines skipped." % skipped
            self._show_completion_mark()
        if unreadable:
            note += (" Could not read: %s."
                     % ", ".join(unreadable))
        self.analyze_window.progress_label.setText(note)
        self.status_label.setText(note)
        self.logger.action("batch analysis finished: %s" % note)

    def on_cancel_batch(self):
        """Stop the batch run (U26): the worker stops between
        lines, and the steps completed so far stay."""
        if self._batch_worker is None:
            return
        self._batch_cancelled = True
        try:
            self._batch_worker.cancelled = True
        except RuntimeError:
            # The worker is already gone: the run has landed.
            pass
        self.analyze_window.cancel_button.setEnabled(False)
        self.analyze_window.progress_label.setText(
            "Stopping the analysis...")
        self.status_label.setText("Stopping the analysis...")
        self._clear_completion_mark()

    def _refresh_analysis(self):
        """Recompute the analysis from the current trace and show
        it in the Analyze window. A loaded session restores the
        analysis too, because the trace is what a session saves."""
        summary = summarize_run(self.engine.trace)
        self.analyze_window.show_results(summary, self.engine.trace)

    def export_trace_to(self, path):
        """Write the trace log as JSON. Returns the path."""
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.engine.trace, fh, indent=2)
        self.logger.action("trace exported to %s" % path)
        return path

    def on_export(self):
        if self._refuse_if_batch_running():
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export trace JSON", "acr-trace.json",
            "JSON files (*.json)")
        if not path:
            return
        try:
            self.export_trace_to(path)
        except OSError:
            self.logger.exception("trace export failed")
            self.status_label.setText("Export failed - could not write.")
            return
        self.status_label.setText("Trace exported to %s" % path)

    # -- sessions -----------------------------------------------------
    def session_dict(self):
        """The whole session as a JSON-ready dict.

        The engine state (W_t factors, previous embedding and
        resonance, trace, buffer), the encoder choice (kind, model,
        dim -- a local session can only resume with the same
        encoder), the current settings, and the heatmap history so
        the views tab restores exactly. The format version lets
        later versions detect old saves.
        """
        return {
            "format": "acr-session",
            "version": SESSION_FORMAT_VERSION,
            "app_version": VERSION,
            "encoder": {
                "kind": self.encoder_kind,
                "model": self.local_model,
                "dim": self.engine.dim,
            },
            "engine": self.engine.state_dict(),
            "settings": {
                "alpha": self._alpha(),
                "cap": self.cap_spin.value(),
                # "frozen" stays for older readers: True exactly
                # when the whole W_t is frozen.
                "frozen": self._freeze_mode() == "whole",
                "freeze_mode": self._freeze_mode(),
                "frozen_ranks": self._frozen_ranks(),
                "preset": self._preset(),
            },
            "heat_history": [[step_id, list(rows)]
                             for step_id, rows in self._heat_history],
            # Full input texts, by step id: the trace holds an
            # 80-character preview, and the curated export needs
            # the whole line. JSON keys are strings; loading
            # turns them back into step ids. Optional: files
            # without it load fine, with previews only.
            "full_texts": {str(step_id): text
                           for step_id, text
                           in self._full_texts.items()},
            # Step locators, by step id: [file name, line number]
            # for batch steps, for the STELLA pack manifest.
            # Optional: files without it load fine, with typed
            # locators for every step.
            "locators": {str(step_id): [name, line_no]
                         for step_id, (name, line_no)
                         in self._locators.items()},
            # The source of the last batch run ("fixture.txt",
            # or "folder data (3 files)"), for the Analyze
            # exports' labels. Optional: files without it load
            # with no source note.
            "source_note": self._batch_source_note,
        }

    def save_session_to(self, path):
        """Write the session as JSON. Returns the path."""
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.session_dict(), fh, indent=2)
        self.logger.action("session saved to %s" % path)
        return path

    def on_save_session(self):
        if self._refuse_if_batch_running():
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save session", "acr-session.json",
            "JSON files (*.json)")
        if not path:
            return
        try:
            self.save_session_to(path)
        except OSError:
            self.logger.exception("session save failed")
            self.status_label.setText("Save failed - could not write.")
            return
        self.status_label.setText("Session saved to %s" % path)

    def load_session_from(self, path):
        """Load a session file, restoring encoder, engine, settings,
        views.

        Validates everything before applying it, so a bad file
        leaves the current session intact. Raises SessionError
        with a plain message on anything malformed; OSError on a
        file that cannot be read at all. Returns the path.

        Version 1 files (no encoder block) load as built-in, dim
        64. Version 2 files carry the encoder: a local session
        only loads when the local service is running and the model
        is installed -- otherwise it refuses with a plain message,
        never a crash and never a silent wrong-dim load.
        """
        with open(path, "r", encoding="utf-8") as fh:
            raw = fh.read()
        try:
            data = json.loads(raw)
        except ValueError:
            raise SessionError("that file is not a session file")
        if not isinstance(data, dict):
            raise SessionError("that file is not a session file")
        if data.get("format") != "acr-session":
            raise SessionError("that file is not an ACR session file")
        if "version" not in data:
            raise SessionError(
                "that session file has no version number")
        version = data["version"]
        if version not in (1, SESSION_FORMAT_VERSION):
            raise SessionError(
                "that session file is version %s, this app reads "
                "versions 1 and %d" % (version,
                                       SESSION_FORMAT_VERSION))
        if "engine" not in data:
            raise SessionError("that session file has no engine state")
        # Encoder: v1 has none (built-in, dim 64); v2 names it.
        encoder_kind = "builtin"
        encoder_model = None
        encoder_dim = DIM
        infos = None
        if version == SESSION_FORMAT_VERSION:
            encoder = data.get("encoder")
            if not isinstance(encoder, dict):
                raise SessionError(
                    "that session file has no encoder")
            encoder_kind = encoder.get("kind")
            if encoder_kind not in ("builtin", "local"):
                raise SessionError(
                    "that session file has a bad encoder")
            encoder_dim = encoder.get("dim")
            if (not isinstance(encoder_dim, int)
                    or isinstance(encoder_dim, bool)
                    or not 8 <= encoder_dim <= 8192):
                raise SessionError(
                    "that session file has a bad encoder dim")
            if encoder_kind == "builtin":
                if encoder_dim != DIM:
                    raise SessionError(
                        "that session file has a bad encoder dim")
            else:
                encoder_model = encoder.get("model")
                if (not isinstance(encoder_model, str)
                        or not encoder_model):
                    raise SessionError(
                        "that session file has a bad encoder model")
                # The session needs its encoder: refuse plainly
                # when the service is down or the model is gone.
                if not self.client.probe():
                    raise SessionError(
                        "this session needs the local encoder, "
                        "which is not running - start it and try "
                        "again")
                try:
                    infos = self.client.list_model_info()
                except LocalModelError:
                    raise SessionError(
                        "this session needs the local encoder, "
                        "which is not running - start it and try "
                        "again")
                if encoder_model not in [info["name"]
                                         for info in infos]:
                    raise SessionError(
                        "this session needs model %s, which is not "
                        "installed" % encoder_model)
        # Validate the engine state on a spare engine first: a bad
        # file must not disturb the live session.
        spare = Engine(dim=encoder_dim)
        spare.load_state_dict(data["engine"])
        settings = data.get("settings")
        if not isinstance(settings, dict):
            raise SessionError("that session file has no settings")
        alpha = settings.get("alpha")
        cap = settings.get("cap")
        if (not isinstance(alpha, (int, float))
                or isinstance(alpha, bool)
                or not 0.01 <= alpha <= 0.50):
            raise SessionError(
                "that session file has a bad learning rate")
        if (not isinstance(cap, (int, float))
                or isinstance(cap, bool)
                or not 0.1 <= cap <= 5.0):
            raise SessionError(
                "that session file has a bad stability cap")
        # Freeze: v0.7 files name the mode and the frozen
        # directions; older files only carry the "frozen" bool,
        # which maps to the whole-W_t mode.
        freeze_mode = settings.get("freeze_mode")
        if freeze_mode is None:
            frozen = settings.get("frozen")
            if not isinstance(frozen, bool):
                raise SessionError(
                    "that session file has a bad freeze setting")
            freeze_mode = "whole" if frozen else "off"
        elif freeze_mode not in ("off", "whole", "ranks"):
            raise SessionError(
                "that session file has a bad freeze setting")
        frozen_ranks = settings.get("frozen_ranks", [])
        if (not isinstance(frozen_ranks, list)
                or any(not isinstance(k, int)
                       or isinstance(k, bool)
                       or not 0 <= k < RANK
                       for k in frozen_ranks)):
            raise SessionError(
                "that session file has a bad freeze setting")
        preset = settings.get("preset", SCHEDULE_STEADY)
        if preset not in SCHEDULES:
            raise SessionError(
                "that session file has a bad schedule preset")
        heat = self._checked_heat_history(data.get("heat_history"),
                                          encoder_dim)
        full_texts = self._checked_full_texts(
            data.get("full_texts"))
        locators = self._checked_locators(data.get("locators"))
        source_note = data.get("source_note", "")
        if not isinstance(source_note, str):
            source_note = ""
        # Everything checked: apply it.
        self.encoder_kind = encoder_kind
        self.local_model = encoder_model
        self.engine = spare
        self._full_texts = full_texts
        self._locators = locators
        self._batch_source_note = source_note
        if encoder_kind == "local":
            self.logger.set_setting(
                "encoder", "local model %s" % encoder_model)
            # The saved dim is a known dim: the picker shows it.
            self._known_dims[encoder_model] = encoder_dim
            self._fill_model_combo(infos)
            index = self.model_combo.findData(encoder_model)
            self.model_combo.blockSignals(True)
            self.model_combo.setCurrentIndex(index)
            self.model_combo.blockSignals(False)
        else:
            self.logger.set_setting("encoder",
                                    "built-in deterministic")
        self.logger.set_setting("dim", encoder_dim)
        self.alpha_slider.setValue(int(round(alpha * 100)))
        self.cap_spin.setValue(cap)
        self.freeze_combo.setCurrentIndex(
            ("off", "whole", "ranks").index(freeze_mode))
        for k, check in enumerate(self.rank_checks):
            check.setChecked(k in frozen_ranks)
        self.schedule_combo.setCurrentIndex(
            SCHEDULES.index(preset))
        self._heat_history = heat
        self._trim_heat_history()
        self._heat_follow = True
        # A loaded session is a different session: the old
        # completion does not carry over.
        self._completion_model = None
        self._completion_text = None
        # A loaded session's flags are old news: mark them known
        # so the resonance moment does not fire for them.
        self._known_flag_steps = {
            entry["id"] for entry in self.engine.trace
            if entry["new_layer"]}
        # v0.10 (R15): the restored source note shows in the
        # Analyze window. The picker is cleared: the loaded
        # session's source is a record, not a picked file, so
        # Go says "Pick a file or a folder first" until the user
        # picks one.
        self.analyze_window._paths = []
        self.analyze_window._source_note = source_note
        if source_note:
            self.analyze_window.source_label.setText(
                "Source: %s" % source_note)
        else:
            self.analyze_window.source_label.setText(
                "No file picked yet.")
        self._set_encoder_ui()
        self.logger.action("session loaded from %s" % path)
        self._refresh_all()
        return path

    def _checked_full_texts(self, value):
        """Validate the full input texts, or return {} if absent.

        The map is step id (as a JSON string key) to full text.
        Entries with a bad key or a non-string value are dropped,
        not fatal: a file with a damaged full-text map still
        loads, with previews for those steps."""
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise SessionError(
                "that session file has bad full texts")
        out = {}
        for key, text in value.items():
            try:
                step_id = int(key)
            except (TypeError, ValueError):
                continue
            if isinstance(text, str):
                out[step_id] = text
        return out

    def _checked_locators(self, value):
        """Validate the step locators, or return {} if absent.

        The map is step id (as a JSON string key) to [file name,
        line number]. Entries with a bad key or a malformed value
        are dropped, not fatal: a file with a damaged locator
        map still loads, with typed locators for those steps."""
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise SessionError(
                "that session file has bad locators")
        out = {}
        for key, locator in value.items():
            try:
                step_id = int(key)
            except (TypeError, ValueError):
                continue
            if (isinstance(locator, list) and len(locator) == 2
                    and isinstance(locator[0], str)
                    and isinstance(locator[1], int)
                    and not isinstance(locator[1], bool)):
                out[step_id] = (locator[0], locator[1])
        return out

    def _checked_heat_history(self, value, dim):
        """Validate the heatmap history, or return [] if absent."""
        if value is None:
            return []
        if not isinstance(value, list):
            raise SessionError(
                "that session file has a bad heatmap history")
        out = []
        for item in value:
            if (not isinstance(item, list) or len(item) != 2
                    or not isinstance(item[0], int)
                    or isinstance(item[0], bool)
                    or not isinstance(item[1], list)
                    or len(item[1]) != dim
                    or not all(isinstance(x, (int, float))
                               and not isinstance(x, bool)
                               for x in item[1])):
                raise SessionError(
                    "that session file has a bad heatmap history")
            out.append([item[0], [float(x) for x in item[1]]])
        return out

    def on_load_session(self):
        if self._refuse_if_batch_running():
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Load session", "", "JSON files (*.json)")
        if not path:
            return
        try:
            self.load_session_from(path)
        except OSError:
            self.logger.exception("session load failed")
            self.status_label.setText(
                "Load failed - could not read the file.")
            return
        except SessionError as exc:
            self.logger.action("session load rejected: %s" % exc)
            self.status_label.setText("Load failed - %s." % exc)
            return
        self.status_label.setText("Session loaded from %s" % path)

    # -- exports ------------------------------------------------------
    def export_trace_csv_to(self, path):
        """Write the trace log as CSV, same columns as the table.

        Returns the path.
        """
        with open(path, "w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(TRACE_COLUMNS)
            for entry in self.engine.trace:
                writer.writerow([
                    entry["id"],
                    entry["input"],
                    "%.4f" % entry["delta_norm"],
                    "%.4f" % entry["old_norm"],
                    "%.4f" % entry["new_norm"],
                    "%.4f" % entry["resonance"],
                    "%.2f" % entry["alpha"],
                    "%.4f" % entry["confidence"],
                    "yes" if entry["new_layer"] else "",
                    "yes" if entry.get("capped") else "",
                ])
        self.logger.action("trace CSV exported to %s" % path)
        return path

    def on_export_csv(self):
        if self._refuse_if_batch_running():
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export trace CSV", "acr-trace.csv",
            "CSV files (*.csv)")
        if not path:
            return
        try:
            self.export_trace_csv_to(path)
        except OSError:
            self.logger.exception("trace CSV export failed")
            self.status_label.setText("Export failed - could not write.")
            return
        self.status_label.setText("Trace exported to %s" % path)

    def export_session_to(self, path):
        """Write the session as readable text: the final state, then
        every input and trace entry in time order. Returns the path.
        """
        if self.encoder_kind == "local":
            encoder_line = ("  Encoder: local, %s (dim %d)"
                            % (self.local_model, self.engine.dim))
        else:
            encoder_line = "  Encoder: built-in (dim 64)"
        lines = ["ACR session export",
                 "App version: %s" % VERSION,
                 "",
                 "Final state",
                 "  Steps: %d" % self.engine.step_count,
                 encoder_line,
                 "  W norm: %.4f" % self.engine.w_norm()]
        if self.engine.trace:
            last = self.engine.trace[-1]
            lines.append("  Resonance: %.4f" % last["resonance"])
        else:
            lines.append("  Resonance: -")
        candidates = [e for e in self.engine.trace if e["new_layer"]]
        lines.append("  Layer candidates: %d" % len(candidates))
        lines.append("")
        if not self.engine.trace:
            lines.append("No steps yet.")
        else:
            lines.append("Steps, in time order")
            for entry in self.engine.trace:
                lines.append("Step %d: %s"
                             % (entry["id"], entry["input"]))
                lines.append(
                    "  Delta norm: %.4f  W norm: %.4f -> %.4f  "
                    "Resonance: %.4f"
                    % (entry["delta_norm"], entry["old_norm"],
                       entry["new_norm"], entry["resonance"]))
                lines.append(
                    "  Alpha: %.2f  Confidence: %.4f  New layer: %s  "
                    "Frozen: %s  Capped: %s"
                    % (entry["alpha"], entry["confidence"],
                       "yes" if entry["new_layer"] else "no",
                       "yes" if entry["frozen"] else "no",
                       "yes" if entry.get("capped") else "no"))
                if entry.get("frozen_ranks"):
                    lines.append(
                        "  Frozen directions: %s"
                        % ", ".join(str(k + 1)
                                    for k in entry["frozen_ranks"]))
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        self.logger.action("session exported to %s" % path)
        return path

    def on_export_session(self):
        if self._refuse_if_batch_running():
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export session", "acr-session.txt",
            "Text files (*.txt)")
        if not path:
            return
        try:
            self.export_session_to(path)
        except OSError:
            self.logger.exception("session export failed")
            self.status_label.setText("Export failed - could not write.")
            return
        self.status_label.setText("Session exported to %s" % path)

    def generate_debug_report(self):
        """Write a debug report. Returns the path."""
        path = self.logger.generate_report()
        self.status_label.setText("Debug report: %s" % path)
        return path

    def on_debug_report(self):
        try:
            path = self.generate_debug_report()
        except OSError:
            self.logger.exception("debug report failed")
            self.status_label.setText(
                "Report failed - could not write.")
            return
        QMessageBox.information(
            self, "Debug report",
            "Report written to:\n%s\n\nSend it to %s if something "
            "broke." % (path, REPORT_EMAIL))

    def do_reset(self):
        self.engine.reset()
        self._heat_history = []
        self._heat_follow = True
        self._full_texts = {}
        self._locators = {}
        self._batch_source_note = ""
        self._completion_model = None
        self._completion_text = None
        self._known_flag_steps = set()
        self._clear_completion_mark()
        self.logger.action("session reset")
        self._refresh_all()
        self.status_label.setText("Reset. W_t is back to its start.")

    def on_reset(self):
        if self._refuse_if_batch_running():
            return
        answer = QMessageBox.question(
            self, "Reset",
            "Clear the session and reset W_t?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if answer == QMessageBox.Yes:
            self.do_reset()

    def on_revert(self):
        """Revert W_t to the selected trace step, after asking.

        Revert discards the current W_t, so it asks first, like
        Reset. Nothing is selected, or the step has aged out of
        the revert history, and the status line says so plainly.
        """
        if self._refuse_if_batch_running():
            return
        row = self.trace_table.currentRow()
        if row < 0 or row >= len(self.engine.trace):
            self.status_label.setText(
                "Pick a step in the trace log first.")
            return
        step_id = self.engine.trace[row]["id"]
        # v0.10 (R11): show the effect before committing: the W
        # norm the revert will land on is the trace entry's
        # new_norm -- the snapshot holds the same state.
        target_norm = self.engine.trace[row]["new_norm"]
        answer = QMessageBox.question(
            self, "Revert W_t",
            "Revert W_t to the state after step %d? W norm will "
            "go from %.4f to %.4f. The current W_t will be "
            "replaced."
            % (step_id, self.engine.w_norm(), target_norm),
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if answer == QMessageBox.Yes:
            self.do_revert(step_id)

    def do_revert(self, step_id):
        """Roll W_t back to the state after one step, no asking.

        Used by on_revert after the confirm dialog, and by tests.
        The trace keeps a revert entry, so the log shows what
        happened; later entries are never deleted. A step with no
        snapshot (aged out of the history) shows a plain message
        instead.
        """
        old_u = [row[:] for row in self.engine.u]
        old_v = [row[:] for row in self.engine.v]
        try:
            entry = self.engine.revert_to(step_id)
        except ValueError as exc:
            self.status_label.setText(str(exc))
            return
        self._heat_history.append(
            (entry["id"], _row_change_norms(
                old_u, old_v, self.engine.u, self.engine.v)))
        self._trim_heat_history()
        self._full_texts[entry["id"]] = entry["input"]
        self.logger.action(
            "reverted W_t to step %d: W norm %.3f -> %.3f"
            % (step_id, entry["old_norm"], entry["new_norm"]))
        self._refresh_all()
        self.status_label.setText(
            "Reverted W_t to step %d." % step_id)

    # -- refresh ------------------------------------------------------
    def _refresh_all(self):
        self.step_label.setText("Step: %d" % self.engine.step_count)
        # v0.10 (R15): at step 0 the W norm and the top singular
        # value are the random starting W_t, not a measurement
        # of anything the user did. They read "-" until the
        # first step lands, like Resonance already did: showing
        # a number would imply a reading that has not happened.
        if self.engine.trace:
            self.norm_label.setText(
                "W norm: %.4f" % self.engine.w_norm())
            last = self.engine.trace[-1]
            self.resonance_label.setText(
                "Resonance: %.4f" % last["resonance"])
            self.spectral_label.setText(
                "Top singular value: %.4f"
                % self.engine.spectral_norm())
        else:
            self.norm_label.setText("W norm: -")
            self.resonance_label.setText("Resonance: -")
            self.spectral_label.setText("Top singular value: -")
        self._refresh_alpha_label()
        self._refresh_trace_table()
        self._refresh_buffer_list()
        self._refresh_synthesis()
        self._refresh_chart()
        self._refresh_heatmap()
        self._refresh_layers()
        self._refresh_analysis()
        self._celebrate_new_flags()
        self._refresh_flag_marker()
        self._refresh_example()

    def _refresh_flag_marker(self):
        """The persistent flag marker (v0.10, R7): a gold dot
        next to the resonance readout, naming the latest
        flagged step on hover. It stays until the next flag or
        a reset. Computed from the trace, so a loaded session
        shows its latest flag too."""
        flagged = [entry["id"] for entry in self.engine.trace
                   if entry["new_layer"]]
        if not flagged:
            self.flag_marker.hide()
            return
        latest = max(flagged)
        self.flag_marker.setToolTip(
            "New layer candidate flagged at step %d. See "
            "Resonance views." % latest)
        self.flag_marker.show()

    def _celebrate_new_flags(self):
        """The resonance moment (v0.9, technique 8): when a step
        flags a new-layer candidate, the chart dot springs in,
        the resonance readout glows gold, and the trace row
        highlights. Called after every refresh; each flag is
        celebrated once. Under reduced motion the state still
        lands (dot, tint, readout) with no motion."""
        flagged = {entry["id"] for entry in self.engine.trace
                   if entry["new_layer"]}
        new = flagged - self._known_flag_steps
        self._known_flag_steps |= flagged
        if not new:
            return
        latest = max(new)
        index = None
        for i, entry in enumerate(self.engine.trace):
            if entry["id"] == latest:
                index = i
                break
        self.norm_chart.pulse_flag(index)
        self.analyze_window.chart.pulse_flag(index)
        self._glow_resonance()
        self._highlight_flag_row(latest)

    def _glow_resonance(self):
        """The resonance readout glows gold briefly when a flag
        lands. Autonomous motion, so plain easing (OutCubic),
        300 ms, then the readout rests at its normal color."""
        if self._glow_anim is not None:
            self._glow_anim.stop()
            self._glow_anim = None
        if reduced_motion():
            return
        anim = QVariantAnimation(self)
        anim.setDuration(300)
        anim.setEasingCurve(QEasingCurve.OutCubic)
        anim.setStartValue(QColor(GLOW_GOLD))
        anim.setEndValue(QColor(THEME_TEXT))
        anim.valueChanged.connect(self._on_glow_frame)
        anim.finished.connect(self._on_glow_done)
        self._glow_anim = anim
        anim.start()

    def _on_glow_frame(self, color):
        self.resonance_label.setStyleSheet(
            "color: %s;" % color.name())

    def _on_glow_done(self):
        self.resonance_label.setStyleSheet("")
        self._glow_anim = None

    def _trace_row_for_step(self, step_id):
        """The trace table row for one step id, or None. Rows
        are in trace order, so the row is the trace index."""
        for i, entry in enumerate(self.engine.trace):
            if entry["id"] == step_id:
                return i
        return None

    def _highlight_flag_row(self, step_id):
        """The flagged trace row fades from a bright tint to the
        calm resting tint over 250 ms. The resting tint is the
        state (applied by _refresh_trace_table); the fade is
        only the motion."""
        if self._flag_anim is not None:
            self._flag_anim.stop()
            self._flag_anim = None
        if reduced_motion():
            return
        self._flag_anim_step = step_id
        anim = QVariantAnimation(self)
        anim.setDuration(250)
        anim.setEasingCurve(QEasingCurve.OutCubic)
        anim.setStartValue(FLAG_TINT_BRIGHT)
        anim.setEndValue(FLAG_TINT)
        anim.valueChanged.connect(self._on_flag_frame)
        anim.finished.connect(self._on_flag_done)
        self._flag_anim = anim
        anim.start()

    def _on_flag_frame(self, color):
        # Re-look the row up every frame: a refresh rebuilds
        # the table's items, so a stored item would go stale.
        row = self._trace_row_for_step(self._flag_anim_step)
        if row is None or row >= self.trace_table.rowCount():
            return
        for col in range(self.trace_table.columnCount()):
            item = self.trace_table.item(row, col)
            if item is not None:
                item.setBackground(color)

    def _on_flag_done(self):
        self._flag_anim = None
        self._flag_anim_step = None

    def _show_completion_mark(self):
        """The completion moment (v0.9, technique 8): a
        checkmark draws itself next to the status text when a
        run finishes or a pack is written."""
        self.completion_mark.draw()

    def _clear_completion_mark(self):
        """Hide the completion mark: the next run or step starts
        fresh."""
        self.completion_mark.clear()

    def _refresh_trace_table(self):
        trace = self.engine.trace
        # v0.10 (R12): follow only when the user is already at
        # the bottom. Read the scroll position BEFORE the
        # rebuild: after it, the scrollbar's range has changed
        # and the old position is gone.
        bar = self.trace_table.verticalScrollBar()
        at_bottom = bar.value() >= bar.maximum()
        self.trace_table.setRowCount(len(trace))
        for row, entry in enumerate(trace):
            values = [
                str(entry["id"]),
                entry["input"],
                "%.4f" % entry["delta_norm"],
                "%.4f" % entry["old_norm"],
                "%.4f" % entry["new_norm"],
                "%.4f" % entry["resonance"],
                "%.2f" % entry["alpha"],
                "%.4f" % entry["confidence"],
                "yes" if entry["new_layer"] else "",
                "yes" if entry.get("capped") else "",
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                if entry["new_layer"]:
                    # The flagged row rests at the calm tint
                    # (the chart's red-dot meaning, as a row).
                    item.setBackground(FLAG_TINT)
                if col == 1:
                    # v0.10 (R3): the full input, on hover: the
                    # column holds an 80-character preview.
                    item.setToolTip(self._entry_text(entry))
                self.trace_table.setItem(row, col, item)
        if trace and at_bottom:
            self.trace_table.scrollToBottom()
        # v0.10 (R15): Revert is disabled with no steps, and the
        # empty window shows guidance instead of blank boxes.
        self.revert_button.setEnabled(bool(trace))
        self.trace_empty_label.setVisible(not trace)
        self._refresh_trace_detail()

    def _on_trace_row_picked(self, current, _previous):
        """A trace row was clicked: show its full input below
        the table (v0.10, R3)."""
        self._refresh_trace_detail()

    def _refresh_trace_detail(self):
        """The full input of the selected trace row, below the
        table. Blank with no selection. Called from the
        selection handler and after every table rebuild: the
        rebuild replaces the items, so the signal alone is not
        enough."""
        row = self.trace_table.currentRow()
        if row < 0 or row >= len(self.engine.trace):
            self.trace_detail.setText("")
            return
        entry = self.engine.trace[row]
        self.trace_detail.setText(
            "Step %d: %s" % (entry["id"],
                             self._entry_text(entry)))

    def _refresh_buffer_list(self):
        self.buffer_list.clear()
        for item in reversed(self.engine.buffer):
            self.buffer_list.addItem(item["text"])
            # v0.10 (R3): the full text, on hover: the list
            # clips long lines.
            row = self.buffer_list.item(
                self.buffer_list.count() - 1)
            row.setToolTip(item["text"])

    def _refresh_synthesis(self):
        # Order: the analogy and counter-examples first, so the
        # new sections are visible without scrolling; the
        # neighbours and midpoint below them; the completion last
        # (on_complete scrolls it into view when it arrives).
        # setPlainText resets the scroll to the top, so after a
        # step the panel always opens on the analogy.
        result = self.engine.synthesize()
        lines = []
        if self.engine.buffer:
            if result["analogy"] is not None:
                lines.append(
                    "Analogy - like \"%s\" is to \"%s\", your "
                    "input points at:"
                    % (result["analogy"]["a"],
                       result["analogy"]["b"]))
                text, score = result["analogy"]["result"]
                lines.append("  %.3f  %s" % (score, text))
            else:
                lines.append(
                    "Analogy: not enough resonant inputs yet "
                    "(need two, plus another entry to point at).")
            if result["counter_examples"]:
                lines.append(
                    "Counter-examples - what this session is not:")
                for text, score in result["counter_examples"]:
                    lines.append("  %.3f  %s" % (score, text))
        if result["neighbours"]:
            lines.append("Closest past inputs:")
            for text, score in result["neighbours"]:
                lines.append("  %.3f  %s" % (score, text))
        if result["midpoint"]:
            lines.append("Midpoint neighbours:")
            for text, score in result["midpoint"]:
                lines.append("  %.3f  %s" % (score, text))
        if self._completion_text is not None:
            lines.append("Local model (%s) said:"
                         % self._completion_model)
            lines.append("  %s" % self._completion_text)
        if not lines:
            lines.append("Step a few inputs and related past inputs "
                         "will show up here.")
        self.synthesis_text.setPlainText("\n".join(lines))

    # -- resonance views ----------------------------------------------
    def _refresh_chart(self):
        trace = self.engine.trace
        self.norm_chart.set_data(
            [entry["new_norm"] for entry in trace],
            [entry["new_layer"] for entry in trace])

    def _trim_heat_history(self):
        """Bound the heat history (v0.10, R13): keep at most
        HEAT_HISTORY_LIMIT entries, thinning the oldest half
        (every other entry) when over. The newest steps stay
        dense; older steps are sampled, never silently dropped:
        the step picker still reaches every step, and a step
        whose data was thinned out says so in the heatmap (see
        _refresh_heatmap). UI-side state only: the engine's
        trace keeps every step."""
        while len(self._heat_history) > HEAT_HISTORY_LIMIT:
            half = len(self._heat_history) // 2
            self._heat_history = (self._heat_history[:half:2]
                                  + self._heat_history[half:])

    def _heat_rows_for(self, step_id):
        """Per-row change norms recorded for one step id."""
        for entry_id, rows in self._heat_history:
            if entry_id == step_id:
                return rows
        return []

    def _refresh_heatmap(self):
        steps = self.engine.step_count
        if steps == 0:
            self.heat_spin.blockSignals(True)
            self.heat_spin.setMaximum(1)
            self.heat_spin.setValue(1)
            self.heat_spin.blockSignals(False)
            self.heat_spin.setEnabled(False)
            self.heatmap.set_data([], 0)
            self.heat_summary.setText("No steps yet.")
            return
        self.heat_spin.setEnabled(True)
        self.heat_spin.blockSignals(True)
        self.heat_spin.setMaximum(steps)
        if self._heat_follow:
            self.heat_spin.setValue(steps)
        self.heat_spin.blockSignals(False)
        step_id = self.heat_spin.value()
        rows = self._heat_rows_for(step_id)
        if not rows:
            # v0.10 (R13): no change data kept for this step --
            # a very long run samples the heat history, or the
            # loaded session did not carry it. Say so, do not
            # claim W_t stood still.
            note = ("Step %d: change data was not kept for this "
                    "step." % step_id)
            self.heatmap.set_data(rows, step_id, note)
            self.heat_summary.setText(note)
            return
        if max(rows) <= 0.0:
            # v0.3: step 1 never moves W_t (no previous state to
            # predict from), so it is not a frozen step. v0.7: a
            # revert that changed nothing is not a frozen step
            # either. One note drives both the summary line and
            # the message painted in the heatmap.
            entry = None
            if 1 <= step_id <= len(self.engine.trace):
                entry = self.engine.trace[step_id - 1]
            if step_id == 1:
                note = ("Step 1: W_t did not move (no previous "
                        "step to learn from).")
            elif entry is not None and "reverted_to" in entry:
                note = ("Step %d: reverted to step %d, W_t did "
                        "not change."
                        % (step_id, entry["reverted_to"]))
            else:
                note = ("Step %d: W_t did not move (frozen step)."
                        % step_id)
            self.heatmap.set_data(rows, step_id, note)
            self.heat_summary.setText(note)
            return
        self.heatmap.set_data(rows, step_id)
        top = self.heatmap.top_rows(3)
        parts = ["row %d (%.3f)" % (i, rows[i]) for i in top]
        self.heat_summary.setText(
            "Step %d: most moved rows %s."
            % (step_id, ", ".join(parts)))

    def _on_heat_step_changed(self, value):
        # Picking the latest step re-enables follow mode; picking
        # an older step pins the heatmap there until the user
        # picks the latest again or a reset happens.
        self._heat_follow = (value >= self.heat_spin.maximum())
        self._refresh_heatmap()

    def _on_chart_step_clicked(self, index):
        """A chart point was clicked or dragged to (v0.10, R6):
        jump the heatmap and the trace selection to that step.
        The index is into the trace; the step id comes from the
        trace itself. The trace scrolls to the row: this is a
        user-initiated jump, not a refresh yanking the place
        (R12)."""
        if index < 0 or index >= len(self.engine.trace):
            return
        step_id = self.engine.trace[index]["id"]
        self.heat_spin.setValue(step_id)
        row = self._trace_row_for_step(step_id)
        if row is not None:
            self.trace_table.setCurrentCell(row, 0)
            item = self.trace_table.item(row, 0)
            if item is not None:
                self.trace_table.scrollToItem(item)

    def _refresh_layers(self):
        selected = None
        current = self.layer_list.currentItem()
        if current is not None:
            selected = current.data(Qt.UserRole)
        self.layer_list.blockSignals(True)
        self.layer_list.clear()
        restore_row = -1
        for entry in self.engine.trace:
            if not entry["new_layer"]:
                continue
            item_text = ("Step %d: %s (resonance %.3f)"
                         % (entry["id"], entry["input"],
                            entry["resonance"]))
            self.layer_list.addItem(item_text)
            item = self.layer_list.item(
                self.layer_list.count() - 1)
            item.setData(Qt.UserRole, entry["id"])
            # v0.10 (R3): the full text, on hover.
            item.setToolTip(self._entry_text(entry))
            if entry["id"] == selected:
                restore_row = self.layer_list.count() - 1
        self.layer_list.blockSignals(False)
        if restore_row >= 0:
            self.layer_list.setCurrentRow(restore_row)
        elif self.layer_list.count() == 0:
            self.layer_detail.setPlainText(
                "No resonance spikes yet. Step a few sentences on "
                "one theme and spikes will show up here.")
        elif current is None:
            self.layer_detail.setPlainText(
                "Click a candidate to see its trace entries.")

    def _format_trace_entry(self, entry, label):
        # v0.10 (R3): the full input, not the 80-character
        # preview: the detail panes exist to show the whole
        # entry.
        return (
            "%s\n"
            "  Input: %s\n"
            "  Delta norm: %.4f\n"
            "  W norm: %.4f -> %.4f\n"
            "  Resonance: %.4f  Alpha: %.2f  Confidence: %.4f\n"
            "  Frozen: %s  Capped: %s"
            % (label, self._entry_text(entry),
               entry["delta_norm"],
               entry["old_norm"], entry["new_norm"],
               entry["resonance"], entry["alpha"],
               entry["confidence"],
               "yes" if entry["frozen"] else "no",
               "yes" if entry.get("capped") else "no"))

    def _on_layer_selected(self, current, _previous):
        if current is None:
            return
        step_id = current.data(Qt.UserRole)
        trace = self.engine.trace
        lines = []
        for entry in trace:
            if entry["id"] == step_id - 1:
                lines.append(self._format_trace_entry(
                    entry, "Step %d (before)" % entry["id"]))
            elif entry["id"] == step_id:
                lines.append(self._format_trace_entry(
                    entry, "Step %d (spike)" % entry["id"]))
            elif entry["id"] == step_id + 1:
                lines.append(self._format_trace_entry(
                    entry, "Step %d (after)" % entry["id"]))
        if not lines:
            self.layer_detail.setPlainText(
                "No trace entries for step %d." % step_id)
            return
        self.layer_detail.setPlainText("\n".join(lines))


def _instance_name():
    """Single-instance name: ACR_INSTANCE_NAME lets tests run a
    private instance without touching a real one."""
    return os.environ.get(INSTANCE_ENV, INSTANCE_DEFAULT)


def main(argv=None):
    app = QApplication(sys.argv if argv is None else argv)
    # v0.9 identity (U15): the dark look, on the whole app.
    apply_theme(app)
    # v0.10 (R9): the big-text preset, when the user saved it.
    apply_big_text(app, big_text_enabled())
    # v0.9 single instance (U27): re-clicking the icon focuses the
    # running app, never starts a second copy. QLockFile plus a
    # QLocalServer: the second launch asks the first to raise,
    # then exits 0. tryLock takes its timeout argument: the bare
    # call raises TypeError in this PySide6 build. removeServer
    # first, or a crashed run leaves its socket behind and the
    # next listen fails.
    name = _instance_name()
    lock_path = os.path.join(tempfile.gettempdir(),
                             "acr-%s.lock" % name)
    lock = QLockFile(lock_path)
    lock.setStaleLockTime(1000)
    if not lock.tryLock(0):
        socket = QLocalSocket()
        socket.connectToServer("ACR-" + name)
        if socket.waitForConnected(2000):
            socket.write(b"raise")
            socket.waitForBytesWritten(2000)
        socket.disconnectFromServer()
        return 0
    QLocalServer.removeServer("ACR-" + name)
    server = QLocalServer()
    server.listen("ACR-" + name)
    window = MainWindow()

    def _on_raise_request():
        sock = server.nextPendingConnection()
        if sock is not None:
            sock.waitForReadyRead(1000)
            sock.readAll()
            sock.disconnectFromServer()
        window.show()
        window.raise_()
        window.activateWindow()
        window.logger.action("single-instance raise requested")

    server.newConnection.connect(_on_raise_request)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
