#!/usr/bin/env python3
"""ACR - Adaptive Contextual Resonance, v0.5.

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
"""

import csv
import json
import math
import os
import shutil
import sys
import time

from PySide6.QtCore import QProcess, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
    QHBoxLayout, QHeaderView, QLabel, QListWidget, QMainWindow,
    QMessageBox, QPlainTextEdit, QPushButton, QSlider, QSpinBox,
    QTabWidget, QTableWidget, QTableWidgetItem, QTextEdit,
    QVBoxLayout, QWidget,
)

from acr_engine import (BUFFER_SIZE, DIM, RANK, Engine,
                        SESSION_FORMAT_VERSION, SessionError)
from debug_log import DebugLogger, check_ollama
from local_encoder import (
    MSG_NO_MODELS, MSG_SERVICE_DOWN, MSG_SERVICE_NOT_INSTALLED,
    MSG_START_FAILED, LocalModelClient, LocalModelError,
    pick_default_model)

APP_NAME = "acr"
VERSION = "0.5"
REPORT_EMAIL = "broke.n.bored.studio@gmail.com"

TRACE_COLUMNS = ["Step", "Input", "Delta norm", "W old", "W new",
                 "Resonance", "Alpha", "Confidence", "New layer"]

CHART_LINE = QColor(31, 78, 121)
CHART_SPIKE = QColor(192, 57, 43)
CHART_GRID = QColor(220, 220, 220)
CHART_AXIS = QColor(120, 120, 120)
HEAT_LOW = QColor(220, 235, 250)
HEAT_HIGH = QColor(31, 78, 121)


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
    It blocks while waiting, pumping the event loop so the window
    stays alive.
    """

    START_TIMEOUT = 15.0
    PROBE_INTERVAL = 0.5

    def __init__(self, client):
        self.client = client

    def start(self, timeout=None):
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
            app = QApplication.instance()
            if app is not None:
                app.processEvents()
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
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(130)
        self.values = []
        self.flags = []

    def set_data(self, values, flags):
        self.values = list(values)
        self.flags = list(flags)
        self.update()

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.white)
        if not self.values:
            painter.setPen(CHART_AXIS)
            painter.drawText(self.rect(), Qt.AlignCenter,
                             "Step some text to see the chart.")
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
                painter.drawEllipse(int(x) - 4, int(y) - 4, 8, 8)
            else:
                painter.setPen(QPen(CHART_LINE, 2))
                painter.setBrush(CHART_LINE)
                painter.drawEllipse(int(x) - 2, int(y) - 2, 4, 4)
        painter.setPen(CHART_SPIKE)
        painter.drawText(left + width - 150, top + 10,
                         "red dot = new-layer candidate")


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

    def set_data(self, values, step_id):
        self.values = list(values)
        self.step_id = step_id
        self.update()

    def top_rows(self, count=3):
        """Row indexes of the biggest movers, biggest first."""
        order = sorted(range(len(self.values)),
                       key=lambda i: self.values[i], reverse=True)
        return order[:count]

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.white)
        if not self.values:
            painter.setPen(CHART_AXIS)
            painter.drawText(self.rect(), Qt.AlignCenter,
                             "Step some text to see the heatmap.")
            return
        top = set(self.top_rows(3))
        peak = max(self.values)
        if peak <= 0.0:
            painter.setPen(CHART_AXIS)
            # v0.3: step 1 never moves W_t (no previous state to
            # predict from), so it is not a frozen step.
            if self.step_id == 1:
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


class MainWindow(QMainWindow):
    def __init__(self, log_dir=None, local_base_url=None):
        super().__init__()
        self.setWindowTitle("ACR - Adaptive Contextual Resonance")
        self.resize(1100, 680)
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
        self._refresh_all()

    # -- UI -----------------------------------------------------------
    def _build_ui(self):
        central = QWidget(self)
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        # Left: input, controls, readouts, synthesis.
        left = QVBoxLayout()
        root.addLayout(left, 2)

        about = QLabel(
            "Small-scale prototype of the adaptive loop. Type text, "
            "press Step, and W_t adapts. This is not a full neural "
            "engine.")
        about.setWordWrap(True)
        left.addWidget(about)

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
        self.model_combo.currentTextChanged.connect(
            self._on_model_changed)
        encoder_row.addWidget(self.model_combo, 1)
        self.start_button = QPushButton("Start service")
        self.start_button.clicked.connect(self.on_start_service)
        encoder_row.addWidget(self.start_button)
        left.addLayout(encoder_row)
        self.encoder_label = QLabel("Built-in encoder, dim 64.")
        left.addWidget(self.encoder_label)
        switch_note = QLabel(
            "Switching encoders starts a fresh W_t (the embedding "
            "sizes differ).")
        switch_note.setWordWrap(True)
        left.addWidget(switch_note)

        left.addWidget(QLabel("Input text"))
        self.input_edit = QPlainTextEdit()
        self.input_edit.setPlaceholderText(
            "Type a sentence, then press Step...")
        self.input_edit.setMaximumHeight(90)
        left.addWidget(self.input_edit)

        self.step_button = QPushButton("Step")
        self.step_button.clicked.connect(self.on_step)
        left.addWidget(self.step_button)

        self.step_label = QLabel("Step: 0")
        self.norm_label = QLabel("W norm: -")
        self.resonance_label = QLabel("Resonance: -")
        self.alpha_label = QLabel("Alpha: 0.30")
        for label in (self.step_label, self.norm_label,
                      self.resonance_label, self.alpha_label):
            left.addWidget(label)

        left.addWidget(QLabel("Learning rate (alpha)"))
        self.alpha_slider = QSlider(Qt.Horizontal)
        self.alpha_slider.setRange(1, 50)
        self.alpha_slider.setValue(30)
        self.alpha_slider.setTickPosition(QSlider.TicksBelow)
        self.alpha_slider.setTickInterval(10)
        self.alpha_slider.valueChanged.connect(self._on_alpha_changed)
        left.addWidget(self.alpha_slider)

        self.freeze_check = QCheckBox("Freeze weights (log only, no update)")
        self.freeze_check.stateChanged.connect(
            lambda _s: self.logger.action(
                "freeze toggled: %s" % self.freeze_check.isChecked()))
        left.addWidget(self.freeze_check)

        cap_row = QHBoxLayout()
        cap_row.addWidget(QLabel("Stability cap"))
        self.cap_spin = QDoubleSpinBox()
        self.cap_spin.setRange(0.1, 5.0)
        self.cap_spin.setSingleStep(0.1)
        self.cap_spin.setValue(1.0)
        cap_row.addWidget(self.cap_spin)
        cap_row.addStretch(1)
        left.addLayout(cap_row)

        left.addWidget(QLabel(
            "Synthesis head (demo) - related past inputs from this "
            "session. Not a language model."))
        self.synthesis_text = QTextEdit()
        self.synthesis_text.setReadOnly(True)
        self.synthesis_text.setMaximumHeight(150)
        left.addWidget(self.synthesis_text)

        btn_row = QHBoxLayout()
        self.export_button = QPushButton("Export trace JSON")
        self.export_button.clicked.connect(self.on_export)
        self.export_csv_button = QPushButton("Export trace CSV")
        self.export_csv_button.clicked.connect(self.on_export_csv)
        self.report_button = QPushButton("Generate debug report")
        self.report_button.clicked.connect(self.on_debug_report)
        btn_row.addWidget(self.export_button)
        btn_row.addWidget(self.export_csv_button)
        btn_row.addWidget(self.report_button)
        left.addLayout(btn_row)

        session_row = QHBoxLayout()
        self.save_button = QPushButton("Save session")
        self.save_button.clicked.connect(self.on_save_session)
        self.load_button = QPushButton("Load session")
        self.load_button.clicked.connect(self.on_load_session)
        self.export_session_button = QPushButton("Export session")
        self.export_session_button.clicked.connect(
            self.on_export_session)
        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.on_reset)
        session_row.addWidget(self.save_button)
        session_row.addWidget(self.load_button)
        session_row.addWidget(self.export_session_button)
        session_row.addWidget(self.reset_button)
        left.addLayout(session_row)

        self.status_label = QLabel("Ready.")
        self.status_label.setWordWrap(True)
        left.addWidget(self.status_label)
        left.addStretch(1)

        # Right: tabs for the trace log and the resonance views.
        self.tabs = QTabWidget()
        root.addWidget(self.tabs, 3)

        trace_page = QWidget()
        trace_layout = QVBoxLayout(trace_page)
        trace_layout.addWidget(QLabel("Trace log - every weight change"))
        self.trace_table = QTableWidget(0, len(TRACE_COLUMNS))
        self.trace_table.setHorizontalHeaderLabels(TRACE_COLUMNS)
        self.trace_table.setEditTriggers(
            QTableWidget.NoEditTriggers)
        # The Step column already numbers the rows, so the row
        # numbers add nothing. Numeric columns size to their
        # contents; the Input column takes the leftover width so
        # all nine columns fit without scrolling.
        self.trace_table.verticalHeader().setVisible(False)
        header = self.trace_table.horizontalHeader()
        for col in range(len(TRACE_COLUMNS)):
            if col == 1:
                header.setSectionResizeMode(
                    col, QHeaderView.Stretch)
            else:
                header.setSectionResizeMode(
                    col, QHeaderView.ResizeToContents)
        trace_layout.addWidget(self.trace_table, 3)

        trace_layout.addWidget(QLabel(
            "Latent state buffer - last %d inputs, newest first"
            % BUFFER_SIZE))
        self.buffer_list = QListWidget()
        trace_layout.addWidget(self.buffer_list, 1)
        self.tabs.addTab(trace_page, "Trace log")

        views_page = QWidget()
        views_layout = QVBoxLayout(views_page)

        views_layout.addWidget(QLabel(
            "W_t norm over time - spikes are resonance spikes"))
        self.norm_chart = NormChart()
        views_layout.addWidget(self.norm_chart, 2)

        views_layout.addWidget(QLabel(
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
        views_layout.addLayout(heat_row)
        self.heatmap = ChangeHeatmap()
        views_layout.addWidget(self.heatmap, 2)

        views_layout.addWidget(QLabel(
            "Layer candidates - resonance spikes and the input "
            "that triggered them"))
        self.layer_list = QListWidget()
        self.layer_list.setMaximumHeight(90)
        self.layer_list.currentItemChanged.connect(
            self._on_layer_selected)
        views_layout.addWidget(self.layer_list)
        self.layer_detail = QTextEdit()
        self.layer_detail.setReadOnly(True)
        self.layer_detail.setMaximumHeight(110)
        views_layout.addWidget(self.layer_detail)
        self.tabs.addTab(views_page, "Resonance views")

    # -- actions ------------------------------------------------------
    def _alpha(self):
        return self.alpha_slider.value() / 100.0

    def _on_alpha_changed(self, value):
        self.alpha_label.setText("Alpha: %.2f" % (value / 100.0))

    # -- encoder ------------------------------------------------------
    def _set_encoder_ui(self):
        """Picker, model list, and label to match the encoder state."""
        self.encoder_combo.blockSignals(True)
        self.encoder_combo.setCurrentIndex(
            1 if self.encoder_kind == "local" else 0)
        self.encoder_combo.blockSignals(False)
        if self.encoder_kind == "local":
            self.model_combo.setEnabled(True)
            self.encoder_label.setText(
                "Local model: %s, dim %d."
                % (self.local_model, self.engine.dim))
        else:
            # No model is running: clear the list so a stale name
            # never sits in the disabled combo.
            self.model_combo.blockSignals(True)
            self.model_combo.clear()
            self.model_combo.blockSignals(False)
            self.model_combo.setEnabled(False)
            self.encoder_label.setText("Built-in encoder, dim 64.")

    def _fresh_engine(self, dim):
        """A fresh engine at the given dim, with views cleared."""
        self.engine = Engine(dim=dim)
        self._heat_history = []
        self._heat_follow = True
        self.logger.set_setting("dim", dim)

    def _on_encoder_changed(self, index):
        if index == 0:
            if self.encoder_kind == "builtin":
                return
            self._switch_to_builtin(
                "Back to the built-in encoder. W_t is fresh.")
            return
        if self.encoder_kind == "local":
            return
        self._switch_to_local()

    def _switch_to_builtin(self, message):
        self.encoder_kind = "builtin"
        self.local_model = None
        self._fresh_engine(DIM)
        self.logger.set_setting("encoder", "built-in deterministic")
        self._set_encoder_ui()
        self._refresh_all()
        self.status_label.setText(message)

    def _switch_to_local(self):
        """Pick the local encoder: health check, model list, then a
        probe embedding to learn the model's dim. Any failure leaves
        the built-in encoder in place with a plain message."""
        if not self.client.probe():
            self._set_encoder_ui()
            self.status_label.setText(
                MSG_SERVICE_DOWN + " Press Start service, or stay "
                "on Built-in.")
            return
        try:
            models = self.client.list_models()
        except LocalModelError as exc:
            self._set_encoder_ui()
            self.status_label.setText(str(exc))
            return
        if not models:
            self._set_encoder_ui()
            self.status_label.setText(MSG_NO_MODELS)
            return
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        self.model_combo.addItems(models)
        self.model_combo.blockSignals(False)
        self._activate_local_model(pick_default_model(models))

    def _activate_local_model(self, model):
        """Make a local model the encoder, with a fresh W_t.

        A probe embedding learns the model's dim first: the engine
        is built at the dim the model actually returns. On failure
        the built-in encoder stays in place with a plain message.
        """
        try:
            probe = self.client.embed(model, "hello")
        except LocalModelError as exc:
            self._set_encoder_ui()
            # A failed model switch leaves the old model active:
            # put its name back in the list so the list never shows
            # a model that is not running.
            if self.encoder_kind == "local" and self.local_model:
                self.model_combo.blockSignals(True)
                self.model_combo.setCurrentText(self.local_model)
                self.model_combo.blockSignals(False)
            self.status_label.setText(str(exc))
            return
        self.encoder_kind = "local"
        self.local_model = model
        self._fresh_engine(len(probe))
        self.logger.set_setting("encoder", "local model %s" % model)
        self.model_combo.blockSignals(True)
        self.model_combo.setCurrentText(model)
        self.model_combo.blockSignals(False)
        self._set_encoder_ui()
        self._refresh_all()
        self.status_label.setText(
            "Local encoder: %s (dim %d). W_t is fresh."
            % (model, len(probe)))

    def _on_model_changed(self, name):
        # Fires while the list is populated too; only a real change
        # to another model while local re-activates (fresh W_t).
        if (self.encoder_kind != "local" or not name
                or name == self.local_model):
            return
        self._activate_local_model(name)

    def on_start_service(self):
        self.start_button.setEnabled(False)
        try:
            ok, message = self.service.start()
        finally:
            self.start_button.setEnabled(True)
        if ok:
            message += " Pick Local model to use it."
        self.status_label.setText(message)

    def _on_embed_retry(self, attempt, delay, kind):
        # Called from the client's retry loop, on the UI thread:
        # show the slow message live instead of after the fact.
        self.status_label.setText(
            "The local model service is slow, retrying "
            "(attempt %d)..." % (attempt + 1))
        app = QApplication.instance()
        if app is not None:
            app.processEvents()

    def on_step(self):
        text = self.input_edit.toPlainText().strip()
        if not text:
            self.status_label.setText("Type something first.")
            return
        h = None
        fallback_message = None
        if self.encoder_kind == "local":
            try:
                h = self.client.embed(
                    self.local_model, text,
                    on_retry=self._on_embed_retry)
            except LocalModelError as exc:
                # Mid-session failure: fall back to the built-in
                # encoder with a fresh W_t, and run this step on
                # it. The user loses the local session, not the
                # app, and the message says what happened.
                self.logger.action(
                    "local encoder failed (%s), fell back to "
                    "built-in" % exc)
                fallback_message = (
                    "The local model service stopped. Back to the "
                    "built-in encoder with a fresh W_t.")
                self._switch_to_builtin(fallback_message)
            else:
                if len(h) != self.engine.dim:
                    self.logger.action(
                        "local model returned dim %d, engine is "
                        "dim %d" % (len(h), self.engine.dim))
                    fallback_message = (
                        "The local model returned the wrong size. "
                        "Back to the built-in encoder with a fresh "
                        "W_t.")
                    self._switch_to_builtin(fallback_message)
                    h = None
        # Snapshot the factors before the step so the heatmap can
        # show which rows of W_t moved (read-only copies).
        old_u = [row[:] for row in self.engine.u]
        old_v = [row[:] for row in self.engine.v]
        try:
            entry = self.engine.step(
                text, alpha=self._alpha(),
                frozen=self.freeze_check.isChecked(),
                cap=self.cap_spin.value(), h=h)
        except Exception:
            self.logger.exception("step failed")
            self.status_label.setText(
                "Step failed - see the debug report for details.")
            return
        self._heat_history.append(
            (entry["id"], _row_change_norms(
                old_u, old_v, self.engine.u, self.engine.v)))
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

    def export_trace_to(self, path):
        """Write the trace log as JSON. Returns the path."""
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.engine.trace, fh, indent=2)
        self.logger.action("trace exported to %s" % path)
        return path

    def on_export(self):
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
                "frozen": self.freeze_check.isChecked(),
            },
            "heat_history": [[step_id, list(rows)]
                             for step_id, rows in self._heat_history],
        }

    def save_session_to(self, path):
        """Write the session as JSON. Returns the path."""
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.session_dict(), fh, indent=2)
        self.logger.action("session saved to %s" % path)
        return path

    def on_save_session(self):
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
        models = None
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
                    models = self.client.list_models()
                except LocalModelError:
                    raise SessionError(
                        "this session needs the local encoder, "
                        "which is not running - start it and try "
                        "again")
                if encoder_model not in models:
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
        frozen = settings.get("frozen")
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
        if not isinstance(frozen, bool):
            raise SessionError(
                "that session file has a bad freeze setting")
        heat = self._checked_heat_history(data.get("heat_history"),
                                          encoder_dim)
        # Everything checked: apply it.
        self.encoder_kind = encoder_kind
        self.local_model = encoder_model
        self.engine = spare
        if encoder_kind == "local":
            self.logger.set_setting(
                "encoder", "local model %s" % encoder_model)
            self.model_combo.blockSignals(True)
            self.model_combo.clear()
            self.model_combo.addItems(models)
            self.model_combo.setCurrentText(encoder_model)
            self.model_combo.blockSignals(False)
        else:
            self.logger.set_setting("encoder",
                                    "built-in deterministic")
        self.logger.set_setting("dim", encoder_dim)
        self.alpha_slider.setValue(int(round(alpha * 100)))
        self.cap_spin.setValue(cap)
        self.freeze_check.setChecked(frozen)
        self._heat_history = heat
        self._heat_follow = True
        self._set_encoder_ui()
        self.logger.action("session loaded from %s" % path)
        self._refresh_all()
        return path

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
                ])
        self.logger.action("trace CSV exported to %s" % path)
        return path

    def on_export_csv(self):
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
                    "Frozen: %s"
                    % (entry["alpha"], entry["confidence"],
                       "yes" if entry["new_layer"] else "no",
                       "yes" if entry["frozen"] else "no"))
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        self.logger.action("session exported to %s" % path)
        return path

    def on_export_session(self):
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
        self.logger.action("session reset")
        self._refresh_all()
        self.status_label.setText("Reset. W_t is back to its start.")

    def on_reset(self):
        answer = QMessageBox.question(
            self, "Reset",
            "Clear the session and reset W_t?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if answer == QMessageBox.Yes:
            self.do_reset()

    # -- refresh ------------------------------------------------------
    def _refresh_all(self):
        self.step_label.setText("Step: %d" % self.engine.step_count)
        self.norm_label.setText(
            "W norm: %.4f" % self.engine.w_norm())
        if self.engine.trace:
            last = self.engine.trace[-1]
            self.resonance_label.setText(
                "Resonance: %.4f" % last["resonance"])
        else:
            self.resonance_label.setText("Resonance: -")
        self._refresh_trace_table()
        self._refresh_buffer_list()
        self._refresh_synthesis()
        self._refresh_chart()
        self._refresh_heatmap()
        self._refresh_layers()

    def _refresh_trace_table(self):
        trace = self.engine.trace
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
            ]
            for col, value in enumerate(values):
                self.trace_table.setItem(
                    row, col, QTableWidgetItem(value))
        if trace:
            self.trace_table.scrollToBottom()

    def _refresh_buffer_list(self):
        self.buffer_list.clear()
        for item in reversed(self.engine.buffer):
            self.buffer_list.addItem(item["text"])

    def _refresh_synthesis(self):
        result = self.engine.synthesize()
        lines = []
        if result["neighbours"]:
            lines.append("Closest past inputs:")
            for text, score in result["neighbours"]:
                lines.append("  %.3f  %s" % (score, text))
        if result["midpoint"]:
            lines.append("Midpoint neighbours:")
            for text, score in result["midpoint"]:
                lines.append("  %.3f  %s" % (score, text))
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
        self.heatmap.set_data(rows, step_id)
        if not rows or max(rows) <= 0.0:
            # v0.3: step 1 never moves W_t (no previous state to
            # predict from), so it is not a frozen step.
            if step_id == 1:
                self.heat_summary.setText(
                    "Step 1: W_t did not move (no previous step "
                    "to learn from).")
            else:
                self.heat_summary.setText(
                    "Step %d: W_t did not move (frozen step)."
                    % step_id)
            return
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
        return (
            "%s\n"
            "  Input: %s\n"
            "  Delta norm: %.4f\n"
            "  W norm: %.4f -> %.4f\n"
            "  Resonance: %.4f  Alpha: %.2f  Confidence: %.4f\n"
            "  Frozen: %s"
            % (label, entry["input"], entry["delta_norm"],
               entry["old_norm"], entry["new_norm"],
               entry["resonance"], entry["alpha"],
               entry["confidence"],
               "yes" if entry["frozen"] else "no"))

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


def main(argv=None):
    app = QApplication(sys.argv if argv is None else argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
