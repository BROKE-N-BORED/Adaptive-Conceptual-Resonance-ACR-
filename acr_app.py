#!/usr/bin/env python3
"""ACR - Adaptive Contextual Resonance, v0.3.

A native desktop window for the ACR adaptive loop at small scale:
type text, step the loop, watch W_t adapt, read the trace log, and
see the resonance views (norm chart, change heatmap, layer
candidates).

What this is: a real, runnable prototype of the blueprint's core
loop (encoder, low-rank W_t updates, resonance detection, trace log).
What it is not: a full self-evolving neural engine. That does not
exist yet, and this app does not claim to be it.
"""

import json
import math
import os
import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QDoubleSpinBox, QFileDialog, QHBoxLayout,
    QHeaderView, QLabel, QListWidget, QMainWindow, QMessageBox,
    QPlainTextEdit, QPushButton, QSlider, QSpinBox, QTabWidget,
    QTableWidget, QTableWidgetItem, QTextEdit, QVBoxLayout, QWidget,
)

from acr_engine import BUFFER_SIZE, DIM, RANK, Engine
from debug_log import DebugLogger, check_ollama

APP_NAME = "acr"
VERSION = "0.3"
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

    Row i of W is (u[i] . v[j]) over j, so each row's change is
    computed directly from the factors without forming either full
    matrix. Read-only: it never touches the engine's factors.
    """
    rows = []
    for i in range(DIM):
        total = 0.0
        for j in range(DIM):
            old = 0.0
            new = 0.0
            for k in range(RANK):
                old += old_u[i][k] * old_v[j][k]
                new += new_u[i][k] * new_v[j][k]
            total += (new - old) ** 2
        rows.append(math.sqrt(total))
    return rows


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
    """Per-row change of W_t for one step, as 64 bars.

    One bar per row of W_t, longest and darkest for the rows that
    moved most on the step. The three biggest movers get their row
    numbers. `values` holds the 64 per-row norms for `step_id`.
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
    def __init__(self, log_dir=None):
        super().__init__()
        self.setWindowTitle("ACR - Adaptive Contextual Resonance")
        self.resize(1100, 680)
        self.engine = Engine()
        # Heatmap history: one (step id, 64 per-row change norms)
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
        self.report_button = QPushButton("Generate debug report")
        self.report_button.clicked.connect(self.on_debug_report)
        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.on_reset)
        btn_row.addWidget(self.export_button)
        btn_row.addWidget(self.report_button)
        btn_row.addWidget(self.reset_button)
        left.addLayout(btn_row)

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

    def on_step(self):
        text = self.input_edit.toPlainText().strip()
        if not text:
            self.status_label.setText("Type something first.")
            return
        # Snapshot the factors before the step so the heatmap can
        # show which rows of W_t moved (read-only copies).
        old_u = [row[:] for row in self.engine.u]
        old_v = [row[:] for row in self.engine.v]
        try:
            entry = self.engine.step(
                text, alpha=self._alpha(),
                frozen=self.freeze_check.isChecked(),
                cap=self.cap_spin.value())
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
            "Clear the trace, the buffer, and W_t back to its start?",
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
