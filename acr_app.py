#!/usr/bin/env python3
"""ACR - Adaptive Contextual Resonance, v0.2.

A native desktop window for the ACR adaptive loop at small scale:
type text, step the loop, watch W_t adapt, read the trace log.

What this is: a real, runnable prototype of the blueprint's core
loop (encoder, low-rank W_t updates, resonance detection, trace log).
What it is not: a full self-evolving neural engine. That does not
exist yet, and this app does not claim to be it.
"""

import json
import os
import sys

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QDoubleSpinBox, QFileDialog, QHBoxLayout,
    QHeaderView, QLabel, QListWidget, QMainWindow, QMessageBox,
    QPlainTextEdit, QPushButton, QSlider, QTableWidget,
    QTableWidgetItem, QTextEdit, QVBoxLayout, QWidget,
)

from acr_engine import BUFFER_SIZE, DIM, RANK, Engine
from debug_log import DebugLogger, check_ollama

APP_NAME = "acr"
VERSION = "0.2"
REPORT_EMAIL = "broke.n.bored.studio@gmail.com"

TRACE_COLUMNS = ["Step", "Input", "Delta norm", "W old", "W new",
                 "Resonance", "Alpha", "Confidence", "New layer"]


class MainWindow(QMainWindow):
    def __init__(self, log_dir=None):
        super().__init__()
        self.setWindowTitle("ACR - Adaptive Contextual Resonance")
        self.resize(1100, 680)
        self.engine = Engine()

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

        # Right: trace table, latent buffer.
        right = QVBoxLayout()
        root.addLayout(right, 3)

        right.addWidget(QLabel("Trace log - every weight change"))
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
        right.addWidget(self.trace_table, 3)

        right.addWidget(QLabel(
            "Latent state buffer - last %d inputs, newest first"
            % BUFFER_SIZE))
        self.buffer_list = QListWidget()
        right.addWidget(self.buffer_list, 1)

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


def main(argv=None):
    app = QApplication(sys.argv if argv is None else argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
