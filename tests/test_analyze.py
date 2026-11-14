"""Batch analysis and STELLA data pack tests (v0.9).

The batch loop is the same Engine.step() path as manual stepping:
a batch over a file of the 16 theme sentences must end in exactly
the state typing them would. The STELLA packs teach a model what
ACR found (themes, closest matches, analogies, resonance
judgments), all measured from the run, never invented.

Run with the Qt test env: LD_LIBRARY_PATH to the extracted libs,
QT_QPA_PLATFORM=offscreen, TMPDIR=/tmp, default HOME.
"""

import hashlib
import json
import os
import random
import re
import sys
import tempfile
import threading
import time
import unittest
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402
from unittest import mock  # noqa: E402

from acr_app import (MainWindow, collect_batch_files,  # noqa: E402
                     extract_batch_text, iter_batch_lines,
                     summarize_run)
from acr_engine import (RESONANT_FLOOR, Engine, _dot,  # noqa: E402
                        _normalize, encode)
import acr_packs  # noqa: E402
from acr_packs import (KIND_ANALOGY, KIND_JUDGMENT, KIND_MATCH,  # noqa: E402
                       KIND_MEMBERSHIP, analogy_for, assign_theme,
                       default_pack_name, export_pack,
                       nearest_neighbour, safe_pack_name,
                       slugify, source_slug)

APP = None

THEME = ["quantum entanglement links particles",
         "entangled particles share a state",
         "quantum particles stay connected",
         "entanglement links distant particles"]
OFF = ["the toaster needs a new heating element",
       "my car battery died this morning",
       "the garden fence fell over in the wind",
       "I need to buy milk and bread",
       "the train was late again today",
       "my phone screen cracked last night",
       "the plumber fixed the leaking tap",
       "we watched a film about sailing"]


def setUpModule():
    global APP
    APP = QApplication.instance()
    if APP is None:
        APP = QApplication(sys.argv)


def theme_engine(lines, alpha=0.3):
    """An engine driven over the given lines, at the window's
    alpha: engine-only steps default to 0.1, which climbs too
    slowly to flag anything."""
    engine = Engine()
    for line in lines:
        engine.step(line, alpha=alpha)
    return engine


class BatchExtractTests(unittest.TestCase):
    def test_extract_plain_line(self):
        self.assertEqual(extract_batch_text("  hello world  ",
                                            False),
                         "hello world")

    def test_extract_blank_line(self):
        self.assertIsNone(extract_batch_text("   ", False))

    def test_extract_jsonl_text_field(self):
        self.assertEqual(extract_batch_text(
            '{"text": "hello"}', True), "hello")

    def test_extract_jsonl_prompt_then_completion(self):
        self.assertEqual(extract_batch_text(
            '{"prompt": "ask this"}', True), "ask this")
        self.assertEqual(extract_batch_text(
            '{"completion": "say that"}', True), "say that")

    def test_extract_jsonl_bad_lines(self):
        self.assertIsNone(extract_batch_text("not json", True))
        self.assertIsNone(extract_batch_text('{"other": 1}', True))
        self.assertIsNone(extract_batch_text('["a"]', True))
        self.assertIsNone(extract_batch_text('{"text": ""}', True))

    def test_iter_batch_lines_yields_locators(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "a.txt")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("first line\n\nthird line\n")
            rows = list(iter_batch_lines([path]))
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0],
                         ("first line", False, path, 1))
        self.assertEqual(rows[1], (None, True, path, 2))
        self.assertEqual(rows[2],
                         ("third line", False, path, 3))

    def test_iter_batch_lines_unreadable(self):
        unreadable = []
        rows = list(iter_batch_lines(["/no/such/file.txt"],
                                     unreadable))
        self.assertEqual(rows, [])
        self.assertEqual(unreadable, ["file.txt"])

    def test_collect_batch_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("b.md", "a.txt", "c.jsonl", "d.png",
                         "e.TXT"):
                with open(os.path.join(tmp, name), "w",
                          encoding="utf-8") as fh:
                    fh.write("x\n")
            found = collect_batch_files(tmp)
        self.assertEqual([os.path.basename(p) for p in found],
                         ["a.txt", "b.md", "c.jsonl", "e.TXT"])


class SummarizeTests(unittest.TestCase):
    def test_theme_run_summary(self):
        lines = [THEME[i % 4] for i in range(16)] + OFF
        engine = theme_engine(lines)
        summary = summarize_run(engine.trace)
        self.assertEqual(summary["steps"], 24)
        self.assertEqual(len(summary["themes"]), 1)
        self.assertEqual(summary["themes"][0]["id"], 15)
        self.assertEqual(summary["themes"][0]["input"],
                         "quantum particles stay connected")
        self.assertEqual([e["id"] for e in summary["shifts"]],
                         [17])
        core_texts = [e["input"] for e in summary["core"]]
        self.assertIn("quantum entanglement links particles",
                      core_texts)
        outlier_texts = [e["input"] for e in summary["outliers"]]
        self.assertIn("the toaster needs a new heating element",
                      outlier_texts)

    def test_pure_theme_run_has_no_shifts(self):
        lines = [THEME[i % 4] for i in range(16)]
        engine = theme_engine(lines)
        summary = summarize_run(engine.trace)
        self.assertEqual(summary["shifts"], [])
        self.assertEqual(len(summary["themes"]), 1)


class BatchWindowCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.window = MainWindow(log_dir=self.tmp.name)
        self.window.show()

    def tearDown(self):
        self.window.close()
        self.window.deleteLater()
        self.tmp.cleanup()

    def _fixture(self, name, lines):
        path = os.path.join(self.tmp.name, name)
        with open(path, "w", encoding="utf-8") as fh:
            for line in lines:
                fh.write(line + "\n")
        return path

    def _run_batch(self, paths, source_note="fixture.txt",
                   timeout=25.0):
        self.window.start_batch(paths, source_note)
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.window._batch_thread is None:
                return True
            QTest.qWait(50)
        return False

    def _step_text(self, text):
        self.window.input_edit.setPlainText(text)
        QTest.mouseClick(self.window.step_button, Qt.LeftButton)

    def _theme_fixture(self):
        return self._fixture(
            "fixture.txt",
            [THEME[i % 4] for i in range(16)] + OFF)


class BatchTests(BatchWindowCase):
    def test_batch_matches_manual_drive(self):
        path = self._fixture(
            "theme16.txt", [THEME[i % 4] for i in range(16)])
        self.assertTrue(self._run_batch([path], "theme16.txt"))
        batch = (self.window.step_label.text(),
                 self.window.norm_label.text(),
                 self.window.resonance_label.text())
        self.assertEqual(batch, ("Step: 16", "W norm: 1.5938",
                                 "Resonance: 0.5216"))
        # The same sentences typed by hand end in exactly the
        # same state: the batch is the same step path.
        manual = MainWindow(log_dir=self.tmp.name)
        manual.show()
        try:
            for i in range(16):
                manual.input_edit.setPlainText(THEME[i % 4])
                QTest.mouseClick(manual.step_button,
                                 Qt.LeftButton)
            typed = (manual.step_label.text(),
                     manual.norm_label.text(),
                     manual.resonance_label.text())
        finally:
            manual.close()
            manual.deleteLater()
        self.assertEqual(typed, batch)

    def test_progress_reaches_100(self):
        path = self._theme_fixture()
        self.assertTrue(self._run_batch([path]))
        bar = self.window.analyze_window.progress_bar
        self.assertEqual(bar.maximum(), 24)
        self.assertEqual(bar.value(), 24)

    def test_batch_results_fill_tabs(self):
        path = self._theme_fixture()
        self.assertTrue(self._run_batch([path]))
        analyze = self.window.analyze_window
        self.assertEqual(analyze.themes_list.count(), 1)
        self.assertGreater(analyze.core_list.count(), 0)
        self.assertGreater(analyze.outlier_list.count(), 0)
        self.assertEqual(analyze.shift_list.count(), 1)
        self.assertEqual(len(analyze.chart.values), 24)
        self.assertTrue(analyze.export_text_button.isEnabled())
        self.assertTrue(analyze.export_packs_button.isEnabled())

    def test_batch_skips_blank_and_bad_lines(self):
        path = self._fixture("messy.jsonl", [
            '{"text": "quantum entanglement links particles"}',
            '',
            'not json at all',
            '{"text": "entanglement links distant particles"}',
        ])
        self.assertTrue(self._run_batch([path], "messy.jsonl"))
        self.assertEqual(self.window.step_label.text(), "Step: 2")
        self.assertIn("2 lines skipped",
                      self.window.status_label.text())

    def test_batch_shares_session_with_manual(self):
        path = self._fixture(
            "theme16.txt", [THEME[i % 4] for i in range(16)])
        self.assertTrue(self._run_batch([path], "theme16.txt"))
        self._step_text("quantum particles stay connected")
        self.assertEqual(self.window.step_label.text(), "Step: 17")
        self.assertEqual(self.window.trace_table.rowCount(), 17)

    def test_curated_export_round_trips(self):
        path = self._theme_fixture()
        self.assertTrue(self._run_batch([path]))
        analyze = self.window.analyze_window
        core = analyze._core_texts()
        text_path = os.path.join(self.tmp.name, "curated.txt")
        analyze.export_curated_text_to(text_path)
        with open(text_path, encoding="utf-8") as fh:
            lines = fh.read().splitlines()
        self.assertEqual(lines, core)
        jsonl_path = os.path.join(self.tmp.name, "curated.jsonl")
        analyze.export_curated_jsonl_to(jsonl_path)
        with open(jsonl_path, encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh]
        self.assertEqual([row["text"] for row in rows], core)
        self.assertTrue(all(set(row) == {"text"} for row in rows))
        report_path = os.path.join(self.tmp.name, "report.txt")
        analyze.export_report_to(report_path)
        with open(report_path, encoding="utf-8") as fh:
            report = fh.read()
        self.assertIn("Themes found", report)
        self.assertIn("Core entries", report)
        self.assertIn("Outliers", report)
        self.assertIn("Theme shifts", report)
        self.assertIn("Source: fixture.txt", report)
        self.assertIn("quantum particles stay connected", report)
        self.assertIn("the toaster needs a new heating element",
                      report)


class FakeSlowHandler(BaseHTTPRequestHandler):
    """A fake local service with a slow embed: one HTTP call per
    line, so a batch over it takes long enough to cancel. Same
    wire shape as test_local_encoder's fake service: POST
    /api/embeddings with {"model", "prompt"}, {"embedding": [...]}
    back."""
    embed_delay = 0.1
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _send_json(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/tags":
            self._send_json(
                {"models": [{"name": "nomic-embed-text:latest"}]})
        else:
            self._send_json({"error": "not found"}, status=404)

    def do_POST(self):
        if self.path != "/api/embeddings":
            self._send_json({"error": "not found"}, status=404)
            return
        length = int(self.headers.get("Content-Length") or 0)
        payload = json.loads(self.rfile.read(length).decode(
            "utf-8"))
        time.sleep(self.embed_delay)
        seed = int(hashlib.sha256(
            payload.get("prompt", "").encode("utf-8")
        ).hexdigest()[:8], 16)
        rng = random.Random(seed)
        vector = [rng.uniform(-1.0, 1.0) for _ in range(16)]
        norm = sum(v * v for v in vector) ** 0.5
        vector = [v / norm for v in vector]
        self._send_json({"embedding": vector})


class BatchCancelTests(BatchWindowCase):
    def setUp(self):
        self.server = ThreadingHTTPServer(
            ("127.0.0.1", 0), FakeSlowHandler)
        self.thread = threading.Thread(
            target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = ("http://127.0.0.1:%d"
                         % self.server.server_address[1])
        self.tmp = tempfile.TemporaryDirectory()
        self.window = MainWindow(log_dir=self.tmp.name,
                                 local_base_url=self.base_url)
        self.window.show()

    def tearDown(self):
        self.window.close()
        self.window.deleteLater()
        self.tmp.cleanup()
        self.server.shutdown()
        self.server.server_close()

    def _go_local(self):
        self.window.encoder_combo.setCurrentIndex(1)
        combo = self.window.model_combo
        combo.setCurrentIndex(
            combo.findData("nomic-embed-text:latest"))

    def test_cancel_stops_mid_file(self):
        self._go_local()
        path = self._fixture(
            "slow.txt",
            ["line number %d about quantum things" % i
             for i in range(50)])
        self.window.start_batch([path], "slow.txt")
        # Wait for the first step to land, then cancel: the run
        # is 50 lines at 0.1s each, so cancelling as soon as a
        # step lands is always mid-file, with real steps kept.
        # The wait uses time.sleep + processEvents, NOT
        # QTest.qWait: in this PySide6 build qWait starves the
        # worker thread of the GIL (a 10-line local batch took
        # 22s under qWait(200), 1.2s under sleep+processEvents),
        # so a fixed qWait can pass with zero steps landed.
        # The app itself is unaffected: the real event loop
        # releases the GIL while it waits.
        deadline = time.time() + 20.0
        while (time.time() < deadline
               and self.window.engine.step_count == 0):
            time.sleep(0.05)
            QApplication.processEvents()
        self.assertGreater(self.window.engine.step_count, 0)
        self.window.on_cancel_batch()
        deadline = time.time() + 25.0
        while time.time() < deadline:
            if self.window._batch_thread is None:
                break
            time.sleep(0.05)
            QApplication.processEvents()
        self.assertIsNone(self.window._batch_thread)
        steps = self.window.engine.step_count
        self.assertGreater(steps, 0)
        self.assertLess(steps, 50)
        self.assertEqual(self.window.trace_table.rowCount(),
                         steps)
        self.assertIn("stopped",
                      self.window.status_label.text())


class PackUnitTests(unittest.TestCase):
    def test_slugify(self):
        self.assertEqual(slugify("Quantum Particles!"),
                         "quantum-particles")
        self.assertEqual(slugify("  a  b  "), "a-b")
        self.assertEqual(slugify("!!!"), "pack")
        self.assertEqual(slugify(""), "pack")
        self.assertEqual(len(slugify("x" * 100)), 40)

    def test_source_slug(self):
        self.assertEqual(source_slug("fixture.txt"), "fixture")
        self.assertEqual(source_slug(""), "session")
        self.assertEqual(source_slug("folder data (3 files)"),
                         "folder-data-3-files")

    def test_safe_pack_name(self):
        self.assertEqual(safe_pack_name("ok-name_1.2"),
                         "ok-name_1.2")
        self.assertEqual(safe_pack_name("a/b"), "ab")
        self.assertEqual(safe_pack_name(".."), "")
        self.assertEqual(safe_pack_name(""), "")

    def test_default_pack_name(self):
        name = default_pack_name("Quantum Particles Stay")
        self.assertTrue(name.startswith(
            "quantum-particles-stay-"))
        self.assertTrue(name.endswith(date.today().isoformat()))

    def test_assign_theme(self):
        themes = [{"id": 10, "resonance": 0.5},
                  {"id": 30, "resonance": 0.5}]
        self.assertEqual(
            assign_theme({"id": 12, "resonance": 0.45}, themes),
            themes[0])
        self.assertEqual(
            assign_theme({"id": 28, "resonance": 0.45}, themes),
            themes[1])
        # Below the floor: no theme, however close the spike.
        self.assertIsNone(
            assign_theme({"id": 11, "resonance": 0.39}, themes))
        # No themes at all: no theme.
        self.assertIsNone(
            assign_theme({"id": 5, "resonance": 0.9}, []))
        # A tie goes to the earlier theme.
        self.assertEqual(
            assign_theme({"id": 20, "resonance": 0.45}, themes),
            themes[0])

    def test_nearest_neighbour(self):
        pool = {"a": [1.0, 0.0], "b": [0.0, 1.0],
                "c": [-1.0, 0.0]}
        # Highest cosine wins.
        self.assertEqual(nearest_neighbour(pool, [0.9, 0.1],
                                           "zzz"),
                         ("a", 0.9))
        # The subject itself is never its own neighbour, and a
        # zero score is not a neighbour (positive only).
        text, score = nearest_neighbour(pool, [0.6, 0.8], "a")
        self.assertEqual((text, score), ("b", 0.8))
        self.assertIsNone(nearest_neighbour(pool, [1.0, 0.0],
                                            "a"))
        # Negative scores are not neighbours.
        self.assertIsNone(nearest_neighbour(
            {"c": [-1.0, 0.0]}, [1.0, 0.0], "zzz"))

    def test_analogy_for(self):
        resonant = [("a", [1.0, 0.0]), ("b", [0.0, 1.0])]
        pool = {"a": [1.0, 0.0], "b": [0.0, 1.0],
                "d": [0.0, 1.0]}
        # Subject a: a + (b - a) = b, but b is skipped, so the
        # result is d (same vector as b, different text).
        result = analogy_for(resonant, pool, [1.0, 0.0], "a")
        self.assertEqual(result["a"], "a")
        self.assertEqual(result["b"], "b")
        self.assertEqual(result["result"][0], "d")
        # Fewer than two resonant entries: no analogy.
        self.assertIsNone(analogy_for(resonant[:1], pool,
                                      [1.0, 0.0], "a"))
        # No other entry to point at: no analogy.
        self.assertIsNone(analogy_for(
            resonant, {"a": [1.0, 0.0], "b": [0.0, 1.0]},
            [1.0, 0.0], "a"))

    def _pack_dict(self, rows):
        return {"name": "test-pack",
                "purpose": "Test pack.",
                "theme": "test theme",
                "rows": [({"prompt": "p%d" % i,
                           "completion": "c%d" % i},
                          {"kind": KIND_JUDGMENT,
                           "source": "f.txt",
                           "locator": "Step %d, line %d" % (i, i),
                           "resonance": 0.5,
                           "text": "c%d" % i})
                         for i in range(rows)],
                "excluded": [],
                "sources": ["f.txt"]}

    def test_export_pack_writes_three_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder, info = export_pack(self._pack_dict(3), tmp)
            self.assertTrue(os.path.isfile(
                os.path.join(folder, "data.jsonl")))
            self.assertTrue(os.path.isfile(
                os.path.join(folder, "README.md")))
            self.assertTrue(os.path.isfile(
                os.path.join(folder, "manifest.json")))
            self.assertEqual(info["items"], 3)
            self.assertFalse(info["under_minimum"])
            with open(os.path.join(folder, "manifest.json"),
                      encoding="utf-8") as fh:
                manifest = json.load(fh)
            self.assertEqual(manifest["pack"],
                             os.path.basename(folder))
            self.assertEqual(manifest["items"], 3)
            self.assertEqual(manifest["counts"],
                             {KIND_JUDGMENT: 3})
            self.assertEqual(manifest["extractors"],
                             ["ACR resonance analysis"])

    def test_export_pack_collision_gets_dash_two(self):
        with tempfile.TemporaryDirectory() as tmp:
            first, _info = export_pack(self._pack_dict(2), tmp)
            second, _info = export_pack(self._pack_dict(2), tmp)
            self.assertEqual(os.path.basename(first), "test-pack")
            self.assertEqual(os.path.basename(second),
                             "test-pack-2")

    def test_export_pack_under_minimum_warning(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder, info = export_pack(self._pack_dict(1), tmp)
            self.assertTrue(info["under_minimum"])
            with open(os.path.join(folder, "README.md"),
                      encoding="utf-8") as fh:
                readme = fh.read()
            self.assertIn("fewer than 2 rows", readme)
            self.assertIn("STELLA needs 2 or more rows", readme)
            self.assertNotIn("ready to import", readme)
        with tempfile.TemporaryDirectory() as tmp:
            folder, info = export_pack(self._pack_dict(2), tmp)
            self.assertFalse(info["under_minimum"])
            with open(os.path.join(folder, "README.md"),
                      encoding="utf-8") as fh:
                readme = fh.read()
            self.assertIn("ready to import into STELLA", readme)


class PackTests(BatchWindowCase):
    def _export(self, lines, name="fixture.txt"):
        path = self._fixture(name, lines)
        self.assertTrue(self._run_batch([path], name))
        packs_dir = os.path.join(self.tmp.name, "packs")
        result = self.window.analyze_window.\
            export_stella_packs_to(packs_dir)
        self.assertIsNone(result["error"])
        return packs_dir, result

    def _read_pack(self, folder):
        with open(os.path.join(folder, "data.jsonl"),
                  encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh]
        with open(os.path.join(folder, "manifest.json"),
                  encoding="utf-8") as fh:
            manifest = json.load(fh)
        with open(os.path.join(folder, "README.md"),
                  encoding="utf-8") as fh:
            readme = fh.read()
        return rows, manifest, readme

    def _theme_and_off_topic(self):
        packs_dir, result = self._export(
            [THEME[i % 4] for i in range(16)] + OFF)
        folders = sorted(os.listdir(packs_dir))
        self.assertEqual(len(folders), 2)
        theme_folder = [f for f in folders
                        if f.startswith(
                            "quantum-particles-stay-connected-")]
        off_folder = [f for f in folders
                      if f.startswith("fixture-off-topic-")]
        self.assertEqual(len(theme_folder), 1)
        self.assertEqual(len(off_folder), 1)
        return (os.path.join(packs_dir, theme_folder[0]),
                os.path.join(packs_dir, off_folder[0]))

    def test_one_pack_per_theme_plus_off_topic(self):
        theme_folder, off_folder = self._theme_and_off_topic()
        for folder in (theme_folder, off_folder):
            self.assertTrue(os.path.isfile(
                os.path.join(folder, "data.jsonl")))
            self.assertTrue(os.path.isfile(
                os.path.join(folder, "README.md")))
            self.assertTrue(os.path.isfile(
                os.path.join(folder, "manifest.json")))
        theme_rows, _m, _r = self._read_pack(theme_folder)
        off_rows, _m, _r = self._read_pack(off_folder)
        self.assertEqual(len(theme_rows), 12)
        self.assertEqual(len(off_rows), 9)

    def test_rows_are_prompt_completion_shaped(self):
        theme_folder, off_folder = self._theme_and_off_topic()
        for folder in (theme_folder, off_folder):
            rows, _m, _r = self._read_pack(folder)
            for row in rows:
                # STELLA's shape: prompt/completion, one format
                # per file, nothing else.
                self.assertEqual(set(row),
                                 {"prompt", "completion"})
                self.assertIsInstance(row["prompt"], str)
                self.assertIsInstance(row["completion"], str)
                self.assertTrue(row["prompt"])
                self.assertTrue(row["completion"])

    def test_theme_rows_assign_planted_theme(self):
        theme_folder, _off = self._theme_and_off_topic()
        rows, _m, _r = self._read_pack(theme_folder)
        membership = [row for row in rows
                      if row["prompt"].startswith(
                          "What theme does this belong to?")]
        self.assertEqual(len(membership), 3)
        for row in membership:
            # The theme label is the trigger text, verbatim.
            self.assertEqual(
                row["completion"],
                "quantum particles stay connected")
            subject = row["prompt"].split("? ", 1)[1]
            self.assertIn(subject, THEME)

    def test_closest_match_is_measured_neighbour(self):
        theme_folder, _off = self._theme_and_off_topic()
        rows, _m, _r = self._read_pack(theme_folder)
        matches = [row for row in rows
                   if row["prompt"].startswith(
                       "What is most related to this?")]
        self.assertEqual(len(matches), 3)
        # The expected neighbour, computed independently from
        # the engine's own buffer and cosine: the distinct text
        # with the highest positive score to the subject, never
        # the subject itself.
        context = self.window._pack_context()
        pool = context["pool"]
        fixture_texts = set(THEME) | set(OFF)
        for row in matches:
            subject = row["prompt"].split("? ", 1)[1]
            self.assertIn(row["completion"], fixture_texts)
            self.assertNotEqual(row["completion"], subject)
            vector = pool[subject]
            best_text = None
            best_score = 0.0
            for text, other in pool.items():
                if text == subject:
                    continue
                score = _dot(vector, other)
                if score > best_score:
                    best_score = score
                    best_text = text
            self.assertEqual(row["completion"], best_text)

    def test_analogy_rows_match_synthesis_head(self):
        # A manual 16-theme drive: the last input is a core entry,
        # so the pack analogy row for it must match the synthesis
        # head's own analogy exactly (same A, B, result).
        for i in range(16):
            self._step_text(THEME[i % 4])
        head = self.window.engine.synthesize()["analogy"]
        self.assertIsNotNone(head)
        packs_dir = os.path.join(self.tmp.name, "packs")
        result = self.window.analyze_window.\
            export_stella_packs_to(packs_dir)
        self.assertIsNone(result["error"])
        folders = os.listdir(packs_dir)
        # Two packs: the theme pack, plus the off-topic pack for
        # the step-14 dip entry (below the floor, so it belongs
        # to no theme). The analogy row lives in the theme pack.
        self.assertEqual(len(folders), 2)
        theme_folder = [f for f in folders
                        if f.startswith(
                            "quantum-particles-stay-connected-")]
        self.assertEqual(len(theme_folder), 1)
        rows, _m, _r = self._read_pack(
            os.path.join(packs_dir, theme_folder[0]))
        subject = THEME[15 % 4]
        analogies = [row for row in rows
                     if row["prompt"].endswith(
                         "as %s is to" % subject)]
        self.assertEqual(len(analogies), 1)
        row = analogies[0]
        self.assertEqual(row["prompt"],
                         "%s is to %s as %s is to"
                         % (head["a"], head["b"], subject))
        self.assertEqual(row["completion"],
                         head["result"][0])

    def test_no_analogy_rows_when_floor_not_met(self):
        packs_dir, _result = self._export([
            "quantum entanglement links particles",
            "the toaster needs a new heating element",
        ], "mixed.txt")
        self.assertLess(
            max(e["resonance"]
                for e in self.window.engine.trace),
            RESONANT_FLOOR)
        for folder in os.listdir(packs_dir):
            rows, manifest, readme = self._read_pack(
                os.path.join(packs_dir, folder))
            self.assertEqual(
                manifest["counts"].get(KIND_ANALOGY, 0), 0)
            self.assertFalse(any(
                " is to " in row["prompt"] and
                row["prompt"].endswith(" is to")
                for row in rows))
        # And the exclusion is listed, not silent.
        folders = os.listdir(packs_dir)
        _rows, _m, readme = self._read_pack(
            os.path.join(packs_dir, folders[0]))
        self.assertIn("No analogy row", readme)

    def test_outlier_judgments_off_topic(self):
        theme_folder, off_folder = self._theme_and_off_topic()
        theme_rows, _m, _r = self._read_pack(theme_folder)
        theme_judgments = [
            row for row in theme_rows
            if row["prompt"].startswith("Is this central")]
        self.assertEqual(len(theme_judgments), 3)
        for row in theme_judgments:
            self.assertEqual(
                row["completion"],
                "Central to quantum particles stay connected")
        off_rows, _m, _r = self._read_pack(off_folder)
        off_judgments = [
            row for row in off_rows
            if row["prompt"].startswith("Is this central")]
        self.assertEqual(len(off_judgments), 9)
        subjects = set()
        for row in off_judgments:
            self.assertEqual(row["completion"], "Off-topic")
            subjects.add(row["prompt"].split("? ", 1)[1])
        for line in OFF:
            self.assertIn(line, subjects)

    def test_manifest_counts_and_locators(self):
        theme_folder, off_folder = self._theme_and_off_topic()
        by_id = {e["id"]: e for e in self.window.engine.trace}
        for folder in (theme_folder, off_folder):
            rows, manifest, _r = self._read_pack(folder)
            self.assertEqual(manifest["items"], len(rows))
            self.assertEqual(sum(manifest["counts"].values()),
                             len(rows))
            self.assertEqual(len(manifest["entries"]), len(rows))
            kinds = {}
            for entry in manifest["entries"]:
                kinds[entry["kind"]] = \
                    kinds.get(entry["kind"], 0) + 1
            self.assertEqual(kinds, manifest["counts"])
            for entry in manifest["entries"]:
                self.assertEqual(
                    set(entry),
                    {"kind", "source", "locator", "resonance",
                     "text"})
                self.assertEqual(entry["source"], "fixture.txt")
                match = re.match(r"Step (\d+), line (\d+)",
                                 entry["locator"])
                self.assertIsNotNone(match, entry["locator"])
                step_id = int(match.group(1))
                line_no = int(match.group(2))
                trace_entry = by_id[step_id]
                self.assertEqual(entry["text"],
                                 self.window._entry_text(
                                     trace_entry))
                self.assertEqual(entry["resonance"],
                                 trace_entry["resonance"])
                # The line number is the entry's line in the
                # fixture file: step ids and line numbers agree
                # here because every line stepped.
                self.assertEqual(line_no, step_id)

    def test_readme_names_source_theme_and_promise(self):
        theme_folder, off_folder = self._theme_and_off_topic()
        _rows, _m, readme = self._read_pack(theme_folder)
        self.assertIn("fixture.txt", readme)
        self.assertIn("quantum particles stay connected", readme)
        self.assertIn("Extractor used: ACR resonance analysis",
                      readme)
        self.assertIn("Every entry is quoted verbatim", readme)
        self.assertIn("Nothing was invented", readme)
        self.assertIn("Theme membership: 3", readme)
        self.assertIn("Resonance judgment: 3", readme)
        _rows, _m, off_readme = self._read_pack(off_folder)
        self.assertIn("fixture.txt", off_readme)
        self.assertIn("## Excluded", off_readme)
        self.assertIn("No theme membership row", off_readme)

    def test_under_minimum_pack_written_with_warning(self):
        path = self._fixture("one.txt",
                             ["quantum entanglement links "
                              "particles"])
        self.assertTrue(self._run_batch([path], "one.txt"))
        packs_dir = os.path.join(self.tmp.name, "packs")
        result = self.window.analyze_window.\
            export_stella_packs_to(packs_dir)
        self.assertEqual(len(result["packs"]), 1)
        folder, rows, under = result["packs"][0]
        self.assertEqual(rows, 1)
        self.assertTrue(under)
        _rows, _m, readme = self._read_pack(folder)
        self.assertIn("fewer than 2 rows", readme)
        self.assertIn("STELLA needs 2 or more rows", readme)
        # And the window says so plainly too.
        with mock.patch(
                "acr_app.QFileDialog.getExistingDirectory",
                return_value=os.path.join(self.tmp.name,
                                          "packs2")):
            self.window.analyze_window.on_export_stella_packs()
        self.assertIn("fewer than 2 rows",
                      self.window.status_label.text())

    def test_pack_export_collision_gets_dash_two(self):
        packs_dir, first = self._export(
            [THEME[i % 4] for i in range(16)] + OFF)
        first_names = sorted(os.listdir(packs_dir))
        second = self.window.analyze_window.\
            export_stella_packs_to(packs_dir)
        self.assertIsNone(second["error"])
        second_names = sorted(os.listdir(packs_dir))
        self.assertEqual(len(second_names), 4)
        for name in first_names:
            self.assertIn(name + "-2", second_names)

    def test_pack_export_after_session_load(self):
        path = self._theme_fixture()
        self.assertTrue(self._run_batch([path]))
        session_path = os.path.join(self.tmp.name, "s.json")
        self.window.save_session_to(session_path)
        self.window.do_reset()
        self.assertEqual(self.window.step_label.text(), "Step: 0")
        self.window.load_session_from(session_path)
        self.assertEqual(self.window.step_label.text(),
                         "Step: 24")
        packs_dir = os.path.join(self.tmp.name, "packs")
        result = self.window.analyze_window.\
            export_stella_packs_to(packs_dir)
        self.assertIsNone(result["error"])
        folders = sorted(os.listdir(packs_dir))
        self.assertEqual(len(folders), 2)
        # Locators and the source name survived the round trip.
        off_folder = [f for f in folders
                      if f.startswith("fixture-off-topic-")]
        self.assertEqual(len(off_folder), 1)
        _rows, manifest, _r = self._read_pack(
            os.path.join(packs_dir, off_folder[0]))
        self.assertTrue(all(
            re.match(r"Step \d+, line \d+", entry["locator"])
            for entry in manifest["entries"]))
        self.assertEqual(manifest["sources"], ["fixture.txt"])


if __name__ == "__main__":
    unittest.main()
