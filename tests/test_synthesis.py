"""Synthesis head tests (v0.6): analogies, counter-examples, and
the optional local-model completion.

Engine tests run against the built-in encoder (deterministic).
Completion tests run against a fake local service (DUO's fake
Ollama shape, extended with /api/generate): /api/tags for the
model list, /api/embeddings for word-overlap vectors at dim 16,
/api/generate for a canned completion. No real local service runs
in the sandbox, so the live completion path is fake-service only.

Run with the Qt test env: LD_LIBRARY_PATH to the extracted libs,
QT_QPA_PLATFORM=offscreen, TMPDIR=/tmp, default HOME.
"""

import hashlib
import json
import math
import os
import random
import re
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

import local_encoder  # noqa: E402
from acr_app import MainWindow  # noqa: E402
from acr_engine import (RESONANT_FLOOR, Engine, _dot,  # noqa: E402
                        _normalize, encode)
from local_encoder import (MSG_MODEL_FAILED, MSG_SERVICE_DOWN,  # noqa: E402
                           MSG_SERVICE_SLOW, LocalModelClient,
                           LocalModelError, pick_completion_model)

APP = None

THEME = ["quantum entanglement links particles",
         "entangled particles share a state",
         "quantum particles stay connected",
         "entanglement links distant particles"]
MIXED = ["quantum entanglement links particles",
         "baking bread needs warm dough",
         "the river runs past the old mill",
         "a bicycle leans on the fence"]


def setUpModule():
    global APP
    APP = QApplication.instance()
    if APP is None:
        APP = QApplication(sys.argv)


def _drive_theme(engine, count, alpha=0.3):
    for i in range(count):
        engine.step(THEME[i % 4], alpha=alpha)


def _drive_mixed(engine, count, alpha=0.3):
    for i in range(count):
        engine.step(MIXED[i % 4], alpha=alpha)


def _fake_vector(text, dim):
    """Deterministic unit vector for a text, with word overlap:
    the mean of per-word hashed vectors, like the built-in
    encoder at a smaller dim. Theme sentences share words, so
    they share direction -- close enough to a real embedding
    model for the resonance climb to work."""
    words = re.findall(r"[a-z0-9']+", text.lower())
    if not words:
        return [0.0] * dim
    acc = [0.0] * dim
    for word in words:
        digest = hashlib.sha256(word.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "big")
        rng = random.Random(seed)
        vec = [rng.gauss(0.0, 1.0) for _ in range(dim)]
        norm = math.sqrt(sum(x * x for x in vec))
        for i in range(dim):
            acc[i] += vec[i] / norm
    norm = math.sqrt(sum(x * x for x in acc))
    return [x / norm for x in acc]


class FakeLocalHandler(BaseHTTPRequestHandler):
    """Fake local model service. Class attrs are the script."""

    models = ["nomic-embed-text:latest", "other:1b"]
    dim = 16
    # Drop the first N /api/embeddings requests (close, no reply).
    fail_embeddings_n = 0
    # Drop the first N /api/generate requests (close, no reply).
    fail_generate_n = 0
    # Drop every /api/generate request; /api/tags still answers.
    fail_generate_always = False
    # Answer /api/generate with a payload that has no response.
    generate_bad_response = False
    generate_response = "Canned completion from the fake service."
    embed_posts = 0
    generate_posts = 0
    generate_prompts = []
    generate_models = []
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
            self._send_json({"models": [{"name": n}
                                        for n in self.models]})
        else:
            self._send_json({"error": "not found"}, status=404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        if self.path == "/api/embeddings":
            self._embeddings(payload)
        elif self.path == "/api/generate":
            self._generate(payload)
        else:
            self._send_json({"error": "not found"}, status=404)

    def _embeddings(self, payload):
        type(self).embed_posts += 1
        if self.fail_embeddings_n > 0:
            type(self).fail_embeddings_n -= 1
            self.close_connection = True
            return
        model = payload.get("model")
        if model not in self.models:
            self._send_json(
                {"error": "model '%s' not found" % model}, status=404)
            return
        try:
            self._send_json(
                {"embedding": _fake_vector(payload.get("prompt", ""),
                                           self.dim)})
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _generate(self, payload):
        type(self).generate_posts += 1
        if (self.fail_generate_always
                or self.fail_generate_n > 0):
            if self.fail_generate_n > 0:
                type(self).fail_generate_n -= 1
            self.close_connection = True
            return
        model = payload.get("model")
        if model not in self.models:
            self._send_json(
                {"error": "model '%s' not found" % model}, status=404)
            return
        type(self).generate_prompts.append(payload.get("prompt", ""))
        type(self).generate_models.append(model)
        if self.generate_bad_response:
            self._send_json({"done": True})
            return
        try:
            self._send_json({"model": model,
                             "response": self.generate_response,
                             "done": True})
        except (BrokenPipeError, ConnectionResetError):
            pass


class FakeServerCase(unittest.TestCase):
    """Base: a fake service per test, fast retry delays."""

    MODELS = ["nomic-embed-text:latest", "other:1b"]
    DIM = 16

    def setUp(self):
        FakeLocalHandler.models = list(self.MODELS)
        FakeLocalHandler.dim = self.DIM
        FakeLocalHandler.fail_embeddings_n = 0
        FakeLocalHandler.fail_generate_n = 0
        FakeLocalHandler.fail_generate_always = False
        FakeLocalHandler.generate_bad_response = False
        FakeLocalHandler.generate_response = (
            "Canned completion from the fake service.")
        FakeLocalHandler.embed_posts = 0
        FakeLocalHandler.generate_posts = 0
        FakeLocalHandler.generate_prompts = []
        FakeLocalHandler.generate_models = []
        self.server = ThreadingHTTPServer(
            ("127.0.0.1", 0), FakeLocalHandler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(
            target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = ("http://127.0.0.1:%d"
                         % self.server.server_address[1])
        self._delays = mock.patch.object(
            local_encoder, "EMBED_RETRY_DELAYS", (0.01, 0.02))
        self._delays.start()
        self._gen_delays = mock.patch.object(
            local_encoder, "GENERATE_RETRY_DELAYS", (0.01, 0.02))
        self._gen_delays.start()

    def tearDown(self):
        self._gen_delays.stop()
        self._delays.stop()
        self.server.shutdown()
        self.server.server_close()

    def _dead_url(self):
        """A URL nothing answers on: this server's port, shut down."""
        self.server.shutdown()
        self.server.server_close()
        return self.base_url


# ---------------------------------------------------------------- engine


class ResonantEntriesTests(unittest.TestCase):
    def test_empty_engine_has_none(self):
        self.assertEqual(Engine().resonant_entries(), [])

    def test_theme_drive_has_resonant_entries(self):
        engine = Engine()
        _drive_theme(engine, 16)
        entries = engine.resonant_entries()
        self.assertGreaterEqual(len(entries), 2)
        for _text, _vec, res, _flag, _sid in entries:
            self.assertGreaterEqual(res, RESONANT_FLOOR)

    def test_layer_candidate_sorts_first(self):
        # Step 15 of the theme drive is the layer candidate
        # (resonance 0.4823); step 16 scores higher (0.5216) but
        # is not a candidate, so the candidate still leads.
        engine = Engine()
        _drive_theme(engine, 16)
        entries = engine.resonant_entries()
        self.assertTrue(entries[0][3])  # new_layer
        self.assertEqual(entries[0][0],
                         "quantum particles stay connected")
        self.assertEqual(entries[0][4], 15)

    def test_entries_are_distinct_texts(self):
        engine = Engine()
        _drive_theme(engine, 16)
        entries = engine.resonant_entries()
        texts = [entry[0] for entry in entries]
        self.assertEqual(len(texts), len(set(texts)))

    def test_mixed_drive_has_none(self):
        # A mixed drive never climbs past the floor (max 0.35).
        engine = Engine()
        _drive_mixed(engine, 8)
        self.assertEqual(engine.resonant_entries(), [])

    def test_count_caps(self):
        engine = Engine()
        _drive_theme(engine, 16)
        self.assertEqual(len(engine.resonant_entries(2)), 2)


class AnalogyTests(unittest.TestCase):
    def test_deterministic(self):
        first = Engine()
        second = Engine()
        _drive_theme(first, 16)
        _drive_theme(second, 16)
        self.assertEqual(first.synthesize()["analogy"],
                         second.synthesize()["analogy"])

    def test_uses_top_resonant_entries(self):
        engine = Engine()
        _drive_theme(engine, 16)
        analogy = engine.synthesize()["analogy"]
        self.assertIsNotNone(analogy)
        # A is the layer candidate (step 15), B the highest
        # resonance (step 16); both pinned by the probe numbers.
        self.assertEqual(analogy["a"],
                         "quantum particles stay connected")
        self.assertEqual(analogy["b"],
                         "entanglement links distant particles")
        self.assertNotEqual(analogy["a"], analogy["b"])

    def test_result_is_nearest_to_h_plus_d(self):
        # Recompute the construction independently from the
        # built-in encoder: the result must be the nearest
        # buffer entry to h + (h_B - h_A), skipping A, B, and
        # the current input's text.
        engine = Engine()
        _drive_theme(engine, 16)
        analogy = engine.synthesize()["analogy"]
        a = encode(analogy["a"])
        b = encode(analogy["b"])
        target = _normalize([h + (y - x) for h, x, y in
                             zip(engine.h_prev, a, b)])
        skip = {analogy["a"], analogy["b"],
                engine.buffer[-1]["text"]}
        best = None
        for item in engine.buffer:
            if item["text"] in skip:
                continue
            score = _dot(target, item["vector"])
            if best is None or score > best[0]:
                best = (score, item["text"])
        self.assertEqual(analogy["result"][0], best[1])
        self.assertAlmostEqual(analogy["result"][1], best[0],
                               places=3)

    def test_result_text_is_pinned(self):
        # The construction lands on a specific entry for the
        # theme drive; pin it so a silent change shows up.
        engine = Engine()
        _drive_theme(engine, 16)
        analogy = engine.synthesize()["analogy"]
        self.assertEqual(analogy["result"][0],
                         "quantum entanglement links particles")
        self.assertAlmostEqual(analogy["result"][1], 0.4696,
                               places=3)

    def test_none_when_too_few_resonant(self):
        engine = Engine()
        _drive_mixed(engine, 8)
        self.assertIsNone(engine.synthesize()["analogy"])
        engine2 = Engine()
        _drive_theme(engine2, 2)
        self.assertIsNone(engine2.synthesize()["analogy"])


class CounterExampleTests(unittest.TestCase):
    def test_farthest_entries_with_correct_cosines(self):
        engine = Engine()
        _drive_mixed(engine, 8)
        result = engine.synthesize()["counter_examples"]
        # Independent recompute: distinct texts except the
        # current input's, lowest cosine first, up to three.
        current = engine.buffer[-1]["text"]
        farthest = {}
        for item in engine.buffer[:-1]:
            if item["text"] == current:
                continue
            score = _dot(engine.h_prev, item["vector"])
            if (item["text"] not in farthest
                    or score < farthest[item["text"]]):
                farthest[item["text"]] = score
        expected = sorted(farthest.items(),
                          key=lambda kv: kv[1])[:3]
        self.assertEqual([t for t, _s in result],
                         [t for t, _s in expected])
        for (_t, score), (_t2, want) in zip(result, expected):
            self.assertAlmostEqual(score, want, places=3)

    def test_distinct_texts(self):
        engine = Engine()
        _drive_theme(engine, 16)
        result = engine.synthesize()["counter_examples"]
        self.assertEqual(len(result), 3)
        texts = [text for text, _score in result]
        self.assertEqual(len(texts), len(set(texts)))

    def test_empty_for_single_step(self):
        engine = Engine()
        engine.step("only one sentence")
        self.assertEqual(engine.synthesize()["counter_examples"],
                         [])

    def test_sorted_lowest_first(self):
        engine = Engine()
        _drive_mixed(engine, 8)
        result = engine.synthesize()["counter_examples"]
        scores = [score for _text, score in result]
        self.assertEqual(scores, sorted(scores))


class SynthesizeShapeTests(unittest.TestCase):
    def test_empty_before_steps(self):
        result = Engine().synthesize()
        self.assertEqual(result["neighbours"], [])
        self.assertEqual(result["midpoint"], [])
        self.assertIsNone(result["analogy"])
        self.assertEqual(result["counter_examples"], [])

    def test_neighbours_and_midpoint_unchanged(self):
        engine = Engine()
        engine.step("quantum entanglement links particles")
        engine.step("baking bread needs warm dough")
        engine.step("quantum particles stay connected")
        result = engine.synthesize()
        self.assertGreater(len(result["neighbours"]), 0)
        top_text, top_score = result["neighbours"][0]
        self.assertEqual(top_text,
                         "quantum particles stay connected")
        self.assertAlmostEqual(top_score, 1.0, places=3)


# ---------------------------------------------------------------- client


class GenerateClientTests(FakeServerCase):
    def test_generate_round_trip(self):
        client = LocalModelClient(self.base_url)
        text = client.generate("other:1b", "say something")
        self.assertEqual(
            text, "Canned completion from the fake service.")
        self.assertEqual(FakeLocalHandler.generate_posts, 1)
        self.assertEqual(FakeLocalHandler.generate_models,
                         ["other:1b"])
        self.assertEqual(FakeLocalHandler.generate_prompts,
                         ["say something"])

    def test_generate_unknown_model_no_retry(self):
        client = LocalModelClient(self.base_url)
        with self.assertRaises(LocalModelError) as ctx:
            client.generate("missing:1b", "hello")
        self.assertEqual(str(ctx.exception), MSG_MODEL_FAILED)
        self.assertEqual(FakeLocalHandler.generate_posts, 1)

    def test_generate_retries_then_succeeds(self):
        FakeLocalHandler.fail_generate_n = 2
        retries = []
        client = LocalModelClient(self.base_url)
        text = client.generate(
            "other:1b", "hello",
            on_retry=lambda *a: retries.append(a))
        self.assertEqual(
            text, "Canned completion from the fake service.")
        self.assertEqual(FakeLocalHandler.generate_posts, 3)
        self.assertEqual(len(retries), 2)
        self.assertEqual(retries[0][0], 1)
        self.assertEqual(retries[1][0], 2)

    def test_generate_exhausted_down_message(self):
        url = self._dead_url()
        client = LocalModelClient(url)
        with self.assertRaises(LocalModelError) as ctx:
            client.generate("other:1b", "hello")
        self.assertEqual(str(ctx.exception), MSG_SERVICE_DOWN)

    def test_generate_exhausted_slow_message(self):
        # Tags answers (probe says up) but generate always drops:
        # slow/busy, never "not running".
        FakeLocalHandler.fail_generate_always = True
        client = LocalModelClient(self.base_url)
        with self.assertRaises(LocalModelError) as ctx:
            client.generate("other:1b", "hello")
        self.assertEqual(str(ctx.exception), MSG_SERVICE_SLOW)
        self.assertEqual(FakeLocalHandler.generate_posts, 3)

    def test_generate_bad_response(self):
        FakeLocalHandler.generate_bad_response = True
        client = LocalModelClient(self.base_url)
        with self.assertRaises(LocalModelError) as ctx:
            client.generate("other:1b", "hello")
        self.assertIn("bad completion", str(ctx.exception))


class PickCompletionModelTests(unittest.TestCase):
    def test_prefers_non_encoder_model(self):
        self.assertEqual(
            pick_completion_model(
                ["nomic-embed-text:latest", "other:1b"],
                "nomic-embed-text:latest"),
            "other:1b")

    def test_first_alphabetical(self):
        self.assertEqual(
            pick_completion_model(["zzz:1b", "aaa:1b", "mmm:1b"],
                                  "mmm:1b"),
            "aaa:1b")

    def test_only_encoder_installed_picks_it(self):
        self.assertEqual(
            pick_completion_model(["nomic-embed-text:latest"],
                                  "nomic-embed-text:latest"),
            "nomic-embed-text:latest")

    def test_none_when_no_models(self):
        self.assertIsNone(pick_completion_model([], "any:1b"))


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

    def _step_mixed(self, count):
        for i in range(count):
            self._step_text(MIXED[i % 4])

    def _synthesis(self):
        return self.window.synthesis_text.toPlainText()


class SynthesisPanelTests(WindowCase):
    def test_panel_shows_analogy_and_counter_examples(self):
        self._step_theme(16)
        text = self._synthesis()
        self.assertIn("Analogy - like", text)
        self.assertIn("quantum particles stay connected", text)
        self.assertIn("entanglement links distant particles", text)
        self.assertIn("your input points at:", text)
        self.assertIn("quantum entanglement links particles", text)
        self.assertIn("Counter-examples - what this session is "
                      "not:", text)
        self.assertIn("entangled particles share a state", text)

    def test_panel_plain_message_when_too_few_resonant(self):
        self._step_mixed(2)
        text = self._synthesis()
        self.assertIn("Analogy: not enough resonant inputs yet",
                      text)
        self.assertNotIn("your input points at:", text)

    def test_analogy_comes_first(self):
        # The new sections sit above the neighbours, so they are
        # visible without scrolling.
        self._step_theme(16)
        text = self._synthesis()
        self.assertLess(text.index("Analogy - like"),
                        text.index("Closest past inputs:"))
        self.assertLess(text.index("Counter-examples"),
                        text.index("Closest past inputs:"))

    def test_panel_label_states_own_subspace(self):
        labels = [child.text() for child in self.window.findChildren(
            type(self.window.encoder_label))]
        self.assertTrue(any(
            "synthesis from this session's own subspace, not "
            "general knowledge" in text for text in labels))

    def test_completion_disabled_under_builtin(self):
        self.assertFalse(self.window.complete_button.isEnabled())
        self.assertFalse(self.window.completion_combo.isEnabled())
        self.assertEqual(self.window.completion_note.text(),
                         "Completion needs the Local model "
                         "encoder.")

    def test_panel_updates_each_step(self):
        self._step_mixed(2)
        self.assertIn("Analogy: not enough resonant inputs yet",
                      self._synthesis())
        self._step_theme(16)
        self.assertIn("your input points at:", self._synthesis())


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
        # v0.9: local steps and completions run in a worker
        # thread. Built-in steps are synchronous, so this returns
        # at once for them.
        deadline = time.time() + timeout
        while time.time() < deadline:
            if (self.window._local_thread is None
                    and self.window._batch_thread is None):
                return True
            QTest.qWait(50)
        return False

    def _step_text(self, text):
        self.window.input_edit.setPlainText(text)
        QTest.mouseClick(self.window.step_button, Qt.LeftButton)
        self.assertTrue(self._wait_idle(), "local step did not land")

    def _step_theme(self, count):
        for i in range(count):
            self._step_text(THEME[i % 4])

    def _switch_local(self):
        # v0.9: picking "Local model" only OFFERS the models. The
        # encoder stays built-in until a model is picked (U39+U6).
        self.window.encoder_combo.setCurrentIndex(1)

    def _pick_model(self, name):
        combo = self.window.model_combo
        index = combo.findData(name)
        self.assertNotEqual(index, -1, "model %r not listed" % name)
        combo.setCurrentIndex(index)

    def _pick_completion(self, name):
        combo = self.window.completion_combo
        index = combo.findData(name)
        self.assertNotEqual(index, -1,
                            "model %r not listed" % name)
        combo.setCurrentIndex(index)

    def _go_local(self, name="nomic-embed-text:latest"):
        self._switch_local()
        self._pick_model(name)

    def _complete(self):
        QTest.mouseClick(self.window.complete_button,
                         Qt.LeftButton)
        self.assertTrue(self._wait_idle(),
                        "completion did not land")

    def _synthesis(self):
        return self.window.synthesis_text.toPlainText()


class CompletionPickerTests(LocalWindowCase):
    def test_picker_lists_all_models_unselected(self):
        # v0.9: nothing is picked for you (U6). The picker lists
        # every installed model; the button enables once a model
        # is picked.
        self._go_local()
        self.assertEqual(self.window.local_model,
                         "nomic-embed-text:latest")
        combo = self.window.completion_combo
        items = [combo.itemData(i) for i in range(combo.count())]
        self.assertEqual(items, ["nomic-embed-text:latest",
                                 "other:1b"])
        self.assertEqual(combo.currentIndex(), -1)
        self.assertTrue(combo.isEnabled())
        self.assertFalse(self.window.complete_button.isEnabled())
        self.assertIn("Pick a model",
                      self.window.completion_note.text())
        self._pick_completion("other:1b")
        self.assertTrue(self.window.complete_button.isEnabled())

    def test_picker_refreshes_when_encoder_model_changes(self):
        self._go_local()
        self._pick_completion("other:1b")
        self._pick_model("other:1b")
        self.assertEqual(self.window.local_model, "other:1b")
        combo = self.window.completion_combo
        items = [combo.itemData(i) for i in range(combo.count())]
        self.assertEqual(items, ["nomic-embed-text:latest",
                                 "other:1b"])
        # The standing pick is kept: the picker is independent of
        # the encoder choice.
        self.assertEqual(combo.currentData(), "other:1b")
        self.assertTrue(self.window.complete_button.isEnabled())

    def test_only_encoder_installed_still_completes(self):
        FakeLocalHandler.models = ["nomic-embed-text:latest"]
        self._go_local()
        self.assertEqual(self.window.encoder_kind, "local")
        combo = self.window.completion_combo
        items = [combo.itemData(i) for i in range(combo.count())]
        self.assertEqual(items, ["nomic-embed-text:latest"])
        self.assertEqual(combo.currentIndex(), -1)
        self.assertTrue(combo.isEnabled())
        self.assertFalse(self.window.complete_button.isEnabled())
        self._pick_completion("nomic-embed-text:latest")
        self.assertTrue(self.window.complete_button.isEnabled())
        self.assertIn("Pick a model",
                      self.window.completion_note.text())

    def test_back_to_builtin_disables_completion(self):
        self._go_local()
        self._pick_completion("other:1b")
        self.assertTrue(self.window.complete_button.isEnabled())
        self.window.encoder_combo.setCurrentIndex(0)
        self.assertFalse(self.window.complete_button.isEnabled())
        self.assertEqual(self.window.completion_combo.count(), 0)
        self.assertEqual(self.window.completion_note.text(),
                         "Completion needs the Local model "
                         "encoder.")


class CompletionTests(LocalWindowCase):
    def test_complete_shows_local_model_words(self):
        self._go_local()
        self._pick_completion("other:1b")
        self._step_theme(16)
        self._complete()
        text = self._synthesis()
        self.assertIn("Local model (other:1b) said:", text)
        self.assertIn("Canned completion from the fake service.",
                      text)
        self.assertIn("Completion from other:1b.",
                      self.window.status_label.text())
        self.assertEqual(FakeLocalHandler.generate_posts, 1)
        self.assertEqual(FakeLocalHandler.generate_models,
                         ["other:1b"])

    def test_complete_scrolls_panel_to_completion(self):
        # The completion sits at the bottom of the panel: after
        # completing, the panel is scrolled to it, not left on
        # the neighbours at the top.
        self._go_local()
        self._pick_completion("other:1b")
        self._step_theme(16)
        self._complete()
        bar = self.window.synthesis_text.verticalScrollBar()
        self.assertGreater(bar.maximum(), 0)
        self.assertEqual(bar.value(), bar.maximum())

    def test_complete_prompt_has_resonant_context(self):
        self._go_local()
        self._pick_completion("other:1b")
        self._step_theme(16)
        self._complete()
        self.assertEqual(len(FakeLocalHandler.generate_prompts), 1)
        prompt = FakeLocalHandler.generate_prompts[0]
        self.assertIn("most resonant inputs", prompt)
        # At least one resonant theme sentence is in the context.
        self.assertTrue(any(sentence in prompt
                            for sentence in THEME))

    def test_complete_needs_resonant_entries(self):
        self._go_local()
        self._pick_completion("other:1b")
        self._step_theme(2)
        QTest.mouseClick(self.window.complete_button,
                         Qt.LeftButton)
        self.assertIn("No resonant inputs yet",
                      self.window.status_label.text())
        self.assertEqual(FakeLocalHandler.generate_posts, 0)
        self.assertNotIn("Local model",
                         self._synthesis())

    def test_complete_service_down_plain_message(self):
        self._go_local()
        self._pick_completion("other:1b")
        self._step_theme(16)
        self.server.shutdown()
        self.server.server_close()
        self._complete()
        self.assertIn("not running",
                      self.window.status_label.text())
        # A failed completion never falls back: the local session
        # is still there, untouched.
        self.assertEqual(self.window.encoder_kind, "local")
        self.assertEqual(self.window.step_label.text(), "Step: 16")
        self.assertNotIn("Local model", self._synthesis())

    def test_completion_clears_on_reset(self):
        self._go_local()
        self._pick_completion("other:1b")
        self._step_theme(16)
        self._complete()
        self.assertIn("Canned completion", self._synthesis())
        self.window.do_reset()
        self.assertNotIn("Canned completion", self._synthesis())

    def test_completion_stays_across_steps(self):
        # The completion is the local model's words about the
        # session; later steps refresh the neighbours but do not
        # wipe it. A reset or a load does.
        self._go_local()
        self._pick_completion("other:1b")
        self._step_theme(16)
        self._complete()
        self._step_text("quantum entanglement links particles")
        self.assertIn("Canned completion", self._synthesis())


if __name__ == "__main__":
    unittest.main()
