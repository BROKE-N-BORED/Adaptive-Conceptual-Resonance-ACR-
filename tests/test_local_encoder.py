"""Local-model encoder tests (v0.5): fake local service, client,
engine dim, encoder switching, Start service, session versions.

The fake service is a small HTTP server in the tests (DUO's fake
Ollama shape): /api/tags for the model list, /api/embeddings for
deterministic vectors. No real local service runs in the sandbox,
so the live path is fake-service only.

Run with the Qt test env: LD_LIBRARY_PATH to the extracted libs,
QT_QPA_PLATFORM=offscreen, TMPDIR=/tmp, default HOME.
"""

import hashlib
import json
import math
import os
import random
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
from acr_app import MainWindow, _row_change_norms  # noqa: E402
from acr_engine import (DIM, RANK, Engine, SessionError,  # noqa: E402
                        _dominant_direction, _dominant_direction_uv,
                        _frobenius, _frobenius_uv, _materialize_w)
from local_encoder import (EMBED_MODEL_DEFAULT, MSG_MODEL_FAILED,  # noqa: E402
                           MSG_NO_MODELS, MSG_SERVICE_DOWN,
                           MSG_SERVICE_SLOW, LocalModelClient,
                           LocalModelError, pick_default_model)

APP = None


def setUpModule():
    global APP
    APP = QApplication.instance()
    if APP is None:
        APP = QApplication(sys.argv)


def _fake_vector(text, dim):
    """Deterministic unit vector for a text, like an embedding."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "big")
    rng = random.Random(seed)
    vec = [rng.gauss(0.0, 1.0) for _ in range(dim)]
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec]


class FakeLocalHandler(BaseHTTPRequestHandler):
    """Fake local model service. Class attrs are the script."""

    models = ["nomic-embed-text:latest", "other:1b"]
    dim = 16
    # Drop the first N /api/embeddings requests (close, no reply).
    fail_embeddings_n = 0
    # Drop every /api/embeddings request; /api/tags still answers.
    fail_embeddings_always = False
    # Seconds to wait before answering /api/embeddings.
    embed_delay = 0.0
    embed_posts = 0
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
        if self.path != "/api/embeddings":
            self._send_json({"error": "not found"}, status=404)
            return
        length = int(self.headers.get("Content-Length") or 0)
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        type(self).embed_posts += 1
        if (self.fail_embeddings_always
                or self.fail_embeddings_n > 0):
            if self.fail_embeddings_n > 0:
                type(self).fail_embeddings_n -= 1
            # Drop: close without a reply, like a cut connection.
            self.close_connection = True
            return
        model = payload.get("model")
        if model not in self.models:
            self._send_json(
                {"error": "model '%s' not found" % model}, status=404)
            return
        if self.embed_delay:
            time.sleep(self.embed_delay)
        try:
            self._send_json(
                {"embedding": _fake_vector(payload.get("prompt", ""),
                                           self.dim)})
        except (BrokenPipeError, ConnectionResetError):
            pass  # Client gave up (timeout test); expected.


class FakeServerCase(unittest.TestCase):
    """Base: a fake service per test, fast retry delays."""

    MODELS = ["nomic-embed-text:latest", "other:1b"]
    DIM = 16

    def setUp(self):
        FakeLocalHandler.models = list(self.MODELS)
        FakeLocalHandler.dim = self.DIM
        FakeLocalHandler.fail_embeddings_n = 0
        FakeLocalHandler.fail_embeddings_always = False
        FakeLocalHandler.embed_delay = 0.0
        FakeLocalHandler.embed_posts = 0
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

    def tearDown(self):
        self._delays.stop()
        self.server.shutdown()
        self.server.server_close()

    def _dead_url(self):
        """A URL nothing answers on: this server's port, shut down."""
        self.server.shutdown()
        self.server.server_close()
        return self.base_url


# ---------------------------------------------------------------- client


class ClientTests(FakeServerCase):
    def test_probe_up(self):
        self.assertTrue(LocalModelClient(self.base_url).probe())

    def test_probe_down(self):
        url = self._dead_url()
        self.assertFalse(LocalModelClient(url).probe(timeout=1))

    def test_list_models(self):
        names = LocalModelClient(self.base_url).list_models()
        self.assertEqual(names, sorted(self.MODELS))

    def test_list_models_down(self):
        url = self._dead_url()
        with self.assertRaises(LocalModelError) as ctx:
            LocalModelClient(url).list_models(timeout=1)
        self.assertEqual(str(ctx.exception), MSG_SERVICE_DOWN)

    def test_embed_round_trip(self):
        client = LocalModelClient(self.base_url)
        vec = client.embed("nomic-embed-text:latest", "hello world")
        self.assertEqual(len(vec), self.DIM)
        self.assertAlmostEqual(
            math.sqrt(sum(x * x for x in vec)), 1.0, places=6)
        # Same text, same vector; different text, different vector.
        again = client.embed("nomic-embed-text:latest", "hello world")
        self.assertEqual(vec, again)
        other = client.embed("nomic-embed-text:latest", "goodbye")
        self.assertNotEqual(vec, other)

    def test_embed_unknown_model_no_retry(self):
        client = LocalModelClient(self.base_url)
        with self.assertRaises(LocalModelError) as ctx:
            client.embed("missing:1b", "hello")
        self.assertEqual(str(ctx.exception), MSG_MODEL_FAILED)
        # A missing model never fixes itself: exactly one request.
        self.assertEqual(FakeLocalHandler.embed_posts, 1)

    def test_embed_retries_then_succeeds(self):
        FakeLocalHandler.fail_embeddings_n = 2
        retries = []
        client = LocalModelClient(self.base_url)
        vec = client.embed("nomic-embed-text:latest", "hello",
                           on_retry=lambda *a: retries.append(a))
        self.assertEqual(len(vec), self.DIM)
        self.assertEqual(FakeLocalHandler.embed_posts, 3)
        self.assertEqual(len(retries), 2)
        self.assertEqual(retries[0][0], 1)  # first failed attempt
        self.assertEqual(retries[1][0], 2)

    def test_embed_exhausted_down_message(self):
        url = self._dead_url()
        client = LocalModelClient(url)
        with self.assertRaises(LocalModelError) as ctx:
            client.embed("nomic-embed-text:latest", "hello")
        self.assertEqual(str(ctx.exception), MSG_SERVICE_DOWN)

    def test_embed_exhausted_slow_message(self):
        # Tags answers (probe says up) but embeddings always drop:
        # slow/busy, never "not running".
        FakeLocalHandler.fail_embeddings_always = True
        client = LocalModelClient(self.base_url)
        with self.assertRaises(LocalModelError) as ctx:
            client.embed("nomic-embed-text:latest", "hello")
        self.assertEqual(str(ctx.exception), MSG_SERVICE_SLOW)
        self.assertEqual(FakeLocalHandler.embed_posts, 3)

    def test_embed_timeout_is_slow(self):
        FakeLocalHandler.embed_delay = 0.5
        kinds = []
        client = LocalModelClient(self.base_url)
        with self.assertRaises(LocalModelError) as ctx:
            client.embed("nomic-embed-text:latest", "hello",
                         timeout=0.05,
                         on_retry=lambda *a: kinds.append(a[2]))
        self.assertEqual(str(ctx.exception), MSG_SERVICE_SLOW)
        self.assertEqual(kinds, ["slow", "slow"])

    def test_parse_tags(self):
        from local_encoder import parse_tags_payload
        payload = {"models": [{"name": "b:2b"}, {"name": "a:2b"},
                              {"name": ""}, {}]}
        self.assertEqual(parse_tags_payload(payload), ["a:2b", "b:2b"])
        self.assertEqual(parse_tags_payload({}), [])
        self.assertEqual(parse_tags_payload({"models": None}), [])

    def test_pick_default_model(self):
        self.assertEqual(
            pick_default_model(["zzz:1b", "nomic-embed-text:latest",
                                "aaa:1b"]),
            "nomic-embed-text:latest")
        self.assertEqual(
            pick_default_model([EMBED_MODEL_DEFAULT, "aaa:1b"]),
            EMBED_MODEL_DEFAULT)
        self.assertEqual(pick_default_model(["bbb:1b", "aaa:1b"]),
                         "aaa:1b")
        self.assertIsNone(pick_default_model([]))


# ---------------------------------------------------------------- engine


class EngineDimTests(unittest.TestCase):
    def test_engine_dim_configurable(self):
        engine = Engine(dim=16)
        self.assertEqual(engine.dim, 16)
        self.assertEqual(len(engine.u), 16)
        self.assertEqual(len(engine.u[0]), RANK)

    def test_step_with_supplied_embedding(self):
        engine = Engine(dim=16)
        entry = engine.step("hello", h=_fake_vector("hello", 16))
        self.assertEqual(entry["id"], 1)
        self.assertEqual(engine.step_count, 1)

    def test_step_without_h_at_other_dim_raises(self):
        engine = Engine(dim=16)
        with self.assertRaises(ValueError):
            engine.step("hello")

    def test_step_wrong_dim_h_raises(self):
        engine = Engine(dim=16)
        with self.assertRaises(ValueError):
            engine.step("hello", h=_fake_vector("hello", 8))

    def test_state_round_trip_at_dim_16(self):
        engine = Engine(dim=16)
        engine.step("first", h=_fake_vector("first", 16))
        engine.step("second", h=_fake_vector("second", 16))
        state = engine.state_dict()
        self.assertEqual(state["dim"], 16)
        other = Engine(dim=16)
        other.load_state_dict(state)
        self.assertAlmostEqual(other.w_norm(), engine.w_norm())
        self.assertEqual(other.trace, engine.trace)
        self.assertEqual(other.buffer, engine.buffer)
        self.assertEqual(other.step_count, 2)

    def test_load_rejects_wrong_dim_state(self):
        engine = Engine(dim=16)
        engine.step("first", h=_fake_vector("first", 16))
        state = engine.state_dict()
        other = Engine()  # dim 64
        with self.assertRaises(SessionError):
            other.load_state_dict(state)

    def test_load_rejects_dim_mismatch_in_state(self):
        engine = Engine(dim=16)
        state = engine.state_dict()
        state["dim"] = 32  # factors are still dim 16
        other = Engine(dim=16)
        with self.assertRaises(SessionError):
            other.load_state_dict(state)

    def test_fast_norm_matches_naive(self):
        for dim in (64, 128):
            rng = random.Random(7)
            u = [[rng.gauss(0, 0.1) for _ in range(RANK)]
                 for _ in range(dim)]
            v = [[rng.gauss(0, 0.1) for _ in range(RANK)]
                 for _ in range(dim)]
            naive = _frobenius(_materialize_w(u, v))
            fast = _frobenius_uv(u, v)
            self.assertAlmostEqual(fast, naive, places=9)

    def test_fast_direction_matches_naive(self):
        rng = random.Random(7)
        u = [[rng.gauss(0, 0.1) for _ in range(RANK)]
             for _ in range(DIM)]
        v = [[rng.gauss(0, 0.1) for _ in range(RANK)]
             for _ in range(DIM)]
        w = _materialize_w(u, v)
        d_naive, lam_naive = _dominant_direction(w)
        d_fast, lam_fast = _dominant_direction_uv(u, v)
        cos = abs(sum(a * b for a, b in zip(d_naive, d_fast)))
        self.assertAlmostEqual(cos, 1.0, places=9)
        self.assertAlmostEqual(lam_fast, lam_naive, places=6)

    def test_fast_direction_is_eigenvector_at_dim_768(self):
        # At dim 768 the naive 30-iteration form under-converges,
        # so check the fast direction directly: it must satisfy
        # W^T W v = lambda v, computed low-rank.
        rng = random.Random(7)
        dim = 768
        u = [[rng.gauss(0, 0.1) for _ in range(RANK)]
             for _ in range(dim)]
        v = [[rng.gauss(0, 0.1) for _ in range(RANK)]
             for _ in range(dim)]
        vec, lam = _dominant_direction_uv(u, v)
        self.assertGreater(lam, 0.0)
        # W^T W v = V (U^T (U (V^T v))).
        from acr_engine import _matvec, _norm, _transpose
        wtw_v = _matvec(
            v, _matvec(_transpose(u),
                       _matvec(u, _matvec(_transpose(v), vec))))
        resid = _norm([a - lam * b for a, b in zip(wtw_v, vec)])
        self.assertLess(resid / lam, 1e-3)

    def test_row_change_norms_match_naive(self):
        rng = random.Random(11)
        dim = 32
        old_u = [[rng.gauss(0, 0.1) for _ in range(RANK)]
                 for _ in range(dim)]
        old_v = [[rng.gauss(0, 0.1) for _ in range(RANK)]
                 for _ in range(dim)]
        new_u = [[x + rng.gauss(0, 0.01) for x in row]
                 for row in old_u]
        new_v = [[x + rng.gauss(0, 0.01) for x in row]
                 for row in old_v]
        fast = _row_change_norms(old_u, old_v, new_u, new_v)
        self.assertEqual(len(fast), dim)
        for i in range(dim):
            total = 0.0
            for j in range(dim):
                old = sum(old_u[i][k] * old_v[j][k]
                          for k in range(RANK))
                new = sum(new_u[i][k] * new_v[j][k]
                          for k in range(RANK))
                total += (new - old) ** 2
            self.assertAlmostEqual(fast[i], math.sqrt(total),
                                   places=9)
        # Identical factors: no row moved at all.
        zeros = _row_change_norms(old_u, old_v, old_u, old_v)
        self.assertEqual(zeros, [0.0] * dim)


# ---------------------------------------------------------------- window


class WindowCase(FakeServerCase):
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

    def _step_text(self, text):
        self.window.input_edit.setPlainText(text)
        QTest.mouseClick(self.window.step_button, Qt.LeftButton)

    def _step_theme(self, count):
        theme = ["quantum entanglement links particles",
                 "entangled particles share a state",
                 "quantum particles stay connected",
                 "entanglement links distant particles"]
        for i in range(count):
            self._step_text(theme[i % 4])

    def _switch_local(self):
        self.window.encoder_combo.setCurrentIndex(1)

    def _path(self, name):
        return os.path.join(self.tmp.name, name)


class EncoderSwitchTests(WindowCase):
    def test_default_is_builtin(self):
        self.assertEqual(self.window.encoder_kind, "builtin")
        self.assertEqual(self.window.encoder_combo.currentIndex(), 0)
        self.assertFalse(self.window.model_combo.isEnabled())
        self.assertEqual(self.window.encoder_label.text(),
                         "Built-in encoder, dim 64.")
        self.assertEqual(self.window.engine.dim, DIM)

    def test_switch_note_states_fresh_w(self):
        labels = [child.text() for child in self.window.findChildren(
            type(self.window.encoder_label))]
        self.assertTrue(any(
            "Switching encoders starts a fresh W_t" in text
            for text in labels))

    def test_switch_to_local(self):
        self._switch_local()
        self.assertEqual(self.window.encoder_kind, "local")
        self.assertEqual(self.window.local_model,
                         "nomic-embed-text:latest")
        self.assertEqual(self.window.engine.dim, self.DIM)
        self.assertTrue(self.window.model_combo.isEnabled())
        self.assertEqual(
            self.window.model_combo.currentText(),
            "nomic-embed-text:latest")
        self.assertIn("Local encoder", self.window.status_label.text())
        self.assertIn("fresh", self.window.status_label.text())
        self.assertIn("nomic-embed-text:latest",
                      self.window.encoder_label.text())
        self.assertIn("dim 16", self.window.encoder_label.text())

    def test_switch_to_local_picks_first_without_embed_model(self):
        FakeLocalHandler.models = ["zzz:1b", "aaa:1b"]
        self._switch_local()
        self.assertEqual(self.window.local_model, "aaa:1b")

    def test_switch_to_local_no_models(self):
        FakeLocalHandler.models = []
        self._switch_local()
        self.assertEqual(self.window.encoder_kind, "builtin")
        self.assertEqual(self.window.encoder_combo.currentIndex(), 0)
        self.assertEqual(self.window.status_label.text(), MSG_NO_MODELS)

    def test_switch_to_local_service_down(self):
        url = self._dead_url()
        window = MainWindow(log_dir=self.tmp.name, local_base_url=url)
        window.show()
        try:
            window.encoder_combo.setCurrentIndex(1)
            self.assertEqual(window.encoder_kind, "builtin")
            self.assertEqual(window.encoder_combo.currentIndex(), 0)
            self.assertIn("not running",
                          window.status_label.text())
        finally:
            window.close()
            window.deleteLater()

    def test_switch_resets_w_and_clears(self):
        self._step_theme(3)
        self.assertEqual(self.window.trace_table.rowCount(), 3)
        self._switch_local()
        self.assertEqual(self.window.step_label.text(), "Step: 0")
        self.assertEqual(self.window.trace_table.rowCount(), 0)
        self.assertEqual(self.window.buffer_list.count(), 0)
        self.assertEqual(len(self.window.norm_chart.values), 0)
        # And back again: fresh again, dim 64.
        self._step_text("local step one")
        self.assertEqual(self.window.step_label.text(), "Step: 1")
        self.window.encoder_combo.setCurrentIndex(0)
        self.assertEqual(self.window.encoder_kind, "builtin")
        self.assertEqual(self.window.engine.dim, DIM)
        self.assertEqual(self.window.step_label.text(), "Step: 0")
        self.assertEqual(self.window.trace_table.rowCount(), 0)

    def test_model_change_resets_w(self):
        self._switch_local()
        self._step_theme(2)
        self.assertEqual(self.window.step_label.text(), "Step: 2")
        self.window.model_combo.setCurrentText("other:1b")
        self.assertEqual(self.window.local_model, "other:1b")
        self.assertEqual(self.window.step_label.text(), "Step: 0")
        self.assertEqual(self.window.trace_table.rowCount(), 0)

    def test_step_with_local(self):
        self._switch_local()
        self._step_theme(3)
        self.assertEqual(self.window.step_label.text(), "Step: 3")
        self.assertEqual(self.window.trace_table.rowCount(), 3)
        self.assertEqual(len(self.window.heatmap.values), self.DIM)
        self.assertEqual(len(self.window.norm_chart.values), 3)

    def test_mid_session_failure_falls_back(self):
        self._switch_local()
        self._step_text("local step one")
        self.assertEqual(self.window.step_label.text(), "Step: 1")
        self.server.shutdown()
        self.server.server_close()
        self._step_text("local step two")
        self.assertEqual(self.window.encoder_kind, "builtin")
        self.assertEqual(self.window.engine.dim, DIM)
        self.assertEqual(self.window.encoder_combo.currentIndex(), 0)
        # The step still ran, on a fresh built-in W_t.
        self.assertEqual(self.window.step_label.text(), "Step: 1")
        self.assertEqual(self.window.trace_table.rowCount(), 1)
        self.assertIn("stopped", self.window.status_label.text())
        self.assertIn("built-in", self.window.status_label.text())


class StartServiceTests(WindowCase):
    def test_start_when_running_says_so(self):
        with mock.patch(
                "acr_app.QProcess.startDetached") as detached:
            QTest.mouseClick(self.window.start_button, Qt.LeftButton)
            detached.assert_not_called()
        self.assertIn("is running", self.window.status_label.text())

    def test_start_when_down_starts_it(self):
        url = self._dead_url()
        window = MainWindow(log_dir=self.tmp.name, local_base_url=url)
        window.show()
        try:
            probe = mock.Mock(side_effect=[False, True])
            with mock.patch.object(window.client, "probe", probe), \
                    mock.patch("acr_app.shutil.which",
                               return_value="/usr/bin/ollama"), \
                    mock.patch("acr_app.QProcess.startDetached",
                               return_value=(True, 123)) as detached:
                QTest.mouseClick(window.start_button, Qt.LeftButton)
            detached.assert_called_once_with("ollama", ["serve"])
            self.assertIn("is running", window.status_label.text())
        finally:
            window.close()
            window.deleteLater()

    def test_start_not_installed(self):
        url = self._dead_url()
        window = MainWindow(log_dir=self.tmp.name, local_base_url=url)
        window.show()
        try:
            with mock.patch("acr_app.shutil.which",
                            return_value=None), \
                    mock.patch(
                        "acr_app.QProcess.startDetached") as detached:
                QTest.mouseClick(window.start_button, Qt.LeftButton)
            detached.assert_not_called()
            self.assertIn("not installed",
                          window.status_label.text())
        finally:
            window.close()
            window.deleteLater()

    def test_start_failed(self):
        url = self._dead_url()
        window = MainWindow(log_dir=self.tmp.name, local_base_url=url)
        window.show()
        try:
            with mock.patch("acr_app.shutil.which",
                            return_value="/usr/bin/ollama"), \
                    mock.patch("acr_app.QProcess.startDetached",
                               return_value=(False, 0)):
                QTest.mouseClick(window.start_button, Qt.LeftButton)
            self.assertIn("did not start", window.status_label.text())
        finally:
            window.close()
            window.deleteLater()

    def test_start_timeout_when_service_never_answers(self):
        url = self._dead_url()
        window = MainWindow(log_dir=self.tmp.name, local_base_url=url)
        window.show()
        try:
            with mock.patch("acr_app.shutil.which",
                            return_value="/usr/bin/ollama"), \
                    mock.patch("acr_app.QProcess.startDetached",
                               return_value=(True, 123)):
                ok, message = window.service.start(timeout=0.5)
            self.assertFalse(ok)
            self.assertIn("did not start", message)
        finally:
            window.close()
            window.deleteLater()


class SessionVersionTests(WindowCase):
    def test_save_file_has_encoder(self):
        self._step_theme(2)
        path = self._path("session.json")
        self.window.save_session_to(path)
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        self.assertEqual(data["version"], 2)
        self.assertEqual(data["encoder"],
                         {"kind": "builtin", "model": None, "dim": 64})
        self.assertEqual(data["engine"]["dim"], 64)

    def test_v1_loads_as_builtin(self):
        self._step_theme(3)
        data = self.window.session_dict()
        data["version"] = 1
        del data["encoder"]
        del data["engine"]["dim"]  # v0.4 engine states had no dim
        path = self._path("v1.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh)
        self.window.do_reset()
        self.window.load_session_from(path)
        self.assertEqual(self.window.encoder_kind, "builtin")
        self.assertEqual(self.window.engine.dim, DIM)
        self.assertEqual(self.window.step_label.text(), "Step: 3")
        self.assertEqual(self.window.trace_table.rowCount(), 3)

    def test_v2_local_round_trip(self):
        self._switch_local()
        self._step_theme(3)
        before = (self.window.step_label.text(),
                  self.window.norm_label.text(),
                  self.window.resonance_label.text())
        path = self._path("local.json")
        self.window.save_session_to(path)
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        self.assertEqual(data["encoder"]["kind"], "local")
        self.assertEqual(data["encoder"]["model"],
                         "nomic-embed-text:latest")
        self.assertEqual(data["encoder"]["dim"], self.DIM)
        self.window.do_reset()
        self.assertEqual(self.window.step_label.text(), "Step: 0")
        self.window.load_session_from(path)
        self.assertEqual(self.window.encoder_kind, "local")
        self.assertEqual(self.window.local_model,
                         "nomic-embed-text:latest")
        self.assertEqual(self.window.engine.dim, self.DIM)
        after = (self.window.step_label.text(),
                 self.window.norm_label.text(),
                 self.window.resonance_label.text())
        self.assertEqual(after, before)
        self.assertEqual(self.window.trace_table.rowCount(), 3)
        self.assertEqual(len(self.window.heatmap.values), self.DIM)

    def test_v2_local_refuses_when_service_down(self):
        self._switch_local()
        self._step_theme(2)
        path = self._path("local.json")
        self.window.save_session_to(path)
        self.server.shutdown()
        self.server.server_close()
        # Live session: two builtin steps to prove it stays put.
        self.window.encoder_combo.setCurrentIndex(0)
        self._step_theme(2)
        with self.assertRaises(SessionError) as ctx:
            self.window.load_session_from(path)
        self.assertIn("needs the local encoder",
                      str(ctx.exception))
        self.assertIn("not running", str(ctx.exception))
        self.assertEqual(self.window.step_label.text(), "Step: 2")
        self.assertEqual(self.window.encoder_kind, "builtin")

    def test_v2_local_refuses_when_model_missing(self):
        self._switch_local()
        self._step_theme(2)
        path = self._path("local.json")
        self.window.save_session_to(path)
        FakeLocalHandler.models = ["other:1b"]
        with self.assertRaises(SessionError) as ctx:
            self.window.load_session_from(path)
        self.assertIn("not installed", str(ctx.exception))

    def test_load_rejects_bad_encoder(self):
        data = self.window.session_dict()
        data["encoder"]["kind"] = "weird"
        path = self._path("bad.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh)
        with self.assertRaises(SessionError):
            self.window.load_session_from(path)

    def test_load_rejects_builtin_wrong_dim(self):
        data = self.window.session_dict()
        data["encoder"]["dim"] = 32
        path = self._path("bad.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh)
        with self.assertRaises(SessionError):
            self.window.load_session_from(path)

    def test_export_session_text_has_encoder(self):
        self._step_theme(2)
        path = self._path("session.txt")
        self.window.export_session_to(path)
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        self.assertIn("Encoder: built-in (dim 64)", text)
        self._switch_local()
        self._step_theme(2)
        self.window.export_session_to(path)
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        self.assertIn("Encoder: local, nomic-embed-text:latest "
                      "(dim 16)", text)


if __name__ == "__main__":
    unittest.main()
