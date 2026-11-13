"""Engine tests: the adaptive loop math, standard library only."""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

from acr_engine import (  # noqa: E402
    BUFFER_SIZE, DIM, Engine, _frobenius, _materialize_w,
    _matvec, encode)


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb)


class EncoderTests(unittest.TestCase):
    def test_deterministic(self):
        self.assertEqual(encode("quantum entanglement"),
                         encode("quantum entanglement"))

    def test_dim(self):
        self.assertEqual(len(encode("hello world")), DIM)

    def test_different_inputs_differ(self):
        a = encode("quantum entanglement")
        b = encode("baking bread with yeast")
        self.assertLess(_cosine(a, b), 0.99)

    def test_empty_text_is_zero_vector(self):
        self.assertEqual(encode(""), [0.0] * DIM)
        self.assertEqual(encode("   !!!"), [0.0] * DIM)


class StepTests(unittest.TestCase):
    def test_step_updates_w_norm(self):
        # v0.2 note: the first step logs without an update (no
        # previous state to predict from), so the update is checked
        # on the second step.
        engine = Engine()
        engine.step("quantum entanglement is strange")
        entry = engine.step("entangled particles share a state")
        self.assertNotEqual(entry["old_norm"], entry["new_norm"])

    def test_resonance_and_confidence_in_range(self):
        engine = Engine()
        texts = ["quantum entanglement links particles",
                 "baking bread needs warm dough",
                 "the river runs past the old mill"]
        for i in range(12):
            entry = engine.step(texts[i % 3])
            self.assertGreaterEqual(entry["resonance"], 0.0)
            self.assertLessEqual(entry["resonance"], 1.0)
            self.assertGreaterEqual(entry["confidence"], 0.0)
            self.assertLessEqual(entry["confidence"], 1.0)

    def test_trace_records_every_step(self):
        engine = Engine()
        for i in range(5):
            engine.step("sentence number %d" % i)
        self.assertEqual(len(engine.trace), 5)
        self.assertEqual([e["id"] for e in engine.trace],
                         [1, 2, 3, 4, 5])

    def test_trace_entry_shape(self):
        engine = Engine()
        entry = engine.step("hello world")
        for key in ("id", "input", "delta_norm", "old_norm",
                    "new_norm", "resonance", "alpha", "confidence",
                    "new_layer", "frozen"):
            self.assertIn(key, entry)

    def test_trace_is_json_serializable(self):
        engine = Engine()
        engine.step("first sentence")
        engine.step("second sentence")
        text = json.dumps(engine.trace)
        self.assertEqual(len(json.loads(text)), 2)

    def test_input_preview_capped(self):
        engine = Engine()
        entry = engine.step("x" * 200)
        self.assertEqual(len(entry["input"]), 80)

    def test_empty_text_step_does_not_crash(self):
        engine = Engine()
        entry = engine.step("")
        self.assertEqual(entry["id"], 1)
        self.assertEqual(entry["delta_norm"], 0.0)

    def test_same_sequence_same_trace(self):
        first = Engine()
        second = Engine()
        for text in ("one", "two", "three"):
            first.step(text)
            second.step(text)
        self.assertEqual(first.trace, second.trace)


class FreezeTests(unittest.TestCase):
    def test_frozen_step_keeps_w(self):
        engine = Engine()
        # v0.2 note: two warm-up steps, because the first step logs
        # without an update (no previous state to predict from).
        engine.step("warm up the weights")
        engine.step("warm them up some more")
        before = engine.w_norm()
        entry = engine.step("frozen input", frozen=True)
        self.assertEqual(entry["old_norm"], entry["new_norm"])
        self.assertEqual(engine.w_norm(), before)
        self.assertTrue(entry["frozen"])

    def test_frozen_step_still_logged(self):
        engine = Engine()
        engine.step("frozen input", frozen=True)
        self.assertEqual(len(engine.trace), 1)


class ResetTests(unittest.TestCase):
    def test_reset_restores_initial_w(self):
        engine = Engine()
        fresh = Engine()
        for i in range(5):
            engine.step("sentence %d about many things" % i)
        engine.reset()
        self.assertEqual(engine.u, fresh.u)
        self.assertEqual(engine.v, fresh.v)
        self.assertEqual(engine.trace, [])
        self.assertEqual(engine.buffer, [])
        self.assertEqual(engine.step_count, 0)


class BufferTests(unittest.TestCase):
    def test_buffer_capped_at_fifty(self):
        engine = Engine()
        for i in range(60):
            engine.step("buffer sentence %d" % i)
        self.assertEqual(len(engine.buffer), BUFFER_SIZE)
        self.assertEqual(engine.buffer[-1]["text"],
                         "buffer sentence 59")


class ResonanceTests(unittest.TestCase):
    def test_repeated_theme_raises_resonance(self):
        engine = Engine()
        texts = ["quantum entanglement links particles",
                 "entangled particles share a state",
                 "quantum particles stay connected",
                 "entanglement links distant particles"]
        early = []
        late = []
        for i in range(30):
            entry = engine.step(texts[i % 4], alpha=0.3)
            if i < 5:
                early.append(entry["resonance"])
            if i >= 20:
                late.append(entry["resonance"])
        self.assertGreater(max(late), 0.3)
        self.assertGreater(max(late), max(early))

    def test_new_layer_flag_fires_on_repeated_theme(self):
        engine = Engine()
        texts = ["quantum entanglement links particles",
                 "entangled particles share a state",
                 "quantum particles stay connected",
                 "entanglement links distant particles"]
        flags = 0
        for i in range(30):
            entry = engine.step(texts[i % 4], alpha=0.3)
            flags += 1 if entry["new_layer"] else 0
        self.assertGreater(flags, 0)


class DeltaRuleTests(unittest.TestCase):
    """v0.2: the update is driven by the drift D = h - h_prev.

    v0.1 shipped an auto-associative substitute (W learns to
    reconstruct h) because the first attempt at the blueprint's
    rule regressed W onto D itself and chased noise. v0.2 implements
    the blueprint's rule with L defined as next-embedding alignment:
    W predicts the next embedding from the previous one, and the
    error driving the update is the unpredicted part of D. These
    tests pin that behaviour, including the regression test that
    would have caught the original failure.
    """

    def test_first_step_logs_without_update(self):
        engine = Engine()
        entry = engine.step("quantum entanglement is strange")
        self.assertEqual(entry["id"], 1)
        self.assertEqual(entry["old_norm"], entry["new_norm"])
        self.assertEqual(len(engine.trace), 1)

    def test_second_step_updates_w(self):
        engine = Engine()
        engine.step("quantum entanglement is strange")
        entry = engine.step("entangled particles share a state")
        self.assertNotEqual(entry["old_norm"], entry["new_norm"])

    def test_update_reduces_prediction_error(self):
        # The rule learns to predict the drifted-to state: after
        # one update, ||W h_prev - h|| must be smaller than before.
        x = "quantum entanglement links particles"
        y = "baking bread needs warm dough"
        engine = Engine()
        engine.step(x, alpha=0.3)
        h_prev, h = encode(x), encode(y)

        def error():
            w = _materialize_w(engine.u, engine.v)
            pred = _matvec(w, h_prev)
            return sum((p - t) ** 2
                       for p, t in zip(pred, h)) ** 0.5

        before = error()
        engine.step(y, alpha=0.3)
        self.assertLess(error(), before)

    def test_update_follows_the_drift(self):
        # Same first step, then a drifting second step vs the same
        # text again (zero drift): the two W_t matrices must end up
        # clearly different, because the update follows D.
        x = "quantum entanglement links particles"
        y = "baking bread needs warm dough"
        drifting = Engine()
        drifting.step(x, alpha=0.3)
        drifting.step(y, alpha=0.3)
        still = Engine()
        still.step(x, alpha=0.3)
        still.step(x, alpha=0.3)
        wa = _materialize_w(drifting.u, drifting.v)
        wb = _materialize_w(still.u, still.v)
        diff = [[a - b for a, b in zip(ra, rb)]
                for ra, rb in zip(wa, wb)]
        self.assertGreater(_frobenius(diff), 0.1)

    def test_repeated_theme_sustains_resonance(self):
        # REGRESSION: the v0.1 drift-target attempt spiked and
        # collapsed (30 theme steps, alpha 0.3: late mean 0.298,
        # late min 0.011). The fixed rule climbs and stays up
        # (late mean 0.523, late min 0.357). A max-only check would
        # have passed the broken rule too (its max late was 0.600),
        # so this pins the mean and the floor.
        engine = Engine()
        texts = ["quantum entanglement links particles",
                 "entangled particles share a state",
                 "quantum particles stay connected",
                 "entanglement links distant particles"]
        res = []
        for i in range(30):
            entry = engine.step(texts[i % 4], alpha=0.3)
            res.append(entry["resonance"])
        early, late = res[:5], res[20:]
        self.assertGreater(max(late), max(early))
        self.assertGreater(sum(late) / len(late), 0.4)
        self.assertGreater(min(late), 0.3)

    def test_theme_switch_drops_resonance(self):
        engine = Engine()
        theme_a = ["quantum entanglement links particles",
                   "entangled particles share a state",
                   "quantum particles stay connected",
                   "entanglement links distant particles"]
        theme_b = ["baking bread needs warm dough",
                   "the oven glows hot at dawn",
                   "flour dust covers the table",
                   "knead the dough until smooth"]
        late_a = []
        for i in range(30):
            entry = engine.step(theme_a[i % 4], alpha=0.3)
            if i >= 20:
                late_a.append(entry["resonance"])
        first_b = []
        for i in range(3):
            entry = engine.step(theme_b[i], alpha=0.3)
            first_b.append(entry["resonance"])
        mean_a = sum(late_a) / len(late_a)
        mean_b = sum(first_b) / len(first_b)
        self.assertLess(mean_b, 0.2)
        self.assertLess(mean_b, mean_a)


class SynthesisTests(unittest.TestCase):
    def test_empty_before_steps(self):
        engine = Engine()
        result = engine.synthesize()
        self.assertEqual(result["neighbours"], [])
        self.assertEqual(result["midpoint"], [])

    def test_neighbours_after_steps(self):
        engine = Engine()
        engine.step("quantum entanglement links particles")
        engine.step("baking bread needs warm dough")
        engine.step("quantum particles stay connected")
        result = engine.synthesize()
        self.assertGreater(len(result["neighbours"]), 0)
        # The latest input is its own nearest neighbour.
        top_text, top_score = result["neighbours"][0]
        self.assertEqual(top_text, "quantum particles stay connected")
        self.assertAlmostEqual(top_score, 1.0, places=3)


if __name__ == "__main__":
    unittest.main()
