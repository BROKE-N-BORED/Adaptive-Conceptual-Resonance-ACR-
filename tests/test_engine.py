"""Engine tests: the adaptive loop math, standard library only."""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

from acr_engine import BUFFER_SIZE, DIM, Engine, encode  # noqa: E402


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
        engine = Engine()
        entry = engine.step("quantum entanglement is strange")
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
        engine.step("warm up the weights")
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
