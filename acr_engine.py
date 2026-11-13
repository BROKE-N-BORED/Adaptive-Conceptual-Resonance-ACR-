"""ACR engine: the adaptive loop at small scale, standard library only.

This is a faithful small-scale implementation of the blueprint's core
loop, not a full neural engine:

- Encoder: deterministic hashed word vectors, dim 64. Same input
  always gives the same vector. No downloads, no model files.
  Since v0.5 the engine also accepts embeddings from the local
  model service, at whatever dim that model returns: the engine
  dim is set at construction (Engine(dim=...)) and step() takes
  an optional precomputed vector h. The update rule is unchanged;
  only the embedding source differs.
- Dynamic weight matrix W_t: low-rank U V^T, rank 8. Each input step
  updates the factors by the blueprint's rule (section 2, steps 2-3):
  the residual signal is the drift D = h - h_prev, and the update is
  a gradient step on L(D, theta). The blueprint's L is "alignment
  between the predicted next token distribution and the observed D".
  There is no token distribution at this scale, so L is defined in
  the closest concrete form: L = -0.5 * ||W h_prev - h||^2, the
  negative prediction error between the predicted next embedding
  (W h_prev) and the observed next embedding (h = h_prev + D).
  Ascending L is descending the prediction loss. The error driving
  the update is (W h_prev - h) = (W h_prev - h_prev) - D: the
  unpredicted part of the drift. So the update is driven by D, not
  by reconstructing h: W_t learns the transition structure of the
  input flow, and repeated themes strengthen the direction they
  share, so resonance climbs for them. The step size is gated by
  the drift size (small drift, small step: nothing new to learn),
  plus a small decay so stale directions fade. The stability cap
  bounds every step. The first step has no previous state, so it
  is logged without an update: there is nothing to predict from.
- Resonance: cosine between the input embedding and the dominant
  direction of W_t (found by power iteration on W^T W). A spike
  flags a new-layer candidate.
- Trace log: every step records what changed and why.

Why the loss is this form and not D as a direct target: v0.1 tried
regressing W onto D itself (L = 0.5 * ||W h_prev - D||^2) and it
failed. The probe numbers (see DELTA-RULE-PROBE.md) show why: at
this scale the per-step drift is dominated by word-sampling noise
(consecutive theme sentences share only cos 0.19-0.30, ||D|| is
about 1.1), the drifts of one theme cycle point in mostly opposing
directions and sum to exactly zero over the cycle, so a D-target
update mostly cancels itself from step to step: resonance spiked
and collapsed (late min 0.011, max 0.600, mean 0.298 over 30 theme
steps) instead of climbing. The next-embedding form keeps the
update driven by D while the theme's stable component accumulates
(late mean 0.523, min 0.357, max 0.590, sustained through 60
steps). v0.1 shipped an auto-associative substitute while this was
unresolved; v0.2 replaces it with the D-driven rule above.

Nothing here trains a language model. The synthesis head in the app
surfaces related past inputs from the session buffer; since v0.6 it
also builds analogies (the direction between the two most resonant
inputs, applied to the current input) and counter-examples (the
past inputs farthest from the current one). It does not generate
text from general knowledge: the optional local-model completion
is the local model's own words, labelled as such.

A note on cost: at the built-in dim 64 the naive O(dim^2) math is
fast, but a local embedding model returns dim 768, where forming
the full W_t and iterating on it costs about 5 seconds per step.
Since W_t is low-rank (rank 8), the norm and the dominant
direction are computed in the rank-8 space instead: the Frobenius
norm from the Gram matrices (||W||_F^2 is the trace of
(V^T V)(U^T U)), and the dominant direction by power iteration on
the 8x8 matrix (U^T U)(V^T V), whose top eigenvector lifts to
W_t's top right singular vector through V. Same numbers, same
rule, a fraction of the work. The naive forms (_materialize_w,
_dominant_direction) stay for the tests that compare the two.

v0.7 adds stability and control, without changing the update
rule: per-direction freezing (individual rank components can be
held still while the rest adapt), learning-rate schedule presets
(steady / fast-adapt / gentle, computed by schedule_alpha), a
spectral-norm readout (the top singular value of W_t), a trace
flag for steps the stability cap actually clipped, and revert:
the engine keeps a bounded snapshot history of (U, V) after each
step, so W_t can be rolled back to any recent step. Snapshots are
not saved in session files: a loaded session starts its revert
history from the loaded state.
"""

import hashlib
import math
import random
import re

DIM = 64
RANK = 8
BUFFER_SIZE = 50
SEED = 20261112
RESONANCE_THRESHOLD = 0.75
RESONANCE_JUMP = 0.25
# v0.6: a step counts as resonant when its resonance is at least
# this. A fresh theme scores about 0.05-0.20 and a settled theme
# holds about 0.45-0.55 (16 theme steps, alpha 0.30: late steps
# 0.46, 0.52, 0.48, 0.52; a mixed drive never passes 0.35), so 0.4
# sits between the noise and the plateau. Resonant entries feed
# the analogy and the local-model completion context.
RESONANT_FLOOR = 0.4
PREVIEW_LEN = 80
SESSION_FORMAT_VERSION = 2
# v0.7: revert keeps the last SNAPSHOT_LIMIT step states. Each
# snapshot is two dim x rank factor matrices; at dim 768 that is
# about 100 KB per step in plain floats, so 200 steps is a
# reasonable bound on an 8 GB machine.
SNAPSHOT_LIMIT = 200
# v0.7: learning-rate schedule presets. Steady is the current
# behaviour (constant base alpha). Fast-adapt starts at twice the
# base and decays toward it over the first steps. Gentle runs at
# one third of the base throughout. See schedule_alpha.
SCHEDULE_STEADY = "steady"
SCHEDULE_FAST = "fast-adapt"
SCHEDULE_GENTLE = "gentle"
SCHEDULES = (SCHEDULE_STEADY, SCHEDULE_FAST, SCHEDULE_GENTLE)

TRACE_KEYS = ("id", "input", "delta_norm", "old_norm", "new_norm",
              "resonance", "alpha", "confidence", "new_layer",
              "frozen")


class SessionError(ValueError):
    """A session file is missing, malformed, or unreadable.

    The message is plain words for the status line, never a
    traceback: loading a bad file must fail gracefully.
    """

_WORD_RE = re.compile(r"[a-z0-9']+")


# ---------------------------------------------------------------- vectors


def _normalize(vec):
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return [0.0] * len(vec)
    return [x / norm for x in vec]


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _norm(vec):
    return math.sqrt(sum(x * x for x in vec))


def _word_vector(word):
    """One deterministic unit vector per word, from its SHA-256 hash."""
    digest = hashlib.sha256(word.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "big")
    rng = random.Random(seed)
    return _normalize([rng.gauss(0.0, 1.0) for _ in range(DIM)])


def encode(text):
    """Encode text as a dim-64 unit vector (mean of word vectors).

    Returns the zero vector for text with no words.
    """
    words = _WORD_RE.findall(text.lower())
    if not words:
        return [0.0] * DIM
    acc = [0.0] * DIM
    for word in words:
        vec = _word_vector(word)
        for i in range(DIM):
            acc[i] += vec[i]
    return _normalize(acc)


# ---------------------------------------------------------------- matrices
# Matrices are lists of rows (lists of floats).


def _matvec(mat, vec):
    return [_dot(row, vec) for row in mat]


def _transpose(mat):
    return [list(col) for col in zip(*mat)]


def _frobenius(mat):
    return math.sqrt(sum(x * x for row in mat for x in row))


def _outer(a, b):
    return [[x * y for y in b] for x in a]


def _materialize_w(u, v):
    """W = U V^T as a full dim x dim matrix.

    Naive form, kept for the tests. The engine itself never forms
    W: the norm and dominant direction use the low-rank forms
    below, which is what makes dim 768 usable.
    """
    dim = len(u)
    return [[sum(u[i][k] * v[j][k] for k in range(RANK))
             for j in range(dim)] for i in range(dim)]


def _gram(a, b):
    """A^T B for two dim x RANK matrices: a RANK x RANK matrix."""
    return [[sum(a[i][p] * b[i][q] for i in range(len(a)))
             for q in range(RANK)] for p in range(RANK)]


def _frobenius_uv(u, v):
    """||U V^T||_F without forming the matrix.

    ||W||_F^2 is the trace of W^T W, which for W = U V^T is the
    trace of (V^T V)(U^T U): a sum over two RANK x RANK Gram
    matrices. Same number as _frobenius(_materialize_w(u, v)),
    up to float rounding.
    """
    utu = _gram(u, u)
    vtv = _gram(v, v)
    total = 0.0
    for p in range(RANK):
        for q in range(RANK):
            total += utu[p][q] * vtv[p][q]
    return math.sqrt(max(0.0, total))


def _dominant_direction_uv(u, v):
    """Top right singular vector of W = U V^T, via the rank space.

    The right singular vectors of W live in the column space of
    V, so the power iteration runs on the RANK x RANK matrix
    M = (U^T U)(V^T V): its top eigenvector c lifts to W's top
    right singular vector as V c, normalized. Returns (vector,
    eigenvalue) like _dominant_direction: the vector is a unit
    vector in R^dim, the eigenvalue is the squared top singular
    value.
    """
    dim = len(u)
    utu = _gram(u, u)
    vtv = _gram(v, v)
    mat = [[sum(utu[i][k] * vtv[k][j] for k in range(RANK))
            for j in range(RANK)] for i in range(RANK)]
    rng = random.Random(SEED + 1)
    coeffs = _normalize([rng.gauss(0.0, 1.0) for _ in range(RANK)])
    # 200 iterations here costs nothing (8x8 matvecs) and drives
    # the direction to the naive form's accuracy. The naive form
    # stops at 30, which under-converges when the top two singular
    # values are close; the cheap form can afford to converge.
    for _ in range(200):
        nxt = _matvec(mat, coeffs)
        n = _norm(nxt)
        if n == 0.0:
            # W is (numerically) zero: no dominant direction.
            # Mirror _dominant_direction's fallback.
            fallback = _normalize(
                [rng.gauss(0.0, 1.0) for _ in range(dim)])
            return fallback, 0.0
        coeffs = [x / n for x in nxt]
    vec = _matvec(v, coeffs)
    n = _norm(vec)
    if n == 0.0:
        fallback = _normalize(
            [rng.gauss(0.0, 1.0) for _ in range(dim)])
        return fallback, 0.0
    vec = [x / n for x in vec]
    # lambda = ||W v||^2, with W v = U (V^T v).
    lam = _norm(_matvec(u, _matvec(_transpose(v), vec))) ** 2
    return vec, max(0.0, lam)


def _dominant_direction(w):
    """Top right singular vector of W, via power iteration on W^T W.

    Returns (vector, eigenvalue). The vector is a unit vector; the
    eigenvalue is the squared top singular value. The iteration
    applies W^T W as two matrix-vector products, so W^T W is never
    formed (that would be 64 x 64 x 64 multiplies per step).
    """
    wt = _transpose(w)
    rng = random.Random(SEED + 1)
    vec = _normalize([rng.gauss(0.0, 1.0) for _ in range(len(w))])
    for _ in range(30):
        nxt = _matvec(wt, _matvec(w, vec))
        n = _norm(nxt)
        if n == 0.0:
            return vec, 0.0
        vec = [x / n for x in nxt]
    # lambda = v^T W^T W v = ||W v||^2
    lam = _norm(_matvec(w, vec)) ** 2
    return vec, max(0.0, lam)


# ------------------------------------------------------- session checks
# Load-time validation for session files. Each helper returns a
# clean deep copy or raises SessionError with a plain message.


def _is_number(value):
    return (isinstance(value, (int, float))
            and not isinstance(value, bool))


def _checked_factors(value, name, dim):
    if (not isinstance(value, list) or len(value) != dim):
        raise SessionError(
            "the session data has a bad %s matrix" % name)
    out = []
    for row in value:
        if (not isinstance(row, list) or len(row) != RANK
                or not all(_is_number(x) for x in row)):
            raise SessionError(
                "the session data has a bad %s matrix" % name)
        out.append([float(x) for x in row])
    return out


def _checked_vector(value, name, dim, allow_none=False):
    if value is None and allow_none:
        return None
    if (not isinstance(value, list) or len(value) != dim
            or not all(_is_number(x) for x in value)):
        raise SessionError(
            "the session data has a bad %s vector" % name)
    return [float(x) for x in value]


def _checked_trace(value, step_count):
    if not isinstance(value, list) or len(value) != step_count:
        raise SessionError("the session data has a bad trace log")
    out = []
    for i, entry in enumerate(value):
        if (not isinstance(entry, dict)
                or entry.get("id") != i + 1
                or any(key not in entry for key in TRACE_KEYS)):
            raise SessionError("the session data has a bad trace log")
        entry = dict(entry)
        # v0.7 keys, absent from older files: fill the defaults
        # (no frozen directions, cap never bit) and validate the
        # types when the keys are present.
        ranks = entry.get("frozen_ranks", [])
        if (not isinstance(ranks, list)
                or any(not isinstance(k, int)
                       or isinstance(k, bool)
                       or not 0 <= k < RANK for k in ranks)):
            raise SessionError("the session data has a bad trace log")
        entry["frozen_ranks"] = sorted(set(ranks))
        capped = entry.get("capped", False)
        if not isinstance(capped, bool):
            raise SessionError("the session data has a bad trace log")
        entry["capped"] = capped
        if "reverted_to" in entry:
            target = entry["reverted_to"]
            if (not isinstance(target, int)
                    or isinstance(target, bool)
                    or not 1 <= target < entry["id"]):
                raise SessionError(
                    "the session data has a bad trace log")
        out.append(entry)
    return out


def _checked_buffer(value, dim):
    if not isinstance(value, list) or len(value) > BUFFER_SIZE:
        raise SessionError("the session data has a bad buffer")
    out = []
    for item in value:
        if (not isinstance(item, dict)
                or not isinstance(item.get("text"), str)):
            raise SessionError("the session data has a bad buffer")
        out.append({"text": item["text"],
                    "vector": _checked_vector(item.get("vector"),
                                              "buffer", dim)})
    return out


def schedule_alpha(preset, base, step_number):
    """Effective learning rate for one step under a preset.

    ``preset`` is one of SCHEDULES, ``base`` the slider value,
    ``step_number`` the 1-based number of the step about to run.
    Steady is the base, constant. Fast-adapt starts at twice the
    base and halves the excess every 3 steps: step 1 is 2x, step
    4 is 1.5x, step 7 is 1.25x, step 10 is 1.125x, settling on the
    base. Gentle is one third of the base, constant. Raises
    ValueError on an unknown preset.
    """
    if preset == SCHEDULE_STEADY:
        return base
    if preset == SCHEDULE_GENTLE:
        return base / 3.0
    if preset == SCHEDULE_FAST:
        return base * (1.0 + 2.0 ** (-(step_number - 1) / 3.0))
    raise ValueError("unknown schedule preset: %r" % (preset,))


# ---------------------------------------------------------------- engine


class Engine:
    """One ACR session: encoder, W_t, buffer, trace log.

    ``dim`` is the embedding size: 64 for the built-in encoder,
    whatever the local model returns otherwise. Switching encoders
    means a new engine: the factors, buffer, and trace of one dim
    cannot run at another.
    """

    def __init__(self, dim=DIM, snapshot_limit=SNAPSHOT_LIMIT):
        self.dim = dim
        self.snapshot_limit = snapshot_limit
        self.reset()

    def reset(self):
        """Back to the initial W_t, empty buffer, empty trace.

        Keeps the dim: a reset is the same encoder starting over,
        not a different encoder. The revert snapshots go too: a
        reset session has no past to roll back to.
        """
        rng = random.Random(SEED)
        self.u = [[rng.gauss(0.0, 0.1) for _ in range(RANK)]
                  for _ in range(self.dim)]
        self.v = [[rng.gauss(0.0, 0.1) for _ in range(RANK)]
                  for _ in range(self.dim)]
        self.trace = []
        self.buffer = []  # list of {"text": preview, "vector": h}
        self.h_prev = None
        self.prev_resonance = None
        self.step_count = 0
        # Revert snapshots: step id -> (u copy, v copy), bounded
        # to the last snapshot_limit steps. Not part of the saved
        # session: a loaded session starts its history from the
        # loaded state.
        self._snapshots = {}

    def state_dict(self):
        """Full engine state as a JSON-ready dict.

        Everything needed to resume this exact session: the dim,
        the W_t factors, the previous embedding and resonance (the
        next step predicts from them), the step count, the whole
        trace log, and the buffer with its vectors. Deep copies:
        mutating the dict never touches the engine. The format
        version lives in the session file around this dict, not
        here.
        """
        return {
            "dim": self.dim,
            "u": [row[:] for row in self.u],
            "v": [row[:] for row in self.v],
            "h_prev": (list(self.h_prev)
                       if self.h_prev is not None else None),
            "prev_resonance": self.prev_resonance,
            "step_count": self.step_count,
            "trace": [dict(entry) for entry in self.trace],
            "buffer": [{"text": item["text"],
                        "vector": list(item["vector"])}
                       for item in self.buffer],
        }

    def load_state_dict(self, state):
        """Restore state from state_dict().

        Validates everything before touching the engine, so a bad
        file leaves the current session intact. Raises SessionError
        with a plain message on anything malformed. The state's
        dim must match this engine's dim: a state saved at another
        dim belongs to a different encoder, never load it into
        this one. (v0.4 states carry no dim key; the app loads
        those into a dim-64 engine, which is what they were.)
        """
        if not isinstance(state, dict):
            raise SessionError("the session data is not an object")
        dim = state.get("dim")
        if dim is not None and dim != self.dim:
            raise SessionError(
                "the session data is for a different embedding "
                "size")
        u = _checked_factors(state.get("u"), "u", self.dim)
        v = _checked_factors(state.get("v"), "v", self.dim)
        h_prev = _checked_vector(state.get("h_prev"), "h_prev",
                                 self.dim, allow_none=True)
        prev_resonance = state.get("prev_resonance")
        if (prev_resonance is not None
                and not isinstance(prev_resonance, (int, float))):
            raise SessionError(
                "the session data has a bad prev_resonance")
        step_count = state.get("step_count")
        if (not isinstance(step_count, int)
                or isinstance(step_count, bool)
                or step_count < 0):
            raise SessionError("the session data has a bad step_count")
        trace = _checked_trace(state.get("trace"), step_count)
        buffer = _checked_buffer(state.get("buffer"), self.dim)
        self.u = u
        self.v = v
        self.h_prev = h_prev
        self.prev_resonance = prev_resonance
        self.step_count = step_count
        self.trace = trace
        self.buffer = buffer
        # Snapshots are not saved in session files: the revert
        # history restarts from the loaded state, which is the
        # state after the last step in the file.
        self._snapshots = {}
        if step_count > 0:
            self._snapshot(step_count)

    def w_norm(self):
        return _frobenius_uv(self.u, self.v)

    def spectral_norm(self):
        """Top singular value of W_t: the size of its strongest
        direction. The square root of the top eigenvalue of
        W^T W, from the rank-space power iteration."""
        _direction, top_eigen = _dominant_direction_uv(self.u,
                                                       self.v)
        return math.sqrt(max(0.0, top_eigen))

    def _snapshot(self, step_id):
        """Record the current (U, V) under one step id, keeping
        only the last snapshot_limit steps."""
        self._snapshots[step_id] = (
            [row[:] for row in self.u],
            [row[:] for row in self.v])
        while len(self._snapshots) > self.snapshot_limit:
            oldest = next(iter(self._snapshots))
            del self._snapshots[oldest]

    def snapshot_step_ids(self):
        """Step ids W_t can still be reverted to, oldest first."""
        return sorted(self._snapshots)

    def revert_to(self, step_id):
        """Roll W_t back to its state after the given step.

        Returns the new trace entry, which is marked with
        "reverted_to" so the log stays honest: later entries are
        never deleted. The input flow is untouched (h_prev and
        the buffer stay), so the next step continues from the
        reverted W_t. Raises ValueError with a plain message when
        the step has no snapshot: either it never existed, or it
        aged out of the bounded history.
        """
        if step_id not in self._snapshots:
            if (not isinstance(step_id, int)
                    or isinstance(step_id, bool)
                    or step_id < 1
                    or step_id > self.step_count):
                raise ValueError(
                    "there is no step %s" % (step_id,))
            raise ValueError(
                "step %d has aged out of the revert history "
                "(the last %d steps are kept)"
                % (step_id, self.snapshot_limit))
        old_norm = self.w_norm()
        snap_u, snap_v = self._snapshots[step_id]
        self.u = [row[:] for row in snap_u]
        self.v = [row[:] for row in snap_v]
        new_norm = _frobenius_uv(self.u, self.v)
        direction, top_eigen = _dominant_direction_uv(self.u,
                                                      self.v)
        if self.h_prev is None or _norm(self.h_prev) == 0.0:
            resonance = 0.0
        else:
            resonance = abs(_dot(self.h_prev, direction))
            resonance = max(0.0, min(1.0, resonance))
        if new_norm > 0.0:
            confidence = top_eigen / (new_norm * new_norm)
            confidence = max(0.0, min(1.0, confidence))
        else:
            confidence = 0.0
        self.step_count += 1
        entry = {
            "id": self.step_count,
            "input": "Revert to step %d" % step_id,
            "delta_norm": 0.0,
            "old_norm": round(old_norm, 6),
            "new_norm": round(new_norm, 6),
            "resonance": round(resonance, 6),
            "alpha": 0.0,
            "confidence": round(confidence, 6),
            "new_layer": False,
            "frozen": False,
            "frozen_ranks": [],
            "capped": False,
            "reverted_to": step_id,
        }
        self.trace.append(entry)
        self.prev_resonance = resonance
        self._snapshot(self.step_count)
        return entry

    def step(self, text, alpha=0.1, frozen=False, cap=1.0, h=None,
             frozen_ranks=None):
        """Run one adaptive step on the given text.

        Returns the trace entry for this step. When frozen, W_t is
        not updated but the step is still logged. The first step is
        also logged without an update: with no previous state there
        is no drift to learn from and nothing to predict.

        ``frozen_ranks`` is a list of rank-component indexes
        (0..RANK-1) to hold still: those components skip both the
        decay and the gradient step, so their values are exactly
        unchanged, while the rest adapt as usual. An index outside
        the range is a programming error and fails loudly.

        ``h`` is a precomputed embedding (from the local model
        service); when it is None the built-in encoder runs. A
        supplied h must match this engine's dim, and an engine
        whose dim is not the built-in 64 has no built-in encoder
        to fall back on: stepping it without h is a programming
        error, and fails loudly instead of silently mixing dims.
        """
        ranks = sorted(set(frozen_ranks or ()))
        for k in ranks:
            if (not isinstance(k, int) or isinstance(k, bool)
                    or not 0 <= k < RANK):
                raise ValueError(
                    "rank direction %r is not in 0..%d"
                    % (k, RANK - 1))
        if h is None:
            if self.dim != DIM:
                raise ValueError(
                    "engine dim %d needs a supplied embedding"
                    % self.dim)
            h = encode(text)
        elif len(h) != self.dim:
            raise ValueError(
                "embedding dim %d does not match engine dim %d"
                % (len(h), self.dim))
        if self.h_prev is None:
            delta = list(h)
        else:
            delta = [a - b for a, b in zip(h, self.h_prev)]
        delta_norm = _norm(delta)
        old_norm = self.w_norm()

        capped = False
        if not frozen and self.h_prev is not None:
            # The residual gates the step: a big semantic drift
            # earns the full learning rate, a tiny drift earns
            # almost nothing.
            gated_alpha = alpha * min(1.0, delta_norm)
            capped = self._update(self.h_prev, h, gated_alpha,
                                  cap, ranks)

        # Norm and dominant direction from the low-rank forms: no
        # full matrix is ever formed, which is what keeps dim 768
        # fast.
        new_norm = _frobenius_uv(self.u, self.v)
        direction, top_eigen = _dominant_direction_uv(self.u,
                                                      self.v)
        if _norm(h) == 0.0:
            resonance = 0.0
        else:
            resonance = abs(_dot(h, direction))
            resonance = max(0.0, min(1.0, resonance))

        # Confidence: share of W_t's energy in its top direction.
        if new_norm > 0.0:
            confidence = top_eigen / (new_norm * new_norm)
            confidence = max(0.0, min(1.0, confidence))
        else:
            confidence = 0.0

        new_layer = False
        if self.prev_resonance is not None:
            crossed = (resonance >= RESONANCE_THRESHOLD
                       and self.prev_resonance < RESONANCE_THRESHOLD)
            jumped = (resonance - self.prev_resonance
                      >= RESONANCE_JUMP)
            new_layer = crossed or jumped

        self.step_count += 1
        entry = {
            "id": self.step_count,
            "input": text[:PREVIEW_LEN],
            "delta_norm": round(delta_norm, 6),
            "old_norm": round(old_norm, 6),
            "new_norm": round(new_norm, 6),
            "resonance": round(resonance, 6),
            "alpha": alpha,
            "confidence": round(confidence, 6),
            "new_layer": new_layer,
            "frozen": frozen,
            "frozen_ranks": ranks,
            "capped": capped,
        }
        self.trace.append(entry)
        self._snapshot(self.step_count)
        self.buffer.append({"text": text[:PREVIEW_LEN], "vector": h})
        if len(self.buffer) > BUFFER_SIZE:
            self.buffer.pop(0)
        self.h_prev = h
        self.prev_resonance = resonance
        return entry

    def _update(self, h_prev, h, alpha, cap, frozen_ranks=()):
        """Gradient step on L = 0.5 * ||W h_prev - h||^2, via U and V.

        This is the blueprint's rule with L defined as next-embedding
        alignment (see the module docstring): W_t predicts the next
        embedding from the previous one, and the error
        (W h_prev - h) is the unpredicted part of the drift
        D = h - h_prev, so the update is driven by D. A small decay
        is applied first so directions W_t has not seen lately fade.
        The step size is capped: if alpha times the update norm
        would exceed the stability cap, the update is scaled down
        to it.

        ``frozen_ranks`` holds rank-component indexes to hold
        still: they skip both the decay and the gradient step, so
        their values are exactly unchanged, and they do not count
        toward the update norm the cap is measured against (they
        are not part of the step being taken). Returns True when
        the cap actually clipped the update, False otherwise.
        """
        frozen = set(frozen_ranks)
        decay = 1.0 - 0.005
        for i in range(self.dim):
            for k in range(RANK):
                if k in frozen:
                    continue
                self.u[i][k] *= decay
                self.v[i][k] *= decay
        # a = V^T h_prev, z = U a, err = z - h
        vt = _transpose(self.v)
        a = _matvec(vt, h_prev)
        z = _matvec(self.u, a)
        err = [zi - hi for zi, hi in zip(z, h)]
        # dL/dU = err (x) a ; dL/dV = h_prev (x) U^T err
        du = _outer(err, a)
        ut = _transpose(self.u)
        ut_err = _matvec(ut, err)
        dv = _outer(h_prev, ut_err)
        for i in range(self.dim):
            for k in frozen:
                du[i][k] = 0.0
                dv[i][k] = 0.0
        update_norm = math.sqrt(_frobenius(du) ** 2
                                + _frobenius(dv) ** 2)
        step_size = alpha * update_norm
        clipped = step_size > cap and step_size > 0.0
        if clipped:
            scale = cap / step_size
        else:
            scale = 1.0
        eff = alpha * scale
        for i in range(self.dim):
            for k in range(RANK):
                self.u[i][k] -= eff * du[i][k]
                self.v[i][k] -= eff * dv[i][k]
        return clipped

    def resonant_entries(self, count=None):
        """Buffer entries from high-resonance steps, best first.

        Returns [(text, vector, resonance, new_layer, step_id),
        ...] for buffer entries whose step scored at least
        RESONANT_FLOOR. Layer candidates (new-layer spikes) sort
        first, then by resonance: the analogy prefers them.
        Repeated inputs collapse to one entry each (the
        best-ranked copy), so an analogy's two ends are always
        two different inputs. `count` caps the list. The buffer
        holds the last N inputs in step order, so buffer item j
        is the j-th of the last N real steps: revert entries are
        not steps and carry no input, so they are filtered out
        before the mapping (without that, a revert would shift
        every buffer item onto the wrong trace entry).
        """
        if not self.buffer:
            return []
        steps = [e for e in self.trace if "reverted_to" not in e]
        n = len(self.buffer)
        ranked = []
        for j, item in enumerate(self.buffer):
            entry = steps[len(steps) - n + j]
            step_id = entry["id"]
            if entry["resonance"] >= RESONANT_FLOOR:
                ranked.append((item["text"], item["vector"],
                               entry["resonance"],
                               entry["new_layer"], step_id))
        ranked.sort(key=lambda e: (e[3], e[2]), reverse=True)
        out = []
        seen = set()
        for entry in ranked:
            if entry[0] in seen:
                continue
            seen.add(entry[0])
            out.append(entry)
        if count is not None:
            out = out[:count]
        return out

    def synthesize(self):
        """Demo synthesis head: neighbours + midpoint + analogy +
        counter-examples.

        Returns {"neighbours": [(text, score), ...], "midpoint":
        [(text, score), ...], "analogy": {"a": text, "b": text,
        "result": (text, score)} or None, "counter_examples":
        [(text, score), ...]}. Neighbours and midpoint are scored
        as before (cosine to the latest input, and to the midpoint
        of the top two neighbours). The analogy takes the two most
        resonant past inputs A and B, forms the direction d =
        h_B - h_A, and finds the nearest other past input to
        h + d: like A is to B, the current input points at the
        result. It is None when fewer than two resonant inputs
        exist, or when no other input is left to point at: no
        analogy is invented. Counter-examples are the distinct
        past inputs farthest from the current input (lowest
        cosine), up to three: what this session is not. Empty
        when the buffer is empty.
        """
        if not self.buffer or self.h_prev is None:
            return {"neighbours": [], "midpoint": [],
                    "analogy": None, "counter_examples": []}
        scored = []
        for item in self.buffer:
            scored.append((_dot(self.h_prev, item["vector"]),
                           item["text"]))
        scored.sort(reverse=True)
        # Only positive cosine scores count as related: a negative
        # score means dissimilar, not a neighbour.
        neighbours = [(text, round(score, 4))
                      for score, text in scored[:3] if score > 0.0]
        midpoint = []
        if len(scored) >= 2:
            top_vecs = []
            for _, text in scored[:2]:
                for item in self.buffer:
                    if item["text"] == text:
                        top_vecs.append(item["vector"])
                        break
            if len(top_vecs) == 2:
                mid = _normalize([(a + b) / 2.0 for a, b in
                                  zip(top_vecs[0], top_vecs[1])])
                top_texts = {text for _, text in scored[:2]}
                mid_scored = []
                for item in self.buffer:
                    if item["text"] in top_texts:
                        continue
                    mid_scored.append((_dot(mid, item["vector"]),
                                       item["text"]))
                mid_scored.sort(reverse=True)
                midpoint = [(text, round(score, 4))
                            for score, text in mid_scored[:2]
                            if score > 0.0]
        analogy = None
        resonant = self.resonant_entries(2)
        if len(resonant) >= 2:
            a_text, a_vec = resonant[0][0], resonant[0][1]
            b_text, b_vec = resonant[1][0], resonant[1][1]
            direction = [y - x for x, y in zip(a_vec, b_vec)]
            target = _normalize([x + y for x, y in
                                 zip(self.h_prev, direction)])
            # Repeated inputs are the same point, so the analogy
            # points at a different input: skip entries whose text
            # is A, B, or the current input.
            skip = {a_text, b_text, self.buffer[-1]["text"]}
            target_scored = []
            for item in self.buffer:
                if item["text"] in skip:
                    continue
                target_scored.append(
                    (_dot(target, item["vector"]), item["text"]))
            if target_scored:
                target_scored.sort(reverse=True)
                score, text = target_scored[0]
                analogy = {"a": a_text, "b": b_text,
                           "result": (text, round(score, 4))}
        # Counter-examples: the distinct past inputs farthest from
        # the current input, lowest cosine first. Duplicates
        # collapse to one entry each, so the three shown are three
        # different inputs. The current input itself never counts.
        current_text = self.buffer[-1]["text"]
        farthest = {}
        for item in self.buffer[:-1]:
            if item["text"] == current_text:
                continue
            score = _dot(self.h_prev, item["vector"])
            if (item["text"] not in farthest
                    or score < farthest[item["text"]]):
                farthest[item["text"]] = score
        ranked = sorted(farthest.items(), key=lambda kv: kv[1])[:3]
        counter_examples = [(text, round(score, 4))
                            for text, score in ranked]
        return {"neighbours": neighbours, "midpoint": midpoint,
                "analogy": analogy,
                "counter_examples": counter_examples}
