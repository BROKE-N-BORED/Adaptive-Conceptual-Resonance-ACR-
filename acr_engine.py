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
surfaces related past inputs from the session buffer; it does not
generate text from general knowledge.

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
PREVIEW_LEN = 80
SESSION_FORMAT_VERSION = 2

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
        out.append(dict(entry))
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


# ---------------------------------------------------------------- engine


class Engine:
    """One ACR session: encoder, W_t, buffer, trace log.

    ``dim`` is the embedding size: 64 for the built-in encoder,
    whatever the local model returns otherwise. Switching encoders
    means a new engine: the factors, buffer, and trace of one dim
    cannot run at another.
    """

    def __init__(self, dim=DIM):
        self.dim = dim
        self.reset()

    def reset(self):
        """Back to the initial W_t, empty buffer, empty trace.

        Keeps the dim: a reset is the same encoder starting over,
        not a different encoder.
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

    def w_norm(self):
        return _frobenius_uv(self.u, self.v)

    def step(self, text, alpha=0.1, frozen=False, cap=1.0, h=None):
        """Run one adaptive step on the given text.

        Returns the trace entry for this step. When frozen, W_t is
        not updated but the step is still logged. The first step is
        also logged without an update: with no previous state there
        is no drift to learn from and nothing to predict.

        ``h`` is a precomputed embedding (from the local model
        service); when it is None the built-in encoder runs. A
        supplied h must match this engine's dim, and an engine
        whose dim is not the built-in 64 has no built-in encoder
        to fall back on: stepping it without h is a programming
        error, and fails loudly instead of silently mixing dims.
        """
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

        if not frozen and self.h_prev is not None:
            # The residual gates the step: a big semantic drift
            # earns the full learning rate, a tiny drift earns
            # almost nothing.
            gated_alpha = alpha * min(1.0, delta_norm)
            self._update(self.h_prev, h, gated_alpha, cap)

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
        }
        self.trace.append(entry)
        self.buffer.append({"text": text[:PREVIEW_LEN], "vector": h})
        if len(self.buffer) > BUFFER_SIZE:
            self.buffer.pop(0)
        self.h_prev = h
        self.prev_resonance = resonance
        return entry

    def _update(self, h_prev, h, alpha, cap):
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
        """
        decay = 1.0 - 0.005
        for i in range(self.dim):
            for k in range(RANK):
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
        update_norm = math.sqrt(_frobenius(du) ** 2
                                + _frobenius(dv) ** 2)
        step_size = alpha * update_norm
        if step_size > cap and step_size > 0.0:
            scale = cap / step_size
        else:
            scale = 1.0
        eff = alpha * scale
        for i in range(self.dim):
            for k in range(RANK):
                self.u[i][k] -= eff * du[i][k]
                self.v[i][k] -= eff * dv[i][k]

    def synthesize(self):
        """Demo synthesis head: neighbours + midpoint neighbours.

        Returns {"neighbours": [(text, score), ...], "midpoint":
        [(text, score), ...]} scored by cosine to the latest input
        vector (neighbours) and to the midpoint of the top two
        neighbours (midpoint). Empty when the buffer is empty.
        """
        if not self.buffer or self.h_prev is None:
            return {"neighbours": [], "midpoint": []}
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
        return {"neighbours": neighbours, "midpoint": midpoint}
