# Why v0.1's drift rule failed, and what v0.2 implements

Probe evidence for the v0.2 engine fix. All runs are deterministic
(fixed seed), 30 steps of a 4-sentence repeated theme unless noted,
alpha 0.3 unless noted. Probe scripts (re-runnable, stdlib only):
`~/workspace/acr-delta-probes/acr_delta_probe.py` (round 1),
`acr_delta_probe2.py` (round 2),
`acr_delta_probe3.py` and `acr_delta_probe3b.py` (round 3); the
numbers below are their output.

## The blueprint's rule

Section 2, steps 2-3 of the blueprint:

- Step 2: residual signal D = h - h_prev (the semantic drift).
- Step 3: theta_{t+1} = theta_t + alpha * grad L(D, theta), where L
  "encourages alignment between the predicted next token distribution
  and the observed D".

The blueprint's L is underspecified for this scale: there is no next
token distribution here. Two readings were tested.

## Reading 1 (v0.1's first attempt): regress W onto D itself

L = 0.5 * ||W h_prev - D||^2 (and the h-input variant
L = 0.5 * ||W h - D||^2). This is the literal "drift-target" loss.
It failed, and the numbers show why:

- Consecutive theme sentences share only cos 0.19-0.30, while
  ||D|| is about 1.1 per step: the drift is dominated by which words
  each sentence happens to sample, not by a stable signal.
- The four drifts of one theme cycle point in mostly opposing
  directions (pairwise cosines -0.71, -0.12, -0.32, -0.48, 0.26,
  -0.57; mean |cos| 0.41) and sum to exactly 0.0000 over the cycle
  (they telescope). So a D-target update mostly cancels itself from
  step to step: W never accumulates a stable direction.
- Result over 30 theme steps: resonance spiked and collapsed
  (5-step bin means 0.06, 0.05, 0.15, 0.17, 0.24, 0.36; late mean
  0.298, late min 0.011, late max 0.600, 4 flags; W norm 1.64 ->
  2.57). A max-only check would have passed it (max late 0.600),
  which is why the v0.2 regression test pins the late mean and the
  late floor instead.

A pure alignment-ascent reading (L = <W h_prev, D>, theta += alpha
grad L) was also tested and rejected: W norm exploded to 13.38 in
30 steps, because ascent on a raw inner product has no bound.

## Reading 2 (v0.2): predict the next embedding, D as the signal

L = -0.5 * ||W h_prev - h||^2: the negative prediction error between
the predicted next embedding (W h_prev) and the observed next
embedding (h = h_prev + D). Ascending L is descending the
prediction loss. The error driving the update is
(W h_prev - h) = (W h_prev - h_prev) - D: the unpredicted part of
the drift. So the update is driven by D, while the theme's stable
component (the part of h that repeats) accumulates instead of
cancelling. Same drift gate (step size scaled by ||D||), same small
decay, same stability cap as v0.1.

Result over 30 theme steps, alpha 0.3 (v0.1's auto-associative
baseline in brackets): 5-step bin means 0.08, 0.12, 0.38, 0.50,
0.53, 0.51 [0.12, 0.28, 0.46, 0.55, 0.58, 0.49]; late mean 0.523
[0.535], late min 0.357 [0.234], late max 0.590 [0.728], 2 flags
[5]; W norm 1.64 -> 1.67. Same climb shape, steadier floor.

- Longer run (60 steps): bins stay 0.51-0.55 through step 60, W
  norm 1.64 -> 1.85. Stable plateau, no blow-up.
- Theme switch (30 theme A steps, then 10 theme B steps): first-3
  B mean 0.125, all-10 B mean 0.075, vs late-A mean 0.523. The
  resonance drops right back down on a new theme, as it should.
- Alpha sensitivity: alpha 0.1 climbs slower (bins 0.08 -> 0.31,
  late mean 0.254, no flags in 30 steps); alpha 0.5 is fast and
  stable (late mean 0.491, late min 0.415, 2 flags).
- Direct pins: one update drops the prediction error
  ||W h_prev - h|| from 1.0358 to 0.9552; after step 2,
  ||W[X,Y] - W[X,X]||_F is 0.1463, so the update follows the drift
  rather than the input alone.

## What changed in v0.2

- `acr_engine.py`: `_update` now takes (h_prev, h) and descends
  L = 0.5 * ||W h_prev - h||^2 via the U/V factors. `step` skips
  the update on the first step (no previous state to predict from)
  and still logs it. Resonance, confidence, flags, trace shape,
  buffer, synthesis, freeze, and reset are unchanged.
- `tests/test_engine.py`: two tests that pinned the old rule updated
  with notes (first step no longer updates; warm-ups use two
  steps). New `DeltaRuleTests`: first step logs without an update,
  second step updates, update reduces the prediction error, update
  follows the drift, repeated theme sustains resonance (the
  regression test the broken rule fails), theme switch drops
  resonance.
- `tests/test_gui.py`: freeze test warms up with two steps, with a
  note; row count updated to match.

## Canonical 16-step drive, v0.11 re-pin

The 16-step theme drive (4 sentences cycled, alpha 0.3) is the
reference run the tests pin. v0.11 changed the flag threshold and
the synthesis head's neighbour list; the update rule is untouched,
so the end state is the same and only the flag/neighbour pins
moved. Old values (v0.2-v0.10) in brackets.

- Step 16: W norm 1.5938, resonance 0.5216. Unchanged.
- Resonance series: 0.058, 0.057, 0.080, 0.133, 0.074, 0.016,
  0.036, 0.169, 0.266, 0.119, 0.242, 0.460, 0.517, 0.219, 0.482,
  0.522. Unchanged.
- Flags: steps 12 and 15 [step 15 only]. The threshold moved
  0.75 -> 0.4, calibrated to the observed scale: no run ever
  reached 0.75 (the 30-step peak is 0.59), so every flag before
  v0.11 came from the jump leg alone. Step 12 crosses 0.4
  (0.242 -> 0.460); step 15 crosses again after the dip and jumps
  0.263.
- Neighbours: 0.721 / 0.302 / 0.217 [1.000 / 0.721 / 0.302]. The
  current input is no longer its own closest match, so the list
  starts at the closest past input.
- Midpoint neighbours: 0.585 / 0.224 [0.448 / 0.222]. The midpoint
  is now between the top two past inputs; its top entry is the
  current input's text, which genuinely sits near that midpoint.
- Analogy: unchanged. A "quantum particles stay connected"
  (step 15), B "entanglement links distant particles" (step 12
  now, same text and vector as the old step-16 copy), result
  "quantum entanglement links particles" (0.470).
- Counter-examples: none for a pure theme run [three theme
  sentences]. The floor is 0.1 now: only inputs actually far from
  the current one count, and a pure theme run has none.
- A frozen whole-W_t step never flags, whatever the resonance
  jump: W_t did not move, so no new layer was found. A partial
  (rank) freeze still adapts, so it can still flag.
