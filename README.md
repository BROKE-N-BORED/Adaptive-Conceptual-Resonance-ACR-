# ACR v0.10 - Adaptive Contextual Resonance

A desktop program for the adaptive loop at small scale. Type text,
step the loop, and watch the weight matrix adapt to what you type.
Or point the loop at a whole file and see what is in it.

## What this is

A real, runnable prototype of the core idea from the blueprint:

- An encoder turns your text into a vector. The built-in encoder is
  deterministic, offline, no downloads, no model files. When the
  local model service is running on this computer, you can switch
  to its embeddings instead (see step 9 below); the built-in
  encoder stays the default and the fallback.
- A weight matrix W_t (low-rank, rank 8) updates a little on every
  step, learning to predict the next input from the previous one.
  The update is driven by the drift between the two (the residual
  signal from the blueprint), so repeated themes strengthen the
  direction they share.
- Resonance is the match between your input and the dominant
  direction of W_t, from 0 to 1. A spike flags a new-layer candidate.
- The trace log records every step: what went in, how far W_t moved,
  the resonance, the confidence, and whether the cap clipped the
  step (10 columns in all).
- Analyze a file runs the same loop over every line of a text file
  or a folder of them, and shows what is in it: the themes, the
  core entries, the outliers, and where the topic changes. The
  core entries export as a curated file, plain text or .jsonl.

## What this is not

This is not a full self-evolving neural engine. There is no trained
language model in here: the synthesis head surfaces related past
inputs, analogies, and counter-examples from your own session, and
the optional completion is the local model's own words, labelled
with the model name in the panel. ACR itself does not generate text
from general knowledge. The honest label is in the window, and it
stays there until a bigger engine actually exists.

## Install

Unzip the release anywhere, then double-click `Install-ACR`.

If the file manager asks, right-click `Install-ACR` first and pick
Allow Launching, then double-click it. Or run this in a terminal:

```
bash install.sh
```

The installer creates a private folder with everything ACR needs,
puts an ACR icon in your menu and on your desktop, and adds an `acr`
command. Nothing is installed system-wide.

## Run

Double-click the ACR icon, or run:

```
acr
```

Or, from the unzipped folder:

```
bash run.sh
```

## Use

The menu bar holds everything: File for sessions and exports, View
for the windows, Help for the How to Use guide. The main window
stays simple: encoder, input, Step, the key readouts, and the
synthesis head. The rest lives in its own window, opened from View.

1. Type a sentence in the input box.
2. Press Step, or Enter in the input box. The readouts update, a
   row appears in the trace log, and the synthesis head shows
   related past inputs. New here? Try an example (on the empty
   main window) steps four sentences on one theme for you, so you
   can see the loop work before typing anything.
3. Step a few sentences on one theme and watch the resonance climb.
   Switch themes and watch it drop.
4. The Controls window holds the learning controls. Alpha (the
   slider) sets the base learning rate: how fast W_t adapts. It
   starts at 0.30, the setting the resonance climb was verified
   at. The Schedule picker scales it: Steady keeps the base,
   Fast-adapt starts at twice the base and settles back to it
   over the first steps, Gentle runs at one third of the base.
   The Alpha readout shows the rate the next step will use, and
   the trace log records it per step. Freeze: Off adapts as
   usual; Whole W_t logs steps without changing W_t; Rank
   directions freezes only the checked directions (1-8) while
   the rest keep adapting. The stability cap bounds each update;
   the top singular value next to it is the size of W_t's
   strongest direction, and the Capped column in the trace log
   marks steps the cap actually clipped. Every control explains
   itself underneath.
5. Export trace JSON saves the whole trace log as a file. Export
   trace CSV saves the same log as a spreadsheet-friendly file.
   Both are on the File menu.
6. The Resonance views window shows the W_t norm over time (red
   dots are resonance spikes), the change heatmap for one step
   (pick a step to look back, or leave it on the latest), and the
   layer candidates: resonance spikes with the input that
   triggered them. Click a candidate to see its trace entries.
   Hover over the chart to see a step's values; click a point
   (or drag across the chart) to jump the heatmap and the trace
   selection to that step. When a layer is flagged, a gold dot
   stays next to the resonance readout on the main window until
   the next flag or a reset.
7. Save session writes the whole session (W_t, buffer, trace,
   settings) to one JSON file. Load session brings it back exactly
   where you left off. Export session writes the same session as
   readable text: the inputs and trace in time order, with the
   final state at the top.
8. Reset asks first, then clears the session and puts W_t back to
   its start.
9. Encoder: the picker offers Built-in or Local model. Built-in
   stays the default, because the app cannot run with no encoder
   at all. Local model uses the embeddings of the local model
   service running on this computer (Ollama, at
   http://127.0.0.1:11434). Picking Local model lists every
   installed model with what it does and whether it fits this PC;
   nothing is picked for you, pick a model to use the local
   encoder. Start service starts the service for you if it is
   not running, so you never need the terminal for it. Switching
   encoders starts a fresh W_t, because the embedding sizes differ:
   the trace and buffer clear, and the window says so. If the
   service stops mid-session, ACR falls back to the built-in
   encoder with a fresh W_t and says so plainly. Sessions remember
   which encoder they used: a session saved with the local encoder
   needs the service running (with that model installed) to load
   back. Every local-model call runs in a worker thread, so the
   window stays responsive while the model works: the model list,
   the activation probe, and starting the service, as well as
   embeddings and completions. The busy light is lit and Cancel
   is offered; starting the service can be abandoned mid-wait.
10. The synthesis head shows more than neighbours now: an Analogy
    (like one resonant input is to another, your input points at
    a third past input), and Counter-examples (the past inputs
    farthest from the current one: what this session is not).
    When the Local model encoder is active, Complete with local
    model asks an installed model to continue the theme from the
    most resonant inputs; its words show in the panel, labelled
    with the model name. The button is off until a model is picked
    in its picker, which lists every installed model the same way
    the encoder picker does. The completion is on-demand only:
    stepping never generates anything.
11. Revert: click a step in the trace log, then Revert W_t to
    selected step, to roll W_t back to its state after that
    step. It asks first, and shows the W norm the revert will
    land on, and the trace log keeps a revert entry so the
    history shows what happened. The last 200 steps are
    kept for reverting on an 8 GB machine; the limit adapts to
    the machine's RAM (fewer on smaller machines, more on bigger
    ones), because snapshots are real memory. Older steps age out.
    A loaded session starts its revert history from the loaded
    state.
12. Analyze a file (the button on the main window, or View, then
    Analyze a file) runs the same loop over every line of a text
    file or a folder of them, in file order. Plain text, .md, and
    .jsonl all work; a .jsonl record uses its text field. The
    progress bar shows how far along it is, Cancel stops it, and
    the steps it finished stay in the trace like typed ones. While
    it runs, the chart and the headline tile update as steps land.
    When it finishes, the results land as tiles: one big tile for
    the headline (themes found and the strongest theme), then tiles
    for the resonance chart over the run, the themes found, the
    core entries (the most resonant: the heart of the file), the
    outliers (the noise and off-topic lines), and the theme shifts
    (where the topic changes). Hover over a finding to see the
    whole entry; double-click one to read it in full. Export
    curated file saves the core entries as plain
    text, one per line; Export curated .jsonl saves them as one
    {"text": ...} object per line; Export report saves the whole
    analysis as readable text. Export STELLA data pack first shows
    a preview of the packs it is about to write (names, row counts,
    first rows); Write these packs saves them. Each pack is one
    training-data pack per theme found (a folder with a .jsonl
    file of training rows, a README, and a manifest), plus one
    off-topic pack for the entries that belong to no theme. The
    packs teach a model what ACR found, not the raw text: theme
    membership, closest matches, analogies, and resonance
    judgments, all measured from the run. Every entry is verbatim
    from the source: nothing is invented. STELLA needs 2 or more
    rows per file, so a pack with fewer says so in its README.

Every window remembers its size and position between runs.
Re-clicking the icon focuses the running app; it never starts a
second copy. The View menu shows which windows are open, and
opens or closes them. Keyboard: Tab moves in visual order, the
input has the focus at launch, and every menu action has a
shortcut (Ctrl+S save, Ctrl+O load, Ctrl+Q quit, Ctrl+1 to Ctrl+4
for the windows, F1 for How to Use; the full list is in the How
to Use guide). The View menu also carries Reduced motion and Big
text: one tap each, saved between runs.

The app has a dark look drawn from its icon: deep dark surfaces,
gold for resonance, blue as the secondary. When the loop flags a
new layer, the chart dot lands with a spring settle, the resonance
readout glows gold, and the trace row highlights, then everything
rests. When a run or a pack export finishes, a checkmark draws
itself next to the status line. The motion is one-shot and under
half a second, and it marks real events only; when nothing is
happening, the UI is still.

## If something breaks

Press Generate debug report in the window. It writes one file with
the version, the system details, the recent actions, and any errors.
Send that file to broke.n.bored.studio@gmail.com. The report never
carries passwords, keys, or tokens.

## How it works, in plain words

Each word gets a fixed vector from its hash, so the same text always
encodes the same way. The sentence vector is the average of its word
vectors. W_t starts random and small. On each step the program
compares the new sentence vector with the last one (the drift), then
takes a small gradient step so W_t gets better at predicting the new
sentence vector from the last one: the error driving the step is the
unpredicted part of the drift. The step size is gated by the drift
and a small decay fades stale directions. The first step has no
previous sentence, so it is logged without an update. Resonance is
the cosine between the sentence vector and the top direction of W_t,
found by power iteration. Confidence is the share of W_t's energy in
that top direction.

The schedule presets scale the base alpha: Steady is the base,
Fast-adapt is the base times (1 + 2^(-(n-1)/3)) for step n (twice
the base on step 1, 1.5x on step 4, 1.25x on step 7, settling on
the base), Gentle is the base divided by 3. The top singular value
is the size of W_t's strongest direction; the stability cap bounds
the size of each update step, not W_t itself. Freezing a rank
direction holds that component of U and V exactly still: it skips
both the decay and the gradient step.

The analogy takes the two most resonant past inputs (layer
candidates first) and applies the direction between them to the
current input: the nearest other past input to that point is the
answer. Counter-examples are the past inputs with the lowest
cosine to the current input. Both are built only from the
session's own buffer and the resonance W_t reports; nothing comes
from outside the session. A step counts as resonant when its
resonance is at least 0.4: a fresh theme scores about 0.05-0.20
and a settled theme holds about 0.45-0.55, so 0.4 sits between
the noise and the plateau. Every synthesis list shows each input
once: repeated inputs are the same point, so they collapse to one
line each instead of repeating.

Two limits in the app have concrete reasons. The latent state
buffer holds the last 50 inputs: the synthesis head works from
recent context, and older inputs have faded from what W_t has
learned, so keeping more would only slow the neighbour search.
The revert history keeps the last 200 steps on an 8 GB machine
(the limit scales with RAM): each snapshot is two factor
matrices, so an unbounded history would grow with every step
until the machine ran out of room.

A note on v0.1: the first version updated W_t with a different rule
(learning to reconstruct each input) because the first attempt at
the blueprint's drift-driven rule chased noise. v0.2 fixes the rule
itself instead: the drift on its own is mostly word-sampling noise
at this scale, so the working form of the rule predicts the next
input from the previous one, with the drift as the signal that
drives each update. The probe numbers behind that call are in
DELTA-RULE-PROBE.md.

## Files

- `acr_app.py` - the window and the resonance views
- `acr_engine.py` - the adaptive loop (standard library only)
- `acr_packs.py` - the STELLA data pack writer: theme
  membership, closest matches, analogies, and resonance
  judgments as prompt/completion rows, one pack per theme
- `local_encoder.py` - the local-model encoder client (standard
  library only)
- `DELTA-RULE-PROBE.md` - the probe numbers behind the v0.2 rule fix
- `HOW-TO-USE.md` - the How to Use guide, shown in the app's Help menu
- `CHANGELOG.md` - what changed in each version, in plain words
- `debug_log.py` - the debug log layer (standard library only)
- `install.sh` / `run.sh` - install and run
- `build_zip.py` / `render_icons.py` - build the release zip; the
  icons render from the SVG when they are missing
- `ACR.desktop` / `Install-ACR.desktop` - menu and desktop launchers
- `icons/` - icon source and images
- `tests/` - the test suite (`python3 -m unittest discover -s tests`)

## Requirements

Linux, Python 3.10 or newer. The installer puts the interface
library in a private folder, so nothing else needs installing.
The Local model encoder needs the local model service (Ollama)
installed and running on this computer, with at least one model
installed; everything else works without it.
