# ACR changelog

What changed in each version, in plain words. The newest is first.

## v0.10 (2026-11-13) - Audit redesign

- Every local-model call runs in a worker thread now, not just
  embeddings and completions: the model list, the activation
  probe, and starting the service. The window stays responsive
  through all three, with the busy light lit, a status line
  saying what is happening, and Cancel offered. Starting the
  service can be abandoned mid-wait.
- Keyboard navigation: a focused control shows a gold ring, Tab
  moves in visual order, the input has the focus at launch, and
  every menu action has a shortcut (Ctrl+S save, Ctrl+O load,
  Ctrl+Q quit, Ctrl+1 to Ctrl+4 for the windows, F1 for How to
  Use; the full list is in the How to Use guide).
- Entry text is readable everywhere: hover over a finding in the
  Analyze tiles to see the whole entry, double-click one to read
  it in full, and click a trace row to see its full input below
  the table.
- Try an example: the empty main window offers four sentences on
  one theme, stepped as real steps, so a first run teaches by
  playing. It goes away once you step anything.
- Analyze runs stream live: the chart and the headline tile
  update as steps land, not only at the end.
- The chart scrubs: hover to see a step's values, click or drag
  to jump the heatmap and the trace selection to that step.
- When a layer is flagged, a gold dot stays next to the resonance
  readout until the next flag or a reset, naming the flagged step
  on hover.
- The Controls window shows the live alpha next to the slider.
- Accessibility in the View menu: a Reduced motion toggle and a
  Big text preset, one tap each, saved between runs. The chart
  legend's red is lightened so its contrast passes the bar.
- In built-in mode the main window collapses to the encoder
  picker, the Start service button, and one line ("Built-in
  encoder, dim 64."). Picking Local model brings the model
  picker, notes, and completion row back.
- Pack export previews before writing: names, row counts per
  kind, and the first rows, then Write these packs saves them.
  The revert confirm shows the W norm the revert will land on.
- The trace table follows the newest row only when you are
  already at the bottom; scrolled up, your place stays put.
- Long runs are cheaper: batch progress updates the bar at most
  every 100 ms, and the heat history is bounded (the oldest half
  is sampled, and a step whose data was sampled out says so).
  Windows open with an opacity fade instead of a geometry
  animation.
- The View menu shows which windows are open, and opens or
  closes them.
- Copy and empty-state polish: the Analyze chart's empty message
  fits Analyze, the hero and progress labels say which steps
  they count, a loaded session's source shows in the Analyze
  window, the empty Trace window guides instead of showing blank
  boxes, Revert is disabled with no steps, How to Use opens with
  no heading highlighted, and the W norm reads "-" at step 0
  like Resonance does.
- No change to the update rule: the 16-theme numbers are still
  exact.

## v0.9 (2026-11-13) - UI rebuild and Analyze a file

- Analyze a file: point the loop at a text file or a folder and see
  what is in it. Every line becomes one step, in file order, the
  same way typing would. The results land as tiles: one big tile
  for the headline (themes found and the strongest theme), then
  tiles for the resonance chart, the themes found, the core
  entries (the heart of the file), the outliers (the noise and
  off-topic lines), and the theme shifts (where the topic
  changes). The core entries export
  as a curated file, plain text or .jsonl, and the whole analysis
  exports as a readable report. Export STELLA data pack writes
  one training-data pack per theme (a folder with a .jsonl file,
  a README, and a manifest), plus one off-topic pack for the
  entries that belong to no theme. The packs teach a model what
  ACR found: theme membership, closest matches, analogies, and
  resonance judgments, all measured from the run. It runs in the
  background with a progress bar and Cancel, and the steps it
  finishes stay in the trace like typed ones.
- A real menu bar: File for sessions and exports, View for the
  windows, Help for the How to Use guide.
- The crowded single window is split into dedicated windows: Main
  (encoder, input, Step, readouts, synthesis), Controls (learning
  rate, freeze, schedule, cap), Trace log, Resonance views, and
  Analyze a file. Nothing was removed, only reorganised. Every
  window remembers its size and position.
- How to Use window: the guide from the Help menu, with an index
  of sections. It reads HOW-TO-USE.md, which ships in the release.
- Re-clicking the icon focuses the running app instead of starting
  a second copy.
- A busy light next to the status line: gold while the app is
  working, grey at rest.
- Local-model calls run in a worker thread now, so the window stays
  responsive while the model works, with Cancel offered.
- The model picker starts unselected and shows what each model does
  and whether it fits this PC. Nothing is picked for you.
- Every field explains itself in plain words underneath it.
- A real visual identity, drawn from the icon: a dark UI with deep
  dark surfaces, gold as the resonance accent and blue as the
  secondary. One look across every window.
- The resonance moment: when the loop flags a new layer, the chart
  dot lands with a spring settle, the resonance readout glows gold,
  and the trace row highlights. When a run or a pack export
  finishes, a checkmark draws itself next to the status line. All
  motion is one-shot and under half a second; when nothing is
  happening, the UI rests.
- The revert history limit adapts to the machine's RAM: 200 steps
  on an 8 GB machine, fewer on smaller machines, more on bigger
  ones.
- No change to the update rule: the 16-theme numbers are still
  exact.

## v0.8.1 (2026-11-13) - Launch fix

- The desktop and menu icon now launch the app. It used to look
  for the program in the wrong folder, so clicking the icon did
  nothing. The icon now points straight at the installed program.
- No new features, no change to the update rule.

## v0.8 (2026-11-13) - Final polish

- README final pass: every number and claim checked against the
  shipped program.
- Icon final: the icon renders cleanly at menu and desktop sizes,
  and the window carries it.
- The build works from a fresh clone: the PNG icons render from
  the SVG when they are missing.
- The chart legend moved to the bottom-left, where a red dot at
  the top-right can no longer land underneath it.
- The neighbour lists show each input once, even when the same
  sentence was stepped more than once.
- Enter in the input box steps the loop, like the Step button.
- The window remembers its size and position between runs.
- No new features, no change to the update rule.

## v0.7 (2026-11-13) - Stability + control

- Freeze modes: freeze the whole W_t, or freeze individual rank
  directions (1-8) while the rest keep adapting.
- Learning-rate schedule presets: Steady (constant), Fast-adapt
  (starts at twice the base and settles down), Gentle (one third
  of the base). The trace log records the effective rate each step.
- Spectral cap visibility: a live top-singular-value readout next
  to the cap, and a new Capped column in the trace log marking
  steps the cap actually clipped. The cap bounds each update step,
  not W_t itself.
- Revert W_t: click any trace row and roll W_t back to its state
  after that step (the last 200 steps are kept). The trace log
  keeps a revert entry, so the history stays honest.

## v0.6 (2026-11-13) - Synthesis head upgrade

- Analogies: like one resonant input is to another, your input
  points at a third past input. All three are shown, with the score.
- Counter-examples: the past inputs farthest from the current one,
  what this session is not.
- Complete with local model: when the Local model encoder is
  active, ask another installed model to continue the theme from
  the most resonant inputs. Its words show in the panel, labelled
  with the model name.
- The window is 1100x700 now, to fit the longer synthesis panel.

## v0.5 (2026-11-13) - Optional local-model encoder

- Encoder picker: Built-in (the default) or Local model, using the
  embeddings of the local model service running on this computer.
- Start service button starts the service from inside ACR, so no
  terminal is needed.
- Health check and retry with plain messages: ACR says whether the
  service is down or just slow.
- If the service stops mid-session, ACR falls back to the built-in
  encoder with a fresh W_t.
- Sessions record which encoder they used. Older session files
  still load as built-in.
- Under the hood, the norm and dominant direction now compute in
  the rank-8 space, so large embeddings stay fast.

## v0.4 (2026-11-13) - Sessions

- Save session writes the whole session (W_t, buffer, trace,
  settings) to one JSON file. Load session brings it back exactly
  where you left off.
- Export trace CSV saves the trace log as a spreadsheet-friendly
  file.
- Export session writes the same session as readable text: the
  inputs and trace in time order, with the final state at the top.
- Reset asks first, then clears the session and puts W_t back to
  its start.

## v0.3 (2026-11-12) - Resonance visualisation

- New Resonance views tab: a chart of the W_t norm over time, with
  red dots on resonance spikes; a heatmap of which rows moved most
  on one step (pick a step to look back); and layer candidates,
  resonance spikes with the input that triggered them. Click a
  candidate to see its trace entries.

## v0.2 (2026-11-12) - The update rule, fixed

- The update rule now follows the blueprint: W_t learns to predict
  the next input from the previous one, with the drift between the
  two driving each update. The first version's substitute rule
  (learning to reconstruct each input) is gone.
- Default alpha raised to 0.30, the setting the resonance climb
  was verified at.

## v0.1 (2026-11-12) - First version

- Native desktop window: type text, press Step, and W_t adapts.
- Deterministic built-in encoder (offline, no downloads), low-rank
  W_t (rank 8), resonance detection, and a trace log recording
  every step.
- Synthesis demo: related past inputs from the session, honestly
  labelled as not a language model.
- Alpha slider, freeze toggle, stability cap, trace export as JSON,
  and a debug report generator.
- Private-folder installer, menu and desktop launchers, and an
  `acr` command. Nothing is installed system-wide.
