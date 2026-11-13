# ACR changelog

What changed in each version, in plain words. The newest is first.

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
