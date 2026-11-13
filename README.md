# ACR - Adaptive Contextual Resonance

A desktop program for the adaptive loop at small scale. Type text,
step the loop, and watch the weight matrix adapt to what you type.

## What this is

A real, runnable prototype of the core idea from the blueprint:

- An encoder turns your text into a vector (deterministic, offline,
  no downloads, no model files).
- A weight matrix W_t (low-rank, rank 8) updates a little on every
  step, learning to reconstruct the input patterns it keeps seeing.
  Repeated themes strengthen the direction they share.
- Resonance is the match between your input and the dominant
  direction of W_t, from 0 to 1. A spike flags a new-layer candidate.
- The trace log records every step: what went in, how far W_t moved,
  the resonance, and the confidence.

## What this is not

This is not a full self-evolving neural engine. There is no trained
language model in here, and the synthesis head only surfaces related
past inputs from your own session. It does not generate text from
general knowledge. The honest label is in the window, and it stays
there until a bigger engine actually exists.

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

1. Type a sentence in the input box.
2. Press Step. The readouts update, a row appears in the trace log,
   and the synthesis head shows related past inputs.
3. Step a few sentences on one theme and watch the resonance climb.
   Switch themes and watch it drop.
4. Alpha (the slider) sets how fast W_t adapts. Freeze weights logs
   steps without changing W_t. The stability cap bounds each update.
5. Export trace JSON saves the whole trace log as a file.

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
takes a small gradient step so W_t gets better at reconstructing the
patterns it sees, with the step size gated by the drift and a small
decay so stale directions fade. Resonance is the cosine between the
sentence vector and the top direction of W_t, found by power
iteration. Confidence is the share of W_t's energy in that top
direction.

## Files

- `acr_app.py` - the window
- `acr_engine.py` - the adaptive loop (standard library only)
- `debug_log.py` - the debug log layer (standard library only)
- `install.sh` / `run.sh` - install and run
- `ACR.desktop` / `Install-ACR.desktop` - menu and desktop launchers
- `icons/` - icon source and images
- `tests/` - the test suite (`python3 -m unittest discover -s tests`)

## Requirements

Linux, Python 3.10 or newer. The installer puts the interface
library in a private folder, so nothing else needs installing.
