# ACR: How to Use and What to Use It For

ACR v0.9. Plain words, everything the app does.

## What ACR is for

ACR is a small-scale working model of an adaptive neural loop, not a full
self-evolving engine. What you actually do with it:

- **Watch a weight matrix learn from text, live.** You feed it sentences one
  at a time and it reshapes its own weights after each one. The trace log
  shows every change.
- **Test whether a set of sentences hangs together.** Feed it 10 to 16
  sentences on one topic and resonance climbs toward 0.5. Feed it something
  off topic and it drops. That is the core experiment.
- **Find relationships in your own inputs.** The synthesis head builds
  analogies, counter-examples, and closest matches from what you typed in
  that session. It only knows your session, not general knowledge.
- **Experiment with the learning itself.** Change the learning rate, freeze
  parts of the weights, try the schedules, cap the update size, revert to an
  earlier step and see what changes.
- **Use real embeddings.** Switch the encoder to Local model and it runs on
  embeddings from a model on your PC instead of the built-in hashed encoder.
- **Analyze a whole file.** Point the loop at a text file or a folder and it
  tells you what is in it: the themes, the core entries, the outliers, and
  where the topic changes. See "Analyze a file" below.

## How to use it

Open it from the desktop icon, or type `acr` in a terminal.

1. Type a sentence in the input box and press Step, or just press Enter.
   Shift+Enter is a newline.
2. Keep going with sentences on one theme. Try quantum entanglement:
   "quantum particles stay connected", "entanglement links distant
   particles", and so on. Watch Resonance climb.
3. Type something on a different topic and watch Resonance fall. That drop is
   the forgetting problem.

The menu bar at the top holds everything: File for sessions and exports,
View for the windows, Help for this guide.

## Reading the numbers

- **Step:** how many inputs you have fed it.
- **W norm:** the size of the weight matrix. It grows as the matrix adapts.
- **Resonance:** 0 to 1. How much the input lines up with what the weights
  have learned. Theme steps land around 0.5 after a dozen steps.
- **Alpha:** the learning rate for the next step. Default 0.30.
- **Top singular value:** how strong the biggest direction in the weights is.
- **Capped:** yes in the trace log when the stability cap clipped that step.
- **Busy light:** the dot next to the status line turns gold while the app is
  working (a local model call, a batch run). Grey means it is idle.

## The windows

Everything lives in its own window, opened from the View menu:

- **Controls:** the learning rate slider, Freeze, Schedule, and the stability
  cap with the top singular value. Each control explains itself underneath.
- **Trace log:** every step, every weight change, 10 columns. Click a row and
  you can revert the weights back to that step.
- **Resonance views:** the chart over time (red dots mark new-layer
  candidates, first one usually lands at step 15 on a theme run), the heatmap
  showing what changed each step, and layer discovery, where you click a
  candidate to see the trace entries around it. When a new layer is flagged,
  the chart dot lands with a spring settle, the resonance readout glows gold,
  and the trace row highlights, so you see the moment it happened.
- **Analyze a file:** see below.
- **How to Use:** this guide.

Every window remembers its size and position between runs.

## Analyze a file

The main window's "Analyze a file" button (or View, then Analyze a file)
opens the Analyze window. Pick a file, press Go, and the loop steps through
every line, in file order, the same way typing would. A folder runs every
text file in it, in name order. Plain text, .md, and .jsonl all work; a
.jsonl record uses its text field.

While it runs, the progress bar shows how far along it is, the busy light
is lit, and Cancel stops it. The steps it finished stay: they are real
steps, in the trace log like typed ones.

When it finishes, the results land as tiles. The big tile is the headline:
how many themes were found, and the strongest one. Then:

- **Chart:** the W_t norm over the whole run.
- **Themes:** the resonance spikes and the entry that triggered each one.
  Click one to see the entries around it.
- **Core:** the most resonant entries overall. The heart of the file.
- **Outliers:** the lowest-resonance entries. The noise and off-topic lines.
- **Shifts:** the steps where resonance drops sharply and stays low. Topic
  boundaries.

A checkmark draws itself next to the status line when the run finishes.

Export curated file saves the core entries as plain text, one per line.
Export curated .jsonl saves them as one {"text": ...} object per line, ready
for training data. Export report saves the whole analysis (themes, core,
outliers, shifts) as a readable text file.

## STELLA data packs

Export STELLA data pack turns the analysis into training data packs for
STELLA. The packs do not teach a model the raw text. They teach it what ACR
found: the themes, the relationships, the analogies, and the resonance
judgments, all measured from this run.

Each pack is a folder with a .jsonl file of training rows, a README, and a
manifest, the same pack shape Data Pack Builder writes. There is one pack per
theme found, named for the theme. Entries that belong to no theme get their
judgment rows in a single off-topic pack. The rows are prompt and completion,
one JSON object per line:

- Theme membership: "What theme does this belong to? <entry>" names the
  theme the entry belongs to. The theme is the triggering entry, verbatim.
- Closest match: "What is most related to this? <entry>" names the closest
  related entry ACR measured.
- Analogy: "<A> is to <B> as <C> is to" names what the entry points at, the
  same analogy the synthesis head shows. No row when too few entries are
  resonant: an analogy is never forced.
- Resonance judgment: "Is this central to the theme or off-topic? <entry>"
  answers "Central to <theme>" or "Off-topic".

The README in each pack says what the pack is for, where the entries came
from, how many rows of each kind it holds, and what was left out and why.
The manifest lists the same details plus every row's source, step and line
number, and resonance score. Every entry is quoted verbatim from the source;
the themes, matches, analogies, and judgments are ACR's measured results.
Nothing is invented. STELLA needs 2 or more rows per file: a pack with fewer
says so in its README, so add more source material before importing it.

## Synthesis head

Below the input box, it shows an analogy ("like A is to B, your input points
at X"), counter-examples (what this session is not), closest past inputs, and
midpoint neighbours. All from your own session. With the Local model encoder
on, "Complete with local model" sends your most resonant entries to a local
model as context and shows what it says.

## Sessions and controls

- Save session and Load session keep the whole thing in one file, from the
  File menu. Export trace CSV and Export session give you readable files.
  Reset asks first.
- **Freeze:** Off, Whole W_t, or Rank directions (freeze individual
  directions 1 to 8 while the rest keep learning).
- **Schedule:** Steady, Fast-adapt (learns fast early, settles down), or
  Gentle (a third of the base rate).
- **Stability cap:** limits how big any single update can be.
- **Generate debug report** if something breaks.
- The encoder picker offers Built-in or Local model. Local model lists every
  installed model with what it does and whether it fits this PC. Nothing is
  picked for you: pick a model to use the local encoder. Built-in stays the
  default, because the app cannot run with no encoder at all.
