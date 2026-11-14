"""STELLA data packs: turn what ACR found into training packs.

A pack is a folder with three files, the same shape as Data Pack
Builder's packs:

- data.jsonl: one JSON object per line, prompt/completion, one
  format per file. Ready to import into STELLA (which needs 2 or
  more rows; a pack with fewer says so plainly in its README and
  in the window).
- README.md: the label. Pack name, date, purpose, sources,
  extractor used, counts, what was excluded and why, and the
  promise that every entry is quoted verbatim and the analysis
  is ACR's measured result, nothing invented.
- manifest.json: the same metadata, machine-readable, including
  every row's source, locator, and resonance score.

The packs do NOT teach a model the raw text. They teach what ACR
found: the themes, the relationships, the analogies, and the
resonance judgments, all measured from the run. Four row kinds,
all prompt/completion:

- Theme membership: "What theme does this belong to? <entry>"
  -> "<theme>". One per core entry. The theme label is the layer
  candidate's triggering entry text, verbatim.
- Closest match: "What is most related to this? <entry>" ->
  "<closest match entry>". One per core entry that has a
  measured nearest neighbour, using the same nearest-neighbour
  logic as the synthesis head (cosine over the buffer, distinct
  texts, positive scores only).
- Analogy: "<A> is to <B> as <C> is to" -> "<D>". Same A/B/result
  logic as Engine.synthesize: A and B are the two most resonant
  entries, the direction runs from A to B, and the result is the
  nearest other entry to the subject plus that direction. One
  per core entry that yields a real analogy; no row when the
  resonant floor is not met, never forced.
- Resonance judgment: "Is this central to the theme or
  off-topic? <entry>" -> "Central to <theme>" or "Off-topic".
  One per entry judged (the core entries and the outliers).

An entry belongs to a theme when its resonance is at or above
the resonant floor (0.4, the same floor the synthesis head uses)
and the theme is the nearest layer candidate by step. A row goes
in the pack of the theme its subject entry belongs to. Entries
in no theme get judgment rows only, in a single off-topic pack;
their other row kinds are listed as excluded in that pack's
README, never silently skipped (U32).

This module is pure: no Qt, no engine import beyond the shared
vector helpers and the resonant floor. The window gathers the
context (vectors, neighbour pool, resonant entries) from the
engine and passes it in.
"""

import json
import os
import re
from datetime import date

from acr_engine import RESONANT_FLOOR, _dot, _normalize

MIN_STELLA_ROWS = 2
EXTRACTOR = "ACR resonance analysis"

KIND_MEMBERSHIP = "Theme membership"
KIND_MATCH = "Closest match"
KIND_ANALOGY = "Analogy"
KIND_JUDGMENT = "Resonance judgment"
KIND_ORDER = (KIND_MEMBERSHIP, KIND_MATCH, KIND_ANALOGY,
              KIND_JUDGMENT)


def default_packs_dir():
    """Where packs live: ~/ACR/Packs by default."""
    return os.path.join(os.path.expanduser("~"), "ACR", "Packs")


def slugify(text):
    """A short folder-safe slug from a theme or source name:
    lower case, non-alphanumeric runs to one hyphen, max 40
    characters. Empty input gives "pack"."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    if not slug:
        slug = "pack"
    return slug[:40].rstrip("-") or "pack"


def safe_pack_name(name):
    """A name that is safe as a folder name, or '' when unusable."""
    cleaned = re.sub(r"[^A-Za-z0-9._ -]+", "", name).strip()
    cleaned = cleaned.strip(".")
    if cleaned in ("", ".", ".."):
        return ""
    return cleaned


def default_pack_name(theme):
    """Pack name from the theme slug plus today's date."""
    return "%s-%s" % (slugify(theme), date.today().isoformat())


def source_slug(source_note):
    """A slug for the source, for the off-topic pack name: the
    file name without its extension, or the note itself, or
    "session" for typed input."""
    if not source_note:
        return "session"
    base = os.path.basename(source_note)
    stem = os.path.splitext(base)[0]
    return slugify(stem or base)


def assign_theme(entry, themes):
    """The theme an entry belongs to, or None: the nearest layer
    candidate by step, when the entry's resonance is at or above
    the resonant floor. Below the floor, or with no themes at
    all, the entry belongs to no theme. Ties go to the earlier
    theme."""
    if not themes or entry["resonance"] < RESONANT_FLOOR:
        return None
    return min(themes,
               key=lambda t: (abs(t["id"] - entry["id"]),
                              t["id"]))


def nearest_neighbour(pool, vector, exclude_text):
    """The measured nearest neighbour: the distinct text in the
    pool (text to vector) with the highest cosine to vector,
    excluding exclude_text. Only positive scores count as
    related, the same rule as the synthesis head. Returns (text,
    score) or None."""
    best_text = None
    best_score = 0.0
    for text, other in pool.items():
        if text == exclude_text:
            continue
        score = _dot(vector, other)
        if score > best_score:
            best_score = score
            best_text = text
    if best_text is None:
        return None
    return best_text, round(best_score, 4)


def analogy_for(resonant, pool, vector, subject_text):
    """The synthesis head's analogy, anchored on one subject
    entry instead of the latest input. resonant is the list of
    (text, vector) most resonant entries, best first; A and B
    are the first two. The direction runs from A to B; the
    result is the nearest other entry to the subject plus that
    direction. Returns {"a", "b", "result": (text, score)} or
    None when fewer than two resonant entries exist, or no other
    entry is left to point at: no analogy is invented."""
    if len(resonant) < 2:
        return None
    a_text, a_vec = resonant[0]
    b_text, b_vec = resonant[1]
    direction = [y - x for x, y in zip(a_vec, b_vec)]
    target = _normalize([x + y for x, y in
                         zip(vector, direction)])
    skip = {a_text, b_text, subject_text}
    best_text = None
    best_score = None
    for text, other in pool.items():
        if text in skip:
            continue
        score = _dot(target, other)
        if best_score is None or score > best_score:
            best_score = score
            best_text = text
    if best_text is None:
        return None
    return {"a": a_text, "b": b_text,
            "result": (best_text, round(best_score, 4))}


def _dedup_entries(entries, item_of):
    """Entries with distinct full texts, in order. The summary
    lists dedup by preview; packs dedup by the full text, so two
    long entries sharing a prefix stay two entries."""
    out = []
    seen = set()
    for entry in entries:
        text = item_of(entry)["text"]
        if text not in seen:
            seen.add(text)
            out.append(entry)
    return out


def build_pack_rows(summary, item_of, context, source_note):
    """Plan the packs for a run. Returns a list of pack dicts:
    name, purpose, theme (trigger text or None for the off-topic
    pack), rows (each a (row dict, manifest entry dict) pair),
    excluded (plain-words reasons), sources (distinct item
    sources). One pack per theme that has rows, plus one
    off-topic pack when any entry belongs to no theme.

    item_of maps a trace entry to {"source", "locator", "text"}.
    context holds "vectors" (step id to embedding, from the
    buffer), "pool" (distinct full text to embedding), and
    "resonant" (the (text, vector) most resonant entries, best
    first). Entries older than the buffer has no vector: their
    closest-match and analogy rows are excluded and listed, not
    guessed."""
    themes = summary["themes"]
    source = source_note or "typed in this session"
    vectors = context["vectors"]
    pool = context["pool"]
    resonant = context["resonant"]
    core = _dedup_entries(summary["core"], item_of)
    outliers = _dedup_entries(summary["outliers"], item_of)

    # One pack per theme, plus the off-topic pack. Packs are
    # created when their first row lands, so a theme with no rows
    # gets no pack.
    packs = {}
    order = []

    def pack_for(theme):
        if theme is None:
            key = None
            name = "%s-off-topic-%s" % (
                source_slug(source_note),
                date.today().isoformat())
            purpose = ("Off-topic entries from the Analyze run "
                       "over %s: entries that belong to no theme."
                       % source)
        else:
            key = theme["id"]
            name = default_pack_name(item_of(theme)["text"])
            purpose = ("Theme '%s' from the Analyze run over %s."
                       % (item_of(theme)["text"], source))
        if key not in packs:
            packs[key] = {"name": name, "purpose": purpose,
                          "theme": (item_of(theme)["text"]
                                    if theme is not None else None),
                          "rows": [], "excluded": [],
                          "sources": []}
            order.append(key)
        return packs[key]

    def add_row(pack, kind, prompt, completion, entry):
        item = item_of(entry)
        row = {"prompt": prompt, "completion": completion}
        manifest_entry = {
            "kind": kind,
            "source": item["source"],
            "locator": item["locator"],
            "resonance": entry["resonance"],
            "text": item["text"],
        }
        pack["rows"].append((row, manifest_entry))
        if item["source"] not in pack["sources"]:
            pack["sources"].append(item["source"])

    def exclude(pack, kind, entry, reason):
        pack["excluded"].append(
            "No %s row for Step %d (%s): %s."
            % (kind.lower(), entry["id"], item_of(entry)["text"],
               reason))

    judged = set()
    for entry in core:
        text = item_of(entry)["text"]
        theme = assign_theme(entry, themes)
        theme_text = (item_of(theme)["text"]
                      if theme is not None else None)
        if theme is not None:
            pack = pack_for(theme)
            add_row(pack, KIND_MEMBERSHIP,
                    "What theme does this belong to? %s" % text,
                    theme_text, entry)
        else:
            off_topic = pack_for(None)
            if not themes:
                reason = "no theme was found in this run"
            else:
                reason = ("resonance %.3f is below the resonant "
                          "floor (%.1f)"
                          % (entry["resonance"], RESONANT_FLOOR))
            exclude(off_topic, KIND_MEMBERSHIP, entry, reason)
        vector = vectors.get(entry["id"])
        if theme is not None and vector is not None:
            neighbour = nearest_neighbour(pool, vector, text)
            if neighbour is not None:
                add_row(pack, KIND_MATCH,
                        "What is most related to this? %s" % text,
                        neighbour[0], entry)
            else:
                exclude(pack, KIND_MATCH, entry,
                        "no related entry was measured")
        elif theme is not None:
            exclude(pack, KIND_MATCH, entry,
                    "the entry has left the buffer")
        elif vector is None:
            exclude(off_topic, KIND_MATCH, entry,
                    "the entry belongs to no theme and has left "
                    "the buffer")
        else:
            exclude(off_topic, KIND_MATCH, entry,
                    "the entry belongs to no theme")
        if theme is not None and vector is not None:
            analogy = analogy_for(resonant, pool, vector, text)
            if analogy is not None:
                add_row(pack, KIND_ANALOGY,
                        "%s is to %s as %s is to"
                        % (analogy["a"], analogy["b"], text),
                        analogy["result"][0], entry)
            elif len(resonant) < 2:
                exclude(pack, KIND_ANALOGY, entry,
                        "fewer than two resonant entries exist")
            else:
                exclude(pack, KIND_ANALOGY, entry,
                        "no other entry to point at")
        elif theme is not None:
            exclude(pack, KIND_ANALOGY, entry,
                    "the entry has left the buffer")
        elif vector is None:
            exclude(off_topic, KIND_ANALOGY, entry,
                    "the entry belongs to no theme and has left "
                    "the buffer")
        else:
            exclude(off_topic, KIND_ANALOGY, entry,
                    "the entry belongs to no theme")
        if theme is not None:
            add_row(pack, KIND_JUDGMENT,
                    "Is this central to the theme or off-topic? "
                    "%s" % text,
                    "Central to %s" % theme_text, entry)
        else:
            add_row(off_topic, KIND_JUDGMENT,
                    "Is this central to the theme or off-topic? "
                    "%s" % text,
                    "Off-topic", entry)
        judged.add(text)

    for entry in outliers:
        text = item_of(entry)["text"]
        if text in judged:
            continue
        judged.add(text)
        theme = assign_theme(entry, themes)
        if theme is not None:
            pack = pack_for(theme)
            add_row(pack, KIND_JUDGMENT,
                    "Is this central to the theme or off-topic? "
                    "%s" % text,
                    "Central to %s" % item_of(theme)["text"],
                    entry)
        else:
            pack = pack_for(None)
            add_row(pack, KIND_JUDGMENT,
                    "Is this central to the theme or off-topic? "
                    "%s" % text,
                    "Off-topic", entry)

    return [packs[key] for key in order]


def _readme_text(pack_name, purpose, sources, rows, excluded,
                 counts, created):
    """The pack label, in plain words: pack name, date, purpose,
    sources, extractor used, counts per row kind, what was
    excluded and why, the files list, and the promise that every
    entry is verbatim and the analysis is ACR's measured result.
    Under-minimum packs say so plainly and tell the reader to
    add more source material before importing."""
    lines = []
    lines.append("# %s" % pack_name)
    lines.append("")
    lines.append("Created: %s" % created)
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append(purpose)
    lines.append("")
    lines.append("## Source")
    lines.append("")
    for source in sources:
        lines.append("- %s" % source)
    lines.append("")
    lines.append("Extractor used: %s" % EXTRACTOR)
    lines.append("")
    lines.append("## Contents")
    lines.append("")
    lines.append("Rows: %d" % len(rows))
    for kind in KIND_ORDER:
        if counts.get(kind):
            lines.append("- %s: %d" % (kind, counts[kind]))
    lines.append("")
    lines.append("## Excluded")
    lines.append("")
    if excluded:
        for note in excluded:
            lines.append("- %s" % note)
    else:
        lines.append("Nothing was excluded.")
    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append("- data.jsonl: the training rows, one JSON object "
                 "per line, prompt and completion.")
    lines.append("- README.md: this label.")
    lines.append("- manifest.json: the same details, machine "
                 "readable, with every row's source, locator, "
                 "and resonance score.")
    lines.append("")
    lines.append("Every entry is quoted verbatim from the source "
                 "listed above. The themes, matches, analogies, "
                 "and judgments are ACR's measured results from "
                 "this run. Nothing was invented.")
    lines.append("")
    if len(rows) < MIN_STELLA_ROWS:
        lines.append("Note: this pack has fewer than %d rows. "
                     "STELLA needs %d or more rows to import a "
                     "pack, so add more source material before "
                     "importing it."
                     % (MIN_STELLA_ROWS, MIN_STELLA_ROWS))
        lines.append("")
    else:
        lines.append("data.jsonl is ready to import into STELLA.")
        lines.append("")
    return "\n".join(lines)


def export_pack(pack, packs_dir):
    """Write one pack folder: data.jsonl, README.md, manifest.json.
    Returns (pack_path, info_dict) with pack_name, items (row
    count), counts per row kind, and under_minimum (True when
    the pack has fewer than 2 rows). An existing folder with the
    same name gets a -2, -3 suffix."""
    name = safe_pack_name(pack["name"]) or default_pack_name(
        pack["theme"] or "pack")
    pack_path = os.path.join(packs_dir, name)
    if os.path.exists(pack_path):
        number = 2
        while os.path.exists("%s-%d" % (pack_path, number)):
            number += 1
        pack_path = "%s-%d" % (pack_path, number)
        name = os.path.basename(pack_path)
    os.makedirs(pack_path)
    created = date.today().isoformat()
    counts = {}
    for _row, entry in pack["rows"]:
        counts[entry["kind"]] = counts.get(entry["kind"], 0) + 1

    jsonl_path = os.path.join(pack_path, "data.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as fh:
        for row, _entry in pack["rows"]:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    readme = _readme_text(name, pack["purpose"], pack["sources"],
                          pack["rows"], pack["excluded"], counts,
                          created)
    with open(os.path.join(pack_path, "README.md"), "w",
              encoding="utf-8") as fh:
        fh.write(readme)

    manifest = {
        "pack": name,
        "created": created,
        "purpose": pack["purpose"],
        "sources": list(pack["sources"]),
        "extractors": [EXTRACTOR],
        "items": len(pack["rows"]),
        "counts": {kind: counts[kind] for kind in KIND_ORDER
                   if counts.get(kind)},
        "entries": [entry for _row, entry in pack["rows"]],
    }
    with open(os.path.join(pack_path, "manifest.json"), "w",
              encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    return pack_path, {"pack_name": name,
                       "items": len(pack["rows"]),
                       "counts": counts,
                       "under_minimum":
                           len(pack["rows"]) < MIN_STELLA_ROWS}


def export_stella_packs(summary, item_of, context, dest_dir,
                        source_note):
    """Write the STELLA data packs for a run under dest_dir: one
    folder per theme that has rows, plus one off-topic pack when
    any entry belongs to no theme. Returns {"packs": [(folder,
    row count, under_minimum)], "error": None or a plain
    message}. Under-minimum packs are still written, with the
    warning in their README."""
    packs = build_pack_rows(summary, item_of, context,
                            source_note)
    if not packs:
        return {"packs": [], "error": "Nothing analyzed yet."}
    written = []
    for pack in packs:
        folder, info = export_pack(pack, dest_dir)
        written.append((folder, info["items"],
                        info["under_minimum"]))
    return {"packs": written, "error": None}
