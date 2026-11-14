#!/usr/bin/env python3
"""Build acr-v0.10.zip with correct stored file modes.

Scripts and launchers are stored as 0o100755 (regular file + exec
bits, so GUI extractors keep the exec bit); everything else is
0o100644. TASK-LIST.md, the venv, logs, and caches stay out.

v0.8: the PNG icons are rendered from the SVG when they are
missing, so the build works from a fresh clone (the repo holds
only icons/acr.svg; the PNGs are zip-only). render_icons.py ships
in the zip for exactly that: without it, a fresh clone would have
nothing to render with.
"""

import hashlib
import os
import sys
import zipfile

ROOT = os.path.dirname(os.path.abspath(__file__))
VERSION = "0.10"
ZIP_NAME = "acr-v0.10.zip"

EXEC_FILES = {
    "install.sh",
    "run.sh",
    "ACR.desktop",
    "Install-ACR.desktop",
}

PLAIN_FILES = [
    "acr_app.py",
    "acr_engine.py",
    "acr_packs.py",
    "local_encoder.py",
    "debug_log.py",
    "build_zip.py",
    "render_icons.py",
    "README.md",
    "CHANGELOG.md",
    "HOW-TO-USE.md",
    "DELTA-RULE-PROBE.md",
    "icons/acr.svg",
    "icons/acr-48.png",
    "icons/acr-256.png",
    "tests/__init__.py",
    "tests/test_engine.py",
    "tests/test_debug_scrub.py",
    "tests/test_gui.py",
    "tests/test_viz.py",
    "tests/test_sessions.py",
    "tests/test_local_encoder.py",
    "tests/test_synthesis.py",
    "tests/test_stability.py",
    "tests/test_polish.py",
    "tests/test_installer.py",
    "tests/test_design_rules.py",
    "tests/test_analyze.py",
    "tests/test_v10.py",
]


def _ensure_icons():
    """Render the PNG icons from the SVG when they are missing.

    The repo holds only icons/acr.svg, so a fresh clone has no
    PNGs and the build used to fail on them. Rendering needs the
    interface library (render_icons.py does the work); when it
    cannot run, the build stops with a plain message instead of
    a bare MISSING line. Returns True when the PNGs are present
    afterwards.
    """
    missing = [name for name in ("icons/acr-48.png",
                                 "icons/acr-256.png")
               if not os.path.exists(os.path.join(ROOT, name))]
    if not missing:
        return True
    print("icons missing (%s), rendering from the SVG..."
          % ", ".join(missing))
    sys.path.insert(0, ROOT)
    try:
        import render_icons
    except ImportError as exc:
        print("cannot render icons: %s" % exc)
        return False
    try:
        return render_icons.main() == 0
    except Exception as exc:
        print("icon render failed: %s" % exc)
        return False


def build():
    if not _ensure_icons():
        return 1
    zip_path = os.path.join(ROOT, ZIP_NAME)
    if os.path.exists(zip_path):
        os.remove(zip_path)
    files = [(name, 0o100755) for name in sorted(EXEC_FILES)]
    files += [(name, 0o100644) for name in PLAIN_FILES]
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, mode in files:
            full = os.path.join(ROOT, name)
            if not os.path.exists(full):
                print("MISSING: %s" % name)
                return 1
            info = zipfile.ZipInfo("acr/" + name)
            info.external_attr = mode << 16
            with open(full, "rb") as fh:
                zf.writestr(info, fh.read())
    # Verify stored modes.
    ok = True
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            mode = info.external_attr >> 16
            want = 0o100755 if os.path.basename(info.filename) \
                in EXEC_FILES else 0o100644
            if mode != want:
                print("BAD MODE: %s has %o, want %o"
                      % (info.filename, mode, want))
                ok = False
    size = os.path.getsize(zip_path)
    digest = hashlib.sha256()
    with open(zip_path, "rb") as fh:
        digest.update(fh.read())
    print("built %s" % zip_path)
    print("size: %d bytes" % size)
    print("sha256: %s" % digest.hexdigest())
    print("modes ok: %s" % ok)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(build())
