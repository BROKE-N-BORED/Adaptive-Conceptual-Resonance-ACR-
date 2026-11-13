#!/usr/bin/env python3
"""Build acr-v0.1.zip with correct stored file modes.

Scripts and launchers are stored as 0o100755 (regular file + exec
bits, so GUI extractors keep the exec bit); everything else is
0o100644. TASK-LIST.md, the venv, logs, and caches stay out.
"""

import hashlib
import os
import sys
import zipfile

ROOT = os.path.dirname(os.path.abspath(__file__))
VERSION = "0.1"
ZIP_NAME = "acr-v0.1.zip"

EXEC_FILES = {
    "install.sh",
    "run.sh",
    "ACR.desktop",
    "Install-ACR.desktop",
}

PLAIN_FILES = [
    "acr_app.py",
    "acr_engine.py",
    "debug_log.py",
    "build_zip.py",
    "README.md",
    "icons/acr.svg",
    "icons/acr-48.png",
    "icons/acr-256.png",
    "tests/__init__.py",
    "tests/test_engine.py",
    "tests/test_debug_scrub.py",
    "tests/test_gui.py",
]


def build():
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
