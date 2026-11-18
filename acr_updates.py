"""Update check, verified download, staging, and apply (v0.12).

Stdlib-only, no Qt: the launcher (acr_launch.py) and the
restart-now updater script both use this module without ever
importing the interface library.

The shape follows Pixel Forge v0.69's updates.py: the check is
one GET to the public releases endpoint, it sends no identifying
information and needs no token, and check_for_update never raises
for a network or parsing problem -- a failed check comes back as
state "unknown" with a plain reason.

Where ACR goes further than Pixel Forge: the check also returns
the release's download links, and the app can pull the update
itself -- but only through the verified path. The release carries
a checksum sidecar (acr-vX.Y.zip.sha256); the downloaded zip's
SHA-256 must match it, or the update is refused and the download
thrown away. A release with no sidecar is refused the same way:
no blind installs, ever.

A verified update is staged in ~/.local/share/ACR/update-pending/
(the zip plus a manifest with its version and checksum), and
applied from there in one of two ways:

- Restart now: the app writes the updater script into the
  staging dir, starts it detached, and quits. The script waits
  for the app to exit, applies the staged update, and relaunches
  run.sh.
- Restart later: the staged update stays put, and the launcher
  applies it before the next start.

apply_staged_update overwrites only the files the zip holds, so
the private .venv and the logs folder are never touched. Files a
newer version dropped linger in the tree (nothing is ever
deleted); that is harmless at this tree's size, and safer than
guessing what to remove. The swap itself is file-by-file, so a
disk failure mid-copy could leave a mixed tree -- the zip is
extracted and checked before anything is touched, which keeps
that to disk errors only; a failed apply drops the staging and
says so in plain words.
"""

import hashlib
import json
import os
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile

LATEST_RELEASE_URL = ("https://api.github.com/repos/BROKE-N-BORED/"
                      "Adaptive-Conceptual-Resonance-ACR-/releases/latest")
RELEASES_PAGE_URL = ("https://github.com/BROKE-N-BORED/"
                     "Adaptive-Conceptual-Resonance-ACR-/releases")
CHECK_TIMEOUT_S = 8
DOWNLOAD_TIMEOUT_S = 10
DOWNLOAD_CHUNK = 65536
ZIP_PREFIX = "acr/"
# Top-level entries the updater never touches: the private venv
# (rebuilt only by install.sh) and the debug logs.
PRESERVE_TOP = (".venv", "logs")


class ACRUpdateError(Exception):
    """An update step failed, with a plain-words reason."""


class UpdateCancelled(Exception):
    """The user cancelled the download."""


def parse_version(text):
    """A version string as a tuple of ints -- "0.12" becomes
    (0, 12), "v0.12" the same -- or None when it is not a
    version at all. Anything but an optional "v" followed by
    numbers and dots is refused, so a tag like "v0.12-beta"
    reads as no usable version rather than a wrong one."""
    if not isinstance(text, str):
        return None
    text = text.strip()
    if text.startswith("v"):
        text = text[1:]
    if not text:
        return None
    parts = text.split(".")
    numbers = []
    for part in parts:
        if not part.isdigit():
            return None
        numbers.append(int(part))
    return tuple(numbers)


def is_newer(latest, current):
    """True when the latest version is newer than the running
    one. Either side unparsable means no -- an unknown version
    is never reported as an update. Shorter tuples are padded
    with zeros, so 0.12 and 0.12.0 compare equal."""
    latest_parsed = parse_version(latest)
    current_parsed = parse_version(current)
    if latest_parsed is None or current_parsed is None:
        return False
    width = max(len(latest_parsed), len(current_parsed))
    latest_padded = latest_parsed + (0,) * (width - len(latest_parsed))
    current_padded = current_parsed + (0,) * (width - len(current_parsed))
    return latest_padded > current_padded


def fetch_latest_release(urlopen=None, url=None):
    """The latest release from GitHub's public endpoint, as
    {"version", "url", "zip_url", "sha_url"}. zip_url and
    sha_url come from the release's assets, matched by name:
    the .zip file and its .zip.sha256 sidecar. sha_url is None
    when the release carries no sidecar -- the download step
    refuses that, it is not a check error. Raises
    ACRUpdateError with a plain reason when the network fails,
    the answer is not JSON, it carries no tag, or it has no
    download file: the caller turns every one of those into
    "unknown", never a crash. urlopen and url are test seams --
    pass a stub or a fake endpoint and no real network is
    touched."""
    opener = urlopen or urllib.request.urlopen
    target = url or LATEST_RELEASE_URL
    try:
        with opener(target, timeout=CHECK_TIMEOUT_S) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        reason = exc.reason
        if isinstance(reason, TimeoutError):
            raise ACRUpdateError("the update check timed out")
        raise ACRUpdateError(
            "could not reach GitHub: %s" % reason)
    except TimeoutError:
        raise ACRUpdateError("the update check timed out")
    except (ValueError, UnicodeDecodeError):
        raise ACRUpdateError(
            "the releases page did not answer in a form "
            "ACR understands")
    if not isinstance(payload, dict):
        raise ACRUpdateError(
            "the releases page did not answer in a form "
            "ACR understands")
    tag = payload.get("tag_name")
    if not isinstance(tag, str) or not tag:
        raise ACRUpdateError(
            "the releases page did not name a version")
    version = parse_version(tag)
    if version is None:
        raise ACRUpdateError(
            "the latest release tag did not look like a version: %s"
            % tag)
    version_text = ".".join(str(part) for part in version)
    page_url = payload.get("html_url") or RELEASES_PAGE_URL
    zip_url = None
    sha_url = None
    assets = payload.get("assets") or []
    if isinstance(assets, list):
        for asset in assets:
            if not isinstance(asset, dict):
                continue
            name = asset.get("name") or ""
            link = asset.get("browser_download_url") or ""
            if not link:
                continue
            if name.endswith(".zip.sha256"):
                if sha_url is None:
                    sha_url = link
            elif name.endswith(".zip"):
                if zip_url is None:
                    zip_url = link
    if zip_url is None:
        raise ACRUpdateError(
            "the latest release has no download file")
    return {"version": version_text, "url": page_url,
            "zip_url": zip_url, "sha_url": sha_url}


def check_for_update(current_version, urlopen=None, url=None):
    """One call, three outcomes, never an exception for a
    network or parsing problem:

    {"state": "newer", "latest": "0.13", "url": "<release page>",
     "zip_url": ..., "sha_url": ...} -- a newer release is out.
    {"state": "current", "latest": "0.12", ...} -- the running
    version is the newest (or newer than the latest release, a
    dev tree).
    {"state": "unknown", "latest": None, "url": None,
     "reason": "<plain words>"} -- the check could not run.
    """
    try:
        release = fetch_latest_release(urlopen=urlopen, url=url)
    except ACRUpdateError as exc:
        return {"state": "unknown", "latest": None, "url": None,
                "reason": str(exc)}
    result = {"latest": release["version"], "url": release["url"],
              "zip_url": release["zip_url"],
              "sha_url": release["sha_url"]}
    if is_newer(release["version"], current_version):
        result["state"] = "newer"
    else:
        result["state"] = "current"
    return result


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(DOWNLOAD_CHUNK)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _download_file(url, dest_path, opener, progress_cb, cancel):
    """Stream one URL to a file. progress_cb(downloaded, total)
    fires per chunk, total None when the server did not say.
    cancel() is checked between chunks; a cancelled download
    raises UpdateCancelled. Network trouble raises
    ACRUpdateError with a plain reason."""
    try:
        response = opener(url, timeout=DOWNLOAD_TIMEOUT_S)
    except urllib.error.URLError as exc:
        raise ACRUpdateError(
            "could not reach the download: %s" % exc.reason)
    except TimeoutError:
        raise ACRUpdateError("the download timed out")
    try:
        with response as resp:
            total = None
            headers = getattr(resp, "headers", None)
            if headers is not None:
                length = headers.get("Content-Length")
                if length:
                    try:
                        total = int(length)
                    except ValueError:
                        total = None
            done = 0
            if progress_cb is not None:
                progress_cb(0, total)
            with open(dest_path, "wb") as fh:
                while True:
                    if cancel is not None and cancel():
                        raise UpdateCancelled()
                    try:
                        chunk = resp.read(DOWNLOAD_CHUNK)
                    except TimeoutError:
                        raise ACRUpdateError(
                            "the download timed out")
                    if not chunk:
                        break
                    fh.write(chunk)
                    done += len(chunk)
                    if progress_cb is not None:
                        progress_cb(done, total)
    except urllib.error.URLError as exc:
        raise ACRUpdateError(
            "the download failed: %s" % exc.reason)


def _download_text(url, opener):
    """One small URL as text (the sidecar)."""
    try:
        with opener(url, timeout=DOWNLOAD_TIMEOUT_S) as resp:
            return resp.read().decode("utf-8")
    except urllib.error.URLError as exc:
        raise ACRUpdateError(
            "could not fetch the checksum file: %s" % exc.reason)
    except TimeoutError:
        raise ACRUpdateError("the download timed out")
    except (ValueError, UnicodeDecodeError):
        raise ACRUpdateError(
            "the checksum file did not read as text")


def download_update(result, dest_dir, progress_cb=None, cancel=None,
                    urlopen=None):
    """Download a newer release and check it against its
    checksum sidecar. Returns {"zip_path", "version", "sha256"}.
    A mismatch, a missing sidecar, or a missing download file
    raises ACRUpdateError and the download is thrown away --
    nothing unverified ever reaches staging. A cancelled
    download raises UpdateCancelled and its partial file is
    removed."""
    zip_url = result.get("zip_url")
    sha_url = result.get("sha_url")
    version = result.get("latest") or result.get("version")
    if not zip_url:
        raise ACRUpdateError("the release has no download file")
    if not sha_url:
        raise ACRUpdateError(
            "the release has no checksum file, so the download "
            "cannot be checked")
    if not version:
        raise ACRUpdateError("the release did not name a version")
    os.makedirs(dest_dir, exist_ok=True)
    zip_name = "acr-v%s.zip" % version
    zip_path = os.path.join(dest_dir, zip_name)
    opener = urlopen or urllib.request.urlopen
    try:
        _download_file(zip_url, zip_path, opener, progress_cb,
                       cancel)
        sidecar = _download_text(sha_url, opener)
    except UpdateCancelled:
        _remove_quietly(zip_path)
        raise
    except ACRUpdateError:
        _remove_quietly(zip_path)
        raise
    want = sidecar.split()[0] if sidecar.split() else ""
    actual = _sha256_file(zip_path)
    if not want or actual != want:
        _remove_quietly(zip_path)
        raise ACRUpdateError(
            "the downloaded file does not match the release "
            "checksum, so it was not used")
    return {"zip_path": zip_path, "version": version,
            "sha256": actual}


def _remove_quietly(path):
    try:
        os.remove(path)
    except OSError:
        pass


def staging_dir():
    """Where a verified update waits: ~/.local/share/ACR/
    update-pending, or under XDG_DATA_HOME when it is set. The
    staging lives outside the app tree, so applying an update
    never touches it until the swap is done."""
    data_home = os.environ.get("XDG_DATA_HOME")
    if data_home:
        base = data_home
    else:
        base = os.path.join(os.path.expanduser("~"), ".local",
                            "share")
    return os.path.join(base, "ACR", "update-pending")


def _drop_staging(staging):
    shutil.rmtree(staging, ignore_errors=True)


def stage_update(zip_path, version, sha256, staging=None):
    """Stage a verified zip for the next restart: the zip plus
    a manifest with its version and checksum. Any older staging
    is cleared first, so versions never mix. Returns the staging
    dir."""
    staging = staging or staging_dir()
    _drop_staging(staging)
    os.makedirs(staging)
    zip_name = os.path.basename(zip_path)
    staged_zip = os.path.join(staging, zip_name)
    shutil.copy2(zip_path, staged_zip)
    manifest = {"version": version, "sha256": sha256,
                "zip": zip_name}
    with open(os.path.join(staging, "manifest.json"), "w",
              encoding="utf-8") as fh:
        json.dump(manifest, fh)
    return staging


def _safe_relpath(name):
    """The install-relative path for a zip entry, or None when
    the entry is not part of the app tree or is unsafe (outside
    the acr/ prefix, absolute, or escaping with ..)."""
    if not name.startswith(ZIP_PREFIX):
        return None
    rel = name[len(ZIP_PREFIX):]
    if not rel or os.path.isabs(rel):
        return None
    parts = rel.replace("\\", "/").split("/")
    if any(part in ("", ".", "..") for part in parts):
        return None
    if parts[0] in PRESERVE_TOP:
        return None
    return "/".join(parts)


def apply_staged_update(install_dir, staging=None):
    """Apply the staged update over the install dir. Returns
    (True, version) on success, (False, reason) when nothing
    was staged or the staging could not be applied. A staged
    update that fails any check is dropped, never applied; the
    staging dir is removed on success and on failure alike.
    Only files the zip holds are overwritten; .venv and logs
    are never touched, and nothing is ever deleted."""
    staging = staging or staging_dir()
    manifest_path = os.path.join(staging, "manifest.json")
    if not os.path.isfile(manifest_path):
        return False, "no update is staged"
    try:
        with open(manifest_path, encoding="utf-8") as fh:
            manifest = json.load(fh)
        version = manifest["version"]
        want_sha = manifest["sha256"]
        zip_name = manifest["zip"]
    except (ValueError, KeyError, OSError) as exc:
        _drop_staging(staging)
        return False, "the staged update is damaged: %s" % exc
    zip_path = os.path.join(staging, zip_name)
    if not os.path.isfile(zip_path):
        _drop_staging(staging)
        return False, "the staged update file is missing"
    if _sha256_file(zip_path) != want_sha:
        _drop_staging(staging)
        return False, ("the staged update does not match its "
                       "checksum")
    tmp = tempfile.mkdtemp(prefix="acr-update-apply-")
    try:
        with zipfile.ZipFile(zip_path) as zf:
            entries = []
            for info in zf.infolist():
                if info.is_dir():
                    continue
                rel = _safe_relpath(info.filename)
                if rel is None:
                    continue
                entries.append((info, rel))
            for info, rel in entries:
                target = os.path.join(tmp, *rel.split("/"))
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with zf.open(info) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst, DOWNLOAD_CHUNK)
                # The zip stores the modes (run.sh and the
                # launchers are 0o755); restore them, so the
                # updated tree stays launchable.
                mode = (info.external_attr >> 16) & 0o777
                if mode:
                    os.chmod(target, mode)
        for _info, rel in entries:
            src = os.path.join(tmp, *rel.split("/"))
            dest = os.path.join(install_dir, *rel.split("/"))
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copy2(src, dest)
    except (OSError, zipfile.BadZipFile, ValueError) as exc:
        _drop_staging(staging)
        return False, "the update could not be applied: %s" % exc
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    _drop_staging(staging)
    return True, version


# The restart-now updater script, written into the staging dir
# at update time and started detached. It waits for the app to
# exit, applies the staged update through this same module (the
# running version's copy, imported from the install dir -- the
# module is in memory before the swap, so overwriting it is
# safe), and relaunches run.sh. If the app never exits, the
# update stays staged for the next launch instead. The script
# takes three arguments: pid, install dir, staging dir.
UPDATER_SCRIPT = '''#!/usr/bin/env python3
"""ACR updater (v0.12): apply a staged update after the app
exits, then relaunch. Started detached by the app; takes pid,
install dir, and staging dir on the command line."""

import os
import sys
import time

WAIT_TIMEOUT_S = 30.0
POLL_S = 0.2


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def main(argv):
    pid = int(argv[1])
    install_dir = argv[2]
    staging_dir = argv[3]
    deadline = time.time() + WAIT_TIMEOUT_S
    while _pid_alive(pid) and time.time() < deadline:
        time.sleep(POLL_S)
    if _pid_alive(pid):
        # The app did not exit: leave the update staged, the
        # next launch applies it.
        return 0
    sys.path.insert(0, install_dir)
    try:
        import acr_updates
    except ImportError:
        return 1
    acr_updates.apply_staged_update(install_dir, staging_dir)
    run_sh = os.path.join(install_dir, "run.sh")
    if os.path.isfile(run_sh):
        os.execv("/bin/bash", ["/bin/bash", run_sh])
    os.execv(sys.executable,
             [sys.executable,
              os.path.join(install_dir, "acr_app.py")])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
'''


def write_updater_script(staging):
    """Write the restart-now updater script into the staging
    dir. Returns the script path. The app starts it detached
    with the pid, install dir, and staging dir as arguments,
    then quits."""
    path = os.path.join(staging, "update.py")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(UPDATER_SCRIPT)
    return path
