#!/usr/bin/env python3
"""Fast launcher for ACR (v0.11, A8): the single-instance check
before the heavy imports.

Importing the app takes seconds (the interface library plus the
app module), and a second launch used to pay all of it before
finding out another ACR was already running. This launcher
imports QtCore only (~1 s), checks the lock, and -- when another
instance is running -- asks it to raise and exits, without ever
importing the app. The socket class lives in QtNetwork, so it is
imported only on the raise path, when the lock is already held:
the common path (no instance running) stays QtCore-only.
Otherwise the launcher execs acr_app.py in the same process, so
the app still gets the lock itself: this is only a fast path,
and main()'s own check stays as the correctness net for the
double-click race and direct launches. A broken Qt import falls
through to the app, which reports the failure itself instead of
this launcher masking it.

run.sh and the installed `acr` command run this, not acr_app.py
directly. The install smoke check runs acr_app.py directly (it
needs a private instance, not the fast path).
"""

import os
import sys
import tempfile


def _instance_name():
    return os.environ.get("ACR_INSTANCE_NAME", "acr")


def _try_raise_existing():
    """True when another instance is running and was asked to
    raise. False when no instance is running, or QtCore cannot
    be imported at all."""
    try:
        from PySide6.QtCore import QLockFile
    except ImportError:
        return False
    name = _instance_name()
    lock_path = os.path.join(tempfile.gettempdir(),
                             "acr-%s.lock" % name)
    lock = QLockFile(lock_path)
    lock.setStaleLockTime(1000)
    if lock.tryLock(0):
        # No instance running. Unlock explicitly before the
        # exec: the lock file descriptor would otherwise survive
        # the exec and block the app's own tryLock (a lock held
        # on one descriptor denies a lock on another, even in
        # the same process). The app takes the lock again in
        # main(); the gap is covered by main()'s own check.
        lock.unlock()
        return False
    # The lock is held: an instance is running. The socket class
    # lives in QtNetwork in this build (not QtCore), and is only
    # needed here, on the raise path.
    try:
        from PySide6.QtNetwork import QLocalSocket
    except ImportError:
        # The instance is running but cannot be raised; say so
        # rather than starting a second one.
        return True
    socket = QLocalSocket()
    socket.connectToServer("ACR-" + name)
    if socket.waitForConnected(2000):
        socket.write(b"raise")
        socket.waitForBytesWritten(2000)
    socket.disconnectFromServer()
    return True


def main(argv=None):
    args = sys.argv if argv is None else argv
    if _try_raise_existing():
        return 0
    here = os.path.dirname(os.path.abspath(__file__))
    os.execv(sys.executable,
             [sys.executable, os.path.join(here, "acr_app.py")]
             + args[1:])


if __name__ == "__main__":
    sys.exit(main())
