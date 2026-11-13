"""Local-model encoder client for ACR. Standard library only.

Talks to the local model service over localhost HTTP
(http://127.0.0.1:11434): /api/tags for the installed model list,
/api/embeddings for embeddings. The service is Ollama; the app
calls it "the local model service" in user-facing text.

Health, retry, and failure wording follow the DUO v0.10 pattern:
a quick probe decides up vs down, transient failures retry a few
times with a pause between attempts, and the final error is
decided by a fresh probe -- a service that answers the probe is
slow or busy, never "not running". Only a probe that fails too
earns the down message.
"""

import json
import os
import time
import urllib.error
import urllib.request

DEFAULT_BASE_URL = "http://127.0.0.1:11434"
LOCAL_MODEL_URL_ENV = "ACR_LOCAL_MODEL_URL"

# Plain-language messages shown in the window. No tracebacks.
MSG_SERVICE_DOWN = ("The local model service is not running. "
                    "Start it and try again.")
MSG_SERVICE_SLOW = ("The local model service is slow or busy right "
                    "now. Try again in a moment.")
MSG_SERVICE_NOT_INSTALLED = ("The local model service is not "
                             "installed on this computer, so ACR "
                             "cannot start it.")
MSG_START_FAILED = ("The local model service did not start. Try "
                    "starting it yourself and try again.")
MSG_MODEL_FAILED = ("That model could not be loaded. Pick another "
                    "model and try again.")
MSG_NO_MODELS = ("No models are installed in the local model "
                 "service.")

# Embeddings retry: up to 3 attempts, pausing between them. An
# embedding call is one short request, so a failure before the
# response arrives is safe to retry: nothing was half-applied.
EMBED_MAX_ATTEMPTS = 3
EMBED_RETRY_DELAYS = (2.0, 5.0)  # pause before attempt 2, then 3
EMBED_TIMEOUT = 60  # first use loads the model, so allow a minute
PROBE_TIMEOUT = 3

# The embedding model to prefer when it is installed. Names carry
# a tag suffix in the list (nomic-embed-text:latest), so matching
# is by exact name or name-plus-colon.
EMBED_MODEL_DEFAULT = "nomic-embed-text"


class LocalModelError(Exception):
    """A plain-language local-service failure. The message is safe
    to show in the window."""


def parse_tags_payload(payload):
    """Parse an /api/tags JSON payload into a sorted list of model
    names."""
    names = []
    for entry in payload.get("models", []) or []:
        name = entry.get("name")
        if name:
            names.append(name)
    return sorted(names)


def pick_default_model(models):
    """The model to use first: the embedding model when installed,
    else the first available model, else None. "First" means first
    alphabetically, so the pick does not depend on the order the
    service happened to list them in."""
    ordered = sorted(models)
    for name in ordered:
        if (name == EMBED_MODEL_DEFAULT
                or name.startswith(EMBED_MODEL_DEFAULT + ":")):
            return name
    if ordered:
        return ordered[0]
    return None


def _failure_kind(exc):
    """Classify a request failure as down, slow, or error.

    Walks the exception chain: a refused connection means the
    service is down, a timeout means it is slow or busy, anything
    else is a plain error.
    """
    current = exc
    while current is not None:
        if isinstance(current, ConnectionRefusedError):
            return "down"
        if isinstance(current, TimeoutError):
            return "slow"
        current = (getattr(current, "reason", None)
                   or getattr(current, "__cause__", None))
    return "error"


class _Retryable(Exception):
    """An embedding attempt failed before any response arrived:
    safe to try again. Carries the failure kind."""

    def __init__(self, error, kind):
        super().__init__(str(error))
        self.error = error
        self.kind = kind


class LocalModelClient:
    """Minimal local-service client over localhost HTTP."""

    def __init__(self, base_url=None):
        self.base_url = (base_url or os.environ.get(LOCAL_MODEL_URL_ENV)
                         or DEFAULT_BASE_URL).rstrip("/")

    def probe(self, timeout=PROBE_TIMEOUT):
        """True when the service answers /api/tags, False otherwise.

        A fast liveness check. Never raises.
        """
        try:
            req = urllib.request.Request(self.base_url + "/api/tags")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                resp.read()
            return True
        except Exception:
            return False

    def list_models(self, timeout=5):
        """Return installed model names, or raise LocalModelError."""
        try:
            req = urllib.request.Request(self.base_url + "/api/tags")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return parse_tags_payload(
                    json.loads(resp.read().decode("utf-8")))
        except Exception:
            raise LocalModelError(MSG_SERVICE_DOWN)

    def embed(self, model, text, timeout=EMBED_TIMEOUT,
              on_retry=None):
        """Embed text with the given model. Returns a list of floats.

        Retries transient failures (refused, timeout, connection
        cut before the response) up to EMBED_MAX_ATTEMPTS times.
        on_retry (optional) is called before each retry with
        (attempt, delay_seconds, kind): attempt is the attempt that
        just failed (1-based), kind is "down", "slow", or "error".

        Raises LocalModelError with a plain message on failure. A
        missing model is not retried: it will not fix itself.
        """
        for attempt in range(1, EMBED_MAX_ATTEMPTS + 1):
            try:
                return self._embed_attempt(model, text, timeout)
            except _Retryable as exc:
                if attempt >= EMBED_MAX_ATTEMPTS:
                    raise self._final_error(exc)
                delay = EMBED_RETRY_DELAYS[attempt - 1]
                if on_retry is not None:
                    on_retry(attempt, delay, exc.kind)
                time.sleep(delay)
        raise LocalModelError(MSG_SERVICE_DOWN)  # unreachable

    def _final_error(self, retryable):
        """The plain error for a call that exhausted its retries.

        A fresh probe decides the message: the service answering
        now means it was slow or busy, not down. Only a probe that
        fails too earns the down message.
        """
        if self.probe():
            return LocalModelError(MSG_SERVICE_SLOW)
        return LocalModelError(MSG_SERVICE_DOWN)

    def _embed_attempt(self, model, text, timeout):
        """One try at an embedding. Returns the vector.

        Raises LocalModelError (model failure, or a bad response)
        or _Retryable (a failure before any response arrived).
        """
        body = json.dumps(
            {"model": model, "prompt": text}).encode("utf-8")
        req = urllib.request.Request(
            self.base_url + "/api/embeddings", data=body,
            headers={"Content-Type": "application/json"},
            method="POST")
        try:
            resp = urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as exc:
            # The service answered with a status: it is up, and a
            # missing model will not fix itself by retrying.
            try:
                detail = json.loads(
                    exc.read().decode("utf-8")).get("error", "")
            except Exception:
                detail = ""
            if "not found" in detail.lower():
                raise LocalModelError(MSG_MODEL_FAILED)
            raise LocalModelError(MSG_MODEL_FAILED)
        except Exception as exc:
            raise _Retryable(exc, _failure_kind(exc))
        try:
            payload = json.loads(resp.read().decode("utf-8"))
        except Exception:
            resp.close()
            raise LocalModelError(MSG_SERVICE_SLOW)
        resp.close()
        embedding = payload.get("embedding")
        if (not isinstance(embedding, list) or not embedding
                or not all(isinstance(x, (int, float))
                           and not isinstance(x, bool)
                           for x in embedding)):
            raise LocalModelError(
                "The local model returned a bad embedding.")
        return [float(x) for x in embedding]
