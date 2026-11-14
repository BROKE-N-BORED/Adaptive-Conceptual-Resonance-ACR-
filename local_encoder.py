"""Local-model encoder client for ACR. Standard library only.

Talks to the local model service over localhost HTTP
(http://127.0.0.1:11434): /api/tags for the installed model list,
/api/embeddings for embeddings, and since v0.6 /api/generate for
the synthesis head's optional completion. The service is Ollama;
the app calls it "the local model service" in user-facing text.

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
# Completion retry: the same shape. Generation takes longer than
# an embedding (first use loads the model, then it writes), so
# the timeout is longer.
GENERATE_MAX_ATTEMPTS = 3
GENERATE_RETRY_DELAYS = (2.0, 5.0)  # pause before attempt 2, then 3
GENERATE_TIMEOUT = 120
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


def parse_tags_info(payload):
    """Parse an /api/tags JSON payload into per-model info dicts.

    Each entry holds what the service actually reports, nothing
    invented: "name", "size" (bytes on disk, or None when the
    service does not say), and "parameter_size" (e.g. "2.0B", or
    None). Sorted by name. Missing or odd fields become None,
    never a crash: older or minimal services report names only.
    """
    infos = []
    for entry in payload.get("models", []) or []:
        name = entry.get("name")
        if not name:
            continue
        size = entry.get("size")
        if not isinstance(size, int) or isinstance(size, bool):
            size = None
        details = entry.get("details") or {}
        parameter_size = details.get("parameter_size")
        if not isinstance(parameter_size, str):
            parameter_size = None
        infos.append({"name": name, "size": size,
                      "parameter_size": parameter_size})
    infos.sort(key=lambda info: info["name"])
    return infos


def _format_bytes(size):
    """Bytes as a short human size: 274 MB, 1.3 GB."""
    if size >= 1024 ** 3:
        return "%.1f GB" % (size / float(1024 ** 3))
    if size >= 1024 ** 2:
        return "%d MB" % (size // (1024 ** 2))
    return "%d KB" % (size // 1024)


def model_display_text(info, total_ram_bytes, known_dim=None):
    """One line for the model picker (v0.9, U39): what the model
    does and whether it fits this PC.

    Kind comes from the name: a model with "embed" in its name is
    an embedding model, anything else is a text model. Size and
    parameter count come from the service's own list. "Fits this
    PC" compares the model's size on disk against this machine's
    RAM: a model loads roughly its own size into memory, so a
    model bigger than RAM cannot load here. The embedding dim is
    shown only once it is known, from a real embedding (never
    guessed). A service that reports names only still gets a
    useful line: name and kind, no invented numbers.
    """
    name = info["name"]
    kind = ("embedding model" if "embed" in name.lower()
            else "text model")
    parts = [kind]
    if info.get("parameter_size"):
        parts.append("%s parameters" % info["parameter_size"])
    if info.get("size"):
        parts.append(_format_bytes(info["size"]))
    if info.get("size") and total_ram_bytes:
        if info["size"] <= total_ram_bytes:
            parts.append("fits this PC")
        else:
            parts.append("too big for this PC")
    if known_dim is not None:
        parts.append("dim %d" % known_dim)
    return "%s (%s)" % (name, ", ".join(parts))


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


def pick_completion_model(models, encoder_model):
    """The model to complete with: the first installed model that
    is not the encoder model, alphabetically. When the encoder
    model is the only one installed, it is the pick: the picker
    shows its name, so nothing is presented as something it is
    not. None when no model is installed at all."""
    ordered = sorted(models)
    for name in ordered:
        if name != encoder_model:
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

    def list_model_info(self, timeout=5):
        """Return per-model info dicts (see parse_tags_info), or
        raise LocalModelError. One /api/tags call: the picker
        shows what each model does and whether it fits this PC
        from this, never from a second guess."""
        try:
            req = urllib.request.Request(self.base_url + "/api/tags")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return parse_tags_info(
                    json.loads(resp.read().decode("utf-8")))
        except Exception:
            raise LocalModelError(MSG_SERVICE_DOWN)

    def _with_retries(self, attempt, max_attempts, delays,
                      on_retry):
        """Run attempt() (a zero-arg callable) with retries.

        Transient failures (refused, timeout, connection cut
        before the response) retry up to max_attempts times.
        on_retry (optional) is called before each retry with
        (attempt, delay_seconds, kind): attempt is the attempt that
        just failed (1-based), kind is "down", "slow", or "error".
        """
        for n in range(1, max_attempts + 1):
            try:
                return attempt()
            except _Retryable as exc:
                if n >= max_attempts:
                    raise self._final_error(exc)
                delay = delays[n - 1]
                if on_retry is not None:
                    on_retry(n, delay, exc.kind)
                time.sleep(delay)
        raise LocalModelError(MSG_SERVICE_DOWN)  # unreachable

    def embed(self, model, text, timeout=EMBED_TIMEOUT,
              on_retry=None):
        """Embed text with the given model. Returns a list of floats.

        Retries transient failures up to EMBED_MAX_ATTEMPTS times.
        Raises LocalModelError with a plain message on failure. A
        missing model is not retried: it will not fix itself.
        """
        return self._with_retries(
            lambda: self._embed_attempt(model, text, timeout),
            EMBED_MAX_ATTEMPTS, EMBED_RETRY_DELAYS, on_retry)

    def generate(self, model, prompt, timeout=GENERATE_TIMEOUT,
                 on_retry=None):
        """Complete a prompt with the given model. Returns the text.

        Retries transient failures up to GENERATE_MAX_ATTEMPTS
        times, with the same down-vs-slow final message as embed:
        a service that answers the probe is slow or busy, never
        "not running". Raises LocalModelError with a plain message
        on failure. A missing model is not retried.
        """
        return self._with_retries(
            lambda: self._generate_attempt(model, prompt, timeout),
            GENERATE_MAX_ATTEMPTS, GENERATE_RETRY_DELAYS, on_retry)

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

    def _generate_attempt(self, model, prompt, timeout):
        """One try at a completion. Returns the response text.

        Raises LocalModelError (model failure, or a bad response)
        or _Retryable (a failure before any response arrived).
        stream is false: the service answers with one JSON object
        holding the whole completion, not a stream of pieces.
        """
        body = json.dumps(
            {"model": model, "prompt": prompt,
             "stream": False}).encode("utf-8")
        req = urllib.request.Request(
            self.base_url + "/api/generate", data=body,
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
        response = payload.get("response")
        if not isinstance(response, str) or not response.strip():
            raise LocalModelError(
                "The local model returned a bad completion.")
        return response
