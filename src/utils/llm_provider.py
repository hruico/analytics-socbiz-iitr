"""
LLM provider wrapper with Groq API key rotation.

Reads up to 4 keys from .env (GROQ_API_KEY_1 … GROQ_API_KEY_4).
When a key hits the rate/quota limit the wrapper transparently switches
to the next available key so the run never dies mid-way.
"""
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Error substrings that mean "this key's DAILY quota is exhausted" (not per-minute 429)
_EXHAUSTION_SIGNALS = (
    "quota_exceeded",
    "tokens per day",
    "requests per day",
    "daily limit",
    "daily_quota",
    "organization_quota",
)


def _is_exhaustion_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(sig.lower() in msg for sig in _EXHAUSTION_SIGNALS)


def _load_keys_from_env() -> List[str]:
    """
    Load Groq API keys from .env file (GROQ_API_KEY_1 … GROQ_API_KEY_4)
    and from the plain GROQ_API_KEY env var as a fallback.
    Returns a deduplicated list of non-empty keys.
    """
    # Try to load .env file
    env_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
    env_path = os.path.normpath(env_path)
    if os.path.exists(env_path):
        try:
            from dotenv import load_dotenv
            load_dotenv(env_path, override=False)
        except ImportError:
            # dotenv not installed — parse manually
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, _, v = line.partition("=")
                        os.environ.setdefault(k.strip(), v.strip())

    seen = set()
    keys = []
    # Numbered keys first (preferred)
    for i in range(1, 5):
        k = os.getenv(f"GROQ_API_KEY_{i}", "").strip()
        if k and k not in seen and not k.startswith("your_"):
            seen.add(k)
            keys.append(k)
    # Plain key as final fallback
    k = os.getenv("GROQ_API_KEY", "").strip()
    if k and k not in seen and not k.startswith("your_"):
        seen.add(k)
        keys.append(k)

    return keys


class LLMProviderWrapper:
    """
    Groq LLM wrapper with automatic API key rotation.

    Usage is identical to the old single-key wrapper — callers just call
    `invoke_with_retry(prompt)` and rotation is transparent.
    """

    def __init__(
        self,
        provider: str = "groq",
        model: str = "llama-3.3-70b-versatile",
        max_retries: int = 2,
        timeout: int = 30,
        api_key: Optional[str] = None,          # kept for backward compat
    ):
        self.provider = provider
        self.model = model
        self.max_retries = max_retries
        self.timeout = timeout

        self.total_calls = 0
        self.failed_calls = 0

        # ---- collect keys ----
        if api_key:
            self._keys: List[str] = [api_key]
        else:
            self._keys = _load_keys_from_env()

        if not self._keys:
            logger.warning(
                "No Groq API keys found. "
                "Add GROQ_API_KEY_1 … GROQ_API_KEY_4 to .env — running in fallback-only mode."
            )
            self._clients: List[Any] = []
            self._key_index = 0
            self.llm = None
            return

        # ---- build one Groq client per key ----
        self._clients = []
        self._key_index = 0
        self._exhausted: List[bool] = [False] * len(self._keys)

        for idx, key in enumerate(self._keys):
            client = self._make_client(key)
            self._clients.append(client)

        self.llm = self._clients[0] if self._clients else None
        logger.info(
            f"LLMProviderWrapper: {len(self._keys)} Groq key(s) loaded, "
            f"active key index=0, model={model}"
        )

    # ------------------------------------------------------------------
    def _make_client(self, key: str):
        try:
            from langchain_groq import ChatGroq
            return ChatGroq(
                model=self.model,
                temperature=0.1,
                groq_api_key=key,
                timeout=self.timeout,
            )
        except ImportError:
            raise ImportError("Run: pip install langchain-groq")
        except Exception as e:
            logger.warning(f"Failed to create Groq client for key ...{key[-4:]}: {e}")
            return None

    def _rotate_key(self) -> bool:
        """
        Mark current key as exhausted and switch to the next available one.
        Returns True if a fresh key is now active, False if all exhausted.
        """
        self._exhausted[self._key_index] = True
        logger.warning(
            f"Key index {self._key_index} (…{self._keys[self._key_index][-4:]}) "
            f"exhausted — rotating."
        )

        for i in range(len(self._keys)):
            next_idx = (self._key_index + i + 1) % len(self._keys)
            if not self._exhausted[next_idx] and self._clients[next_idx] is not None:
                self._key_index = next_idx
                self.llm = self._clients[next_idx]
                logger.info(
                    f"Switched to key index {next_idx} (…{self._keys[next_idx][-4:]})"
                )
                return True

        logger.error("All Groq API keys exhausted — falling back to deterministic mode.")
        self.llm = None
        return False

    # ------------------------------------------------------------------
    def invoke_with_retry(
        self, prompt: str, response_format: str = "json"
    ) -> Optional[Dict[str, Any]]:
        """
        Invoke the LLM with retry + key rotation.

        On rate-limit / quota errors the current key is marked exhausted
        and the next key is tried immediately (no wasted sleep).
        On transient errors we back off briefly and retry.
        """
        self.total_calls += 1

        if self.llm is None:
            self.failed_calls += 1
            return None

        attempts_remaining = self.max_retries + 1

        while attempts_remaining > 0 and self.llm is not None:
            try:
                t0 = time.time()
                response = self.llm.invoke(prompt)
                latency_ms = (time.time() - t0) * 1000

                content = response.content if hasattr(response, "content") else str(response)
                logger.debug(f"LLM response ({latency_ms:.0f}ms): {content[:120]}")

                if response_format == "json":
                    # Strip markdown fences if present
                    if "```json" in content:
                        content = content.split("```json")[1].split("```")[0].strip()
                    elif "```" in content:
                        content = content.split("```")[1].split("```")[0].strip()
                    # Some models wrap with extra prose — grab the JSON object
                    start = content.find("{")
                    end   = content.rfind("}") + 1
                    if start != -1 and end > start:
                        content = content[start:end]
                    try:
                        parsed = json.loads(content)
                        logger.info(
                            f"LLM call OK (key={self._key_index}, {latency_ms:.0f}ms)"
                        )
                        return parsed
                    except json.JSONDecodeError as e:
                        logger.warning(f"JSON parse error: {e} — content: {content[:200]}")
                        attempts_remaining -= 1
                        if attempts_remaining > 0:
                            time.sleep(0.5)
                        continue
                else:
                    return {"text": content}

            except Exception as exc:
                if _is_exhaustion_error(exc):
                    # Key exhausted — rotate immediately, do NOT decrement attempts
                    rotated = self._rotate_key()
                    if not rotated:
                        break  # all keys gone
                    # retry same prompt with new key right away
                else:
                    # Transient error — back off and retry
                    attempts_remaining -= 1
                    logger.warning(
                        f"LLM error (attempt left={attempts_remaining}): {exc}"
                    )
                    if attempts_remaining > 0:
                        time.sleep(1.5)

        self.failed_calls += 1
        logger.error("LLM call failed after all attempts/key rotations — using fallback.")
        return None

    # ------------------------------------------------------------------
    def get_stats(self) -> Dict[str, Any]:
        success_rate = (
            (self.total_calls - self.failed_calls) / self.total_calls * 100
            if self.total_calls > 0
            else 0.0
        )
        return {
            "total_calls": self.total_calls,
            "failed_calls": self.failed_calls,
            "success_rate": success_rate,
            "active_key_index": self._key_index,
            "keys_loaded": len(self._keys),
            "keys_exhausted": sum(getattr(self, "_exhausted", [False])),
        }
