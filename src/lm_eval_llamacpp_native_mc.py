from __future__ import annotations

import logging
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests
from requests import Response
from requests.exceptions import RequestException
from tqdm import tqdm

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model


logger = logging.getLogger(__name__)


@register_model("llamacpp-native-mc")
class LlamaCppNativeMultipleChoiceLM(LM):
    """Minimal llama.cpp native backend for multiple-choice scoring.

    This backend targets llama.cpp's native `/completion` endpoint instead of
    the OpenAI-compatible `/v1/completions` endpoint. It is primarily intended
    for tasks such as MMLU where each answer choice is scored as the next token
    after a shared context prompt.
    """

    def __init__(
        self,
        base_url: str,
        model: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3,
        retry_delay_s: float = 1.0,
        n_probs: int = 20,
        temperature: float = 0.0,
        max_gen_toks: int = 256,
        **_: object,
    ) -> None:
        super().__init__()
        if not base_url:
            raise ValueError("base_url is required")
        self.base_url = base_url.rstrip("/")
        self.model = model or ""
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay_s = retry_delay_s
        self.n_probs = n_probs
        self.temperature = temperature
        self.max_gen_toks = max_gen_toks
        self._session = requests.Session()

    def _completion_url(self) -> str:
        if self.base_url.endswith("/completion"):
            return self.base_url
        return f"{self.base_url}/completion"

    def _post_completion(self, payload: Dict[str, object]) -> Dict[str, object]:
        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response: Response = self._session.post(
                    self._completion_url(),
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                result = response.json()
                if not isinstance(result, dict):
                    raise RuntimeError(f"unexpected non-dict response: {type(result)!r}")
                return result
            except (RequestException, ValueError, RuntimeError) as exc:
                last_error = exc
                if attempt == self.max_retries:
                    break
                time.sleep(self.retry_delay_s)

        raise RuntimeError(
            f"llama.cpp native completion request failed after {self.max_retries} attempts: {last_error}"
        )

    @staticmethod
    def _extract_top_logprobs(response: Dict[str, object]) -> Tuple[str, List[Dict[str, object]]]:
        probabilities = response.get("completion_probabilities")
        if not isinstance(probabilities, list) or not probabilities:
            raise RuntimeError(
                "native /completion response does not contain completion_probabilities"
            )
        first = probabilities[0]
        if not isinstance(first, dict):
            raise RuntimeError("invalid completion_probabilities entry")
        top_logprobs = first.get("top_logprobs")
        if not isinstance(top_logprobs, list) or not top_logprobs:
            raise RuntimeError("native /completion response does not contain top_logprobs")
        token = first.get("token")
        if not isinstance(token, str):
            raise RuntimeError("native /completion response does not contain token text")
        typed_top_logprobs = [entry for entry in top_logprobs if isinstance(entry, dict)]
        return token, typed_top_logprobs

    @staticmethod
    def _match_token(
        continuation: str, top_logprobs: Iterable[Dict[str, object]]
    ) -> Optional[Dict[str, object]]:
        for entry in top_logprobs:
            if entry.get("token") == continuation:
                return entry
        return None

    def loglikelihood(
        self, requests: List[Any], disable_tqdm: bool = False
    ) -> List[Tuple[float, bool]]:
        if not requests:
            return []

        results: List[Tuple[float, bool]] = []
        cache: Dict[str, Tuple[str, List[Dict[str, object]]]] = {}

        for context, continuation in tqdm(
            [req.args for req in requests],
            disable=disable_tqdm,
            desc="Requesting llama.cpp native API",
        ):
            if not isinstance(context, str) or not isinstance(continuation, str):
                raise TypeError("llamacpp-native-mc expects string context/continuation pairs")

            if context not in cache:
                payload = {
                    "prompt": context,
                    "n_predict": 0,
                    "n_probs": self.n_probs,
                    "temperature": self.temperature,
                }
                response = self._post_completion(payload)
                cache[context] = self._extract_top_logprobs(response)

            greedy_token, top_logprobs = cache[context]
            matched = self._match_token(continuation, top_logprobs)
            if matched is None:
                available = ", ".join(
                    repr(entry.get("token")) for entry in top_logprobs[: min(10, len(top_logprobs))]
                )
                raise RuntimeError(
                    f"continuation {continuation!r} not present in native top_logprobs; "
                    f"increase n_probs or use a different backend. Available: {available}"
                )

            logprob = matched.get("logprob")
            if not isinstance(logprob, (int, float)):
                raise RuntimeError(f"missing numeric logprob for continuation {continuation!r}")
            results.append((float(logprob), greedy_token == continuation))

        return results

    def generate_until(
        self, requests: List[Any], disable_tqdm: bool = False
    ) -> List[str]:
        if not requests:
            return []

        results: List[str] = []
        for context, gen_kwargs in tqdm(
            [req.args for req in requests],
            disable=disable_tqdm,
            desc="Generating via llama.cpp native API",
        ):
            if not isinstance(context, str) or not isinstance(gen_kwargs, dict):
                raise TypeError("llamacpp-native-mc generate_until expects (str, dict) requests")
            stop = gen_kwargs.get("until", [])
            if isinstance(stop, str):
                stop = [stop]
            payload = {
                "prompt": context,
                "n_predict": int(gen_kwargs.get("max_gen_toks", self.max_gen_toks)),
                "temperature": float(gen_kwargs.get("temperature", self.temperature)),
                "stop": stop,
            }
            response = self._post_completion(payload)
            content = response.get("content")
            if not isinstance(content, str):
                raise RuntimeError("native /completion generation response missing content")
            results.append(content)
        return results

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False) -> List[float]:
        raise NotImplementedError(
            "llamacpp-native-mc does not support loglikelihood_rolling"
        )
