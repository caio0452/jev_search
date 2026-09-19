import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

try:
    import requests
    from requests.adapters import HTTPAdapter
except ImportError:
    requests = None
    HTTPAdapter = None


@dataclass
class OpenRouterQuestionResponse:
    question_identifier: str
    noul_score: float


class OpenRouterDecisionsClient:
    def __init__(
        self,
        api_key: str | None = None,
        endpoint_url: str = "https://openrouter.ai/api/alpha/decisions",
        model: str = "~typesafe/jev-latest",
        timeout_seconds: int = 30,
        pool_size: int = 50,
        max_retries: int = 4,
    ):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("TYPESAFE_API_KEY", "")
        self.endpoint_url = endpoint_url
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self._reported_errors: set[str] = set()

        self._session = None
        if requests is not None and HTTPAdapter is not None:
            self._session = requests.Session()
            adapter = HTTPAdapter(pool_connections=pool_size, pool_maxsize=pool_size)
            self._session.mount("https://", adapter)
            self._session.mount("http://", adapter)

    def _log_error_once(self, message: str) -> None:
        if message not in self._reported_errors:
            self._reported_errors.add(message)
            sys.stderr.write(f"\n[API Error] {message}\n")
            sys.stderr.flush()

    def evaluate(self, state: str, question_instructions: dict[str, str]) -> dict[str, float]:
        # Send state and questions to OpenRouter Decisions API with rate-limit retry and error logging.
        if not question_instructions:
            return {}

        if not self.api_key:
            raise ValueError(
                "No API key provided. Set OPENROUTER_API_KEY environment variable or pass --api-key."
            )

        payload_questions = {
            identifier: {
                "type": "noul",
                "instructions": instruction,
                "criteria": {
                    "true": "Explicitly matches criteria",
                    "false": "Does not match criteria",
                },
            }
            for identifier, instruction in question_instructions.items()
        }

        request_body = {
            "model": self.model,
            "state": state,
            "questions": payload_questions,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        for attempt in range(self.max_retries):
            try:
                if self._session is not None:
                    response = self._session.post(
                        self.endpoint_url,
                        json=request_body,
                        headers=headers,
                        timeout=self.timeout_seconds,
                    )
                    status_code = response.status_code
                    response_text = response.text
                    retry_header = response.headers.get("Retry-After")
                else:
                    encoded_data = json.dumps(request_body).encode("utf-8")
                    http_request = urllib.request.Request(
                        url=self.endpoint_url,
                        data=encoded_data,
                        headers=headers,
                        method="POST",
                    )
                    try:
                        with urllib.request.urlopen(http_request, timeout=self.timeout_seconds) as raw_response:
                            status_code = raw_response.status
                            response_text = raw_response.read().decode("utf-8")
                            retry_header = raw_response.headers.get("Retry-After")
                    except urllib.error.HTTPError as http_error:
                        status_code = http_error.code
                        response_text = http_error.read().decode("utf-8")
                        retry_header = http_error.headers.get("Retry-After")

                if status_code == 200:
                    response_data = json.loads(response_text)
                    answers = response_data.get("answers", {})
                    return {
                        identifier: float(answers.get(identifier, {}).get("noul", 0.0))
                        for identifier in question_instructions
                    }

                if status_code == 429:
                    sleep_duration = float(retry_header) if retry_header and retry_header.isdigit() else (1.5 * (2 ** attempt))
                    time.sleep(sleep_duration)
                    continue

                if status_code in (401, 402, 403):
                    self._log_error_once(f"Status {status_code}: {response_text.strip()}")
                    raise PermissionError(f"OpenRouter API returned {status_code}: {response_text.strip()}")

                self._log_error_once(f"Status {status_code}: {response_text.strip()[:250]}")
                return {identifier: 0.0 for identifier in question_instructions}

            except (PermissionError, ValueError):
                raise
            except Exception as exception_instance:
                if attempt == self.max_retries - 1:
                    self._log_error_once(f"Request failed: {exception_instance}")
                    return {identifier: 0.0 for identifier in question_instructions}
                time.sleep(1.0 * (attempt + 1))

        return {identifier: 0.0 for identifier in question_instructions}


JevHttpClient = OpenRouterDecisionsClient
