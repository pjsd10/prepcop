"""Small, single-attempt Tavily requests. The runtime owns retries and budgets.

Only the fixed Tavily API endpoint is contacted by this process. Candidate page
URLs are validated before passing them to Tavily; untrusted page content is data.
"""
from __future__ import annotations

import ipaddress
import json
import math
import re
import socket
import urllib.error
import urllib.request
from typing import Any, Callable
from urllib.parse import unquote, urlsplit, urlunsplit


MAX_PAGE_CHARS = 18_000
MAX_RESPONSE_BYTES = 2_000_000
MAX_FAILURE_CHARS = 400
CONTACT_PATTERN = re.compile(
    r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|(?:\+?\d[\d ().-]{7,}\d)"
)
HTTP_STATUS_SEQUENCE = re.compile(r"(?i)\bHTTP(?:\s+status(?:\s+codes?)?)?\s+(?:[1-5]\d{2}[\s,/]+){1,}[1-5]\d{2}\b")
PRIVATE_SUFFIXES = (".localhost", ".local", ".internal", ".lan", ".home", ".onion")


def validate_query(query: str) -> str:
    """Keep searches focused and prevent obvious contact-data disclosure."""
    if not isinstance(query, str) or not 3 <= len(query.strip()) <= 500:
        raise ValueError("Search query must contain 3 to 500 characters.")
    query = " ".join(query.split())
    if redact_contacts(query) != query:
        raise ValueError("Search query contains contact information; use topic terms only.")
    return query


def redact_contacts(value: str) -> str:
    """Preserve explicit HTTP status-code lists while removing contact details."""
    parts, position = [], 0
    for match in HTTP_STATUS_SEQUENCE.finditer(value):
        parts.append(CONTACT_PATTERN.sub("[contact omitted]", value[position:match.start()]))
        parts.append(match.group())
        position = match.end()
    parts.append(CONTACT_PATTERN.sub("[contact omitted]", value[position:]))
    return "".join(parts)


def validate_public_url(url: str, *, resolve: bool = False) -> str:
    """Reject local, credential-bearing and non-web targets.

    DNS checking is optional: Tavily performs the fetch on its infrastructure,
    so local DNS cannot guarantee Tavily's resolution or redirect handling. The
    app never opens a supplied page URL itself. Hostname and literal-IP checks
    are always applied, including to each returned URL.
    """
    if not isinstance(url, str) or not 8 <= len(url) <= 2048:
        raise ValueError("A public HTTP or HTTPS page URL is required.")
    if any(ord(char) < 33 for char in url) or "\\" in url:
        raise ValueError("URL contains invalid characters.")
    parsed = urlsplit(url)
    host = (parsed.hostname or "").rstrip(".").lower()
    if parsed.scheme not in ("http", "https") or not host:
        raise ValueError("Only public HTTP or HTTPS page URLs are allowed.")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("URLs containing credentials are not allowed.")
    if parsed.port not in (None, 80, 443):
        raise ValueError("Only standard public web ports are allowed.")
    if CONTACT_PATTERN.search(unquote(parsed.query)):
        raise ValueError("URL query contains contact information.")
    if host == "localhost" or host.endswith(PRIVATE_SUFFIXES):
        raise ValueError("Private or local URLs are not allowed.")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        if "." not in host or not re.fullmatch(r"[a-z0-9.-]+", host):
            raise ValueError("A public DNS hostname is required.") from None
        # Reject ambiguous inet_aton forms (decimal/hex/octal shortened IPv4).
        if re.fullmatch(r"(?:0x[0-9a-f]+|[0-9]+)(?:\.(?:0x[0-9a-f]+|[0-9]+))*", host):
            raise ValueError("Ambiguous numeric hosts are not allowed.")
        if resolve:
            addresses = socket.getaddrinfo(host, parsed.port or 443, type=socket.SOCK_STREAM)
            if not addresses or any(not ipaddress.ip_address(item[4][0]).is_global for item in addresses):
                raise ValueError("URL does not resolve exclusively to public addresses.")
    else:
        if not address.is_global:
            raise ValueError("Private or reserved IP addresses are not allowed.")
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path or "/", parsed.query, ""))


def _failure(message: str, *, retryable: bool = False, **extra: Any) -> dict:
    return {"status": "failed", "ok": False, "error": message, "retryable": retryable, **extra}


def _safe_failure_reason(value: Any, api_key: str) -> str:
    """Keep provider diagnostics useful without echoing contacts or credentials."""
    if not isinstance(value, str):
        return ""
    message = value.replace(api_key, "[credential omitted]") if api_key else value
    message = re.sub(r"(?i)\b(?:bearer|basic)\s+[A-Za-z0-9._~+/=-]+", "[authorization omitted]", message)
    message = re.sub(r"(?i)([\"']?(?:api[-_ ]?key|access[-_ ]?token|token|authorization|password|secret)[\"']?\s*[:=]\s*)(?:[\"'][^\"']*[\"']|[^\s,;]+)",
                     r"\1[credential omitted]", message)
    message = re.sub(r"(?i)\b(?:tvly|sk|fw)[-_][A-Za-z0-9_-]+", "[credential omitted]", message)
    message = re.sub(r"(?i)(https?://)[^/\s]+@", r"\1[credentials omitted]@", message)
    message = redact_contacts(message)
    return " ".join(message.split())[:MAX_FAILURE_CHARS]


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(req.full_url, code, "Unexpected API redirect", headers, fp)


class TavilyRetrieval:
    """Tavily Search/Extract thin wrapper with explicit bounded request options.

    transport is injectable for controlled tests. It accepts endpoint URL,
    JSON body, headers, timeout and returns a decoded JSON object. No SDK or
    transport retries are enabled, so one invocation is one counted attempt.
    """

    def __init__(self, api_key: str, *, transport: Callable | None = None):
        self._api_key = api_key
        self._transport = transport or self._post

    @staticmethod
    def _post(endpoint: str, body: dict, headers: dict, timeout: float) -> dict:
        request = urllib.request.Request(endpoint, data=json.dumps(body).encode("utf-8"),
                                         headers=headers, method="POST")
        opener = urllib.request.build_opener(_NoRedirect())
        with opener.open(request, timeout=timeout) as response:
            data = response.read(MAX_RESPONSE_BYTES + 1)
        if len(data) > MAX_RESPONSE_BYTES:
            raise ValueError("Tavily response exceeded the configured size limit.")
        result = json.loads(data)
        if not isinstance(result, dict):
            raise ValueError("Tavily returned an invalid response object.")
        return result

    def _request(self, endpoint: str, body: dict, timeout: float) -> dict:
        if not self._api_key:
            return _failure("Tavily API key is not configured.")
        if not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            return _failure("Retrieval deadline has been reached.")
        try:
            response = self._transport(
                "https://api.tavily.com/" + endpoint, body,
                {"Authorization": "Bearer " + self._api_key, "Content-Type": "application/json"},
                min(float(timeout), 20.0),
            )
            if not isinstance(response, dict):
                return _failure("Tavily returned an invalid response object.")
            return response
        except urllib.error.HTTPError as exc:
            # Never echo response bodies, request headers, keys or provider URLs.
            return _failure(f"Tavily returned HTTP {exc.code}.", retryable=exc.code in (408, 429) or exc.code >= 500)
        except (TimeoutError, socket.timeout):
            return _failure("Tavily retrieval timed out; the provider outcome may be unknown.", unknown_outcome=True)
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, (TimeoutError, socket.timeout)):
                return _failure("Tavily retrieval timed out; the provider outcome may be unknown.", unknown_outcome=True)
            return _failure("Tavily could not be reached.", retryable=True)
        except (ValueError, TypeError, KeyError):
            return _failure("Tavily returned an unreadable or oversized response.")

    @staticmethod
    def _metadata(response: dict) -> dict:
        metadata = {"request_id": str(response.get("request_id", ""))[:200],
                    "usage": response.get("usage") if isinstance(response.get("usage"), dict) else {}}
        response_time = response.get("response_time")
        if (isinstance(response_time, (int, float)) and not isinstance(response_time, bool)
                and math.isfinite(response_time) and response_time >= 0):
            metadata["response_time"] = response_time
        return metadata

    def search_sources(self, query: str, timeout: float = 20) -> dict:
        try:
            query = validate_query(query)
        except ValueError as exc:
            return _failure(str(exc))
        response = self._request("search", {
            "query": query, "search_depth": "basic", "topic": "general", "max_results": 5,
            "include_answer": False, "include_raw_content": False, "include_images": False,
            "auto_parameters": False, "include_usage": True,
        }, timeout)
        if response.get("ok") is False:
            return response
        if not isinstance(response.get("results"), list):
            return _failure("Tavily search response has no results list.")
        results, rejected = [], 0
        for item in response["results"][:5]:
            try:
                url = validate_public_url(item["url"])
                results.append({"url": url, "title": str(item.get("title", ""))[:300],
                                "content": str(item.get("content", ""))[:1500]})
            except (ValueError, TypeError, KeyError):
                rejected += 1
        return {"status": "success", "ok": True, "query": query, "results": results,
                "rejected_results": rejected, **self._metadata(response)}

    def extract_page(self, url: str, timeout: float = 20) -> dict:
        try:
            url = validate_public_url(url, resolve=True)
        except ValueError as exc:
            return _failure(str(exc), content_available=False)
        except OSError:
            return _failure("Page hostname could not be resolved safely.", content_available=False)
        if timeout < 1:
            return _failure("Less than one second remains for extraction.", content_available=False)
        response = self._request("extract", {
            "urls": [url], "extract_depth": "basic", "format": "markdown",
            "include_images": False, "include_usage": True, "timeout": min(float(timeout), 20.0),
        }, timeout)
        if response.get("ok") is False:
            return {**response, "content_available": False}
        results = response.get("results")
        if not isinstance(results, list) or not results:
            failed = response.get("failed_results")
            first_failure = failed[0] if isinstance(failed, list) and failed and isinstance(failed[0], dict) else {}
            reason = _safe_failure_reason(first_failure.get("error"), self._api_key)
            message = "Tavily extraction failed: " + reason if reason else "Page extraction failed; Tavily returned no diagnostic reason."
            return _failure(message, content_available=False,
                            **self._metadata(response))
        item = results[0]
        try:
            returned_url = validate_public_url(item.get("url", url))
        except (ValueError, TypeError, AttributeError):
            return _failure("Extraction returned an unsafe page URL.", content_available=False)
        text = item.get("raw_content")
        if not isinstance(text, str) or len(text.strip()) < 80:
            return _failure("Page has too little extracted content to assess.", content_available=False,
                            **self._metadata(response))
        text = text.strip()
        # Content availability is transport information. The Coach decides
        # readability, including login/blocked screens, from the actual text.
        return {"status": "success", "ok": True, "content_available": True,
                "results": [{"url": returned_url, "title": str(item.get("title", ""))[:300],
                             "raw_content": text[:MAX_PAGE_CHARS]}],
                "truncated": len(text) > MAX_PAGE_CHARS, **self._metadata(response)}
