"""Controlled transport tests. These make no live provider or DNS calls."""
import socket
import urllib.error

import pytest

from interview_prep.retrieval import MAX_FAILURE_CHARS, MAX_PAGE_CHARS, TavilyRetrieval, validate_public_url, validate_query


@pytest.fixture(autouse=True)
def public_dns(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", lambda *args, **kwargs: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))])


@pytest.mark.parametrize("url", [
    "file:///etc/passwd", "http://localhost/a", "http://localhost./a",
    "http://127.0.0.1/a", "https://10.0.0.1/", "https://169.254.169.254/latest/",
    "http://[::1]/", "https://[fc00::1]/", "http://2130706433/",
    "http://0177.0.0.1/", "http://0x7f000001/", "http://127.1/",
    "https://server.internal/a", "https://intranet/a", "https://user:pass@example.com/",
    "https://example.com:8443/a", "https://example.com/\nsecret", "https://example.com\\@localhost/a",
    "https://example.com/?email=person%40example.com",
])
def test_rejects_unsafe_page_urls(url):
    with pytest.raises(ValueError):
        validate_public_url(url)


def test_public_url_removes_fragment():
    assert validate_public_url("https://docs.python.org/3/tutorial/#section") == "https://docs.python.org/3/tutorial/"


def test_dns_to_private_address_is_rejected(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", lambda *args, **kwargs: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 80))])
    with pytest.raises(ValueError, match="public addresses"):
        validate_public_url("https://public-name.example.com/path", resolve=True)


@pytest.mark.parametrize("query", ["a", "x" * 501, "Find person@example.com", "Interview 212-555-1234"])
def test_search_query_blocks_bad_size_and_contacts(query):
    with pytest.raises(ValueError):
        validate_query(query)


def test_http_status_code_sequence_is_not_mistaken_for_a_phone_number():
    assert validate_query("HTTP 401 403 404 negative testing") == "HTTP 401 403 404 negative testing"
    with pytest.raises(ValueError, match="contact"):
        validate_query("HTTP 401 403 404 call 212-555-1234")


def test_search_is_one_fixed_endpoint_request_with_safe_options():
    calls = []
    def transport(endpoint, body, headers, timeout):
        calls.append((endpoint, body, headers, timeout))
        return {"results": [{"url": "https://docs.example.com/guide", "title": "Guide", "content": "Snippet"},
                            {"url": "http://localhost/private", "title": "Bad"}],
                "request_id": "request-1", "usage": {"credits": 1}}
    result = TavilyRetrieval("secret-test-key", transport=transport).search_sources("API testing documentation", timeout=99)
    assert len(calls) == 1
    endpoint, body, headers, timeout = calls[0]
    assert endpoint == "https://api.tavily.com/search"
    assert body["include_answer"] is False and body["include_raw_content"] is False
    assert body["auto_parameters"] is False and body["max_results"] == 5
    assert timeout == 20
    assert headers["Authorization"] == "Bearer secret-test-key"
    assert result["ok"] and len(result["results"]) == 1 and result["rejected_results"] == 1
    assert result["usage"] == {"credits": 1}
    assert "secret-test-key" not in str(result)


def test_extract_sends_only_one_url_and_preserves_capped_content():
    calls = []
    def transport(endpoint, body, headers, timeout):
        calls.append((endpoint, body, timeout))
        return {"results": [{"url": body["urls"][0], "raw_content": "A" * (MAX_PAGE_CHARS + 30)}]}
    result = TavilyRetrieval("test-key", transport=transport).extract_page("https://example.com/page", timeout=7)
    assert calls[0][0] == "https://api.tavily.com/extract"
    assert calls[0][1]["urls"] == ["https://example.com/page"]
    assert calls[0][1]["timeout"] == 7 and calls[0][2] == 7
    assert calls[0][1]["extract_depth"] == "basic"
    assert result["content_available"] and result["truncated"]
    assert len(result["results"][0]["raw_content"]) == MAX_PAGE_CHARS


@pytest.mark.parametrize("response", [
    {"results": [], "failed_results": [{"url": "https://example.com", "error": "blocked"}]},
    {"results": [{"url": "https://example.com", "raw_content": "Login"}]},
    {"results": [{"url": "http://127.0.0.1", "raw_content": "Readable " * 30}]},
])
def test_failed_or_empty_extraction_reports_no_content(response):
    result = TavilyRetrieval("test-key", transport=lambda *args: response).extract_page("https://example.com/")
    assert result["ok"] is False and result["content_available"] is False


def test_empty_results_preserve_per_url_failure_and_timing_without_retry():
    calls = []
    def transport(endpoint, body, headers, timeout):
        calls.append((endpoint, body, timeout))
        return {"results": [], "failed_results": [{"url": "https://example.com/guide",
                    "error": "The page requires JavaScript rendering."}],
                "response_time": 1.25, "request_id": "extract-request-1", "usage": {"credits": 0}}
    result = TavilyRetrieval("test-key", transport=transport).extract_page("https://example.com/guide")
    assert result["error"] == "Tavily extraction failed: The page requires JavaScript rendering."
    assert result["response_time"] == 1.25 and result["request_id"] == "extract-request-1"
    assert result["usage"] == {"credits": 0} and result["retryable"] is False
    assert result["content_available"] is False and len(calls) == 1
    assert calls[0][1]["extract_depth"] == "basic" and calls[0][1]["timeout"] == calls[0][2] == 20


def test_provider_failure_reason_redacts_credentials_contacts_and_caps_length():
    secret = "actual-local-test-key"
    diagnostic = ("Access check failed. " + secret + " Authorization: Bearer bearer-secret "
                  'api_key="quoted-secret" token=query-secret&next=1 password=private-password '
                  "tvly-foreign-secret https://name:privatepass@example.com/page "
                  "person@example.com 212-555-1234 " + "additional details " * 80)
    response = {"results": [], "failed_results": [{"url": "https://example.com/", "error": diagnostic}]}
    result = TavilyRetrieval(secret, transport=lambda *args: response).extract_page("https://example.com/")
    error = result["error"]
    assert error.startswith("Tavily extraction failed: Access check failed.")
    for sensitive in (secret, "bearer-secret", "quoted-secret", "query-secret", "private-password",
                      "tvly-foreign-secret", "privatepass", "person@example.com", "212-555-1234"):
        assert sensitive not in str(result)
    assert len(error) <= len("Tavily extraction failed: ") + MAX_FAILURE_CHARS


@pytest.mark.parametrize("failed_results", [None, [], ["unexpected text"], [{"error": {"details": "untrusted object"}}]])
def test_missing_or_malformed_failure_reason_uses_honest_generic_message(failed_results):
    response = {"results": [], "failed_results": failed_results, "response_time": float("nan")}
    result = TavilyRetrieval("test-key", transport=lambda *args: response).extract_page("https://example.com/")
    assert result["error"] == "Page extraction failed; Tavily returned no diagnostic reason."
    assert "response_time" not in result and "untrusted object" not in str(result)
    assert result["retryable"] is False


@pytest.mark.parametrize("content", [
    "# Sign in\nPlease sign in to continue to this private resource.\nEmail\nPassword\nForgot your password?\nRemember me\nSign in",
    "Access denied\nYou do not have permission to access this resource. Your request has been blocked. Contact the site administrator.",
    "# Just a moment...\nPlease verify you are human. Enable JavaScript and cookies to continue. This browser check may take a few seconds.",
])
def test_access_gate_text_is_preserved_for_coach_assessment(content):
    assert len(content) > 80
    response = {"results": [{"url": "https://example.com/guide", "raw_content": content}], "request_id": "gate-1"}
    result = TavilyRetrieval("test-key", transport=lambda *args: response).extract_page("https://example.com/guide")
    assert result["ok"] is True and result["content_available"] is True
    assert "readable" not in result and result["request_id"] == "gate-1"
    assert result["results"][0]["raw_content"] == content


@pytest.mark.parametrize("content", [
    "# Sign in\nAuthentication documentation: a login form accepts a password and establishes a server-side session. Always validate credentials securely and expire the session.",
    '# Authentication guide\nA gate can say "Sign in to continue", "Password", and "Forgot your password". This guide explains how to test the redirect and session behavior.',
    "# Access denied\nThis documentation explains HTTP 403 authorization errors. Check the user's permissions and the server's access-control policy when debugging a rejected request.",
    "# Gardening guide\nPlant seedlings in fertile soil, water them consistently, and ensure adequate drainage. This is readable content even when unrelated to an interview topic.",
])
def test_authentication_documentation_and_wrong_topic_prose_are_returned_without_semantic_filter(content):
    response = {"results": [{"url": "https://example.com/guide", "raw_content": content}]}
    result = TavilyRetrieval("test-key", transport=lambda *args: response).extract_page("https://example.com/guide")
    assert result["ok"] is True and result["content_available"] is True and "readable" not in result
    assert result["results"][0]["raw_content"] == content


@pytest.mark.parametrize("exception,retryable", [
    (TimeoutError("test-secret-key"), False),
    (urllib.error.URLError("test-secret-key"), True),
    (urllib.error.HTTPError("url", 429, "test-secret-key", {}, None), True),
    (urllib.error.HTTPError("url", 401, "test-secret-key", {}, None), False),
    (ValueError("test-secret-key"), False),
])
def test_transport_errors_are_redacted_and_never_retried(exception, retryable):
    calls = []
    def transport(*args):
        calls.append(True)
        raise exception
    result = TavilyRetrieval("test-secret-key", transport=transport).search_sources("public testing topics")
    assert len(calls) == 1
    assert result["ok"] is False and result["retryable"] is retryable
    assert "test-secret-key" not in str(result)


def test_invalid_arguments_are_blocked_before_transport():
    calls = []
    client = TavilyRetrieval("key", transport=lambda *args: calls.append(args))
    assert not client.search_sources("person@example.com")["ok"]
    assert not client.extract_page("https://127.0.0.1/")["ok"]
    assert not calls


def test_missing_configuration_and_malformed_provider_shape():
    assert not TavilyRetrieval("").search_sources("public topic")["ok"]
    assert not TavilyRetrieval("key", transport=lambda *args: []).search_sources("public topic")["ok"]
    assert not TavilyRetrieval("key", transport=lambda *args: {}).search_sources("public topic")["ok"]


def test_socket_timeout_is_an_unknown_outcome_requiring_deliberate_recovery():
    def transport(*args):
        raise TimeoutError("the HTTP request may still have been processed")
    result = TavilyRetrieval("key", transport=transport).search_sources("public documentation")
    assert result["unknown_outcome"] is True and result["retryable"] is False
