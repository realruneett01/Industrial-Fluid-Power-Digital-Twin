"""Live integration tests for ERPNext CMMS work order dispatch and query.

Opt-in test module: skipped unless ERPNEXT_URL, ERPNEXT_API_KEY, and
ERPNEXT_API_SECRET environment variables are configured.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
import warnings
from typing import Any

import pytest

from hydro.online.cmms import build_work_order, dispatch_cmms_work_order

_REQUIRED_ENV = ("ERPNEXT_URL", "ERPNEXT_API_KEY", "ERPNEXT_API_SECRET")
_MISSING_ENV = [var for var in _REQUIRED_ENV if not os.environ.get(var)]

pytestmark = pytest.mark.skipif(
    bool(_MISSING_ENV),
    reason=f"ERPNext credentials missing: {', '.join(_MISSING_ENV)}",
)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse redirects so credentials cannot be forwarded to untrusted hosts."""

    def redirect_request(self, *_args: Any, **_kwargs: Any) -> None:
        return None


_OPENER = urllib.request.build_opener(_NoRedirect)


def _auth_headers() -> dict[str, str]:
    """Return authorization headers without logging or exposing credentials."""
    key = os.environ.get("ERPNEXT_API_KEY", "")
    secret = os.environ.get("ERPNEXT_API_SECRET", "")
    return {
        "Authorization": f"token {key}:{secret}",
        "Accept": "application/json",
    }


def _query_issues(base_url: str, subject_pattern: str) -> list[dict[str, Any]]:
    """Query Issue doctype records matching subject pattern via GET."""
    params = {
        "filters": json.dumps([["subject", "like", subject_pattern]]),
        "fields": json.dumps(["*"]),
    }
    url = f"{base_url.rstrip('/')}/api/resource/Issue?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers=_auth_headers(), method="GET")
    with _OPENER.open(req, timeout=5.0) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body.get("data", [])


def _delete_issue(base_url: str, name: str) -> None:
    """Delete an Issue record by name via DELETE."""
    quoted_name = urllib.parse.quote(name, safe="")
    url = f"{base_url.rstrip('/')}/api/resource/Issue/{quoted_name}"
    req = urllib.request.Request(url, headers=_auth_headers(), method="DELETE")
    try:
        with _OPENER.open(req, timeout=5.0):
            pass
    except urllib.error.HTTPError as err:
        if err.code != 404:
            raise
    except (urllib.error.URLError, TimeoutError, OSError):
        pass


def _delete_matching_issues(base_url: str, subject_pattern: str) -> None:
    """Find and delete issues matching subject pattern."""
    try:
        issues = _query_issues(base_url, subject_pattern)
    except (urllib.error.URLError, TimeoutError, OSError):
        return
    for issue in issues:
        name = issue.get("name")
        if name:
            _delete_issue(base_url, str(name))


@pytest.fixture
def cleanup_issues():
    """Cleanup fixture deleting test issues before and after live tests."""
    base_url = os.environ.get("ERPNEXT_URL", "")
    if base_url:
        _delete_matching_issues(base_url, "%HYD-LIVE-TEST%")
    yield
    if base_url:
        _delete_matching_issues(base_url, "%HYD-LIVE-TEST%")


def _find_matching_issue(
    issues: list[dict[str, Any]], expected_subject: str
) -> dict[str, Any] | None:
    """Find issue with matching subject from query results."""
    for issue in issues:
        if issue.get("subject") == expected_subject:
            return issue
    return None


def _assert_standard_fields(issue: dict[str, Any], payload: dict[str, Any]) -> None:
    """Assert standard fields round-trip correctly."""
    assert issue.get("subject") == payload["subject"]
    assert issue.get("priority") == payload["priority"]
    assert issue.get("description") == payload["description"]


def _report_custom_fields(issue: dict[str, Any], payload: dict[str, Any]) -> None:
    """Report persisted hydro_* custom fields or warn if dropped."""
    for key in payload:
        if not key.startswith("hydro_"):
            continue
        if key not in issue or issue[key] is None:
            warnings.warn(
                f"ERPNext dropped custom field {key!r} (needs Custom Field on Issue doctype)",
                UserWarning,
                stacklevel=2,
            )


def test_cmms_live_dispatch_and_round_trip(cleanup_issues: None) -> None:
    """Test dispatching a work order and reading it back from live ERPNext."""
    payload = build_work_order("HYD-LIVE-TEST", "valve", 3, 0.26)
    base_url = os.environ["ERPNEXT_URL"]

    status = dispatch_cmms_work_order(payload, base_url=base_url)
    assert 200 <= status < 300, f"Expected 2xx status from CMMS, got {status}"

    issues = _query_issues(base_url, "%HYD-LIVE-TEST%")
    assert len(issues) >= 1, "No Issue found matching '%HYD-LIVE-TEST%'"

    target = _find_matching_issue(issues, payload["subject"])
    assert target is not None, f"Issue with subject {payload['subject']!r} not found"

    _assert_standard_fields(target, payload)
    _report_custom_fields(target, payload)
