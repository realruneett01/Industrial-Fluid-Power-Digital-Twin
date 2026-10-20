"""Level-3 CMMS bridge: advisory work orders posted to an ERPNext (Frappe) REST endpoint.

Only outbound HTTP POSTs leave Level 2; nothing is ever written to Level 1. Credentials come
from the environment (``ERPNEXT_URL``, ``ERPNEXT_API_KEY``, ``ERPNEXT_API_SECRET``) and are
never logged. Frappe's REST API is ``POST /api/resource/<DocType>`` with the header
``Authorization: token <key>:<secret>``. The default DocType is ``Issue``; its field mapping
has not been checked against a live ERPNext instance.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime

log = logging.getLogger(__name__)

DOCTYPE = "Issue"
TIMEOUT_S = 2.0
ACTIONS = {
    "accumulator": "Recharge nitrogen bladder / check accumulator pre-charge",
    "valve": "Inspect proportional valve for switching lag; service or replace",
    "pump": "Inspect main pump for internal leakage; check bypass orifice V9",
    "cooler": "Clean or service oil cooler C1; check cooling water flow",
}
PRIORITY = {1: "Low", 2: "Medium", 3: "High"}


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse redirects so the Authorization header can never reach another host."""

    def redirect_request(self, *_args, **_kwargs):
        return None


_OPENER = urllib.request.build_opener(_NoRedirect)


def build_work_order(
    asset_tag: str, fault_type: str, severity: int, metric: float, timestamp: datetime | None = None
) -> dict:
    """Structured JSON payload: asset tag, fault type, severity, timestamp, degradation metric."""
    if fault_type not in ACTIONS:
        raise ValueError(f"unknown fault type {fault_type!r}; expected one of {sorted(ACTIONS)}")
    sev = int(severity)
    met = float(metric)
    ts = (timestamp or datetime.now(UTC)).isoformat()
    return {
        "subject": f"[{asset_tag}] {fault_type} degradation (severity {sev})",
        "priority": PRIORITY.get(sev, "High"),
        "description": f"{ACTIONS[fault_type]}. Degradation metric {met:.4f} at {ts}.",
        "hydro_asset_tag": asset_tag,
        "hydro_fault_type": fault_type,
        "hydro_severity": sev,
        "hydro_metric": met,
        "hydro_timestamp": ts,
    }


def _request(payload: dict, base_url: str) -> urllib.request.Request:
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/resource/{DOCTYPE}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    key, secret = os.environ.get("ERPNEXT_API_KEY"), os.environ.get("ERPNEXT_API_SECRET")
    if key and secret:
        req.add_header("Authorization", f"token {key}:{secret}")
    return req


def dispatch_cmms_work_order(fault_payload: dict, base_url: str | None = None) -> int:
    """POST a work order; return the HTTP status, or 0 if the CMMS is unreachable.

    Never raises on network failure, so the supervisory loop cannot stall.
    """
    url = base_url or os.environ.get("ERPNEXT_URL", "http://127.0.0.1:8000")
    try:
        with _OPENER.open(_request(fault_payload, url), timeout=TIMEOUT_S) as resp:
            return resp.status
    except urllib.error.HTTPError as err:
        log.warning("CMMS rejected work order: HTTP %s", err.code)
        return err.code
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as err:
        log.warning("CMMS unreachable: %s", type(err).__name__)
        return 0


HYDRO_CUSTOM_FIELDS: tuple[dict[str, str], ...] = (
    {"fieldname": "hydro_asset_tag", "label": "Hydro Asset Tag", "fieldtype": "Data"},
    {"fieldname": "hydro_fault_type", "label": "Hydro Fault Type", "fieldtype": "Data"},
    {"fieldname": "hydro_severity", "label": "Hydro Severity", "fieldtype": "Int"},
    {"fieldname": "hydro_metric", "label": "Hydro Metric", "fieldtype": "Float"},
    {"fieldname": "hydro_timestamp", "label": "Hydro Timestamp", "fieldtype": "Data"},
)


def _send_cmms_request(url: str, method: str, payload: dict | None = None) -> int:
    """Send an HTTP request with CMMS token auth; return status or 404 on missing GET."""
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method=method,
    )
    key, secret = os.environ.get("ERPNEXT_API_KEY"), os.environ.get("ERPNEXT_API_SECRET")
    if key and secret:
        req.add_header("Authorization", f"token {key}:{secret}")
    try:
        with _OPENER.open(req, timeout=TIMEOUT_S) as resp:
            return resp.status
    except urllib.error.HTTPError as err:
        if err.code == 404 and method == "GET":
            return 404
        raise RuntimeError(f"CMMS schema request failed: HTTP {err.code}") from None
    except (urllib.error.URLError, TimeoutError, OSError) as err:
        raise RuntimeError(f"CMMS unreachable: {type(err).__name__}") from None


def _ensure_cmms_record(base_url: str, doctype: str, record_id: str, payload: dict) -> bool:
    """Ensure a CMMS record exists; return True if created, False if existing."""
    quoted_dt = urllib.parse.quote(doctype)
    get_url = f"{base_url.rstrip('/')}/api/resource/{quoted_dt}/{urllib.parse.quote(record_id)}"
    if _send_cmms_request(get_url, method="GET") != 404:
        return False
    post_url = f"{base_url.rstrip('/')}/api/resource/{quoted_dt}"
    _send_cmms_request(post_url, method="POST", payload=payload)
    return True


def ensure_erpnext_schema(base_url: str | None = None) -> dict[str, int]:
    """Ensure Issue Priority records and hydro_* Custom Fields exist in ERPNext."""
    url = base_url or os.environ.get("ERPNEXT_URL", "http://127.0.0.1:8000")
    created, existing = 0, 0
    for name in ("Low", "Medium", "High"):
        if _ensure_cmms_record(url, "Issue Priority", name, {"name": name}):
            created += 1
        else:
            existing += 1
    for cf in HYDRO_CUSTOM_FIELDS:
        record_id = f"Issue-{cf['fieldname']}"
        payload = {"dt": "Issue", "insert_after": "description", **cf}
        if _ensure_cmms_record(url, "Custom Field", record_id, payload):
            created += 1
        else:
            existing += 1
    return {"created": created, "existing": existing}
