"""Telco business logic with mock data.

Provides in-memory session/OTP stores and functions for customer lookup,
authentication, package management, roaming, billing, and addons.
Data is loaded from JSON fixtures in ../mock_data/.
"""

import json
import os
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


_FIXTURE_CACHE: Dict[str, Any] = {}
_SESSIONS: Dict[str, Dict[str, Any]] = {}
_OTP_DB: Dict[str, Dict[str, Any]] = {}


def _fixtures_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "mock_data"


def _load_fixture(name: str) -> Any:
    if name in _FIXTURE_CACHE:
        return _FIXTURE_CACHE[name]
    p = _fixtures_dir() / name
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    _FIXTURE_CACHE[name] = data
    return data


def _normalize_msisdn(msisdn: Optional[str]) -> Optional[str]:
    if not isinstance(msisdn, str) or not msisdn.strip():
        return None
    s = msisdn.strip()
    digits = "".join(ch for ch in s if ch.isdigit() or ch == "+")
    if digits.startswith("+"):
        return digits
    return f"+{digits}"


def _get_customer(msisdn: str) -> Dict[str, Any]:
    ms = _normalize_msisdn(msisdn) or ""
    data = _load_fixture("customers.json")
    return dict((data.get("customers", {}) or {}).get(ms, {}))


def _get_package(package_id: str) -> Dict[str, Any]:
    pkgs = _load_fixture("packages.json").get("packages", [])
    for p in pkgs:
        if str(p.get("id")) == str(package_id):
            return dict(p)
    return {}


def _get_roaming_country(country_code: str) -> Dict[str, Any]:
    data = _load_fixture("roaming_rates.json")
    return dict(
        (data.get("countries", {}) or {}).get((country_code or "").upper(), {})
    )


def _mask_phone(msisdn: str) -> str:
    s = _normalize_msisdn(msisdn) or ""
    tail = s[-2:] if len(s) >= 2 else s
    return f"***-***-**{tail}"


# ---------------------------------------------------------------------------
# Identity via SMS OTP
# ---------------------------------------------------------------------------

def start_login(session_id: str, msisdn: str) -> Dict[str, Any]:
    ms = _normalize_msisdn(msisdn)
    if not ms:
        return {"sent": False, "error": "invalid_msisdn"}
    cust = _get_customer(ms)
    if not cust:
        return {"sent": False, "reason": "not_found"}

    static = None
    try:
        data = _load_fixture("otps.json")
        if isinstance(data, dict):
            byn = data.get("by_number", {}) or {}
            static = byn.get(ms) or data.get("default")
    except Exception:
        static = None

    code = str(static or f"{uuid.uuid4().int % 1000000:06d}").zfill(6)
    _OTP_DB[ms] = {"otp": code, "created_at": datetime.utcnow().isoformat() + "Z"}
    _SESSIONS[session_id] = {"verified": False, "msisdn": ms}

    resp: Dict[str, Any] = {"sent": True, "masked": _mask_phone(ms), "destination": "sms"}
    if os.getenv("TELCO_DEBUG_OTP", "0").lower() not in ("", "0", "false"):
        resp["debug_code"] = code
    return resp


def verify_login(session_id: str, msisdn: str, otp: str) -> Dict[str, Any]:
    ms = _normalize_msisdn(msisdn) or ""
    rec = _OTP_DB.get(ms) or {}
    ok = str(rec.get("otp")) == str(otp)
    sess = _SESSIONS.get(session_id) or {"verified": False}
    if ok:
        rec["used_at"] = datetime.utcnow().isoformat() + "Z"
        _OTP_DB[ms] = rec
        sess["verified"] = True
        sess["msisdn"] = ms
    _SESSIONS[session_id] = sess
    return {"session_id": session_id, "verified": ok, "msisdn": ms}


def is_verified(session_id: str) -> bool:
    sess = _SESSIONS.get(session_id) or {}
    return bool(sess.get("verified"))


def get_session_msisdn(session_id: str) -> Optional[str]:
    sess = _SESSIONS.get(session_id) or {}
    return sess.get("msisdn")


def require_verified_session(
    session_id: str, msisdn: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Require a verified session for account operations.
    Returns None if OK; returns an error dict if not verified or msisdn mismatch.
    If msisdn is omitted, the session's msisdn is used (caller should use return value).
    """
    if not session_id or not session_id.strip():
        return {"error": "auth_required", "message": "Session ID required for this operation."}
    sess = _SESSIONS.get(session_id) or {}
    if not sess.get("verified"):
        return {"error": "not_authenticated", "message": "Please complete login (OTP verification) first."}
    session_msisdn = sess.get("msisdn")
    if msisdn is not None and msisdn != "":
        ms_norm = _normalize_msisdn(msisdn)
        if ms_norm != session_msisdn:
            return {"error": "auth_mismatch", "message": "This session is for a different number."}
    return None


# ---------------------------------------------------------------------------
# Customer and package information
# ---------------------------------------------------------------------------

def get_current_package(session_id: str, msisdn: str) -> Dict[str, Any]:
    err = require_verified_session(session_id, msisdn)
    if err is not None:
        return err
    cust = _get_customer(msisdn)
    if not cust:
        return {"error": "not_found"}
    pkg = _get_package(cust.get("package_id", ""))
    return {
        "msisdn": _normalize_msisdn(msisdn),
        "package": pkg,
        "contract": cust.get("contract"),
        "addons": list(cust.get("addons", [])),
    }


def get_data_balance(session_id: str, msisdn: str) -> Dict[str, Any]:
    err = require_verified_session(session_id, msisdn)
    if err is not None:
        return err
    cust = _get_customer(msisdn)
    if not cust:
        return {"error": "not_found"}
    pkg = _get_package(cust.get("package_id", ""))
    usage = (cust.get("usage", {}) or {}).get("current_month", {})
    included = (
        float(pkg.get("data_gb", 0))
        if not bool(pkg.get("unlimited", False))
        else None
    )
    used = float(usage.get("data_gb_used", 0.0))
    remaining = None if included is None else max(0.0, included - used)
    return {
        "msisdn": _normalize_msisdn(msisdn),
        "unlimited": bool(pkg.get("unlimited", False)),
        "included_gb": included,
        "used_gb": round(used, 2),
        "remaining_gb": None if remaining is None else round(remaining, 2),
        "resets_day": int((cust.get("billing", {}) or {}).get("cycle_day", 1)),
    }


def list_available_packages() -> List[Dict[str, Any]]:
    return list(_load_fixture("packages.json").get("packages", []))


def _estimate_monthly_cost(
    pkg: Dict[str, Any], avg_data: float, avg_min: int, avg_sms: int
) -> float:
    if bool(pkg.get("unlimited", False)):
        return float(pkg.get("monthly_fee", 0.0))
    fee = float(pkg.get("monthly_fee", 0.0))
    rates = pkg.get("overage", {}) or {}
    data_over = max(0.0, avg_data - float(pkg.get("data_gb", 0.0)))
    min_over = max(0, avg_min - int(pkg.get("minutes", 0)))
    sms_over = max(0, avg_sms - int(pkg.get("sms", 0)))
    over = (
        data_over * float(rates.get("per_gb", 0.0))
        + min_over * float(rates.get("per_min", 0.0))
        + sms_over * float(rates.get("per_sms", 0.0))
    )
    return round(fee + over, 2)


def recommend_packages(
    session_id: str, msisdn: str, preferences: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    err = require_verified_session(session_id, msisdn)
    if err is not None:
        return err
    cust = _get_customer(msisdn)
    if not cust:
        return {"error": "not_found"}
    prefs = preferences or {}
    hist = (cust.get("usage", {}) or {}).get("history", [])
    if hist:
        last = hist[-3:]
        avg_data = sum(float(m.get("data_gb", 0)) for m in last) / len(last)
        avg_min = int(sum(int(m.get("minutes", 0)) for m in last) / len(last))
        avg_sms = int(sum(int(m.get("sms", 0)) for m in last) / len(last))
    else:
        avg = (cust.get("usage", {}) or {}).get("monthly_avg", {})
        avg_data = float(avg.get("data_gb", 10.0))
        avg_min = int(avg.get("minutes", 300))
        avg_sms = int(avg.get("sms", 100))

    wants_5g = bool(prefs.get("need_5g", False))
    travel_country = (prefs.get("travel_country") or "").upper()
    budget = float(prefs.get("budget", 9999))

    pkgs = list_available_packages()
    scored: List[Tuple[float, Dict[str, Any], Dict[str, Any]]] = []
    for p in pkgs:
        if wants_5g and not bool(p.get("fiveg", False)):
            continue
        if budget < float(p.get("monthly_fee", 0.0)):
            continue
        est = _estimate_monthly_cost(p, avg_data, avg_min, avg_sms)
        feature_bonus = 0.0
        if travel_country and travel_country in set(
            p.get("roam_included_countries", [])
        ):
            feature_bonus -= 5.0
        if bool(p.get("data_rollover", False)):
            feature_bonus -= 1.0
        score = est + feature_bonus
        rationale = f"Estimated monthly cost {est:.2f}; {'5G' if p.get('fiveg') else '4G'}"
        if travel_country:
            rationale += (
                "; roam-included"
                if travel_country in (p.get("roam_included_countries") or [])
                else "; roam-paygo"
            )
        scored.append((score, p, {"estimated_cost": est, "rationale": rationale}))

    scored.sort(key=lambda x: x[0])
    top = [
        {
            "package": pkg,
            "estimated_monthly_cost": meta["estimated_cost"],
            "rationale": meta["rationale"],
        }
        for _, pkg, meta in scored[:3]
    ]
    return {
        "msisdn": _normalize_msisdn(msisdn),
        "based_on": {
            "avg_data_gb": round(avg_data, 2),
            "avg_minutes": avg_min,
            "avg_sms": avg_sms,
        },
        "recommendations": top,
    }


def get_roaming_info(session_id: str, msisdn: str, country_code: str) -> Dict[str, Any]:
    err = require_verified_session(session_id, msisdn)
    if err is not None:
        return err
    cust = _get_customer(msisdn)
    if not cust:
        return {"error": "not_found"}
    pkg = _get_package(cust.get("package_id", ""))
    country = _get_roaming_country(country_code)
    included = country_code.upper() in set(
        pkg.get("roam_included_countries") or []
    ) or (bool(pkg.get("eu_roaming", False)) and country.get("region") == "EU")
    return {
        "msisdn": _normalize_msisdn(msisdn),
        "package": {"id": pkg.get("id"), "name": pkg.get("name")},
        "roaming": {
            "country": country_code.upper(),
            "included": bool(included),
            "paygo": country.get("paygo"),
            "passes": country.get("passes", []),
        },
    }


def close_contract(session_id: str, msisdn: str, confirm: bool = False) -> Dict[str, Any]:
    err = require_verified_session(session_id, msisdn)
    if err is not None:
        return err
    cust = _get_customer(msisdn)
    if not cust:
        return {"error": "not_found"}
    contract = dict(cust.get("contract", {}))
    if contract.get("status") == "closed":
        return {"status": "already_closed"}
    try:
        end = contract.get("end_date")
        future = (
            datetime.fromisoformat(end) if isinstance(end, str) and end else datetime.max
        )
    except Exception:
        future = datetime.max
    fee = (
        float(contract.get("early_termination_fee", 0.0))
        if future > datetime.utcnow()
        else 0.0
    )
    summary = {
        "msisdn": _normalize_msisdn(msisdn),
        "current_status": contract.get("status", "active"),
        "early_termination_fee": round(fee, 2),
        "will_cancel": bool(confirm),
    }
    if confirm:
        contract["status"] = "closed"
        contract["closed_at"] = datetime.utcnow().isoformat() + "Z"
        cust["contract"] = contract
        data = _load_fixture("customers.json")
        ms = _normalize_msisdn(msisdn)
        data.setdefault("customers", {})[ms] = cust
        _FIXTURE_CACHE["customers.json"] = data
        summary["new_status"] = "closed"
    return summary


# ---------------------------------------------------------------------------
# Extended utilities
# ---------------------------------------------------------------------------

def list_addons(session_id: str, msisdn: str) -> Dict[str, Any]:
    err = require_verified_session(session_id, msisdn)
    if err is not None:
        return err
    cust = _get_customer(msisdn)
    if not cust:
        return {"error": "not_found"}
    return {"msisdn": _normalize_msisdn(msisdn), "addons": list(cust.get("addons", []))}


def purchase_roaming_pass(
    session_id: str, msisdn: str, country_code: str, pass_id: str
) -> Dict[str, Any]:
    err = require_verified_session(session_id, msisdn)
    if err is not None:
        return err
    cust = _get_customer(msisdn)
    if not cust:
        return {"error": "not_found"}
    country = _get_roaming_country(country_code)
    passes = country.get("passes", [])
    sel = next((p for p in passes if str(p.get("id")) == str(pass_id)), None)
    if not sel:
        return {"error": "invalid_pass"}
    valid_days = int(sel.get("valid_days", 1))
    addon = {
        "type": "roaming_pass",
        "country": (country_code or "").upper(),
        "data_mb": int(sel.get("data_mb", 0)),
        "price": float(sel.get("price", 0.0)),
        "purchased_at": datetime.utcnow().isoformat() + "Z",
        "expires": (datetime.utcnow() + timedelta(days=valid_days)).date().isoformat(),
    }
    cust.setdefault("addons", []).append(addon)
    data = _load_fixture("customers.json")
    ms = _normalize_msisdn(msisdn)
    data.setdefault("customers", {})[ms] = cust
    _FIXTURE_CACHE["customers.json"] = data
    return {"msisdn": _normalize_msisdn(msisdn), "added": addon}


def change_package(
    session_id: str, msisdn: str, package_id: str, effective: str = "next_cycle"
) -> Dict[str, Any]:
    err = require_verified_session(session_id, msisdn)
    if err is not None:
        return err
    cust = _get_customer(msisdn)
    if not cust:
        return {"error": "not_found"}
    new_pkg = _get_package(package_id)
    if not new_pkg:
        return {"error": "invalid_package"}
    effective_when = (effective or "next_cycle").lower()
    if effective_when not in ("now", "next_cycle"):
        effective_when = "next_cycle"
    summary = {
        "msisdn": _normalize_msisdn(msisdn),
        "current_package_id": cust.get("package_id"),
        "new_package_id": new_pkg.get("id"),
        "effective": effective_when,
    }
    data = _load_fixture("customers.json")
    ms = _normalize_msisdn(msisdn)
    if effective_when == "now":
        cust["package_id"] = new_pkg.get("id")
        summary["status"] = "changed"
    else:
        contract = dict(cust.get("contract", {}))
        contract["pending_change"] = {
            "package_id": new_pkg.get("id"),
            "requested_at": datetime.utcnow().isoformat() + "Z",
        }
        cust["contract"] = contract
        summary["status"] = "scheduled"
    data.setdefault("customers", {})[ms] = cust
    _FIXTURE_CACHE["customers.json"] = data
    return summary


def get_billing_summary(session_id: str, msisdn: str) -> Dict[str, Any]:
    err = require_verified_session(session_id, msisdn)
    if err is not None:
        return err
    cust = _get_customer(msisdn)
    if not cust:
        return {"error": "not_found"}
    pkg = _get_package(cust.get("package_id", ""))
    bill = dict(cust.get("billing", {}))
    return {
        "msisdn": _normalize_msisdn(msisdn),
        "last_bill_amount": bill.get("last_bill_amount"),
        "cycle_day": bill.get("cycle_day"),
        "monthly_fee": float(pkg.get("monthly_fee", 0.0)),
    }


def set_data_alerts(
    session_id: str,
    msisdn: str,
    threshold_percent: Optional[int] = None,
    threshold_gb: Optional[float] = None,
) -> Dict[str, Any]:
    err = require_verified_session(session_id, msisdn)
    if err is not None:
        return err
    if threshold_percent is None and threshold_gb is None:
        return {"error": "invalid_threshold"}
    cust = _get_customer(msisdn)
    if not cust:
        return {"error": "not_found"}
    alerts = dict(cust.get("alerts", {}))
    if isinstance(threshold_percent, int):
        alerts["data_threshold_percent"] = max(1, min(100, threshold_percent))
    if isinstance(threshold_gb, (int, float)):
        alerts["data_threshold_gb"] = max(0.1, float(threshold_gb))
    cust["alerts"] = alerts
    data = _load_fixture("customers.json")
    ms = _normalize_msisdn(msisdn)
    data.setdefault("customers", {})[ms] = cust
    _FIXTURE_CACHE["customers.json"] = data
    return {"msisdn": _normalize_msisdn(msisdn), "alerts": alerts}
