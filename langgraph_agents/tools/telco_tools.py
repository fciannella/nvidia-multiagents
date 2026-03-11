"""Telco workflow tools.

LangChain @tool functions for the telco workflow agent. These wrap the
business logic in telco_logic.py and include `ask_user` for human-in-the-loop
via interrupt().

The telco workflow uses `ask_user` for:
  - Collecting the caller's phone number (MSISDN)
  - Asking for the OTP code
  - Getting confirmation before destructive actions (close contract, etc.)
"""

import json
from typing import Optional

from langchain_core.tools import tool
from langgraph.types import interrupt

from tools import telco_logic


@tool
def ask_user(question: str) -> str:
    """Ask the user a follow-up question and wait for their answer.
    Use this when you need information from the user to proceed:
    phone number, OTP code, confirmation, preferences, country, etc.
    The question should be clear and specific.
    """
    return interrupt({"question": question})


@tool
def check_auth_tool(session_id: str) -> str:
    """Check whether the current session is already authenticated.
    Call this FIRST before starting login. If verified=true, skip login
    and use the msisdn from the response.
    """
    verified = telco_logic.is_verified(session_id)
    msisdn = telco_logic.get_session_msisdn(session_id)
    return json.dumps({"verified": verified, "msisdn": msisdn or ""})


@tool
def start_login_tool(session_id: str, msisdn: str) -> str:
    """Send a one-time SMS code to the given mobile number for authentication.
    Returns JSON with {sent, masked, destination} on success.
    """
    return json.dumps(telco_logic.start_login(session_id, msisdn))


@tool
def verify_login_tool(session_id: str, msisdn: str, otp: str) -> str:
    """Verify the one-time code the user received via SMS.
    Returns JSON with {verified, session_id, msisdn}.
    """
    return json.dumps(telco_logic.verify_login(session_id, msisdn, otp))


@tool
def get_current_package_tool(session_id: str, msisdn: str) -> str:
    """Get the customer's current mobile package, contract status, and addons. Requires authenticated session."""
    return json.dumps(telco_logic.get_current_package(session_id, msisdn))


@tool
def get_data_balance_tool(session_id: str, msisdn: str) -> str:
    """Get the customer's current data usage and remaining allowance this month. Requires authenticated session."""
    return json.dumps(telco_logic.get_data_balance(session_id, msisdn))


@tool
def list_available_packages_tool() -> str:
    """List all available mobile packages with fees and features."""
    return json.dumps(telco_logic.list_available_packages())


@tool
def recommend_packages_tool(session_id: str, msisdn: str, preferences_json: Optional[str] = None) -> str:
    """Recommend up to 3 packages based on the customer's usage history
    and optional preferences (JSON with need_5g, travel_country, budget). Requires authenticated session.
    """
    prefs = {}
    if isinstance(preferences_json, str) and preferences_json.strip():
        try:
            prefs = json.loads(preferences_json)
        except Exception:
            pass
    return json.dumps(telco_logic.recommend_packages(session_id, msisdn, prefs))


@tool
def get_roaming_info_tool(session_id: str, msisdn: str, country_code: str) -> str:
    """Get roaming pricing and available passes for a specific country.
    Shows whether roaming is included in the current package. Requires authenticated session.
    """
    return json.dumps(telco_logic.get_roaming_info(session_id, msisdn, country_code))


@tool
def close_contract_tool(session_id: str, msisdn: str, confirm: bool = False) -> str:
    """Close the customer's contract. First call with confirm=false to show
    the early termination fee. Only set confirm=true after explicit user
    confirmation. Requires authenticated session.
    """
    return json.dumps(telco_logic.close_contract(session_id, msisdn, bool(confirm)))


@tool
def list_addons_tool(session_id: str, msisdn: str) -> str:
    """List the customer's active addons (roaming passes, etc.). Requires authenticated session."""
    return json.dumps(telco_logic.list_addons(session_id, msisdn))


@tool
def purchase_roaming_pass_tool(session_id: str, msisdn: str, country_code: str, pass_id: str) -> str:
    """Purchase a roaming pass for a country. Use get_roaming_info_tool first
    to find available pass_id values. Requires authenticated session.
    """
    return json.dumps(telco_logic.purchase_roaming_pass(session_id, msisdn, country_code, pass_id))


@tool
def change_package_tool(session_id: str, msisdn: str, package_id: str, effective: str = "next_cycle") -> str:
    """Change the customer's mobile package. Set effective='now' for immediate
    change or 'next_cycle' (default) to take effect at the next billing cycle. Requires authenticated session.
    """
    return json.dumps(telco_logic.change_package(session_id, msisdn, package_id, effective))


@tool
def get_billing_summary_tool(session_id: str, msisdn: str) -> str:
    """Get the customer's billing summary including monthly fee and last bill. Requires authenticated session."""
    return json.dumps(telco_logic.get_billing_summary(session_id, msisdn))


@tool
def set_data_alerts_tool(session_id: str, msisdn: str, threshold_percent: Optional[int] = None, threshold_gb: Optional[float] = None) -> str:
    """Set data usage alerts. Specify threshold_percent (e.g. 80) and/or
    threshold_gb (e.g. 5.0) to get notified when usage exceeds the limit. Requires authenticated session.
    """
    return json.dumps(telco_logic.set_data_alerts(session_id, msisdn, threshold_percent, threshold_gb))


ALL_TOOLS = [
    ask_user,
    check_auth_tool,
    start_login_tool,
    verify_login_tool,
    get_current_package_tool,
    get_data_balance_tool,
    list_available_packages_tool,
    recommend_packages_tool,
    get_roaming_info_tool,
    close_contract_tool,
    list_addons_tool,
    purchase_roaming_pass_tool,
    change_package_tool,
    get_billing_summary_tool,
    set_data_alerts_tool,
]

TOOL_NAMES = [t.name for t in ALL_TOOLS]
TOOLS_BY_NAME = {t.name: t for t in ALL_TOOLS}

TELCO_TOOL_DESCRIPTIONS = [
    {"name": "telco_account", "description": "Manage mobile account: check package, data balance, billing, roaming, change plans, OTP login", "duration_secs": 10},
]
