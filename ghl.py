import requests
import logging
from config import GHL_BASE, GHL_HEADERS, FIELD_IDS

log = logging.getLogger(__name__)


def _request(method: str, path: str, **kwargs):
    """One place for every GHL API call, with retries."""
    for attempt in range(3):
        try:
            r = requests.request(
                method, f"{GHL_BASE}{path}",
                headers=GHL_HEADERS, timeout=10, **kwargs
            )
            r.raise_for_status()
            return r.json() if r.text else {}
        except requests.HTTPError as e:
            code = e.response.status_code
            if code == 401:
                raise ValueError("GHL auth failed — check API key in .env")
            if code < 500:
                log.error("GHL %s %s -> %s: %s",
                          method, path, code, e.response.text)
                raise
            log.warning("GHL 5xx (attempt %s), retrying", attempt + 1)
        except requests.Timeout:
            log.warning("GHL timeout (attempt %s), retrying", attempt + 1)
    raise RuntimeError(f"GHL request failed after 3 attempts: {path}")


def update_contact(contact_id: str, values: dict):
    """
    values: dict keyed by our friendly names, e.g.
        {"cover_type": "Life Insurance", "readiness_score": 85}
    We translate those into GHL's customField payload using the IDs.
    """
    custom_fields = []
    for name, value in values.items():
        if value is None:
            continue
        field_id = FIELD_IDS.get(name)
        if not field_id:
            log.warning("No field ID for '%s', skipping", name)
            continue
        custom_fields.append({"id": field_id, "field_value": value})

    payload = {"customFields": custom_fields}
    return _request("PUT", f"/contacts/{contact_id}", json=payload)

def create_contact(first_name=None, last_name=None, phone=None,
                   email=None, source="Referral"):
    """Create a NEW contact in GHL and return (id, contact_dict)."""
    from config import GHL_LOCATION_ID

    payload = {
        "locationId": GHL_LOCATION_ID,
        "firstName": first_name,
        "lastName": last_name,
        "phone": phone,
        "email": email,
        "source": source,
    }
    # GHL's create endpoint rejects null values — strip them
    payload = {k: v for k, v in payload.items() if v is not None}

    result = _request("POST", "/contacts/", json=payload)
    contact = result.get("contact", result)
    return contact.get("id"), contact

def send_sms(contact_id: str, message: str):
    return _request("POST", "/conversations/messages", json={
        "type": "SMS",
        "contactId": contact_id,
        "message": message,
    })


def add_note(contact_id: str, body: str):
    return _request("POST", f"/contacts/{contact_id}/notes",
                    json={"body": body})


def get_contact(contact_id: str):
    return _request("GET", f"/contacts/{contact_id}")