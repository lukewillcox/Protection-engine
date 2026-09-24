import hmac
import logging
import os
import threading
import time
from collections import deque
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
import json as _json
import config
from copilot import extract_lead_from_email, generate_prep

import ghl
import booking
from qualify import qualify
from copilot import register_copilot_routes
from messaging import write_sms
from config import WEBHOOK_SECRET

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = Flask(__name__)
# No CORS: the co-pilot page is served by this same app (/copilot), so every
# API call is same-origin. Allowing every origin only widened the attack surface.

import hashlib
from config import COPILOT_PASSWORD

# ---------- co-pilot session tokens ----------
# Tokens are signed and expire (12h), instead of being a permanent hash of the
# password. Set COPILOT_SECRET_KEY in Railway for a signing key independent of
# the password; without it, a key is derived from the password, so changing the
# password still invalidates every issued token.
TOKEN_MAX_AGE = 12 * 3600
_signing_key = os.getenv("COPILOT_SECRET_KEY") or hashlib.sha256(
    ("copilot-token-signing:" + (COPILOT_PASSWORD or "")).encode()).hexdigest()
_signer = URLSafeTimedSerializer(_signing_key, salt="copilot-session")


def _valid_token(token):
    if not COPILOT_PASSWORD or not token:
        return False
    try:
        _signer.loads(token, max_age=TOKEN_MAX_AGE)
        return True
    except (BadSignature, SignatureExpired):
        return False


# Simple login rate limit: 5 failed attempts per minute per client IP, plus a
# global ceiling. In-memory and per gunicorn worker - enough to stop scripted
# guessing without adding a dependency.
_login_failures = {}
_login_failures_all = deque()
_login_lock = threading.Lock()
LOGIN_MAX_PER_IP = 5
LOGIN_MAX_GLOBAL = 120      # high enough that one attacker can't easily lock the adviser out
LOGIN_WINDOW = 60
LOGIN_MAX_TRACKED_IPS = 5000


def _client_ip():
    # Use the right-most X-Forwarded-For entry: the one added by Railway's proxy.
    # Left-most entries are whatever the caller sent, so they can be faked.
    fwd = request.headers.get("X-Forwarded-For", "")
    parts = [p.strip() for p in fwd.split(",") if p.strip()]
    return (parts[-1] if parts else request.remote_addr) or "?"


def _prune(q, now):
    while q and now - q[0] > LOGIN_WINDOW:
        q.popleft()


def _login_blocked(ip):
    now = time.time()
    with _login_lock:
        _prune(_login_failures_all, now)
        q = _login_failures.get(ip)
        if q is not None:
            _prune(q, now)
            if not q:
                _login_failures.pop(ip, None)
                q = None
        return (q is not None and len(q) >= LOGIN_MAX_PER_IP) or len(_login_failures_all) >= LOGIN_MAX_GLOBAL


def _record_login_failure(ip):
    now = time.time()
    with _login_lock:
        if ip not in _login_failures and len(_login_failures) >= LOGIN_MAX_TRACKED_IPS:
            # bounded memory: drop IPs with no recent failures, then the oldest if still full
            for k in [k for k, v in _login_failures.items() if not v or now - v[-1] > LOGIN_WINDOW]:
                _login_failures.pop(k, None)
            while len(_login_failures) >= LOGIN_MAX_TRACKED_IPS:
                _login_failures.pop(next(iter(_login_failures)))
        _login_failures.setdefault(ip, deque()).append(now)
        _login_failures_all.append(now)


@app.post("/copilot/login")
def copilot_login():
    ip = _client_ip()
    if _login_blocked(ip):
        return jsonify({"error": "too many attempts - wait a minute"}), 429
    data = request.get_json(silent=True) or {}
    pw = str(data.get("password") or "")
    if COPILOT_PASSWORD and hmac.compare_digest(pw.encode(), COPILOT_PASSWORD.encode()):
        return jsonify({"token": _signer.dumps({"u": "adviser"}), "expires_in": TOKEN_MAX_AGE})
    _record_login_failure(ip)
    return jsonify({"error": "wrong password"}), 401

@app.before_request
def _guard_copilot():
    path = request.path
    # only guard /copilot paths; allow the login and the page shell through
    if path.startswith("/copilot") and path not in ("/copilot", "/copilot/login"):
        token = request.headers.get("X-Copilot-Token", "")
        if not _valid_token(token):
            return jsonify({"error": "unauthorised"}), 401
        
register_copilot_routes(app)


@app.get("/copilot/session")
def copilot_session():
    # The guard above has already checked the token. The page calls this on load
    # (so an expired or old-style token shows the login straight away) and every
    # so often while in use, getting a fresh 12-hour token back - so an open tab
    # never expires in the middle of a call.
    return jsonify({"ok": True, "token": _signer.dumps({"u": "adviser"}), "expires_in": TOKEN_MAX_AGE})

# Temporary in-memory store of slot offers per contact.
# (Fine for now; a real deployment would use a database.)
_pending_offers = {}


@app.get("/health")
def health():
    return {"ok": True}

@app.get("/copilot")
def serve_copilot():
    return send_from_directory("static", "copilot.html")

@app.post("/webhook/inbound-email")
def inbound_email():
    if not hmac.compare_digest(
        request.headers.get("X-Webhook-Secret", ""), WEBHOOK_SECRET or ""
    ):
        return jsonify({"error": "unauthorised"}), 401

    data = request.get_json(silent=True) or {}
    email_text = (data.get("body") or data.get("text")
                  or data.get("email") or data.get("plain") or "").strip()
    if not email_text:
        return jsonify({"error": "no email body"}), 400

    try:
        lead = extract_lead_from_email(email_text)

        if not lead.get("client_details_present"):
            log.info("Referral email had no client details — skipping")
            return jsonify({"ok": True, "note": "no client details"}), 200

        contact_id, contact = ghl.create_contact(
            first_name=lead.get("first_name") or None,
            last_name=lead.get("last_name") or None,
            phone=lead.get("phone") or None,
            email=lead.get("email") or None,
            source="Mortgage Broker Referral",
        )

        prep = generate_prep(email_text)
        ghl.add_note(contact_id, "PREP_JSON:" + _json.dumps(prep))
        ghl.add_note(
            contact_id,
            f"Referral from {lead.get('referred_by') or 'mortgage broker'}\n"
            f"Occupation: {lead.get('occupation')}\n"
            f"Summary: {lead.get('message')}"
        )

        log.info("Created lead %s from referral email, prep stored", contact_id)
        return jsonify({"ok": True, "contact_id": contact_id})

    except Exception:
        log.exception("inbound-email processing failed")
        return jsonify({"ok": False}), 200

@app.get("/copilot/recent-referrals")
def recent_referrals():
    try:
        result = ghl._request(
            "GET",
            f"/contacts/?locationId={config.GHL_LOCATION_ID}&limit=20&query=",
            timeout=6, attempts=1,
        )
        contacts = result.get("contacts", [])
        out = []
        for c in contacts:
            src = (c.get("source") or "")
            if "referral" in src.lower():
                out.append({
                    "id": c.get("id"),
                    "name": f"{c.get('firstName','')} {c.get('lastName','')}".strip(),
                    "added": c.get("dateAdded") or "",
                })
        return jsonify({"referrals": out})
    except Exception:
        log.exception("recent-referrals failed")
        return jsonify({"referrals": []}), 200


@app.get("/copilot/load/<contact_id>")
def load_prep(contact_id):
    try:
        result = ghl._request("GET", f"/contacts/{contact_id}/notes", timeout=6, attempts=1)
        notes = result.get("notes", [])
        for n in notes:
            body = n.get("body", "")
            if body.startswith("PREP_JSON:"):
                return jsonify(_json.loads(body[len("PREP_JSON:"):]))
        return jsonify({"error": "no prep found"}), 200
    except Exception:
        log.exception("load prep failed")
        return jsonify({"error": "load failed"}), 200


@app.post("/webhook/new-lead")
def new_lead():
    if not hmac.compare_digest(
        request.headers.get("X-Webhook-Secret", ""), WEBHOOK_SECRET or ""
    ):
        return jsonify({"error": "unauthorised"}), 401

    data = request.get_json(silent=True) or {}
    contact_id = data.get("contact_id") or data.get("id")
    if not contact_id:
        return jsonify({"error": "no contact_id"}), 400

    try:
        q = qualify(data)

        # Read real availability and build a slot offer
        slots = booking.get_free_slots()
        chosen = booking.pick_slots(slots, urgency_driver=q.get("urgency_driver"))
        offer_text, labels = booking.format_offer(chosen)

        # Store which slots we offered, so the reply handler can resolve them
        _pending_offers[contact_id] = {
            k: v.isoformat() for k, v in labels.items()
        }

        # Personalised intro + the concrete slot options
        intro = write_sms(data, q)
        sms = f"{intro}\n\nEarliest I can offer: {offer_text}\nReply 1, 2 or 3."

        ghl.update_contact(contact_id, {
            "cover_type":               q.get("cover_type"),
            "cover_amount":             q.get("cover_amount"),
            "term_years":               q.get("term_years"),
            "urgency_driver":           q.get("urgency_driver"),
            "mortgage_completion_date": q.get("mortgage_completion_date"),
            "readiness_score":          q.get("readiness_score"),
            "impaired_life_flag":       ["Yes"] if q.get("impaired_life_flag") else None,
            "ai_qualified":             ["Yes"],
            "ai_sms_content":           sms,
        })

        try:
            ghl.send_sms(contact_id, sms)
        except Exception:
            log.warning("SMS send failed for %s (DND or no number) — continuing", contact_id)

        ghl.add_note(
            contact_id,
            f"AI qualification\nScore: {q.get('readiness_score')}\n"
            f"Reasoning: {q.get('reasoning')}\nOffered: {offer_text}"
        )

        log.info("Processed %s score=%s", contact_id, q.get("readiness_score"))
        return jsonify({"ok": True, "score": q.get("readiness_score")})

    except Exception:
        log.exception("Failed processing %s", contact_id)
        return jsonify({"ok": False}), 200


@app.post("/webhook/reply")
def handle_reply():
    if not hmac.compare_digest(
        request.headers.get("X-Webhook-Secret", ""), WEBHOOK_SECRET or ""
    ):
        return jsonify({"error": "unauthorised"}), 401

    data = request.get_json(silent=True) or {}
    contact_id = data.get("contact_id") or data.get("id")
    reply_text = (data.get("message") or "").strip()

    if not contact_id or not reply_text:
        return jsonify({"error": "missing contact_id or message"}), 400

    offered = _pending_offers.get(contact_id)
    if not offered:
        return jsonify({"ok": True, "note": "no pending offer"}), 200

    labels = {k: datetime.fromisoformat(v) for k, v in offered.items()}

    slot = booking.parse_reply(reply_text, labels)
    if not slot:
        try:
            ghl.send_sms(contact_id,
                         "Sorry, I didn't catch which slot — reply 1, 2 or 3.")
        except Exception:
            log.warning("Clarification SMS failed for %s — continuing", contact_id)
        return jsonify({"ok": True, "note": "unparsed reply"}), 200

    try:
        booking.book(contact_id, slot)
        _pending_offers.pop(contact_id, None)
        nice = slot.strftime("%A %-d %B at %-I:%M%p").replace("AM", "am").replace("PM", "pm")
        try:
            ghl.send_sms(contact_id, f"Booked you in for {nice}. Speak then.")
        except Exception:
            log.warning("Confirmation SMS failed for %s — booking still made", contact_id)
        ghl.add_note(contact_id, f"Appointment booked via SMS reply: {nice}")
        log.info("Booked %s for %s", contact_id, slot)
        return jsonify({"ok": True, "booked": nice})
    except Exception:
        log.exception("Booking failed for %s", contact_id)
        return jsonify({"ok": False}), 200


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)