import hmac
import logging
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
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
CORS(app)

import hashlib
from config import COPILOT_PASSWORD

def _valid_token(token):
    # token is a hash of the password; simple shared-secret scheme
    expected = hashlib.sha256((COPILOT_PASSWORD or "").encode()).hexdigest()
    return token == expected and bool(COPILOT_PASSWORD)

@app.post("/copilot/login")
def copilot_login():
    data = request.get_json(silent=True) or {}
    if data.get("password") and data["password"] == COPILOT_PASSWORD:
        token = hashlib.sha256(COPILOT_PASSWORD.encode()).hexdigest()
        return jsonify({"token": token})
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
        )
        contacts = result.get("contacts", [])
        out = []
        for c in contacts:
            src = (c.get("source") or "")
            if "referral" in src.lower():
                out.append({
                    "id": c.get("id"),
                    "name": f"{c.get('firstName','')} {c.get('lastName','')}".strip(),
                })
        return jsonify({"referrals": out})
    except Exception:
        log.exception("recent-referrals failed")
        return jsonify({"referrals": []}), 200


@app.get("/copilot/load/<contact_id>")
def load_prep(contact_id):
    try:
        result = ghl._request("GET", f"/contacts/{contact_id}/notes")
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