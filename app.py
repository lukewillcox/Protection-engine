import hmac
import logging
from datetime import datetime
from flask import Flask, request, jsonify

import ghl
import booking
from qualify import qualify
from messaging import write_sms
from config import WEBHOOK_SECRET

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = Flask(__name__)

# Temporary in-memory store of slot offers per contact.
# (Fine for now; a real deployment would use a database.)
_pending_offers = {}


@app.get("/health")
def health():
    return {"ok": True}


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