import json
import logging
from flask import request, jsonify
from openai import OpenAI
from config import OPENAI_API_KEY

log = logging.getLogger(__name__)
client = OpenAI(api_key=OPENAI_API_KEY)

FACTS = ["Dependants", "Employment", "Existing cover",
         "Health / lifestyle", "Budget & affordability", "Protection need"]

# --- Distilled adviser methodology + product knowledge ---
# Drawn from the broker's own handbook: the advice philosophy,
# structural approach, and factual claim statistics. Deliberately
# excludes sales-pressure tactics — the adviser brings those; the
# assistant stays factual and need-focused.
KNOWLEDGE = """
ADVICE PHILOSOPHY (how this adviser works):
- Protection is advice, not a sale. The client came for advice, not just a mortgage.
- The goal isn't the mortgage, it's the home — and protection is what secures the ability to keep it.
- Income is the foundation. Assess need based on household income, not mortgage payments (they aren't linked).
- Anchor everything to the client's real goals and why they're buying this home.
- Protection underpins the mortgage; it is essential, not a bolt-on.
- Frame as education: the adviser is an expert working for the client, helping them make an informed decision.
- The adviser charges no fees to the client. Using an FCA-regulated broker means recourse if a claim is wrongly declined.

STRUCTURAL APPROACH:
- Establish the need early, woven through the conversation — never sprung at the end.
- Income stress test: if income stopped, how many months until the client runs out; could they clear the debt.
- Two natural touchpoints: income protection at affordability stage; mortgage-related cover at mortgage offer.
- Employer benefits are conditional — tied to staying with that employer, can be cut, often underwritten only at point of claim. Treat as a bonus, not a foundation.

FACTUAL EVIDENCE (use to inform, never to pressure):
- Average income protection claim lasts around 7 years 8 months — a reason full-term cover matters.
- Over half of IP claims (~56%) are for mental health or musculoskeletal conditions, which critical illness cover typically would NOT cover.
- Less than 1% of decreasing critical illness cover ends up paying off the mortgage.
- Standard/ordinary underwriting terms protect against future uninsurability as health changes.
- Child critical illness cover can fund time off work if something happens to a child.

Use this knowledge to make prompts specific, credible and need-focused. NEVER use it to pressure,
obscure, or push a larger sale. Surface a relevant fact only where it genuinely helps the client
understand their real need.
"""

ANALYZE_SYSTEM = f"""You assist a UK FCA-regulated protection adviser during a client fact-find.
You help them gather and assess the client's NEEDS thoroughly and document them.
You are a fact-find and compliance assistant, NOT a sales tool.

You draw on this adviser's methodology and product knowledge:
{KNOWLEDGE}

You NEVER:
- suggest pushing a larger or more expensive product
- recommend a specific product, provider, or premium
- optimise for the size of the sale
- treat the client's wealth or an inheritance as a reason to upsell
- use pressure, urgency tactics, or deliberately obscure information

You ALWAYS:
- orient prompts toward assessing genuine protection need
- identify what information is still needed for a complete fact-find
- surface relevant factual evidence (e.g. claim statistics) where it helps the client understand their need
- flag compliance-relevant points the adviser should document
- note underwriting-relevant disclosures (health, occupation) factually
- keep the client's best interest central

Return ONLY valid JSON matching:
{{
  "captured": {{"Dependants":"","Employment":"","Existing cover":"","Health / lifestyle":"","Budget & affordability":"","Protection need":""}},
  "prompts": ["advice-led suggestion"],
  "flags": ["point to document"]
}}
For captured, fill only fields the notes evidence; leave others "".
Max 3 prompts, max 3 flags, each under 30 words."""

SUMMARY_SYSTEM = """You write a factual fact-find summary for a UK protection adviser's suitability record.
Summarise ONLY what the notes evidence. Do not recommend products, quote premiums, or give advice.
Structure: Client & referral / Circumstances / Protection needs identified / Information still to gather / Points to document.
Plain, professional, concise."""


def register_copilot_routes(app):

    @app.post("/copilot/analyze")
    def copilot_analyze():
        data = request.get_json(silent=True) or {}
        notes = (data.get("notes") or "").strip()
        if not notes:
            return jsonify({"error": "no notes"}), 400
        try:
            resp = client.chat.completions.create(
                model="gpt-4o",
                response_format={"type": "json_object"},
                temperature=0.3,
                messages=[
                    {"role": "system", "content": ANALYZE_SYSTEM},
                    {"role": "user", "content": "Fact-find notes so far:\n\n" + notes},
                ],
            )
            out = json.loads(resp.choices[0].message.content)
            return jsonify(out)
        except Exception:
            log.exception("copilot analyze failed")
            return jsonify({"error": "analyze failed"}), 200

    @app.post("/copilot/summary")
    def copilot_summary():
        data = request.get_json(silent=True) or {}
        notes = (data.get("notes") or "").strip()
        name = data.get("client_name") or "the client"
        ref = data.get("referred_by") or "—"
        if not notes:
            return jsonify({"error": "no notes"}), 400
        try:
            resp = client.chat.completions.create(
                model="gpt-4o",
                temperature=0.3,
                messages=[
                    {"role": "system", "content": SUMMARY_SYSTEM},
                    {"role": "user", "content": f"Client: {name}\nReferred by: {ref}\n\nNotes:\n{notes}"},
                ],
            )
            return jsonify({"summary": resp.choices[0].message.content.strip()})
        except Exception:
            log.exception("copilot summary failed")
            return jsonify({"error": "summary failed"}), 200