import json
import logging
from flask import request, jsonify
from openai import OpenAI
from config import OPENAI_API_KEY
from rag import retrieve
from llm import claude_json, claude_text, MODEL_ANALYZE, MODEL_PREPARE, MODEL_SUMMARY

log = logging.getLogger(__name__)
client = OpenAI(api_key=OPENAI_API_KEY)

FACTS = ["Dependants", "Employment", "Existing cover",
         "Health / lifestyle", "Budget & affordability", "Protection need"]

KNOWLEDGE = """
ADVICE PHILOSOPHY (how this adviser works):
- Protection is advice, not a sale. The client came for advice, not just a mortgage.
- The goal isn't the mortgage, it's the home — and protection secures the ability to keep it.
- Income is the foundation. Assess need based on household income, not mortgage payments.
- Anchor everything to the client's real goals and why they're buying this home.
- Protection underpins the mortgage; it is essential, not a bolt-on.
- Frame as education: the adviser is an expert working for the client.
- No fees to the client. An FCA-regulated broker means recourse if a claim is wrongly declined.

NEED-BASED QUESTIONING (tailor to the client's situation):
- Make questioning specific to their circumstances, occupation, and dependants.
- Occupation-aware: probe how their particular work would be affected by illness/injury. E.g. a surgeon whose hands are their livelihood — explore what happens to income if they couldn't operate, and whether an own-occupation definition matters. A manual worker vs desk worker differ greatly.
- Dependants drive term and type: how long would the family need support (e.g. until youngest is independent), shaping term and whether Family Income Benefit suits.
- Income stress test: if income stopped, how many months of savings before they run out; could they clear the debt; who depends on that income.

VALUE FRAMING (help the adviser frame cost against income — NEVER quote a premium):
- Anchor to what is being protected: their income and family security.
- Frame cost as a small proportion of what it safeguards.
- Prompt value questions: "what would it be worth to protect this income?" rather than leading with price.
- IMPORTANT: never state, estimate, or suggest an actual premium or a specific recommended cover amount. The adviser sources real prices and makes the recommendation. You help frame VALUE and NEED only.

FACTUAL EVIDENCE (use to inform, never to pressure):
- Average income protection claim lasts around 7 years 8 months — a reason full-term cover matters.
- Over half of IP claims (~56%) are for mental health or musculoskeletal conditions, which critical illness cover typically would NOT cover.
- Less than 1% of decreasing critical illness cover ends up paying off the mortgage.
- Standard underwriting terms protect against future uninsurability as health changes.
- Child critical illness cover can fund time off if something happens to a child.

Use this knowledge to make prompts specific, credible and need-focused. NEVER use it to pressure,
obscure, or push a larger sale. Never output a premium figure or a specific cover amount as an answer.

PROTECTION MATHS (grounded UK 2026 figures — use to reason, never to quote):

INCOME REPLACEMENT RATIO (IRR):
  Insurers cap IP benefit at 65-70% of gross (some to 80% of net).
  Target benefit = net monthly income x 0.70 to 0.80.
  Example: £4,000 net/mo -> aim for £2,800-3,200/mo IP benefit.
  State SSP: £507/mo for first 28 weeks only.

DEFERRED PERIOD SELECTION:
  Match the deferred period to when employer sick pay actually ends:
  Self-employed (no sick pay) -> 4-week deferred.
  1-3 months sick pay -> 4 or 8-week deferred.
  3-6 months sick pay -> 13-week deferred.
  6+ months or NHS -> 26-week deferred.
  Shorter deferred = higher premium but less reliance on savings.

IHT EXPOSURE (2026/27 frozen to April 2030):
  Nil-Rate Band: £325,000 per person.
  Residence Nil-Rate Band: £175,000 if home passes to direct descendants.
  Couple combined allowances: up to £1,000,000.
  IHT rate: 40% on excess. Quick screen: (equity + savings - NRB - RNRB) x 40%.
  Whole-of-life written in trust is the standard planning tool.

CIC BENEFIT NEED ESTIMATE:
  Gross income x 2-5 years (3 years is typical) + debts to clear - savings.
  CI pays a tax-free lump sum on diagnosis, not linked to inability to work.
  Distinct from IP: CI clears debts and funds recovery; IP replaces monthly income.

MORTGAGE PAYMENT SHOCK:
  FCA requires lenders to stress-test at SVR (+3% minimum).
  Typical SVR 2026: 6-8.5%. A 2yr fix reverting to SVR adds ~25-35% to monthly payment.
  Useful frame: can the client meet the mortgage at reversion rate if income also dips?
  This justifies IP cover even when the current payment feels manageable.
"""
def get_relevant_knowledge(situation_text, k=5):
    """Retrieve the most relevant knowledge chunks for a situation,
    formatted for injection into a prompt. Fails safe: if retrieval
    errors, returns empty string so the co-pilot still works."""
    try:
        hits = retrieve(situation_text, k=k)
        if not hits:
            return ""
        lines = []
        for cid, doc in hits:
            # tag the source so the model knows where it came from
            source = cid.split("::")[0] if "::" in cid else cid
            lines.append(f"[{source}] {doc}")
        return "\n\n".join(lines)
    except Exception:
        log.exception("knowledge retrieval failed")
        return ""

ANALYZE_SYSTEM = f"""You assist a UK FCA-regulated protection adviser during a client fact-find.
You help them gather and assess the client's NEEDS thoroughly and document them.
You are a fact-find and compliance assistant, NOT a sales tool.

You draw on this adviser's methodology and product knowledge:
{KNOWLEDGE}

You NEVER:
- suggest pushing a larger or more expensive product
- recommend a specific product, provider, or premium
- state or estimate an actual premium figure or a specific recommended cover amount
- optimise for the size of the sale
- treat the client's wealth or an inheritance as a reason to upsell
- use pressure, urgency tactics, or deliberately obscure information

You ALWAYS:
- orient prompts toward assessing genuine protection need
- identify what information is still needed for a complete fact-find
- surface relevant factual evidence or value framing where it helps the client understand their need
- flag compliance-relevant points the adviser should document
- note underwriting-relevant disclosures (health, occupation) factually
- keep the client's best interest central

Return ONLY valid JSON matching:
{{
  "captured": {{"Dependants":"","Employment":"","Existing cover":"","Health / lifestyle":"","Budget & affordability":"","Protection need":""}},
  "prompts": [
    {{"headline":"5-8 WORD HEADLINE","detail":"one actionable sentence","severity":"green|amber|red"}}
  ],
  "flags": ["must-document compliance point, under 20 words"],
  "relevant_facts": ["a short factual point with source in brackets"]
}}
For captured, fill only fields the notes evidence; leave others "".
For prompts: 2-4 items. Each MUST be a JSON object with:
  headline: 5-8 words MAX, caps, the broker scans this first
  detail: one actionable sentence explaining why this matters
  severity: green (informational), amber (gap/action needed), red (must document/compliance risk)
  NOT EVERYTHING IS RED. Use green for factual prompts, amber for gaps, red only for compliance.
For flags: must-document items only, under 20 words each. Always shown as red.
Max 4 prompts, max 3 flags.

For "relevant_facts": include 0-3 SHORT factual points drawn ONLY from the
provider knowledge supplied in the user message (if any is provided). Each
should be a concrete, sourced fact that helps the adviser understand the
product landscape - e.g. "Some CIC plans cover 50+ conditions; definitions
vary by insurer [Zurich, L&G]". Put the provider/source in brackets. NEVER
state a premium or recommend a specific product. If no provider knowledge is
relevant or supplied, return an empty array"""

PROMPTS_SYSTEM = """You help a UK protection adviser during a live client call by
generating 3 ready-to-say conversation prompts for the step they are about to move to.

You are given: the step they're moving to, and everything captured so far in the
fact-find (fields and notes across all previous steps).

Generate 3 prompts the adviser can say out loud to the client at this step. They must:
- Be specific to THIS client, using details already captured (name, job, family,
  mortgage, health, goals, figures). Reference real facts where you can.
- Sound natural and warm — like a skilled adviser speaking, not a script.
- Move the conversation forward at this step.
- Never state a premium, a specific product/provider recommendation, or a cover amount.
- Be concise — one or two sentences each, ready to say aloud.

If little has been captured yet, give sensible general prompts for the step.

Return ONLY valid JSON: {"prompts": ["...", "...", "..."]}
Each string is a complete phrase the adviser can say. UK English."""


SUMMARY_SYSTEM = """You write a factual fact-find summary for a UK protection adviser's suitability record.
Summarise ONLY what the notes evidence. Do not recommend products, quote premiums, or give advice.
Structure: Client & referral / Circumstances / Protection needs identified / Information still to gather / Points to document.
Plain, professional, concise."""

EMAIL_SYSTEM = """You draft a warm, professional follow-up EMAIL from a UK protection
adviser to their client after a fact-find call. The adviser will review and send it.

Purpose: thank the client, recap what was discussed, and confirm the next steps.
Use the fact-find brief provided.

TONE: warm, clear, human, professional. Short paragraphs. UK English. Address the
client by first name where known. Sign off from the adviser.

STRICT COMPLIANCE BOUNDARIES:
- This is a RECAP, not advice in writing. Do NOT state a recommendation, a premium, a
  specific product or provider, or a recommended cover amount. Those belong in the
  adviser's formal suitability process, not this recap email.
- Do NOT make guarantees or claims about cover or outcomes.
- Recap the protection AREAS discussed (e.g. protecting income, protecting the family/
  mortgage) in plain English, and the agreed next steps.
- Keep it concise (roughly 120-200 words).

Return ONLY valid JSON: {"subject": "...", "body": "..."}
The body should use \n for line breaks, ready to paste into an email."""


PREPARE_SYSTEM = f"""You prepare a UK protection adviser for a client call by reading a
referral email from a mortgage broker.

You draw on this adviser's methodology and product knowledge:
{KNOWLEDGE}

From the email, extract every relevant detail: name, age or date of birth, income (and whether
salaried, self-employed, or bonus/variable), occupation/role, mortgage debt, deposit, monthly
mortgage payment, repayment vs interest-only, lender, and dependants.

Then reason from EACH detail to the protection need and the questions it raises:
- Occupation → how THIS job's income could be lost or interrupted; own-occupation IP relevance.
- Age → appropriate cover term, future insurability, acting sooner while healthy.
- Income → how much to replace and for how long; bonus/variable income is harder to protect.
- Mortgage debt → who clears it on death or serious illness; level vs decreasing cover.
- Interest-only vs repayment → interest-only means the full debt persists.
- Monthly payment → could they keep meeting it if income stopped (income protection need).
- Deposit → capital at risk in the home; indicator of savings buffer for the income stress test.
- Dependants → how long support is needed (e.g. until youngest is independent); FIB relevance.

NEED ILLUSTRATIONS: using their REAL figures, produce short plain-English illustrations that make
the EXPOSURE vivid — help the client SEE why protection matters. Examples of the style:
- "£600k mortgage debt would fall to the family to clear if the worst happened."
- "£2,800/month still has to be found even if illness stopped your income."
- "Kids aged 4 and 6 mean roughly 15+ years of dependency to plan around."
These illustrate the SIZE and SHAPE of the need using their numbers.

CRITICAL: You NEVER compute or state a recommended cover amount, never quote or estimate a premium
or cost, never recommend a product. You frame the NEED and the EXPOSURE only. The adviser sources
real prices and makes the recommendation. Do arithmetic only to restate their own figures as
exposure (e.g. "the debt is £600k"), never to produce "you need £X of cover".

Return ONLY valid JSON:
{{
  "client_name": "extracted name or empty",
  "occupation": "extracted occupation or empty",
  "key_facts": {{"age":"","income":"","mortgage_debt":"","monthly_payment":"","mortgage_type":"","deposit":"","dependants":""}},
  "situation": "one-line summary of their circumstances",
  "need_illustrations": ["plain-English exposure framing using their real figures"],
  "opening_questions": ["question tailored to a specific fact about them"],
  "watch_for": ["need or risk area worth exploring given their specific situation"]
}}
For key_facts, fill only what the email evidences; leave others "".
Max 4 need_illustrations, max 6 opening_questions, max 4 watch_for, each concise."""

LEAD_EXTRACT_SYSTEM = """You extract a structured lead from a referral email sent to a
UK protection adviser by a mortgage broker.

Pull out the CLIENT's details (not the broker's). Return ONLY valid JSON:
{
  "first_name": "",
  "last_name": "",
  "phone": "",
  "email": "",
  "occupation": "",
  "referred_by": "the mortgage broker's name if present",
  "message": "a one-paragraph plain summary of the client's situation from the email",
  "client_details_present": true
}
- phone/email: the CLIENT's, if present. Empty string if not found.
- client_details_present: false if the email doesn't contain the client's contact details.
- Never invent details. Empty string for anything not in the email."""


def extract_lead_from_email(email_text: str) -> dict:
    return claude_json(LEAD_EXTRACT_SYSTEM, "Referral email:\n\n" + email_text, MODEL_PREPARE)


def generate_prep(email_text: str) -> dict:
    return claude_json(PREPARE_SYSTEM, "Referral email:\n\n" + email_text, MODEL_PREPARE)

def register_copilot_routes(app):

    @app.post("/copilot/analyze")
    def copilot_analyze():
        data = request.get_json(silent=True) or {}
        notes = (data.get("notes") or "").strip()
        if not notes:
            return jsonify({"error": "no notes"}), 400
        step = (data.get("step") or "").strip()
        try:
            kb = get_relevant_knowledge(notes, k=5)
            user_content = ""
            if step:
                user_content += (
                    "The adviser is currently on the '" + step + "' step of the call. "
                    "Weight your guidance toward what matters at THIS step, while staying "
                    "aware of the whole conversation.\n\n"
                )
            user_content += "Fact-find notes so far (across all steps):\n\n" + notes
            if kb:
                user_content += (
                    "\n\n---\nRelevant factual knowledge from provider "
                    "documents and guidance (use only if genuinely relevant; "
                    "never quote a premium or recommend a specific product):\n\n" + kb
                )
            return jsonify(claude_json(ANALYZE_SYSTEM, user_content, MODEL_ANALYZE))
        except Exception:
            log.exception("copilot analyze failed")
            return jsonify({"error": "analyze failed"}), 200

    @app.post("/copilot/prompts")
    def copilot_prompts():
        data = request.get_json(silent=True) or {}
        step = (data.get("step") or "").strip()
        notes = (data.get("notes") or "").strip()
        if not step:
            return jsonify({"prompts": []}), 200
        try:
            user_content = "Step the adviser is moving to: " + step + "\n\n"
            if notes:
                user_content += "Everything captured so far:\n\n" + notes
            else:
                user_content += "Nothing captured yet — give sensible general prompts for this step."
            result = claude_json(PROMPTS_SYSTEM, user_content, MODEL_ANALYZE)
            return jsonify(result)
        except Exception:
            log.exception("copilot prompts failed")
            return jsonify({"prompts": []}), 200


    @app.post("/copilot/summary")
    def copilot_summary():
        data = request.get_json(silent=True) or {}
        notes = (data.get("notes") or "").strip()
        name = data.get("client_name") or "the client"
        ref = data.get("referred_by") or "—"
        ff = data.get("fact_find") or {}
        if not notes and not ff:
            return jsonify({"error": "no notes"}), 400
        try:
            parts = [f"Client: {name}", f"Referred by: {ref}"]
            if ff.get("clients"):
                parts.append("Clients:\n" + "\n".join("- " + c for c in ff["clients"]))
            if ff.get("income"):
                parts.append("Income: " + ff["income"])
            if ff.get("mortgage"):
                parts.append("Mortgage: " + ff["mortgage"])
            if ff.get("existing"):
                ex = [e for e in ff["existing"] if e and e.strip()]
                parts.append("Existing cover: " + ("; ".join(ex) if ex else "none recorded"))
            if ff.get("quotes"):
                qs = [q for q in ff["quotes"] if q and q.strip()]
                parts.append("Quotes captured by adviser (from their sourcing tool):\n"
                             + ("\n".join("- " + q for q in qs) if qs else "none"))
            if ff.get("total"):
                parts.append("Total monthly premium captured: " + ff["total"])
            if ff.get("notesByStep"):
                parts.append("Call notes by step:\n" + ff["notesByStep"])
            elif notes:
                parts.append("Notes:\n" + notes)
            user_content = "\n\n".join(parts)
            return jsonify({"summary": claude_text(SUMMARY_SYSTEM, user_content, MODEL_SUMMARY).strip()})
        except Exception:
            log.exception("copilot summary failed")
            return jsonify({"error": "summary failed"}), 200

    @app.post("/copilot/summary-pdf")
    def copilot_summary_pdf():
        from flask import send_file
        import io
        import config as _cfg
        from pdf_summary import build_summary_pdf
        data = request.get_json(silent=True) or {}
        notes = (data.get("notes") or "").strip()
        name = data.get("client_name") or "the client"
        ref = data.get("referred_by") or "\u2014"
        ff = data.get("fact_find") or {}
        if not notes and not ff:
            return jsonify({"error": "no notes"}), 400
        try:
            parts = [f"Client: {name}", f"Referred by: {ref}"]
            if ff.get("clients"):
                parts.append("Clients:\n" + "\n".join("- " + c for c in ff["clients"]))
            if ff.get("income"):
                parts.append("Income: " + ff["income"])
            if ff.get("mortgage"):
                parts.append("Mortgage: " + ff["mortgage"])
            if ff.get("existing"):
                ex = [e for e in ff["existing"] if e and e.strip()]
                parts.append("Existing cover: " + ("; ".join(ex) if ex else "none recorded"))
            if ff.get("quotes"):
                qs = [q for q in ff["quotes"] if q and q.strip()]
                parts.append("Quotes captured by adviser:\n" + ("\n".join("- " + q for q in qs) if qs else "none"))
            if ff.get("total"):
                parts.append("Total monthly premium captured: " + ff["total"])
            if ff.get("notesByStep"):
                parts.append("Call notes by step:\n" + ff["notesByStep"])
            elif notes:
                parts.append("Notes:\n" + notes)
            user_content = "\n\n".join(parts)

            summary_text = claude_text(SUMMARY_SYSTEM, user_content, MODEL_ANALYZE).strip()

            meta = {
                "client": name,
                "adviser": getattr(_cfg, "BROKER_NAME", "Adviser"),
                "firm": getattr(_cfg, "BROKER_COMPANY", ""),
            }
            pdf_bytes = build_summary_pdf(summary_text, meta)
            safe = name.replace(" ", "-")
            return send_file(io.BytesIO(pdf_bytes), mimetype="application/pdf",
                             as_attachment=True,
                             download_name="suitability-summary-" + safe + ".pdf")
        except Exception:
            log.exception("copilot summary-pdf failed")
            return jsonify({"error": "pdf failed"}), 200

    @app.post("/copilot/summary-email")
    def copilot_summary_email():
        data = request.get_json(silent=True) or {}
        notes = (data.get("notes") or "").strip()
        name = data.get("client_name") or "the client"
        ff = data.get("fact_find") or {}
        if not notes and not ff:
            return jsonify({"error": "no notes"}), 400
        try:
            parts = [f"Client: {name}"]
            if ff.get("clients"):
                parts.append("Clients:\n" + "\n".join("- " + c for c in ff["clients"]))
            if ff.get("mortgage"):
                parts.append("Mortgage: " + ff["mortgage"])
            if ff.get("notesByStep"):
                parts.append("Call notes by step:\n" + ff["notesByStep"])
            elif notes:
                parts.append("Notes:\n" + notes)
            user_content = "\n\n".join(parts)
            result = claude_json(EMAIL_SYSTEM, user_content, MODEL_PREPARE)
            return jsonify(result)
        except Exception:
            log.exception("copilot summary-email failed")
            return jsonify({"error": "email failed"}), 200



    @app.post("/copilot/prepare")
    def copilot_prepare():
        data = request.get_json(silent=True) or {}
        email = (data.get("email") or "").strip()
        if not email:
            return jsonify({"error": "no email"}), 400
        try:
            kb = get_relevant_knowledge(email, k=5)
            user_content = "Referral email:\n\n" + email
            if kb:
                user_content += (
                    "\n\n---\nRelevant factual knowledge from provider "
                    "documents (use only if genuinely relevant; never quote a "
                    "premium or recommend a specific product):\n\n" + kb
                )
            return jsonify(claude_json(PREPARE_SYSTEM, user_content, MODEL_PREPARE))
        except Exception:
            log.exception("copilot prepare failed")
            return jsonify({"error": "prepare failed"}), 200