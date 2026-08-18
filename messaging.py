from openai import OpenAI
from config import OPENAI_API_KEY, BROKER_NAME, BROKER_COMPANY
from llm import claude_text, MODEL_SMS

client = OpenAI(api_key=OPENAI_API_KEY)

SYSTEM = f"""You write SMS messages for {BROKER_COMPANY}, a UK
protection broker. You are an admin assistant, not an adviser.
You never quote premiums, never recommend products, never give
financial advice. You sound like a real person, never like a bot.
You never use emojis."""


def write_sms(contact: dict, q: dict) -> str:
    driver = q.get("urgency_driver")
    completion = q.get("mortgage_completion_date")
    score = q.get("readiness_score") or 0   # None-safe

    if driver == "Mortgage Completion" and completion:
        tone = (f"They have a mortgage completion on {completion}. "
                f"Reference it naturally. Convey quiet urgency "
                f"without pressure.")
    elif score >= 70:
        tone = "Life event driven. Warm, prompt, human."
    else:
        tone = "Early research stage. Helpful, zero pressure."

    prompt = f"""
Write one SMS to this enquirer.

Name: {contact.get('first_name', 'there')}
Enquiry type: {q.get('cover_type')}
Context: {tone}

Rules:
- Under 300 characters
- Use their first name once, at the start
- Reference their specific situation naturally
- No premium figures, no product recommendations
- End with: reply YES and I'll send you some times
- Sign off exactly: {BROKER_NAME} @ {BROKER_COMPANY}

Return the SMS text only, no quotation marks.
"""
    return claude_text(SYSTEM, prompt, MODEL_SMS).strip()