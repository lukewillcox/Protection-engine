import json
from datetime import date
from openai import OpenAI
from config import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)

SCHEMA = {
    "cover_type": "Life Insurance | Critical Illness | Income Protection | Mortgage Protection | Family Income Benefit | Not Sure",
    "cover_amount": "integer GBP or null",
    "term_years": "integer or null",
    "urgency_driver": "Mortgage Completion | New Baby | New Job | Recently Married | General Review | Just Researching",
    "mortgage_completion_date": "YYYY-MM-DD or null",
    "impaired_life_flag": "boolean",
    "readiness_score": "integer 0-100",
    "reasoning": "one short sentence",
}

SYSTEM = """You are an intake assistant for a UK protection broker.
You extract structured data from enquiries. You never give financial
advice, never recommend a product, and never estimate a premium.
You return only valid JSON."""


def qualify(enquiry: dict) -> dict:
    prompt = f"""
Extract structured data from this protection enquiry.

Enquiry:
{json.dumps(enquiry, indent=2)}

Today's date: {date.today().isoformat()}

Return ONLY valid JSON matching this schema:
{json.dumps(SCHEMA, indent=2)}

Rules:
- Unknown fields must be null, never guessed.
- impaired_life_flag: true only if the client explicitly
  disclosed a health condition, current medication, or a
  recent medical investigation. Do not infer from age.
  Record only the flag, never the condition itself.
- readiness_score:
    90-100  mortgage completion within 6 weeks
    70-89   life event driven (baby, marriage, new job)
    40-69   general review, no deadline
    0-39    early research, no stated need
"""
    resp = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        temperature=0.2,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
    )
    return json.loads(resp.choices[0].message.content)