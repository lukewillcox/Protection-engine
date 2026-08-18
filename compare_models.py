"""
compare_models.py — run the co-pilot's REAL prompts through GPT-4o and Claude
side by side, so you can judge quality before committing to a switch.

Setup:
    pip install anthropic
    add ANTHROPIC_API_KEY to .env (never paste keys in chat)

Run:
    python compare_models.py
"""

import json
from openai import OpenAI
from anthropic import Anthropic
from config import OPENAI_API_KEY
import os
from dotenv import load_dotenv
load_dotenv()

# import your REAL prompts + RAG so the comparison is fair
from copilot import ANALYZE_SYSTEM, get_relevant_knowledge

openai_client = OpenAI(api_key=OPENAI_API_KEY)
anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

OPENAI_MODEL = "gpt-4o"
CLAUDE_MODEL = "claude-sonnet-4-6"   # strong, fast; adjust if you prefer

# --- test cases: realistic call-note snippets ---
TEST_CASES = [
    {
        "step": "Eligibility & Health",
        "notes": "[Eligibility & Health] client is a self-employed plumber, 45, "
                 "type 2 diabetes checked regularly, mother had cancer at 75, two young kids, no cover",
    },
    {
        "step": "Income Protection",
        "notes": "[Income Protection] self-employed graphic designer, no sick pay, "
                 "main earner, worried about being unable to work, 285k mortgage",
    },
    {
        "step": "Essential vs Needed",
        "notes": "[Essential vs Needed] couple, one wants to stop work to raise kids, "
                 "650k mortgage, both employed, client says premiums feel like a lot",
    },
]


def build_user_content(notes, step):
    """Mirror exactly what copilot_analyze builds, so the test is fair."""
    kb = get_relevant_knowledge(notes, k=5)
    content = ""
    if step:
        content += (
            "The adviser is currently on the '" + step + "' step of the call. "
            "Weight your guidance toward what matters at THIS step, while staying "
            "aware of the whole conversation.\n\n"
        )
    content += "Fact-find notes so far (across all steps):\n\n" + notes
    if kb:
        content += (
            "\n\n---\nRelevant factual knowledge from provider documents and guidance "
            "(use only if genuinely relevant; never quote a premium or recommend a specific "
            "product):\n\n" + kb
        )
    return content


def run_openai(system, user):
    resp = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        response_format={"type": "json_object"},
        temperature=0.3,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
    )
    return resp.choices[0].message.content


def run_claude(system, user):
    # Claude: system is a top-level param; ask for JSON explicitly
    resp = anthropic_client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        temperature=0.3,
        system=system + "\n\nReturn ONLY valid JSON, no preamble or markdown.",
        messages=[{"role": "user", "content": user}],
    )
    return resp.content[0].text


def pretty(label, raw):
    print("\n" + "=" * 32 + f" {label} " + "=" * 32)
    try:
        print(json.dumps(json.loads(raw), indent=2))
    except Exception:
        print(raw)


for i, tc in enumerate(TEST_CASES, 1):
    print("\n\n" + "#" * 78)
    print(f"# TEST {i}: step = {tc['step']}")
    print(f"# notes = {tc['notes']}")
    print("#" * 78)
    user = build_user_content(tc["notes"], tc["step"])
    try:
        pretty("GPT-4o", run_openai(ANALYZE_SYSTEM, user))
    except Exception as e:
        print("GPT-4o error:", e)
    try:
        pretty("CLAUDE", run_claude(ANALYZE_SYSTEM, user))
    except Exception as e:
        print("Claude error:", e)

print("\n\nDone. Compare the two outputs for each test — sharper coverage reasoning? "
      "better use of the provider facts? better compliance instinct? Judge on YOUR prompts.")