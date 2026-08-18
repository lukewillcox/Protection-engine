"""
compare_claude_models.py — run the co-pilot's REAL analyze prompt through
Claude Opus 5, Sonnet 5, and Haiku 4.5 side by side, on the same test cases,
so you can judge quality-per-cost before choosing.

Setup:
    pip install anthropic
    ANTHROPIC_API_KEY in .env

Run:
    python compare_claude_models.py
"""

import json, os
from anthropic import Anthropic
from dotenv import load_dotenv
load_dotenv()

from copilot import ANALYZE_SYSTEM, get_relevant_knowledge

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# current model IDs (Aug 2026)
MODELS = [
    ("OPUS 5",   "claude-opus-5"),
    ("SONNET 5", "claude-sonnet-5"),
    ("HAIKU 4.5","claude-haiku-4-5-20251001"),
]

# rough per-call cost estimate (input/output per MTok), for a ~4k in / 500 out call
RATES = {
    "claude-opus-5":            (5.0, 25.0),
    "claude-sonnet-5":          (2.0, 10.0),
    "claude-haiku-4-5-20251001":(1.0, 5.0),
}

TEST_CASES = [
    {"step": "Eligibility & Health",
     "notes": "[Eligibility & Health] self-employed plumber, 45, type 2 diabetes checked "
              "regularly, mother had cancer at 75, two young kids, no cover"},
    {"step": "Income Protection",
     "notes": "[Income Protection] self-employed graphic designer, no sick pay, main earner, "
              "worried about being unable to work, 285k mortgage"},
]


def build_user(notes, step):
    kb = get_relevant_knowledge(notes, k=5)
    c = (f"The adviser is currently on the '{step}' step of the call. "
         "Weight your guidance toward what matters at THIS step, while staying "
         "aware of the whole conversation.\n\n")
    c += "Fact-find notes so far (across all steps):\n\n" + notes
    if kb:
        c += ("\n\n---\nRelevant factual knowledge from provider documents and guidance "
              "(use only if genuinely relevant; never quote a premium or recommend a specific "
              "product):\n\n" + kb)
    return c


def run(model, system, user):
    resp = client.messages.create(
        model=model, max_tokens=1024,
        system=system + "\n\nReturn ONLY valid JSON, no preamble or markdown.",
        messages=[{"role": "user", "content": user}],
    )
    # newer models may return a ThinkingBlock before the text block — grab the text one
    text = ""
    for block in resp.content:
        if getattr(block, "type", None) == "text":
            text = block.text
            break
    if not text:
        # fallback: first block that has .text
        for block in resp.content:
            if hasattr(block, "text"):
                text = block.text
                break
    usage = resp.usage
    ins, outs = RATES[model]
    cost = (usage.input_tokens/1e6)*ins + (usage.output_tokens/1e6)*outs
    return text, usage.input_tokens, usage.output_tokens, cost


def show(label, raw):
    print(f"\n------------------------ {label} ------------------------")
    t = raw.strip()
    if t.startswith("```"):
        t = t.split("```")[1].replace("json", "", 1).strip()
    try:
        print(json.dumps(json.loads(t), indent=2))
    except Exception:
        print(raw)


for i, tc in enumerate(TEST_CASES, 1):
    print("\n\n" + "#"*70)
    print(f"# TEST {i}: {tc['step']}")
    print(f"# {tc['notes']}")
    print("#"*70)
    user = build_user(tc["notes"], tc["step"])
    for label, model in MODELS:
        try:
            raw, ti, to, cost = run(model, ANALYZE_SYSTEM, user)
            show(f"{label}  (in {ti} / out {to} tok  ≈ ${cost:.4f}/call)", raw)
        except Exception as e:
            print(f"\n{label} error: {e}")

print("\n\nCompare: is Opus's reasoning/facts noticeably sharper than Sonnet's "
      "for YOUR task? Is Haiku good enough? Weigh quality against the per-call cost shown.")