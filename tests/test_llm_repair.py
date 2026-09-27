"""Unit tests for llm.py's truncated-JSON recovery, using the exact response text
that broke /copilot/prepare live (see the "James Whitcombe" case): Claude hit
max_tokens mid-way through key_facts and the plain json.loads() failed, wiping
out everything the adviser had just read out. claude_json() should now recover
whatever finished before the cut instead of losing the lot."""
import os, sys, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("ANTHROPIC_API_KEY", "x")

import llm
from llm import _repair_truncated_json, claude_json

fails = []
def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond: fails.append(msg)

# ---- _repair_truncated_json directly ----
check(_repair_truncated_json('{"a":"1","b":{"c":"2","d":"unterm') == '{"a":"1","b":{"c":"2"}}',
      "drops the unfinished field, closes the objects that were open")
check(_repair_truncated_json('{"a":"1"') is None,
      "nothing ever safely completed (no comma reached) -> gives up, doesn't fabricate")
check(_repair_truncated_json('{"a":["1","2","unfin') == '{"a":["1","2"]}',
      "handles a truncation inside a nested array too, keeping every element that finished")
comma_in_string = '{"a":"has, a comma","b":"unterm'
check(_repair_truncated_json(comma_in_string) == '{"a":"has, a comma"}',
      "a comma inside a string value isn't mistaken for a structural one")

# ---- the real production case, verbatim from the Railway log ----
REAL_TRUNCATED = ('{\n  "client_name": "James Whitcombe",\n  "occupation": "Former logistics '
    'business owner, now living off investment dividends",\n  "key_facts": {\n    "age": "52",\n'
    '    "income": "£120,000/year (dividends from business sale proceeds)",\n    '
    '"mortgage_debt": "£1.2 million",\n    "monthly_payment": "£6,700",\n    '
    '"mortgage_term": "25 years, 5-year fixed",\n    "purchase_price": "£2.85 million",\n    '
    '"mortgage_type": "",\n    "deposit": "",\n    "dependants": "two children, ages 21 and 16",\n'
    '    "youngest_child')
repaired = _repair_truncated_json(REAL_TRUNCATED)
import json as _json
parsed = _json.loads(repaired) if repaired else None
check(parsed is not None, f"the real truncated log line repairs to valid JSON (got {repaired!r})")
if parsed:
    check(parsed.get("client_name") == "James Whitcombe" and parsed.get("occupation","").startswith("Former logistics"),
          f"top-level fields survive ({parsed.get('client_name')!r}, {parsed.get('occupation')!r})")
    kf = parsed.get("key_facts", {})
    check(kf.get("mortgage_debt") == "£1.2 million" and kf.get("dependants") == "two children, ages 21 and 16",
          f"key_facts fields that finished before the cut survive ({kf})")
    check("youngest_child_age" not in kf, "the field mid-write when it got cut is simply absent, not garbled")

# ---- claude_json end-to-end: a fake truncated response should recover, not raise ----
class FakeBlock:
    def __init__(self, text): self.type = "text"; self.text = text
class FakeUsage:
    output_tokens = 4000; cache_creation_input_tokens = 0; cache_read_input_tokens = 0
class FakeResp:
    def __init__(self, text, stop_reason): self.content=[FakeBlock(text)]; self.stop_reason=stop_reason; self.usage=FakeUsage()
class FakeMessages:
    def __init__(self, text, stop_reason): self._text=text; self._stop=stop_reason
    def create(self, **kw): return FakeResp(self._text, self._stop)

real_client = llm.anthropic_client
try:
    llm.anthropic_client = type("C", (), {"messages": FakeMessages(REAL_TRUNCATED, "max_tokens")})()
    result = claude_json("sys", "user", "some-model", max_tokens=4000)
    check(result.get("client_name") == "James Whitcombe", f"claude_json recovers a usable dict instead of raising ({result})")

    # a genuinely broken response (not a truncation) should still raise, not silently
    # return a guessed partial - that would hide a real formatting bug in the model output
    llm.anthropic_client = type("C", (), {"messages": FakeMessages("not json at all { broken", "end_turn")})()
    threw = False
    try: claude_json("sys", "user", "some-model")
    except ValueError: threw = True
    check(threw, "a non-truncated parse failure still raises - repair only kicks in for stop_reason=max_tokens")
finally:
    llm.anthropic_client = real_client

print("\n%d failure(s)" % len(fails))
for f in fails: print("  -", f)
sys.exit(1 if fails else 0)
