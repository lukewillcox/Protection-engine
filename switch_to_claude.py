"""
switch_to_claude.py — convert all chat/generation calls from OpenAI to Claude
(via the llm.py helper). Embeddings stay on OpenAI (rag.py untouched).

Run ONCE from the project folder:  python switch_to_claude.py
It reports each replacement. If any says NOT FOUND, tell me and stop.
Make sure llm.py is in the folder first.
"""

def patch(path, replacements, add_imports=None):
    src = open(path).read()
    ok = True
    for label, old, new in replacements:
        if old in src:
            src = src.replace(old, new, 1)
            print(f"  [{path}] {label}: applied")
        else:
            print(f"  [{path}] {label}: NOT FOUND")
            ok = False
    if add_imports:
        # insert imports after the last existing 'from ... import' near the top
        lines = src.splitlines()
        insert_at = 0
        for i, ln in enumerate(lines[:30]):
            if ln.startswith("from ") or ln.startswith("import "):
                insert_at = i + 1
        for imp in add_imports:
            if imp not in src:
                lines.insert(insert_at, imp)
                insert_at += 1
        src = "\n".join(lines)
        print(f"  [{path}] imports added")
    open(path, "w").write(src)
    return ok


print("Patching copilot.py ...")
patch("copilot.py",
    [
        ("extract_lead call",
'''    resp = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        temperature=0.2,
        messages=[
            {"role": "system", "content": LEAD_EXTRACT_SYSTEM},
            {"role": "user", "content": "Referral email:\\n\\n" + email_text},
        ],
    )
    return json.loads(resp.choices[0].message.content)''',
'''    return claude_json(LEAD_EXTRACT_SYSTEM, "Referral email:\\n\\n" + email_text, MODEL_PREPARE)'''),

        ("generate_prep call",
'''    resp = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        temperature=0.3,
        messages=[
            {"role": "system", "content": PREPARE_SYSTEM},
            {"role": "user", "content": "Referral email:\\n\\n" + email_text},
        ],
    )
    return json.loads(resp.choices[0].message.content)''',
'''    return claude_json(PREPARE_SYSTEM, "Referral email:\\n\\n" + email_text, MODEL_PREPARE)'''),

        ("analyze call",
'''            resp = client.chat.completions.create(
                model="gpt-4o",
                response_format={"type": "json_object"},
                temperature=0.3,
                messages=[
                    {"role": "system", "content": ANALYZE_SYSTEM},
                    {"role": "user", "content": user_content},
                ],
            )
            return jsonify(json.loads(resp.choices[0].message.content))''',
'''            return jsonify(claude_json(ANALYZE_SYSTEM, user_content, MODEL_ANALYZE))'''),

        ("summary call",
'''            resp = client.chat.completions.create(
                model="gpt-4o",
                temperature=0.3,
                messages=[
                    {"role": "system", "content": SUMMARY_SYSTEM},
                    {"role": "user", "content": user_content},
                ],
            )
            return jsonify({"summary": resp.choices[0].message.content.strip()})''',
'''            return jsonify({"summary": claude_text(SUMMARY_SYSTEM, user_content, MODEL_SUMMARY).strip()})'''),

        ("prepare route call",
'''            resp = client.chat.completions.create(
                model="gpt-4o",
                response_format={"type": "json_object"},
                temperature=0.3,
                messages=[
                    {"role": "system", "content": PREPARE_SYSTEM},
                    {"role": "user", "content": user_content},
                ],
            )
            return jsonify(json.loads(resp.choices[0].message.content))''',
'''            return jsonify(claude_json(PREPARE_SYSTEM, user_content, MODEL_PREPARE))'''),
    ],
    add_imports=["from llm import claude_json, claude_text, MODEL_ANALYZE, MODEL_PREPARE, MODEL_SUMMARY"],
)

print("Patching qualify.py ...")
patch("qualify.py",
    [
        ("qualify call",
'''    resp = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        temperature=0.2,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
    )
    return json.loads(resp.choices[0].message.content)''',
'''    return claude_json(SYSTEM, prompt, MODEL_QUALIFY)'''),
    ],
    add_imports=["from llm import claude_json, MODEL_QUALIFY"],
)

print("Patching messaging.py ...")
patch("messaging.py",
    [
        ("sms call",
'''    resp = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.7,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[0].message.content.strip()''',
'''    return claude_text(SYSTEM, prompt, MODEL_SMS).strip()'''),
    ],
    add_imports=["from llm import claude_text, MODEL_SMS"],
)

print("\nDone. Now run:  python -c \"import copilot, qualify, messaging; print('all import OK')\"")