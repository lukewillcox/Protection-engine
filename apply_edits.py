"""
apply_edits.py — apply the step-aware analyze + structured summary edits
to copilot.py safely (no manual indentation). Run once: python apply_edits.py
"""

src = open("copilot.py").read()

# ---- EDIT 1: step-aware analyze ----
old_analyze = '''        try:
            kb = get_relevant_knowledge(notes, k=5)
            user_content = "Fact-find notes so far:\\n\\n" + notes
            if kb:'''

new_analyze = '''        step = (data.get("step") or "").strip()
        try:
            kb = get_relevant_knowledge(notes, k=5)
            user_content = ""
            if step:
                user_content += (
                    "The adviser is currently on the '" + step + "' step of the call. "
                    "Weight your guidance toward what matters at THIS step, while staying "
                    "aware of the whole conversation.\\n\\n"
                )
            user_content += "Fact-find notes so far (across all steps):\\n\\n" + notes
            if kb:'''

if old_analyze in src:
    src = src.replace(old_analyze, new_analyze, 1)
    print("EDIT 1 (step-aware analyze): applied")
else:
    print("EDIT 1: PATTERN NOT FOUND — no change")

# ---- EDIT 2: structured summary body ----
old_summary = '''        name = data.get("client_name") or "the client"
        ref = data.get("referred_by") or "—"
        if not notes:
            return jsonify({"error": "no notes"}), 400
        try:
            resp = client.chat.completions.create(
                model="gpt-4o",
                temperature=0.3,
                messages=[
                    {"role": "system", "content": SUMMARY_SYSTEM},
                    {"role": "user", "content": f"Client: {name}\\nReferred by: {ref}\\n\\nNotes:\\n{notes}"},
                ],
            )
            return jsonify({"summary": resp.choices[0].message.content.strip()})'''

new_summary = '''        name = data.get("client_name") or "the client"
        ref = data.get("referred_by") or "—"
        ff = data.get("fact_find") or {}
        if not notes and not ff:
            return jsonify({"error": "no notes"}), 400
        try:
            parts = [f"Client: {name}", f"Referred by: {ref}"]
            if ff.get("clients"):
                parts.append("Clients:\\n" + "\\n".join("- " + c for c in ff["clients"]))
            if ff.get("income"):
                parts.append("Income: " + ff["income"])
            if ff.get("mortgage"):
                parts.append("Mortgage: " + ff["mortgage"])
            if ff.get("existing"):
                ex = [e for e in ff["existing"] if e and e.strip()]
                parts.append("Existing cover: " + ("; ".join(ex) if ex else "none recorded"))
            if ff.get("quotes"):
                qs = [q for q in ff["quotes"] if q and q.strip()]
                parts.append("Quotes captured by adviser (from their sourcing tool):\\n"
                             + ("\\n".join("- " + q for q in qs) if qs else "none"))
            if ff.get("total"):
                parts.append("Total monthly premium captured: " + ff["total"])
            if ff.get("notesByStep"):
                parts.append("Call notes by step:\\n" + ff["notesByStep"])
            elif notes:
                parts.append("Notes:\\n" + notes)
            user_content = "\\n\\n".join(parts)
            resp = client.chat.completions.create(
                model="gpt-4o",
                temperature=0.3,
                messages=[
                    {"role": "system", "content": SUMMARY_SYSTEM},
                    {"role": "user", "content": user_content},
                ],
            )
            return jsonify({"summary": resp.choices[0].message.content.strip()})'''

if old_summary in src:
    src = src.replace(old_summary, new_summary, 1)
    print("EDIT 2 (structured summary): applied")
else:
    print("EDIT 2: PATTERN NOT FOUND — no change")

open("copilot.py", "w").write(src)
print("Saved copilot.py")