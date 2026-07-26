import booking

slots = booking.get_free_slots()
print("Total real slots found:", len(slots))

chosen = booking.pick_slots(slots, urgency_driver="Mortgage Completion")
text, labels = booking.format_offer(chosen)
print("OFFER SMS:", text)

for reply in ["2", "the first one", "tuesday please"]:
    match = booking.parse_reply(reply, labels)
    print(f"  reply {reply!r} -> {match}")