import booking

slots = booking.get_free_slots()
chosen = booking.pick_slots(slots, urgency_driver="Mortgage Completion")
text, labels = booking.format_offer(chosen)
print("Offered:", text)

# Simulate the lead picking option 1
slot = booking.parse_reply("1", labels)
print("Lead picked:", slot)

result = booking.book("taTLsvWdHbfckD2T5Ru8", slot)
print("Booked — appointment id:", result.get("id") or result)