import requests
from datetime import datetime, timedelta
from config import GHL_BASE, GHL_HEADERS, CALENDAR_ID, GHL_LOCATION_ID

BUSINESS_START = 8    # 08:00
BUSINESS_END   = 18   # 18:00


def get_free_slots(days_ahead: int = 10):
    """Return a flat, sorted list of datetime objects for real free slots."""
    now = datetime.now()
    start_ms = int(now.timestamp() * 1000)
    end_ms   = int((now + timedelta(days=days_ahead)).timestamp() * 1000)

    r = requests.get(
        f"{GHL_BASE}/calendars/{CALENDAR_ID}/free-slots",
        headers=GHL_HEADERS,
        params={"startDate": start_ms, "endDate": end_ms,
                "timezone": "Europe/London"},
        timeout=10,
    )
    r.raise_for_status()
    data = r.json()

    slots = []
    for day, payload in data.items():
        if not isinstance(payload, dict):
            continue
        for iso in payload.get("slots", []):
            dt = datetime.fromisoformat(iso)
            # defensive: only business hours, never weekends
            if dt.weekday() < 5 and BUSINESS_START <= dt.hour < BUSINESS_END:
                slots.append(dt)
    return sorted(slots)


def pick_slots(slots, urgency_driver=None, n=3):
    """Choose n slots across DIFFERENT time-of-day / days for a real choice."""
    if not slots:
        return []

    # For urgent leads: bias toward soonest, but still spread across
    # at least two days and mix morning/afternoon.
    chosen = []
    seen = set()  # (date, half-of-day) so we don't offer 3 near-identical slots

    for s in slots:
        half = "am" if s.hour < 13 else "pm"
        key = (s.date(), half)
        if key in seen:
            continue
        chosen.append(s)
        seen.add(key)
        if len(chosen) == n:
            break

    # Fallback if the calendar is sparse
    if len(chosen) < n:
        for s in slots:
            if s not in chosen:
                chosen.append(s)
            if len(chosen) == n:
                break

    return chosen

def format_offer(slots):
    """Turn chosen slots into a human SMS line + a lookup map."""
    labels = {}
    lines = []
    for i, s in enumerate(slots, start=1):
        # e.g. "Mon 27 Jul 8:00am" / "Tue 28 Jul 2:30pm"
        label = s.strftime("%a %-d %b %-I:%M%p").replace("AM", "am").replace("PM", "pm")
        lines.append(f"{i}) {label}")
        labels[str(i)] = s
    text = "  ".join(lines)
    return text, labels


def parse_reply(reply: str, labels: dict):
    """Map a lead's reply back to a chosen slot datetime, or None."""
    reply = reply.strip().lower()
    # direct number
    for key in labels:
        if reply == key or reply.startswith(key):
            return labels[key]
    # word numbers
    words = {"one": "1", "first": "1", "two": "2", "second": "2",
             "three": "3", "third": "3"}
    for w, k in words.items():
        if w in reply and k in labels:
            return labels[k]
    # day name match (e.g. "tuesday works")
    for key, dt in labels.items():
        if dt.strftime("%a").lower() in reply or dt.strftime("%A").lower() in reply:
            return dt
    return None


def book(contact_id: str, slot: datetime):
    """Create the appointment in GHL."""
    end = slot + timedelta(minutes=30)
    r = requests.post(
        f"{GHL_BASE}/calendars/events/appointments",
        headers=GHL_HEADERS,
        json={
            "calendarId": CALENDAR_ID,
            "locationId": GHL_LOCATION_ID,
            "contactId": contact_id,
            "startTime": slot.isoformat(),
            "endTime": end.isoformat(),
            "title": "Protection Consultation",
        },
        timeout=10,
    )
    r.raise_for_status()
    return r.json()