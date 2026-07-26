import requests
from datetime import datetime, timedelta
from config import GHL_HEADERS, CALENDAR_ID

# GHL wants epoch milliseconds
now = datetime.now()
start_ms = int(now.timestamp() * 1000)
end_ms   = int((now + timedelta(days=7)).timestamp() * 1000)

r = requests.get(
    f"https://services.leadconnectorhq.com/calendars/{CALENDAR_ID}/free-slots",
    headers=GHL_HEADERS,
    params={
        "startDate": start_ms,
        "endDate":   end_ms,
        "timezone":  "Europe/London",
    },
    timeout=10,
)

print("Status:", r.status_code)
print(r.text[:1500])