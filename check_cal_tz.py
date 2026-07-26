import requests
from config import GHL_HEADERS, CALENDAR_ID

r = requests.get(
    f"https://services.leadconnectorhq.com/calendars/{CALENDAR_ID}",
    headers=GHL_HEADERS,
    timeout=10,
)
import json
data = r.json().get("calendar", r.json())
print("Calendar name:", data.get("name"))
print("Timezone:", data.get("timezone", "NOT SET"))