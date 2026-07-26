import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY     = os.getenv("GHL_API_KEY")
LOCATION_ID = os.getenv("GHL_LOCATION_ID")

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Version": "2021-07-28",
}

r = requests.get(
    "https://services.leadconnectorhq.com/calendars/",
    headers=headers,
    params={"locationId": LOCATION_ID},
    timeout=10,
)

print("Status:", r.status_code)
if r.status_code != 200:
    print(r.text)
    raise SystemExit

cals = r.json().get("calendars", [])
print(f"\nFound {len(cals)} calendars\n")
for c in cals:
    print(f"{c.get('name','?'):<35} {c.get('id')}")
    print(f"    active: {c.get('isActive')}   slug: {c.get('slug','')}")