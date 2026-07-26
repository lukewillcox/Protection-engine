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
    f"https://services.leadconnectorhq.com/locations/{LOCATION_ID}/customFields",
    headers=headers,
    timeout=10,
)

print("Status:", r.status_code)

if r.status_code != 200:
    print(r.text)
    raise SystemExit

fields = r.json().get("customFields", [])
print(f"\nFound {len(fields)} custom fields\n")
print(f"{'NAME':<30}{'KEY':<36}ID")
print("-" * 100)

for f in sorted(fields, key=lambda x: x.get("name", "")):
    print(f"{f.get('name',''):<30}{f.get('fieldKey',''):<36}{f.get('id','')}")