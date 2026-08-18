import os
from dotenv import load_dotenv

load_dotenv()

# --- GHL credentials ---
GHL_API_KEY     = os.getenv("GHL_API_KEY")
GHL_LOCATION_ID = os.getenv("GHL_LOCATION_ID")

# --- Calendar ---
CALENDAR_ID = "qSsD7GaoZnIXWHDvQM4C"

# --- OpenAI ---
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")

# --- Webhook security ---
WEBHOOK_SECRET  = os.getenv("WEBHOOK_SECRET")

# --- Copilot security ---
COPILOT_PASSWORD = os.getenv("COPILOT_PASSWORD")

# --- Broker identity (used in SMS sign-off) ---
BROKER_NAME     = os.getenv("BROKER_NAME", "Andrew")
BROKER_COMPANY  = os.getenv("BROKER_COMPANY", "AndrewTEST")
BROKER_PHONE    = os.getenv("BROKER_PHONE", "")

# --- GHL API basics ---
GHL_BASE = "https://services.leadconnectorhq.com"
GHL_HEADERS = {
    "Authorization": f"Bearer {GHL_API_KEY}",
    "Version": "2021-07-28",
    "Content-Type": "application/json",
}

# --- Custom field keys (from check_fields.py, confirmed) ---
# The API wants the key WITHOUT the "contact." prefix when writing.
FIELD_KEYS = {
    "cover_type":               "cover_type",
    "cover_amount":             "cover_amount",
    "term_years":               "term_years",
    "smoker_status":            "smoker_status",
    "dependants":               "dependants",
    "existing_cover":           "existing_cover",
    "employment_status":        "employment_status",
    "urgency_driver":           "urgency_driver",
    "mortgage_completion_date": "mortgage_completion_date",
    "readiness_score":          "readiness_score",
    "impaired_life_flag":       "impaired_life_flag",
    "ai_qualified":             "ai_qualified",
    "ai_sms_content":           "ai_sms_content",
}

# Field IDs (fallback — some API endpoints prefer the ID over the key)
FIELD_IDS = {
    "cover_type":               "alQfk7lizzGTtB2VjBxB",
    "cover_amount":             "VTofE8nDADuloecYcGgS",
    "term_years":               "TrFGsieB0OoEVlObLhPa",
    "smoker_status":            "v01vRTYIOUBdPM61KDIg",
    "dependants":               "ZDiDi3Rtjd0OXZ141RC8",
    "existing_cover":           "q494YI4eVKpBOwzgC0Fr",
    "employment_status":        "zsBEje21rTmCQYRNAmGe",
    "urgency_driver":           "XDTohgLvU7iwjzQcM7r1",
    "mortgage_completion_date": "FlJVTNIi4MRzL6whbtte",
    "readiness_score":          "989Cpywqw9bfH4AuCXMy",
    "impaired_life_flag":       "cPgFr4VDYmnPYvsVrCSf",
    "ai_qualified":             "xM2ihaEvgB0rUskl6XpY",
    "ai_sms_content":           "2D1kWElWLbncWK0LDb8H",
}

# --- Fail early if essentials missing ---
if not GHL_API_KEY or not GHL_LOCATION_ID:
    raise SystemExit("Missing GHL credentials in .env")