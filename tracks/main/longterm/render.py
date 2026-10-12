"""Dashboard HTML: the template with the night's data embedded (no external data requests)."""
import json
from pathlib import Path

TEMPLATE = Path(__file__).with_name("dashboard_template.html")


def render(data):
    payload = json.dumps(data, default=str).replace("</", "<\\/")
    return TEMPLATE.read_text(encoding="utf-8").replace("/*__DATA__*/null", payload)
