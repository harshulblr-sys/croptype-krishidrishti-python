"""Earth Engine initialization for every pipeline entry point.

Uses the service-account key when one is present (deployment), else falls
back to the developer's personal `earthengine authenticate` login. The key
is looked for at $GEE_KEY_FILE, defaulting to gee_key.json next to this
file — that name is gitignored, so it can never be committed.
"""
import json
import os

PROJECT = "crop-identification-501611"
ROOT = os.path.dirname(os.path.abspath(__file__))
KEY_FILE = os.environ.get("GEE_KEY_FILE", os.path.join(ROOT, "gee_key.json"))


def init(project=PROJECT):
    """Initialize ee; returns a short description of the auth mode used."""
    import ee
    if os.path.exists(KEY_FILE):
        with open(KEY_FILE) as f:
            email = json.load(f)["client_email"]
        ee.Initialize(ee.ServiceAccountCredentials(email, KEY_FILE),
                      project=project)
        return f"service-account ({email})"
    ee.Initialize(project=project)
    return "personal login"
