"""Earth Engine initialization for every pipeline entry point.

Auth resolution order:
  1. $GEE_KEY_JSON  — the key file's *content* as an env var (container
     hosts like Hugging Face Spaces pass secrets as strings)
  2. $GEE_KEY_FILE / ./gee_key.json — a key file on disk (VM deployment;
     the default name is gitignored so it can never be committed)
  3. the developer's personal `earthengine authenticate` login
"""
import json
import os

PROJECT = "crop-identification-501611"
ROOT = os.path.dirname(os.path.abspath(__file__))
KEY_FILE = os.environ.get("GEE_KEY_FILE", os.path.join(ROOT, "gee_key.json"))


def init(project=PROJECT):
    """Initialize ee; returns a short description of the auth mode used."""
    import ee
    key_json = os.environ.get("GEE_KEY_JSON")
    if key_json:
        email = json.loads(key_json)["client_email"]
        ee.Initialize(ee.ServiceAccountCredentials(email, key_data=key_json),
                      project=project)
        return f"service-account via env ({email})"
    if os.path.exists(KEY_FILE):
        with open(KEY_FILE) as f:
            email = json.load(f)["client_email"]
        ee.Initialize(ee.ServiceAccountCredentials(email, KEY_FILE),
                      project=project)
        return f"service-account ({email})"
    ee.Initialize(project=project)
    return "personal login"
