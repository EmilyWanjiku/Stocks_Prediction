"""
Lightweight file-based authentication helpers for the Streamlit app.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Dict, Tuple

from werkzeug.security import check_password_hash, generate_password_hash

CREDENTIALS_PATH = Path("app") / "credentials.json"
_LOCK = Lock()


def ensure_credentials_file() -> None:
    """
    Ensure the credentials file exists with the expected structure.
    """
    CREDENTIALS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not CREDENTIALS_PATH.exists():
        CREDENTIALS_PATH.write_text(json.dumps({"users": []}, indent=2), encoding="utf-8")


def _load_credentials() -> Dict[str, list]:
    ensure_credentials_file()
    with CREDENTIALS_PATH.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def _save_credentials(payload: Dict[str, list]) -> None:
    with CREDENTIALS_PATH.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)


def authenticate_user(email: str, password: str) -> bool:
    payload = _load_credentials()
    for entry in payload.get("users", []):
        if entry.get("email") == email and check_password_hash(entry.get("password_hash", ""), password):
            return True
    return False


def register_user(email: str, password: str, is_admin: bool = False) -> Tuple[bool, str]:
    if not email or not password:
        return False, "Email and password are required."

    with _LOCK:
        payload = _load_credentials()
        if any(entry.get("email") == email for entry in payload.get("users", [])):
            return False, "An account with that email already exists."

        password_hash = generate_password_hash(password)
        payload.setdefault("users", []).append({
            "email": email,
            "password_hash": password_hash,
            "is_admin": is_admin,
            "created_at": datetime.now().isoformat(),
        })
        _save_credentials(payload)
    return True, "Account created successfully."


def is_admin_user(email: str) -> bool:
    """Check if a user is an admin."""
    payload = _load_credentials()
    for entry in payload.get("users", []):
        if entry.get("email") == email:
            return entry.get("is_admin", False)
    return False


def get_all_users() -> list:
    """Get list of all users (admin only)."""
    payload = _load_credentials()
    users = []
    for entry in payload.get("users", []):
        users.append({
            "email": entry.get("email"),
            "is_admin": entry.get("is_admin", False),
            "created_at": entry.get("created_at", "Unknown"),
        })
    return users


def delete_user(email: str) -> Tuple[bool, str]:
    """Delete a user account (admin only)."""
    with _LOCK:
        payload = _load_credentials()
        users = payload.get("users", [])
        original_count = len(users)
        payload["users"] = [u for u in users if u.get("email") != email]
        
        if len(payload["users"]) == original_count:
            return False, "User not found."
        
        _save_credentials(payload)
    return True, f"User {email} deleted successfully."

