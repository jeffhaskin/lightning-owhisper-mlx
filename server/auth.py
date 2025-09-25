"""Simple Deepgram-style API key validation."""

from __future__ import annotations

import base64
from typing import Optional

from fastapi import HTTPException, status


def extract_token(header_value: str) -> Optional[str]:
    if not header_value:
        return None
    parts = header_value.strip().split()
    if len(parts) == 1:
        return parts[0]
    if len(parts) >= 2:
        scheme = parts[0].lower()
        token = " ".join(parts[1:])
        if scheme in {"token", "bearer"}:
            return token
        if scheme == "basic":
            try:
                decoded = base64.b64decode(token).decode("utf-8")
            except Exception:
                return None
            username, _, password = decoded.partition(":")
            return password or username
    return None

__all__ = ['extract_token', 'require_api_key']


def require_api_key(header_value: Optional[str], expected: Optional[str]) -> None:
    """Validate that ``header_value`` contains the configured API key."""
    if expected is None:
        return
    token = extract_token(header_value or "")
    if not token or token != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing Deepgram API key",
        )
