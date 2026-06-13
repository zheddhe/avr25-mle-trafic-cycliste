"""Authentication dependencies for the prediction serving API."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, HTTPException
from fastapi.security import HTTPBasic, HTTPBasicCredentials

security = HTTPBasic()


def _default_credentials() -> dict[str, dict[str, str]]:
    """Return local demo credentials without embedding secret constants."""

    return {
        username: {"password": username, "role": role}
        for username, role in {
            "admin1": "admin",
            "admin2": "admin",
            "user1": "user",
            "user2": "user",
        }.items()
    }


dict_credentials = _default_credentials()


def check_credentials(
    credentials: Annotated[HTTPBasicCredentials, Depends(security)],
) -> dict[str, str]:
    """Validate Basic Auth credentials and return sanitized user details."""

    if credentials.username not in dict_credentials:
        raise HTTPException(
            status_code=403,
            detail=f"Unknown user [{credentials.username}]",
        )

    user_info = dict_credentials[credentials.username]
    if user_info["password"] != credentials.password:
        raise HTTPException(status_code=403, detail="Invalid password.")

    return {"username": credentials.username, "role": user_info["role"]}


def check_admin_role(
    user_info: Annotated[dict[str, str], Depends(check_credentials)],
) -> dict[str, str]:
    """Ensure the current user has the admin role."""

    if user_info["role"] != "admin":
        raise HTTPException(
            status_code=403,
            detail=(f"Access denied. Admin role required. Current role: {user_info['role']}"),
        )
    return user_info


def check_user_or_admin_role(
    user_info: Annotated[dict[str, str], Depends(check_credentials)],
) -> dict[str, str]:
    """Ensure the current user has user or admin privileges."""

    if user_info["role"] not in ["user", "admin"]:
        raise HTTPException(
            status_code=403,
            detail=(
                f"Access denied. User or Admin role required. Current role: {user_info['role']}"
            ),
        )
    return user_info
