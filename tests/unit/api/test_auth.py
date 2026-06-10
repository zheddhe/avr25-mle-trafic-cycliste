# tests/api/test_auth.py
from __future__ import annotations

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPBasicCredentials

from src.api.auth import (
    check_admin_role,
    check_credentials,
    check_user_or_admin_role,
)


class TestApiAuth:
    def test_check_credentials_accepts_known_user(self) -> None:
        credentials = HTTPBasicCredentials(username="user1", password="user1")

        user_info = check_credentials(credentials)

        assert user_info == {"username": "user1", "role": "user"}

    def test_check_credentials_rejects_unknown_user(self) -> None:
        credentials = HTTPBasicCredentials(username="unknown", password="unknown")

        with pytest.raises(HTTPException) as exc_info:
            check_credentials(credentials)

        assert exc_info.value.status_code == 403
        assert exc_info.value.detail == "Unknown user [unknown]"

    def test_check_credentials_rejects_invalid_password(self) -> None:
        credentials = HTTPBasicCredentials(username="user1", password="bad")

        with pytest.raises(HTTPException) as exc_info:
            check_credentials(credentials)

        assert exc_info.value.status_code == 403
        assert exc_info.value.detail == "Invalid password."

    def test_check_admin_role_accepts_admin(self) -> None:
        user_info = {"username": "admin1", "role": "admin"}

        result = check_admin_role(user_info)

        assert result == user_info

    def test_check_admin_role_rejects_user(self) -> None:
        user_info = {"username": "user1", "role": "user"}

        with pytest.raises(HTTPException) as exc_info:
            check_admin_role(user_info)

        assert exc_info.value.status_code == 403
        assert "Admin role required" in exc_info.value.detail

    def test_check_user_or_admin_role_accepts_user(self) -> None:
        user_info = {"username": "user1", "role": "user"}

        result = check_user_or_admin_role(user_info)

        assert result == user_info

    def test_check_user_or_admin_role_rejects_other_role(self) -> None:
        user_info = {"username": "reader1", "role": "reader"}

        with pytest.raises(HTTPException) as exc_info:
            check_user_or_admin_role(user_info)

        assert exc_info.value.status_code == 403
        assert "User or Admin role required" in exc_info.value.detail
