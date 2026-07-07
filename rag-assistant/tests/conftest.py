from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.auth import decode_access_token


client = TestClient(app)


@pytest.fixture
def auth_headers():
    email = f"test-{uuid4()}@example.com"
    password = "test_password_123"
    register_response = client.post("/auth/register", json={"email": email, "password": password})
    assert register_response.status_code == 200
    login_response = client.post("/auth/login", json={"email": email, "password": password})
    assert login_response.status_code == 200
    token = login_response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def auth_user_id(auth_headers):
    token = auth_headers["Authorization"].split(" ", 1)[1]
    return decode_access_token(token)["sub"]
