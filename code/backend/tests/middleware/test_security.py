import pytest
import os
import jwt
from datetime import datetime, timedelta
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from unittest.mock import patch, MagicMock

# Adjust the import path based on your project structure
from backend.middleware.security import validate_jwt, RoleChecker

# Test JWT_SECRET
TEST_JWT_SECRET = "testsecret"

@pytest.fixture(autouse=True)
def set_test_jwt_secret(monkeypatch):
    monkeypatch.setenv("JWT_SECRET", TEST_JWT_SECRET)

def create_test_token(payload, secret=TEST_JWT_SECRET, algorithm="HS256"):
    return jwt.encode(payload, secret, algorithm=algorithm)

@pytest.mark.asyncio
async def test_validate_jwt_valid_token():
    payload = {"sub": "testuser", "role": "admin", "exp": datetime.utcnow() + timedelta(minutes=5)}
    token = create_test_token(payload)
    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
    
    decoded_payload = await validate_jwt(credentials)
    assert decoded_payload["sub"] == "testuser"
    assert decoded_payload["role"] == "admin"

@pytest.mark.asyncio
async def test_validate_jwt_invalid_secret():
    payload = {"sub": "testuser", "exp": datetime.utcnow() + timedelta(minutes=5)}
    token = create_test_token(payload, secret="wrongsecret")
    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
    
    with pytest.raises(HTTPException) as exc_info:
        await validate_jwt(credentials)
    assert exc_info.value.status_code == 403
    assert exc_info.value.detail == "Invalid token"

@pytest.mark.asyncio
async def test_validate_jwt_expired_token():
    payload = {"sub": "testuser", "exp": datetime.utcnow() - timedelta(minutes=5)}
    token = create_test_token(payload)
    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
    
    with pytest.raises(HTTPException) as exc_info:
        await validate_jwt(credentials)
    assert exc_info.value.status_code == 403
    assert exc_info.value.detail == "Invalid token" # PyJWT raises ExpiredSignatureError

@pytest.mark.asyncio
async def test_validate_jwt_malformed_token():
    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="not.a.jwt")
    
    with pytest.raises(HTTPException) as exc_info:
        await validate_jwt(credentials)
    assert exc_info.value.status_code == 403
    assert exc_info.value.detail == "Invalid token" # PyJWT raises DecodeError

@pytest.mark.asyncio
async def test_validate_jwt_unsupported_algorithm():
    payload = {"sub": "testuser", "exp": datetime.utcnow() + timedelta(minutes=5)}
    # This test is a bit tricky as HS256 is hardcoded. 
    # If the library strictly checks the "alg" header against the provided algorithm, this would be relevant.
    # However, jwt.decode with algorithms=["HS256"] will reject tokens not signed with HS256.
    # Let's simulate a token that *claims* to be RS256 but is passed to an HS256-only validator.
    # For simplicity, we will just test if a token with a different alg in its header (if possible to craft easily)
    # or simply a token that cannot be decoded by HS256 (like one signed by RS256) fails.
    # The current implementation of validate_jwt only accepts HS256.
    # If a token is signed with RS256, it will fail decoding with an HS256 secret.
    try:
        # Attempt to create a token that might look like it has a different alg, or is just incompatible
        # This is a simplification; a true RS256 token would need public/private keys.
        # We rely on the fact that a non-HS256 token will likely cause a JWTError with HS256 validation.
        token_bad_alg = jwt.encode(payload, "rs256_key_material_placeholder", algorithm="RS256")
    except Exception: # Fallback if RS256 isn't easily usable without full key setup
        token_bad_alg = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ0ZXN0dXNlciJ9.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c" # Example invalid for HS256

    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token_bad_alg)
    with pytest.raises(HTTPException) as exc_info:
        await validate_jwt(credentials)
    assert exc_info.value.status_code == 403
    assert exc_info.value.detail == "Invalid token"

@pytest.mark.asyncio
async def test_role_checker_allowed_role():
    checker = RoleChecker(allowed_roles=["admin", "editor"])
    mock_user_payload = {"sub": "testuser", "role": "admin"}
    
    # Mock the Depends(validate_jwt) part
    # The checker itself takes the user payload as input when called.
    # In a real FastAPI app, Depends resolves this. For unit test, we pass it directly.
    try:
        await checker(user=mock_user_payload) # Pass the payload directly for unit testing the __call__ logic
    except HTTPException:
        pytest.fail("HTTPException was raised unexpectedly for allowed role")

@pytest.mark.asyncio
async def test_role_checker_disallowed_role():
    checker = RoleChecker(allowed_roles=["admin", "editor"])
    mock_user_payload = {"sub": "testuser", "role": "viewer"}
    
    with pytest.raises(HTTPException) as exc_info:
        await checker(user=mock_user_payload)
    assert exc_info.value.status_code == 403
    assert exc_info.value.detail == "Insufficient privileges"

@pytest.mark.asyncio
async def test_role_checker_missing_role_key():
    checker = RoleChecker(allowed_roles=["admin"])
    mock_user_payload = {"sub": "testuser"} # No "role" key
    
    with pytest.raises(KeyError): # Expecting a KeyError as the code directly accesses user["role"]
        await checker(user=mock_user_payload)

@pytest.mark.asyncio
async def test_role_checker_empty_allowed_roles():
    checker = RoleChecker(allowed_roles=[])
    mock_user_payload = {"sub": "testuser", "role": "admin"}
    
    with pytest.raises(HTTPException) as exc_info:
        await checker(user=mock_user_payload)
    assert exc_info.value.status_code == 403
    assert exc_info.value.detail == "Insufficient privileges"

