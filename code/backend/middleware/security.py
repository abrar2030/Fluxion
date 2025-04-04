from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

async def validate_jwt(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    try:
        # JWT validation logic with HS256
        payload = jwt.decode(
            credentials.credentials,
            os.getenv("JWT_SECRET"),
            algorithms=["HS256"]
        )
        return payload
    except JWTError:
        raise HTTPException(status_code=403, detail="Invalid token")

class RoleChecker:
    def __init__(self, allowed_roles):
        self.allowed_roles = allowed_roles
    
    def __call__(self, user: dict = Depends(validate_jwt)):
        if user["role"] not in self.allowed_roles:
            raise HTTPException(403, "Insufficient privileges")