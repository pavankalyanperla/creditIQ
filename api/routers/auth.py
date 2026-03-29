from datetime import datetime, timedelta
from typing import Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

from config import settings

router = APIRouter()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

# Lazy-loaded demo users (avoid hashing at import time)
_demo_users_cache: Optional[Dict] = None


def get_demo_users() -> Dict:
    """Get demo users with lazy-loaded password hashes."""
    global _demo_users_cache
    if _demo_users_cache is None:
        _demo_users_cache = {
            "admin": {
                "username": "admin",
                "hashed_password": pwd_context.hash("creditiq2024"),
                "role": "admin",
            },
            "analyst": {
                "username": "analyst",
                "hashed_password": pwd_context.hash("analyst2024"),
                "role": "analyst",
            },
        }
    return _demo_users_cache


class Token(BaseModel):
    access_token: str
    token_type: str


def create_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(
        minutes=settings.jwt_access_token_expire_minutes
    )
    to_encode.update({"exp": expire})
    return jwt.encode(
        to_encode, settings.jwt_secret_key, algorithm=settings.jwt_algorithm
    )


def get_current_user(token: str = Depends(oauth2_scheme)):
    demo_users = get_demo_users()
    try:
        payload = jwt.decode(
            token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm]
        )
        username = payload.get("sub")
        if username not in demo_users:
            raise HTTPException(status_code=401)
        return demo_users[username]
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
        )


@router.post("/token", response_model=Token)
async def login(form: OAuth2PasswordRequestForm = Depends()):
    demo_users = get_demo_users()
    user = demo_users.get(form.username)
    if not user or not pwd_context.verify(form.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )
    token = create_token({"sub": form.username})
    return {"access_token": token, "token_type": "bearer"}
