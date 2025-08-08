from datetime import datetime
from fastapi import APIRouter


router = APIRouter()


@router.get("/")
async def root():
    return {"message": "AI Trading Platform API", "version": "1.0.0"}


@router.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


