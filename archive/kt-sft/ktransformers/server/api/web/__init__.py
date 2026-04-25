from fastapi import APIRouter
from .system import router as system_router


router = APIRouter()
router.include_router(system_router)
