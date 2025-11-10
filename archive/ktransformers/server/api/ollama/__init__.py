from fastapi import APIRouter

from .completions import router as completions_router

router = APIRouter()
router.include_router(completions_router)
