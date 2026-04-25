from fastapi import APIRouter

from . import completions

router = APIRouter()
router.include_router(completions.router)