from fastapi import APIRouter

from .ollama import router as ollama_router
from .openai import router as openai_router,post_db_creation_operations
from .web import router as web_router

router = APIRouter()
router.include_router(ollama_router)
router.include_router(openai_router)
router.include_router(web_router)
