from fastapi import APIRouter

from .assistants import router as assistants_router,create_default_assistant
from .endpoints.chat import router as chat_router
from .legacy import router as legacy_router

router = APIRouter(prefix='/v1')


router.include_router(assistants_router)
router.include_router(chat_router)
router.include_router(legacy_router)

def post_db_creation_operations():
    create_default_assistant()
