from fastapi import APIRouter

from .assistants import router as assistants_router, create_default_assistant
from .messages import router as messages_router
from .runs import router as runs_router
from .threads import router as threads_router

router = APIRouter()

threads_router.include_router(runs_router)
threads_router.include_router(messages_router)

router.include_router(assistants_router)
router.include_router(threads_router)
