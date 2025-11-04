from typing import Optional

from pydantic import BaseModel

from .assistants.assistants import AssistantObject
from .assistants.threads import ThreadObject
from .assistants.messages import MessageObject

class ThreadPreview(BaseModel):
    assistant: Optional[AssistantObject] = None
    thread: ThreadObject
    first_message: Optional[MessageObject] = None
