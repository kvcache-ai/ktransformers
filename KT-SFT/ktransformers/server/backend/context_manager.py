from asyncio import Lock
from typing import Dict, Optional

from ktransformers.server.backend.base import ThreadContext, BackendInterfaceBase
from ktransformers.server.schemas.assistants.runs import RunObject
from ktransformers.server.schemas.base import ObjectID
from ktransformers.server.config.log import logger
from ktransformers.server.backend.interfaces.transformers import TransformersThreadContext
from ktransformers.server.backend.interfaces.ktransformers import KTransformersThreadContext
from ktransformers.server.backend.interfaces.exllamav2 import ExllamaThreadContext


from ktransformers.server.backend.interfaces.exllamav2 import ExllamaInterface
from ktransformers.server.backend.interfaces.transformers import TransformersInterface
from ktransformers.server.backend.interfaces.ktransformers import KTransformersInterface

class ThreadContextManager:
    lock: Lock
    threads_context: Dict[ObjectID, ThreadContext]
    interface: BackendInterfaceBase
    
    def __init__(self,interface) -> None:
        logger.debug(f"Creating Context Manager")
        self.lock = Lock()
        self.threads_context = {}
        self.interface = interface
        pass

    async def get_context_by_run_object(self, run: RunObject) -> ThreadContext:
        async with self.lock:
            logger.debug(f"keys {self.threads_context.keys()}")
            if run.thread_id not in self.threads_context:
                logger.debug(f"new inference context {run.thread_id}")
                if isinstance(self.interface, ExllamaInterface):
                    new_context = ExllamaThreadContext(run, self.interface)
                elif isinstance(self.interface, KTransformersInterface):
                    new_context = KTransformersThreadContext(run, self.interface)
                elif isinstance(self.interface, TransformersInterface):
                    new_context = TransformersThreadContext(run, self.interface)
                else:
                    from ktransformers.server.backend.interfaces.balance_serve import BalanceServeThreadContext
                    from ktransformers.server.backend.interfaces.balance_serve import BalanceServeInterface
                    if isinstance(self.interface, BalanceServeInterface):
                        new_context = BalanceServeThreadContext(run, self.interface)
                    else:
                        raise NotImplementedError
                # elif isinstance(self.interface, BalanceServeInterface):
                #     new_context = BalanceServeThreadContext(run, self.interface)
                # else:
                #     raise NotImplementedError
                self.threads_context[run.thread_id] = new_context
                # self.threads_context[run.thread_id] = ExllamaInferenceContext(run)
            re = self.threads_context[run.thread_id]
            re.update_by_run(run)
            return re

    async def get_context_by_thread_id(self, thread_id: ObjectID) -> Optional[ThreadContext]:
        async with self.lock:
            if thread_id in self.threads_context:
                logger.debug(f'found context for thread {thread_id}')
                return self.threads_context[thread_id]
            else:
                logger.debug(f'no context for thread {thread_id}')
                return None
            