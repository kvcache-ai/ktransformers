from time import time
from uuid import uuid4

from ktransformers.server.models.assistants.runs import Run
from ktransformers.server.schemas.assistants.runs import RunCreate,RunObject
from ktransformers.server.schemas.base import ObjectID
from ktransformers.server.utils.sql_utils import SQLUtil


class RunsDatabaseManager:
    def __init__(self) -> None:
        self.sql_util = SQLUtil()

    def create_run_object(self, thread_id: ObjectID, run: RunCreate) -> RunObject:
        run_obj = RunObject(
            **run.model_dump(mode='json', exclude={"stream"}),
            id=str(uuid4()),
            object='run',
            created_at=int(time()),
            thread_id=thread_id,
            status=RunObject.Status.queued,
        )
        run_obj.set_compute_save(0)
        return run_obj

    def db_create_run(self, thread_id: str, run: RunCreate):
        db_run = Run(
            **run.model_dump(mode="json", exclude={"stream"}),
            id=str(uuid4()),
            created_at=int(time()),
            status="queued",
            thread_id=thread_id,
        )
        with self.sql_util.get_db() as db:
            self.sql_util.db_add_commit_refresh(db, db_run)
            run_obj = RunObject.model_validate(db_run.__dict__)
            run_obj.set_compute_save(0)
        return run_obj

    def db_sync_run(self, run: RunObject) -> None:
        db_run = Run(
            **run.model_dump(mode='json'),
        )
        with self.sql_util.get_db() as db:
            self.sql_util.db_merge_commit(db, db_run)

    def db_get_run(self, run_id: ObjectID) -> RunObject:
        with self.sql_util.get_db() as db:
            db_run = db.query(Run).filter(Run.id == run_id).first()
            return RunObject.model_validate(db_run.__dict__)
