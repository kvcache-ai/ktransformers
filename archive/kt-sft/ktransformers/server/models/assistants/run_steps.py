from sqlalchemy import JSON, Column, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from ktransformers.server.utils.sql_utils import Base


class RunStep(Base):
    __tablename__ = "run_steps"
    # todo
    id = Column(String, primary_key=True, index=True)
    object = Column(String, default="thread.run.step")
    created_at = Column(Integer)

    assistant_id = Column(String, ForeignKey("assistants.id"))
    thread_id = Column(String, ForeignKey("threads.id"))
    run_id = Column(String, ForeignKey("runs.id"))
    type = Column(String)
    status = Column(String)
    step_details = Column(JSON)
    last_error = Column(JSON, nullable=True)
    expires_at = Column(Integer, nullable=True)
    cancelled_at = Column(Integer, nullable=True)
    failed_at = Column(Integer, nullable=True)
    completed_at = Column(Integer, nullable=True)

    meta_data = Column(JSON, nullable=True)
    usage = Column(JSON, nullable=True)

    assistant = relationship("Assistant", back_populates="run_steps")
    thread = relationship("Thread", back_populates="run_steps")
    run = relationship("Run", back_populates="run_steps")
