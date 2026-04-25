from sqlalchemy import JSON, Column, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from ktransformers.server.utils.sql_utils import Base


class Message(Base):
    __tablename__ = "messages"

    id = Column(String, primary_key=True, index=True)
    object = Column(String, default="thread.message")
    created_at = Column(Integer)

    thread_id = Column(String, ForeignKey("threads.id"))
    status = Column(String, default="in_progress")
    incomplete_details = Column(JSON, nullable=True)
    completed_at = Column(Integer, nullable=True)
    incomplete_at = Column(Integer, nullable=True)
    role = Column(JSON)
    content = Column(JSON)
    assistant_id = Column(String, ForeignKey("assistants.id"), nullable=True)
    run_id = Column(String, ForeignKey("runs.id"), nullable=True)
    attachments = Column(JSON, nullable=True)
    meta_data = Column(JSON, nullable=True)

    thread = relationship("Thread", back_populates="messages")
    assistant = relationship("Assistant", back_populates="messages")
    run = relationship("Run", back_populates="message")
