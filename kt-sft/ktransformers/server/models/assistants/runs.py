from sqlalchemy import JSON, Column, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from ktransformers.server.utils.sql_utils import Base


class Run(Base):
    __tablename__ = "runs"

    id = Column(String, primary_key=True, index=True)
    object = Column(String, default="thread.run")
    created_at = Column(Integer)
    thread_id = Column(String, ForeignKey("threads.id"))
    assistant_id = Column(String, ForeignKey("assistants.id"))
    status = Column(String)
    required_action = Column(JSON, nullable=True)
    last_error = Column(JSON, nullable=True)
    expires_at = Column(Integer, nullable=True)
    started_at = Column(Integer, nullable=True)
    cancelled_at = Column(Integer, nullable=True)
    failed_at = Column(Integer, nullable=True)
    completed_at = Column(Integer, nullable=True)
    incomplete_details = Column(JSON, nullable=True)
    # get from assistant
    model = Column(String)
    instructions = Column(Text, nullable=True)
    tools = Column(JSON)
    meta_data = Column(JSON, nullable=True)
    usage = Column(JSON, nullable=True)
    temperature = Column(Float, nullable=True)
    top_p = Column(Float, nullable=True)
    max_propmp_tokens = Column(Integer, nullable=True)
    truncation_strategy = Column(JSON)
    tool_choice = Column(JSON)
    response_format = Column(JSON, default="auto")

    thread = relationship("Thread", back_populates="runs")
    assistant = relationship("Assistant", back_populates="runs")
    message = relationship("Message", back_populates="run")
