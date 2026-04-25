from sqlalchemy import JSON, Column, Float, Integer, String, Text
from sqlalchemy.orm import relationship

from ktransformers.server.utils.sql_utils import Base


class Assistant(Base):
    __tablename__ = "assistants"

    id = Column(String, primary_key=True, index=True)
    object = Column(String, default="assistant")
    created_at = Column(Integer)

    name = Column(String, nullable=True)
    description = Column(String, nullable=True)
    model = Column(String)
    instructions = Column(Text, nullable=True)
    tools = Column(JSON)
    tool_resources = Column(JSON)
    temperature = Column(Float, nullable=True)
    meta_data = Column(JSON, nullable=True)
    top_p = Column(Float, nullable=True)
    response_format = Column(JSON, default="auto")

    build_status = Column(JSON, nullable=True)

    runs = relationship("Run", back_populates="assistant")

    messages = relationship("Message", back_populates="assistant")
