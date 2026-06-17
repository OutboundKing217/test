import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
     name = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False,
                        default=lambda: datetime.now(timezone.utc))
    sessions = relationship("Session", back_populates="user", lazy="selectin")


class Session(Base):
    __tablename__ = "sessions"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    started_at = Column(DateTime(timezone=True), nullable=False)
    duration_s = Column(Float, nullable=True)
    sample_count = Column(Integer, nullable=False, default=0)
    raw_samples = Column(JSONB, nullable=False, default=list)
    user = relationship("User", back_populates="sessions")
    analysis = relationship("Analysis", back_populates="session", uselist=False, lazy="selectin")

class Analysis(Base):
    __tablename__ = "analyses"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("sessions.id"), nullable=False, unique=True)
    computed_at = Column(DateTime(timezone=True), nullable=False,
                         default=lambda: datetime.now(timezone.utc))
    tau = Column(Float, nullable=True)
    power_law_range = Column(Float, nullable=True)
    goodness_of_fit = Column(Float, nullable=True)
    is_scale_free = Column(Boolean, nullable=True)
    n_events = Column(Integer, nullable=True)
    error = Column(String, nullable=True)
    session = relationship("Session", back_populates="analysis")
    __table_args__ = (UniqueConstraint("session_id", name="uq_analyses_session_id"),)
