import uuid
from datetime import datetime
from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, relationship

class Base(DeclarativeBase): pass

class User(Base):
    __tablename__ = "users"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    sessions = relationship("Session", back_populates="user", lazy="select")

class Session(Base):
    __tablename__ = "sessions"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    started_at = Column(DateTime, nullable=False)
    duration_s = Column(Float, nullable=True)
    sample_count = Column(Integer, nullable=False, default=0)
    raw_samples = Column(JSONB, nullable=False, default=list)
    user = relationship("User", back_populates="sessions")
    analysis = relationship("Analysis", back_populates="session", uselist=False, lazy="select")

class Analysis(Base):
    __tablename__ = "analyses"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("sessions.id"), nullable=False, unique=True)
    beta_x = Column(Float, nullable=False)
    beta_y = Column(Float, nullable=False)
    beta_z = Column(Float, nullable=False)
    beta_magnitude = Column(Float, nullable=False)
    r2_x = Column(Float, nullable=False)
    r2_y = Column(Float, nullable=False)
    r2_z = Column(Float, nullable=False)
    r2_magnitude = Column(Float, nullable=False)
    computed_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    session = relationship("Session", back_populates="analysis")
    __table_args__ = (UniqueConstraint("session_id", name="uq_analyses_session_id"),)
