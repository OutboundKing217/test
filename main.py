import os, uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import selectinload

from analysis import analyze_samples
from models import Analysis, Base, Session, User

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql+asyncpg://localhost/neuromotion")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+asyncpg://", 1)
elif DATABASE_URL.startswith("postgresql://") and "+asyncpg" not in DATABASE_URL:
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)

engine = create_async_engine(DATABASE_URL, echo=False, future=True)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    await engine.dispose()

app = FastAPI(title="NeuroMotion API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

class SampleIn(BaseModel):
    t: float; x: float; y: float; z: float; magnitude: float

class CreateSessionRequest(BaseModel):
    user_id: str; started_at: str; samples: list[SampleIn]

class AnalysisOut(BaseModel):
    beta_x: float; beta_y: float; beta_z: float; beta_magnitude: float
    r2_x: float; r2_y: float; r2_z: float; r2_magnitude: float

class SessionOut(BaseModel):
    id: str; started_at: str; duration_s: float | None; sample_count: int
    analysis: AnalysisOut | None

class CreateSessionResponse(BaseModel):
    session_id: str; analysis: AnalysisOut | None

class CreateUserResponse(BaseModel):
    user_id: str

def _analysis_to_out(a):
    if a is None: return None
    return AnalysisOut(beta_x=a.beta_x, beta_y=a.beta_y, beta_z=a.beta_z,
                       beta_magnitude=a.beta_magnitude, r2_x=a.r2_x, r2_y=a.r2_y,
                       r2_z=a.r2_z, r2_magnitude=a.r2_magnitude)

def _session_to_out(s):
    return SessionOut(id=str(s.id), started_at=s.started_at.isoformat(),
                      duration_s=s.duration_s, sample_count=s.sample_count,
                      analysis=_analysis_to_out(s.analysis))

@app.get("/health")
async def health(): return {"status": "ok"}

@app.post("/users", response_model=CreateUserResponse, status_code=201)
async def create_user():
    async with AsyncSessionLocal() as db:
        user = User(id=uuid.uuid4(), created_at=datetime.now(timezone.utc))
        db.add(user); await db.commit()
        return CreateUserResponse(user_id=str(user.id))

@app.post("/sessions", response_model=CreateSessionResponse, status_code=201)
async def create_session(body: CreateSessionRequest):
    try: user_uuid = uuid.UUID(body.user_id)
    except ValueError: raise HTTPException(400, "Invalid user_id format")

    async with AsyncSessionLocal() as db:
           user = (await db.execute(select(User).where(User.id == user_uuid))).scalar_one_or_none()
        if user is None:
            user = User(id=user_uuid, created_at=datetime.now(timezone.utc))
            db.add(user)
            await db.flush()

        try: started_at_dt = datetime.fromisoformat(body.started_at.replace("Z", "+00:00"))
        except ValueError: raise HTTPException(400, "Invalid started_at format")

        samples_list: list[dict[str, Any]] = [s.model_dump() for s in body.samples]
        times = [s["t"] for s in samples_list]
        duration_s = (max(times) - min(times)) if len(times) >= 2 else None

        session_id = uuid.uuid4()
        session = Session(id=session_id, user_id=user_uuid, started_at=started_at_dt,
                          duration_s=duration_s, sample_count=len(samples_list),
                          raw_samples=samples_list)
        db.add(session); await db.flush()

        analysis_out = None
        if len(samples_list) >= 16:
            try:
                results = analyze_samples(samples_list)
                analysis = Analysis(id=uuid.uuid4(), session_id=session_id,
                                    beta_x=results["x"]["beta"], beta_y=results["y"]["beta"],
                                    beta_z=results["z"]["beta"], beta_magnitude=results["magnitude"]["beta"],
                                    r2_x=results["x"]["r2"], r2_y=results["y"]["r2"],
                                    r2_z=results["z"]["r2"], r2_magnitude=results["magnitude"]["r2"],
                                    computed_at=datetime.now(timezone.utc))
                db.add(analysis); analysis_out = _analysis_to_out(analysis)
            except Exception as exc:
                print(f"Analysis error for session {session_id}: {exc}")

        await db.commit()
    return CreateSessionResponse(session_id=str(session_id), analysis=analysis_out)

@app.get("/users/{user_id}/sessions", response_model=list[SessionOut])
async def list_sessions(user_id: str):
    try: user_uuid = uuid.UUID(user_id)
    except ValueError: raise HTTPException(400, "Invalid user_id format")
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(Session).where(Session.user_id == user_uuid)
            .options(selectinload(Session.analysis)).order_by(Session.started_at.desc()))
        return [_session_to_out(s) for s in result.scalars().all()]

@app.get("/sessions/{session_id}", response_model=SessionOut)
async def get_session(session_id: str):
    try: session_uuid = uuid.UUID(session_id)
    except ValueError: raise HTTPException(400, "Invalid session_id format")
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(Session).where(Session.id == session_uuid)
            .options(selectinload(Session.analysis)))
        session = result.scalar_one_or_none()
    if session is None: raise HTTPException(404, "Session not found")
    return _session_to_out(session)

