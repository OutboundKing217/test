import os
import uuid
from datetime import datetime, timezone

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from models import Analysis, Base, Session as DBSession, User
from analysis import analyze_samples

DATABASE_URL = os.environ.get("DATABASE_URL", "")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+asyncpg://", 1)
elif DATABASE_URL.startswith("postgresql://") and "+asyncpg" not in DATABASE_URL:
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)

engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

app = FastAPI()


@app.on_event("startup")
async def startup():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db():
    async with AsyncSessionLocal() as session:
        yield session


@app.get("/health")
async def health():
    return {"status": "ok"}


class SampleIn(BaseModel):
    t: float
    x: float
    y: float
    z: float
    magnitude: float

class SessionIn(BaseModel):
    user_id: str
    user_name: str = ""
    started_at: str
    samples: list[SampleIn]


@app.post("/sessions")
async def create_session(body: SessionIn, db: AsyncSession = Depends(get_db)):
    user_uuid = body.user_id

    user = (await db.execute(select(User).where(User.id == user_uuid))).scalar_one_or_none()
    if user is None:
        user = User(id=user_uuid, name=body.user_name, created_at=datetime.now(timezone.utc))
        db.add(user)
        await db.flush()
    elif body.user_name and user.name != body.user_name:
        user.name = body.user_name
        await db.flush()

    started_at_str = body.started_at.replace("Z", "+00:00")
    try:
        started_at_dt = datetime.fromisoformat(started_at_str)
    except ValueError:
        started_at_dt = datetime.now(timezone.utc)
    if started_at_dt.tzinfo is None:
        started_at_dt = started_at_dt.replace(tzinfo=timezone.utc)

    samples = [s.model_dump() for s in body.samples]

    session_id = str(uuid.uuid4())
    duration_s = samples[-1]["t"] - samples[0]["t"] if len(samples) >= 2 else 0.0

    db_session = DBSession(
        id=session_id,
        user_id=user_uuid,
        started_at=started_at_dt,
        duration_s=duration_s,
        sample_count=len(samples),
        raw_samples=samples,
    )
    db.add(db_session)
    await db.flush()

    analysis_result: dict = {"error": "Too few samples (need 64+)"}
    if len(samples) >= 64:
        try:
            analysis_result = analyze_samples(samples)
        except Exception as exc:
            analysis_result = {"error": str(exc)}

    analysis = Analysis(
        session_id=session_id,
        computed_at=datetime.now(timezone.utc),
        tau=analysis_result.get("tau"),
        power_law_range=analysis_result.get("power_law_range"),
        goodness_of_fit=analysis_result.get("goodness_of_fit"),
        is_scale_free=analysis_result.get("is_scale_free", False),
        n_events=analysis_result.get("n_events"),
        error=analysis_result.get("error"),
    )
    db.add(analysis)
    await db.commit()

    return {
        "session_id": session_id,
        "sample_count": len(samples),
        "analysis": analysis_result,
    }


@app.get("/users/{user_id}/sessions")
async def get_user_sessions(user_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(DBSession)
        .where(DBSession.user_id == user_id)
        .order_by(DBSession.started_at.desc())
    )
    sessions = result.scalars().all()

    out = []
    for s in sessions:
        a = s.analysis
        out.append({
            "session_id": s.id,
            "started_at": s.started_at.isoformat(),
            "duration_s": s.duration_s,
            "sample_count": s.sample_count,
            "analysis": {
                "tau": a.tau,
                "power_law_range": a.power_law_range,
                "goodness_of_fit": a.goodness_of_fit,
                "is_scale_free": a.is_scale_free,
                "n_events": a.n_events,
                "error": a.error,
            } if a else None,
        })
    return out


@app.get("/sessions/{session_id}")
async def get_session(session_id: str, db: AsyncSession = Depends(get_db)):
    s = (await db.execute(
        select(DBSession).where(DBSession.id == session_id)
    )).scalar_one_or_none()
    if s is None:
        raise HTTPException(status_code=404, detail="Session not found")

    a = s.analysis
    return {
        "session_id": s.id,
        "started_at": s.started_at.isoformat(),
        "duration_s": s.duration_s,
        "sample_count": s.sample_count,
        "analysis": {
            "tau": a.tau,
            "power_law_range": a.power_law_range,
            "goodness_of_fit": a.goodness_of_fit,
            "is_scale_free": a.is_scale_free,
            "n_events": a.n_events,
            "error": a.error,
        } if a else None,
    }
