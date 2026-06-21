import asyncio
import os
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from scipy.signal import butter, lfilter, lfilter_zi
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from analysis import analyze_samples
from models import Analysis, Base, Session as DBSession, User

DATABASE_URL = os.environ.get("DATABASE_URL", "")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+asyncpg://", 1)
elif DATABASE_URL.startswith("postgresql://") and "+asyncpg" not in DATABASE_URL:
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)

engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

app = FastAPI()

_watchers: dict[str, list[WebSocket]] = defaultdict(list)
_filter_states: dict[str, Any] = {}

LIVE_FS = 30.0
LIVE_CUTOFF = 0.3
LIVE_ORDER = 4


def _make_causal_filter():
    nyq = 0.5 * LIVE_FS
    b, a = butter(LIVE_ORDER, LIVE_CUTOFF / nyq, btype="low")
    return {"b": b, "a": a, "zi_x": lfilter_zi(b, a), "zi_y": lfilter_zi(b, a), "zi_z": lfilter_zi(b, a)}


def _process_live_samples(user_id: str, samples: list[dict]) -> list[dict]:
    if user_id not in _filter_states:
        state = _make_causal_filter()
        if samples:
            s0 = samples[0]
            state["zi_x"] = state["zi_x"] * s0["x"]
            state["zi_y"] = state["zi_y"] * s0["y"]
            state["zi_z"] = state["zi_z"] * s0["z"]
        _filter_states[user_id] = state

    state = _filter_states[user_id]
    b, a = state["b"], state["a"]

    x_arr = np.array([s["x"] for s in samples], dtype=float)
    y_arr = np.array([s["y"] for s in samples], dtype=float)
    z_arr = np.array([s["z"] for s in samples], dtype=float)
    t_arr = [s["t"] for s in samples]

    grav_x, state["zi_x"] = lfilter(b, a, x_arr, zi=state["zi_x"])
    grav_y, state["zi_y"] = lfilter(b, a, y_arr, zi=state["zi_y"])
    grav_z, state["zi_z"] = lfilter(b, a, z_arr, zi=state["zi_z"])

    dyn_mag = np.sqrt((x_arr - grav_x)**2 + (y_arr - grav_y)**2 + (z_arr - grav_z)**2)
    return [{"t": t_arr[i], "dynamic_mag": round(float(dyn_mag[i]), 6)} for i in range(len(samples))]


@app.on_event("startup")
async def startup():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db():
    async with AsyncSessionLocal() as session:
        yield session


async def _run_analysis_background(session_id: str, samples: list[dict]):
    """Run heavy analysis after upload returns — never blocks the HTTP response."""
    await asyncio.sleep(0)  # yield control so response is sent first
    async with AsyncSessionLocal() as db:
        try:
            analysis_result: dict = {"error": "Too few samples (need 64+)", "dynamic_signal_csv": None}
            if len(samples) >= 64:
                try:
                    loop = asyncio.get_event_loop()
                    analysis_result = await loop.run_in_executor(None, analyze_samples, samples)
                except Exception as exc:
                    analysis_result = {"error": str(exc), "dynamic_signal_csv": None}

            existing = (await db.execute(
                select(Analysis).where(Analysis.session_id == session_id)
            )).scalar_one_or_none()

            if existing:
                existing.computed_at = datetime.now(timezone.utc)
                existing.tau = analysis_result.get("tau")
                existing.power_law_range = analysis_result.get("power_law_range")
                existing.goodness_of_fit = analysis_result.get("goodness_of_fit")
                existing.is_scale_free = analysis_result.get("is_scale_free", False)
                existing.n_events = analysis_result.get("n_events")
                existing.error = analysis_result.get("error")
                existing.dynamic_signal = analysis_result.get("dynamic_signal_csv")
            else:
                analysis = Analysis(
                    session_id=session_id,
                    computed_at=datetime.now(timezone.utc),
                    tau=analysis_result.get("tau"),
                    power_law_range=analysis_result.get("power_law_range"),
                    goodness_of_fit=analysis_result.get("goodness_of_fit"),
                    is_scale_free=analysis_result.get("is_scale_free", False),
                    n_events=analysis_result.get("n_events"),
                    error=analysis_result.get("error"),
                    dynamic_signal=analysis_result.get("dynamic_signal_csv"),
                )
                db.add(analysis)

            await db.commit()
        except Exception as exc:
            print(f"Background analysis failed for session {session_id}: {exc}")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/researcher")
async def researcher_dashboard():
    return FileResponse("researcher.html")


@app.get("/dashboard/patients")
async def get_all_patients(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).order_by(User.name))
    users = result.scalars().all()
    out = []
    for u in users:
        session_result = await db.execute(
            select(DBSession).where(DBSession.user_id == u.id).order_by(DBSession.started_at.desc())
        )
        sessions = session_result.scalars().all()
        session_list = []
        for s in sessions:
            a = s.analysis
            session_list.append({
                "session_id": str(s.id),
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
        out.append({
            "user_id": str(u.id),
            "name": u.name or "Unknown",
            "session_count": len(sessions),
            "sessions": session_list,
            "is_live": str(u.id) in _filter_states,
        })
    return out


@app.get("/sessions/{session_id}/signal.csv")
async def download_signal_csv(session_id: str, db: AsyncSession = Depends(get_db)):
    s = (await db.execute(
        select(DBSession).where(DBSession.id == session_id)
    )).scalar_one_or_none()
    if s is None:
        raise HTTPException(status_code=404, detail="Session not found")
    a = s.analysis
    if a is None or not a.dynamic_signal:
        raise HTTPException(status_code=404, detail="No signal data yet — analysis may still be running")
    return StreamingResponse(
        iter([a.dynamic_signal]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=session_{session_id[:8]}_signal.csv"},
    )


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

    # Save a placeholder analysis row immediately
    placeholder = Analysis(
        session_id=session_id,
        computed_at=datetime.now(timezone.utc),
        error="Analysis pending...",
    )
    db.add(placeholder)
    await db.commit()

    # Fire analysis in background — does NOT block the HTTP response
    asyncio.create_task(_run_analysis_background(session_id, samples))
    _filter_states.pop(user_uuid, None)

    return {"session_id": session_id, "sample_count": len(samples), "status": "uploaded"}


@app.get("/users/{user_id}/sessions")
async def get_user_sessions(user_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(DBSession).where(DBSession.user_id == user_id).order_by(DBSession.started_at.desc())
    )
    sessions = result.scalars().all()
    user = (await db.execute(select(User).where(User.id == user_id))).scalar_one_or_none()
    user_name = user.name if user else None
    out = []
    for s in sessions:
        a = s.analysis
        out.append({
            "session_id": s.id,
            "user_name": user_name,
            "started_at": s.started_at.isoformat(),
            "duration_s": s.duration_s,
            "sample_count": s.sample_count,
            "analysis": {
                "user_name": user_name,
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
    user = (await db.execute(select(User).where(User.id == s.user_id))).scalar_one_or_none()
    user_name = user.name if user else None
    a = s.analysis
    return {
        "session_id": s.id,
        "user_name": user_name,
        "started_at": s.started_at.isoformat(),
        "duration_s": s.duration_s,
        "sample_count": s.sample_count,
        "analysis": {
            "user_name": user_name,
            "tau": a.tau,
            "power_law_range": a.power_law_range,
            "goodness_of_fit": a.goodness_of_fit,
            "is_scale_free": a.is_scale_free,
            "n_events": a.n_events,
            "error": a.error,
        } if a else None,
    }


@app.websocket("/ws/ingest/{user_id}")
async def ws_ingest(websocket: WebSocket, user_id: str):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            samples = data.get("samples", [])
            if not samples:
                continue
            processed = _process_live_samples(user_id, samples)
            dead = []
            for ws in _watchers.get(user_id, []):
                try:
                    await ws.send_json({"user_id": user_id, "points": processed})
                except Exception:
                    dead.append(ws)
            for ws in dead:
                _watchers[user_id].remove(ws)
    except WebSocketDisconnect:
        _filter_states.pop(user_id, None)


@app.websocket("/ws/watch/{user_id}")
async def ws_watch(websocket: WebSocket, user_id: str):
    await websocket.accept()
    _watchers[user_id].append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in _watchers.get(user_id, []):
            _watchers[user_id].remove(websocket)
