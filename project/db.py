# db.py
import os, uuid
from datetime import datetime
from contextlib import contextmanager
from typing import Any, Dict, Iterable, Optional

from sqlalchemy import (
    create_engine, Column, String, Integer, Float, DateTime, Text, JSON, Boolean, ForeignKey
)
from sqlalchemy.orm import DeclarativeBase, relationship, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./db.sqlite")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
    future=True,
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)

class Base(DeclarativeBase):
    pass

class Run(Base):
    __tablename__ = "runs"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    task_id = Column(String, index=True, nullable=True)

    status = Column(String, index=True, default="QUEUED")  # QUEUED, RUNNING, COMPLETE, FAILED
    ablation = Column(String, nullable=True)
    overrides_json = Column(JSON, nullable=True)

    epochs = Column(Integer, default=3)
    batch_size = Column(Integer, default=8)
    max_len = Column(Integer, default=128)
    device = Column(String, nullable=True)

    save_dir = Column(String, nullable=True)
    save_artifacts = Column(Boolean, default=False)
    early_stop_json = Column(JSON, nullable=True)

    best_val_acc = Column(Float, nullable=True)
    result_json = Column(JSON, nullable=True)
    error = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    epochs_rel = relationship("EpochMetric", back_populates="run", cascade="all, delete-orphan")
    artifacts = relationship("Artifact", back_populates="run", cascade="all, delete-orphan")

class EpochMetric(Base):
    __tablename__ = "epochs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String, ForeignKey("runs.id", ondelete="CASCADE"), index=True)
    epoch = Column(Integer)

    time_sec = Column(Float)
    train_acc = Column(Float)
    train_loss_ema = Column(Float)
    val_acc = Column(Float)
    val_f1_macro = Column(Float)
    lr = Column(Float)
    gates_json = Column(JSON)

    created_at = Column(DateTime, default=datetime.utcnow)

    run = relationship("Run", back_populates="epochs_rel")

class Artifact(Base):
    __tablename__ = "artifacts"
    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String, ForeignKey("runs.id", ondelete="CASCADE"), index=True)
    kind = Column(String)   # e.g., 'report', 'checkpoint', 'nsys', 'ncu', 'csv'
    path = Column(String)
    bytes = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    run = relationship("Run", back_populates="artifacts")

def init_db() -> None:
    Base.metadata.create_all(engine)

@contextmanager
def get_session():
    s = SessionLocal()
    try:
        yield s
        s.commit()
    except:
        s.rollback()
        raise
    finally:
        s.close()

# ----------------- convenience helpers -----------------
def create_run(
    s,
    *,
    ablation: Optional[str],
    overrides: Optional[Dict[str, Any]],
    epochs: int,
    batch_size: int,
    max_len: int,
    device: Optional[str],
    save_dir: Optional[str],
    save_artifacts: bool,
    early_stop: Optional[Dict[str, Any]],
) -> Run:
    run = Run(
        ablation=ablation,
        overrides_json=overrides or {},
        epochs=epochs,
        batch_size=batch_size,
        max_len=max_len,
        device=device,
        save_dir=save_dir,
        save_artifacts=save_artifacts,
        early_stop_json=early_stop or {},
        status="QUEUED",
    )
    s.add(run)
    s.flush()  # alloc id
    return run

def set_task_id(s, run_id: str, task_id: str) -> None:
    r = s.get(Run, run_id)
    if r:
        r.task_id = task_id
        r.status = "QUEUED"

def update_status(s, run_id: str, status: str, error: Optional[str] = None) -> None:
    r = s.get(Run, run_id)
    if r:
        r.status = status
        if error:
            r.error = error

def append_epoch(s, run_id: str, e: Dict[str, Any]) -> None:
    row = EpochMetric(
        run_id=run_id,
        epoch=int(e.get("epoch")),
        time_sec=e.get("time_sec"),
        train_acc=e.get("train_acc"),
        train_loss_ema=e.get("train_loss_ema"),
        val_acc=e.get("val_acc"),
        val_f1_macro=e.get("val_f1_macro"),
        lr=e.get("lr"),
        gates_json=e.get("gates"),
    )
    s.add(row)

def complete_run(s, run_id: str, *, result_json: Dict[str, Any], best_val_acc: Optional[float]) -> None:
    r = s.get(Run, run_id)
    if r:
        r.status = "COMPLETE"
        r.result_json = result_json
        r.best_val_acc = best_val_acc

def add_artifact(s, run_id: str, kind: str, path: str, bytes: Optional[int] = None) -> None:
    s.add(Artifact(run_id=run_id, kind=kind, path=path, bytes=bytes))

def list_runs(s, limit: int = 50, status: Optional[str] = None) -> Iterable[Run]:
    q = s.query(Run).order_by(Run.created_at.desc())
    if status:
        q = q.filter(Run.status == status)
    return q.limit(limit).all()

def get_run_row(s, run_id: str) -> Optional[Run]:
    return s.get(Run, run_id)

# ---------- simple serializers for API ----------
def serialize_run(run: Run, with_children: bool = False) -> Dict[str, Any]:
    d = {
        "run_id": run.id,
        "task_id": run.task_id,
        "status": run.status,
        "ablation": run.ablation,
        "overrides": run.overrides_json,
        "epochs": run.epochs,
        "batch_size": run.batch_size,
        "max_len": run.max_len,
        "device": run.device,
        "save_dir": run.save_dir,
        "save_artifacts": run.save_artifacts,
        "early_stop": run.early_stop_json,
        "best_val_acc": run.best_val_acc,
        "created_at": run.created_at.isoformat() if run.created_at else None,
        "updated_at": run.updated_at.isoformat() if run.updated_at else None,
        "error": run.error,
    }
    if with_children:
        d["epochs_log"] = [
            {
                "epoch": e.epoch,
                "time_sec": e.time_sec,
                "train_acc": e.train_acc,
                "train_loss_ema": e.train_loss_ema,
                "val_acc": e.val_acc,
                "val_f1_macro": e.val_f1_macro,
                "lr": e.lr,
                "gates": e.gates_json,
                "created_at": e.created_at.isoformat() if e.created_at else None,
            }
            for e in run.epochs_rel
        ]
        d["artifacts"] = [{"kind": a.kind, "path": a.path, "bytes": a.bytes} for a in run.artifacts]
        d["result"] = run.result_json
    return d
