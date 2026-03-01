"""
Active Learning Feedback Loop.

Stores curator approve/reject decisions in SQLite and computes penalty
adjustments for recommendation types that are consistently rejected,
enabling the system to learn curator preferences over time.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

_REPO_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "feedback.db"


def _resolve_db_path() -> Path:
    """Return a writable path for the SQLite database.

    On Streamlit Cloud the repo is mounted read-only at /mount/src/.
    We try the repo path first; if it's not writable we fall back to a
    temp directory so the app never crashes on a write."""
    import tempfile, os

    # Fast check: if the repo db already exists and is writable, use it.
    if _REPO_DB_PATH.exists():
        try:
            _REPO_DB_PATH.touch()
            return _REPO_DB_PATH
        except OSError:
            pass

    # Try creating the parent directory (works locally, fails on Cloud).
    try:
        _REPO_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        _REPO_DB_PATH.touch()
        return _REPO_DB_PATH
    except OSError:
        pass

    # Fallback: writable temp directory (ephemeral on Cloud, fine for demo).
    tmp = Path(tempfile.gettempdir()) / "pulse_data"
    tmp.mkdir(parents=True, exist_ok=True)
    return tmp / "feedback.db"


DB_PATH = _resolve_db_path()


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS human_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            persona_tier TEXT NOT NULL,
            signal TEXT NOT NULL,
            product_code TEXT NOT NULL,
            confidence REAL NOT NULL,
            governance_tier TEXT,
            action TEXT NOT NULL CHECK (action IN ('approved', 'rejected', 'pending')),
            reason TEXT,
            macro_reasons TEXT,
            timestamp TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn


def record_feedback(
    user_id: str,
    persona_tier: str,
    signal: str,
    product_code: str,
    confidence: float,
    governance_tier: str,
    action: str,
    reason: str = "",
    macro_reasons: str = "",
) -> None:
    """Insert a curator decision into the feedback table."""
    conn = _get_conn()
    conn.execute(
        """INSERT INTO human_feedback
           (user_id, persona_tier, signal, product_code, confidence,
            governance_tier, action, reason, macro_reasons, timestamp)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            user_id, persona_tier, signal, product_code, confidence,
            governance_tier, action, reason, macro_reasons,
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()
    conn.close()


def get_rejection_rates() -> dict[str, dict]:
    """
    Compute rejection rate by (persona_tier, signal, product_code).

    Returns dict keyed by "persona_tier:signal:product_code" with
    {total, approved, rejected, rejection_rate}.
    """
    conn = _get_conn()
    rows = conn.execute("""
        SELECT persona_tier, signal, product_code, action, COUNT(*) as cnt
        FROM human_feedback
        WHERE action IN ('approved', 'rejected')
        GROUP BY persona_tier, signal, product_code, action
    """).fetchall()
    conn.close()

    agg: dict[str, dict] = {}
    for persona, signal, product, action, cnt in rows:
        key = f"{persona}:{signal}:{product}"
        if key not in agg:
            agg[key] = {"total": 0, "approved": 0, "rejected": 0, "rejection_rate": 0.0}
        agg[key][action] = cnt
        agg[key]["total"] += cnt

    for v in agg.values():
        v["rejection_rate"] = round(v["rejected"] / max(v["total"], 1), 3)

    return agg


def apply_feedback_penalty(
    confidence: float,
    persona_tier: str,
    signal: str,
    product_code: str,
    rejection_threshold: float = 0.60,
    max_penalty: float = 0.15,
) -> tuple[float, str | None]:
    """
    Reduce confidence if curators consistently reject this recommendation type.

    If rejection rate for this (persona, signal, product) exceeds `rejection_threshold`,
    apply a proportional penalty up to `max_penalty`.

    Returns (adjusted_confidence, feedback_reason_or_None).
    """
    rates = get_rejection_rates()
    key = f"{persona_tier}:{signal}:{product_code}"
    info = rates.get(key)

    if info is None or info["total"] < 3:
        return confidence, None

    if info["rejection_rate"] > rejection_threshold:
        excess = info["rejection_rate"] - rejection_threshold
        penalty = min(excess * 0.5, max_penalty)
        adjusted = max(confidence - penalty, 0.0)
        reason = (
            f"Active learning: curators rejected {info['rejection_rate']:.0%} of "
            f"similar recommendations ({info['rejected']}/{info['total']}). "
            f"Confidence reduced by {penalty:.2%}."
        )
        return round(adjusted, 4), reason

    return confidence, None


def get_feedback_stats() -> dict:
    """Return overall feedback statistics for the UI."""
    conn = _get_conn()
    total = conn.execute("SELECT COUNT(*) FROM human_feedback").fetchone()[0]
    approved = conn.execute("SELECT COUNT(*) FROM human_feedback WHERE action='approved'").fetchone()[0]
    rejected = conn.execute("SELECT COUNT(*) FROM human_feedback WHERE action='rejected'").fetchone()[0]
    pending = conn.execute("SELECT COUNT(*) FROM human_feedback WHERE action='pending'").fetchone()[0]
    conn.close()
    return {
        "total": total,
        "approved": approved,
        "rejected": rejected,
        "pending": pending,
        "approval_rate": round(approved / max(approved + rejected, 1), 3),
    }
