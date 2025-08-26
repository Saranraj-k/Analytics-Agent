import io
import uuid
import pandas as pd
from werkzeug.datastructures import FileStorage

_SESSIONS = {}

def new_session_id() -> str:
    return uuid.uuid4().hex

def set_session(store: dict):
    sid = new_session_id()
    _SESSIONS[sid] = store
    return sid

def get_session(sid: str) -> dict:
    return _SESSIONS.get(sid, {})

def update_session(sid: str, **kwargs):
    if sid not in _SESSIONS:
        _SESSIONS[sid] = {}
    _SESSIONS[sid].update(kwargs)

def load_csv_to_df(file: FileStorage) -> pd.DataFrame:
    """Read uploaded CSV (utf-8 or fallback) into DataFrame."""
    raw = file.read()
    try:
        return pd.read_csv(io.BytesIO(raw))
    except UnicodeDecodeError:
        return pd.read_csv(io.BytesIO(raw), encoding='latin-1')