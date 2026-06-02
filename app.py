"""DCOL secure surveillance backend with PostgreSQL persistence and Render-ready camera proxy."""

import json
import os
import secrets
import time
from functools import wraps
from urllib.parse import urlparse

import cv2
import requests
from flask import Flask, Response, jsonify, make_response, request, send_from_directory, stream_with_context
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.security import check_password_hash

import config
from input_validator import validate_login_payload
from rate_limiter import block_ip, clear_failed_logins, get_blocked_ips, is_allowed, record_failed_login
from security_headers import init_security_headers
from security_logger import Event, security_logger


# ── Helpers ───────────────────────────────────────────────────────────────────

def _camera_source():
    source = (config.CAMERA_SOURCE or "").strip()
    return int(source) if source.isdigit() else source


def _camera_backend():
    if config.CAMERA_BACKEND == "CAP_DSHOW" and os.name == "nt":
        return cv2.CAP_DSHOW
    return cv2.CAP_ANY


def _client_ip() -> str:
    forwarded = request.headers.get("X-Forwarded-For", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.remote_addr or ""


def _session_token() -> str:
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        return auth_header.removeprefix("Bearer ").strip()
    return request.cookies.get(config.SESSION_COOKIE_NAME, "")


def _is_allowed_origin() -> bool:
    origin = request.headers.get("Origin")
    if not origin:
        return True

    current_origin = request.host_url.rstrip("/")
    if origin == current_origin:
        return True

    return origin in config.ALLOWED_ORIGINS


def _is_remote_camera_source() -> bool:
    source = str(config.CAMERA_SOURCE or "")
    return urlparse(source).scheme in {"http", "https", "rtsp", "rtmp"}


# ── App setup ─────────────────────────────────────────────────────────────────

app = Flask(__name__, static_folder=None)
app.secret_key = config.SECRET_KEY or os.urandom(32)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

if config.ALLOWED_ORIGINS:
    CORS(app, origins=config.ALLOWED_ORIGINS, supports_credentials=True)

init_security_headers(app)

db_url = config.DATABASE_URL
if db_url and db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)

app.config["SQLALCHEMY_DATABASE_URI"] = db_url or "sqlite:///dcol_local.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)


# ── Database models ───────────────────────────────────────────────────────────

class DBLog(db.Model):
    __tablename__ = "security_events"
    id = db.Column(db.Integer, primary_key=True)
    time = db.Column(db.String(20), nullable=False)
    timestamp = db.Column(db.String(50), nullable=False)
    event = db.Column(db.String(50), nullable=False)
    ip = db.Column(db.String(45), nullable=False)
    username = db.Column(db.String(50), default="")
    status = db.Column(db.String(100), nullable=False)
    is_threat = db.Column(db.Boolean, default=False)


class DBSession(db.Model):
    __tablename__ = "active_sessions"
    id = db.Column(db.Integer, primary_key=True)
    token = db.Column(db.String(128), unique=True, nullable=False, index=True)
    ip = db.Column(db.String(45), nullable=False)
    username = db.Column(db.String(50), nullable=False)
    created_at = db.Column(db.Float, nullable=False, default=time.time)
    last_seen = db.Column(db.Float, nullable=False, default=time.time)
    revoked = db.Column(db.Boolean, default=False)


# ── Logging/session functions ─────────────────────────────────────────────────

def log_security_event(event_type: str, ip: str, username: str | None = None, extra: dict | None = None):
    entry = security_logger.log(event_type, ip, username=username, extra=extra)
    try:
        db.session.add(DBLog(
            time=entry.get("time", ""),
            timestamp=entry.get("timestamp", ""),
            event=entry.get("event", ""),
            ip=entry.get("ip", ""),
            username=entry.get("username", ""),
            status=entry.get("status", ""),
            is_threat=entry.get("is_threat", False),
        ))
        db.session.commit()
    except Exception as exc:
        print(f"[DB ERROR] Failed to record security event: {exc}", flush=True)
        db.session.rollback()
    return entry


def cleanup_sessions() -> None:
    cutoff = time.time() - config.SESSION_TTL
    DBSession.query.filter((DBSession.created_at < cutoff) | (DBSession.revoked.is_(True))).delete(synchronize_session=False)
    db.session.commit()

def revoke_sessions_for_ip(ip: str) -> int:
    count = DBSession.query.filter_by(ip=ip, revoked=False).update({"revoked": True})
    db.session.commit()
    return count

def validate_db_session(token: str, client_ip: str | None = None, touch: bool = True) -> DBSession | None:
    if not token:
        return None

    sess = DBSession.query.filter_by(token=token, revoked=False).first()
    if not sess:
        return None

    now = time.time()
    if (now - sess.created_at) > config.SESSION_TTL:
        sess.revoked = True
        db.session.commit()
        return None

    if config.SESSION_BIND_IP and client_ip and sess.ip != client_ip:
        sess.revoked = True
        db.session.commit()
        log_security_event(Event.TOKEN_HIJACK, client_ip, username=sess.username)
        return None

    if touch:
        sess.last_seen = now
        db.session.commit()

    return sess

def require_auth(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        ip = _client_ip()

        if ip in get_blocked_ips():
            revoke_sessions_for_ip(ip)
            return jsonify({"success": False, "message": "IP blocked."}), 403

        sess = validate_db_session(_session_token(), ip)
        if not sess:
            if request.path == "/video_feed":
                log_security_event(Event.STREAM_DENIED, ip)
            return jsonify({"success": False, "message": "Login required."}), 401

        request.current_session = sess
        return fn(*args, **kwargs)
    return wrapper

def _password_ok(password: str) -> bool:
    if config.ADMIN_PASSWORD_HASH:
        return check_password_hash(config.ADMIN_PASSWORD_HASH, password)
    if config.ADMIN_PASSWORD:
        return secrets.compare_digest(config.ADMIN_PASSWORD, password)
    return secrets.compare_digest(password, "admin")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(os.path.dirname(__file__), "frontend.html")


@app.get("/api/health")
def health():
    db_ok = False
    try:
        db.session.execute(db.text("SELECT 1"))
        db_ok = True
    except Exception:
        db.session.rollback()

    source = str(config.CAMERA_SOURCE or "").strip()
    return jsonify({
        "status": "ok",
        "database": db_ok,
        "camera_source_set": bool(source),
        "camera_source_type": "remote" if _is_remote_camera_source() else "local",
        "camera_backend": config.CAMERA_BACKEND,
    })


@app.post("/api/login")
def login():
    ip = _client_ip()
    if not _is_allowed_origin():
        return jsonify({"success": False, "message": "Origin not allowed."}), 403

    # Keep rate limiting only on login to prevent brute-force attacks.
    allowed, reason = is_allowed(ip)
    if not allowed:
        log_security_event(Event.RATE_LIMIT, ip, extra={"status": reason})
        return jsonify({"success": False, "message": reason}), 429

    data = request.get_json(silent=True) or {}
    username, password, threat = validate_login_payload(data)
    if threat:
        block_ip(ip, config.BLOCK_DURATION)
        revoke_sessions_for_ip(ip)
        log_security_event(
            threat,
            ip,
            username=username,
            extra={"status": "Instant IP ban for injection attempt"}
    )
        return jsonify({"success": False, "message": "Threat detected. IP blocked."}), 403

    if username != config.ADMIN_USERNAME or not _password_ok(password):
        blocked, count = record_failed_login(ip)
        log_security_event(Event.BRUTE_FORCE if blocked else Event.LOGIN_FAILURE, ip, username=username)
        message = "Too many failed attempts. Temporarily blocked." if blocked else f"Invalid credentials. Attempt {count}."
        return jsonify({"success": False, "message": message}), 401

    clear_failed_logins(ip)
    cleanup_sessions()

    active_count = DBSession.query.filter_by(revoked=False).count()
    if active_count >= config.MAX_SESSIONS:
        return jsonify({"success": False, "message": "Maximum active sessions reached."}), 429

    token = secrets.token_hex(config.TOKEN_BYTES)
    db.session.add(DBSession(token=token, ip=ip, username=username))
    db.session.commit()
    log_security_event(Event.LOGIN_SUCCESS, ip, username=username)

    resp = make_response(jsonify({"success": True, "username": username}))
    resp.set_cookie(
        config.SESSION_COOKIE_NAME,
        token,
        max_age=config.SESSION_TTL,
        httponly=True,
        secure=config.SESSION_COOKIE_SECURE,
        samesite="Lax",
    )
    return resp


@app.post("/api/logout")
@require_auth
def logout():
    token = _session_token()
    DBSession.query.filter_by(token=token).update({"revoked": True})
    db.session.commit()

    log_security_event(Event.SESSION_REVOKED, _client_ip(), username=request.current_session.username)

    resp = make_response(jsonify({"success": True}))
    resp.delete_cookie(config.SESSION_COOKIE_NAME, secure=config.SESSION_COOKIE_SECURE, samesite="Lax")
    return resp


@app.get("/api/me")
@require_auth
def me():
    return jsonify({"success": True, "username": request.current_session.username})


@app.get("/api/security_logs")
@require_auth
def security_logs():
    rows = DBLog.query.order_by(DBLog.id.desc()).limit(20).all()
    return jsonify([{
        "time": r.time,
        "timestamp": r.timestamp,
        "event": r.event,
        "ip": r.ip,
        "username": r.username,
        "status": r.status,
        "is_threat": r.is_threat,
    } for r in rows])


@app.get("/api/blocked_ips")
@require_auth
def blocked_ips():
    return jsonify(get_blocked_ips())


@app.get("/api/active_sessions")
@require_auth
def active_sessions():
    cleanup_sessions()
    rows = DBSession.query.filter_by(revoked=False).order_by(DBSession.last_seen.desc()).all()
    now = time.time()
    return jsonify([{
        "token_prefix": r.token[:8] + "...",
        "ip": r.ip,
        "username": r.username,
        "age_seconds": int(now - r.created_at),
        "last_seen_seconds": int(now - r.last_seen),
    } for r in rows if (now - r.created_at) <= config.SESSION_TTL])


@app.get("/api/events")
@require_auth
def events():
    @stream_with_context
    def event_stream():
        while True:
            now = time.time()
            cleanup_sessions()

            log_rows = DBLog.query.order_by(DBLog.id.desc()).limit(20).all()
            session_rows = DBSession.query.filter_by(revoked=False).order_by(DBSession.last_seen.desc()).all()

            payload = {
                "logs": [{
                    "time": r.time,
                    "timestamp": r.timestamp,
                    "event": r.event,
                    "ip": r.ip,
                    "username": r.username,
                    "status": r.status,
                    "is_threat": r.is_threat,
                } for r in log_rows],
                "blocked": get_blocked_ips(),
                "sessions": [{
                    "token_prefix": r.token[:8] + "...",
                    "ip": r.ip,
                    "username": r.username,
                    "age_seconds": int(now - r.created_at),
                    "last_seen_seconds": int(now - r.last_seen),
                } for r in session_rows if (now - r.created_at) <= config.SESSION_TTL],
            }

            yield f"data: {json.dumps(payload)}\n\n"
            time.sleep(1)

    return Response(event_stream(), mimetype="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    })


def _proxy_remote_camera():
    """Proxy a remote Cloudflare/MJPEG stream without OpenCV decoding."""
    source = str(config.CAMERA_SOURCE or "").strip()
    if not source:
        def error_stream():
            yield (
                b"--frame\r\n"
                b"Content-Type: text/plain\r\n\r\n"
                b"DCOL_CAMERA_SOURCE is not set.\r\n"
            )
        return Response(error_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")

    upstream = requests.get(
        source,
        stream=True,
        timeout=(10, None),
        headers={
            "User-Agent": "DCOL-Monitor/1.0",
            "Accept": "multipart/x-mixed-replace,image/jpeg,*/*",
        },
    )

    content_type = upstream.headers.get("Content-Type", "multipart/x-mixed-replace; boundary=frame")

    def generate():
        try:
            for chunk in upstream.iter_content(chunk_size=8192):
                if chunk:
                    yield chunk
        finally:
            upstream.close()

    return Response(stream_with_context(generate()), mimetype=content_type, headers={"Cache-Control": "no-store"})


def _opencv_camera_frames():
    """Fallback for local webcams or RTSP streams that OpenCV can decode."""
    cap = cv2.VideoCapture(_camera_source(), _camera_backend())
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), config.JPEG_QUALITY]

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.25)
                continue

            frame = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))
            ok, buffer = cv2.imencode(".jpg", frame, encode_params)
            if not ok:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )
    finally:
        cap.release()


@app.get("/video_feed")
@require_auth
def video_feed():
    log_security_event(Event.STREAM_ACCESS, _client_ip(), username=request.current_session.username)

    source = config.CAMERA_SOURCE
    if not source:
        return jsonify({"success": False, "message": "DCOL_CAMERA_SOURCE is not set."}), 500

    upstream = requests.get(source, stream=True, timeout=(10, None))
    return Response(
        upstream.iter_content(chunk_size=8192),
        content_type=upstream.headers.get("Content-Type", "multipart/x-mixed-replace; boundary=frame"),
        headers={"Cache-Control": "no-store"},
    )   


with app.app_context():
    db.create_all()


if __name__ == "__main__":
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG, use_reloader=config.USE_RELOADER, threaded=True)
