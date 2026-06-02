"""DCOL secure surveillance backend with PostgreSQL persistence and shared camera stream."""

import json
import os
import secrets
import threading
import time
from functools import wraps

import cv2
from flask import Flask, Response, jsonify, make_response, request, send_from_directory, stream_with_context
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.security import check_password_hash, generate_password_hash

import config
from input_validator import validate_login_payload
from rate_limiter import clear_failed_logins, get_blocked_ips, is_allowed, record_failed_login
from security_headers import init_security_headers
from security_logger import Event, security_logger


def _camera_source():
    source = config.CAMERA_SOURCE
    return int(source) if str(source).isdigit() else source


def _camera_backend():
    if config.CAMERA_BACKEND == "CAP_DSHOW" and os.name == "nt":
        return cv2.CAP_DSHOW
    return cv2.CAP_ANY


def _client_ip() -> str:
    return request.headers.get("X-Forwarded-For", request.remote_addr or "").split(",")[0].strip()


def _session_token() -> str:
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        return auth_header.removeprefix("Bearer ").strip()
    return request.cookies.get(config.SESSION_COOKIE_NAME, "")


def _is_allowed_origin() -> bool:
    origin = request.headers.get("Origin")
    return not origin or origin in config.ALLOWED_ORIGINS


app = Flask(__name__, static_folder=None)
app.secret_key = config.SECRET_KEY or os.urandom(32)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

if config.ALLOWED_ORIGINS:
    CORS(app, origins=config.ALLOWED_ORIGINS, supports_credentials=True)

init_security_headers(app)

# Database
db_url = config.DATABASE_URL
if db_url and db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)
app.config["SQLALCHEMY_DATABASE_URI"] = db_url or "sqlite:///dcol_local.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)


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


class SharedCamera:
    """One capture thread shared by all viewers. Prevents each user opening the camera."""

    def __init__(self):
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._frame: bytes | None = None
        self._running = False
        self._thread: threading.Thread | None = None
        self._last_frame_at = 0.0

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True
            self._thread = threading.Thread(target=self._capture_loop, daemon=True)
            self._thread.start()

    def _open_capture(self):
        cap = cv2.VideoCapture(_camera_source(), _camera_backend())
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        return cap

    def _capture_loop(self) -> None:
        cap = None
        while True:
            if cap is None or not cap.isOpened():
                if cap is not None:
                    cap.release()
                cap = self._open_capture()
                time.sleep(0.2)

            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.25)
                if cap is not None:
                    cap.release()
                cap = None
                continue

            frame = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))
            ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY])
            if not ok:
                continue

            with self._condition:
                self._frame = buffer.tobytes()
                self._last_frame_at = time.time()
                self._condition.notify_all()

    def frames(self):
        self.start()
        last_sent = 0.0
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._last_frame_at > last_sent, timeout=10)
                if self._frame is None:
                    continue
                frame = self._frame
                last_sent = self._last_frame_at
            yield b"--frame\r\nContent-Type: image/jpeg\r\nCache-Control: no-store\r\n\r\n" + frame + b"\r\n"


camera = SharedCamera()


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


def validate_db_session(token: str, client_ip: str | None = None) -> DBSession | None:
    if not token:
        return None
    sess = DBSession.query.filter_by(token=token, revoked=False).first()
    if not sess:
        return None
    if (time.time() - sess.created_at) > config.SESSION_TTL:
        sess.revoked = True
        db.session.commit()
        return None
    if config.SESSION_BIND_IP and client_ip and sess.ip != client_ip:
        sess.revoked = True
        db.session.commit()
        log_security_event(Event.TOKEN_HIJACK, client_ip, username=sess.username)
        return None
    sess.last_seen = time.time()
    db.session.commit()
    return sess


def require_auth(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        ip = _client_ip()
        allowed, reason = is_allowed(ip)
        if not allowed:
            log_security_event(Event.RATE_LIMIT, ip, extra={"status": reason})
            return jsonify({"success": False, "message": reason}), 429
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
    # Local fallback only, so the app is testable before Render env vars exist.
    return secrets.compare_digest(password, "admin")


@app.route("/")
def index():
    return send_from_directory(os.path.dirname(__file__), "frontend.html")


@app.post("/api/login")
def login():
    ip = _client_ip()
    if not _is_allowed_origin():
        return jsonify({"success": False, "message": "Origin not allowed."}), 403
    allowed, reason = is_allowed(ip)
    if not allowed:
        log_security_event(Event.RATE_LIMIT, ip, extra={"status": reason})
        return jsonify({"success": False, "message": reason}), 429

    data = request.get_json(silent=True) or {}
    username, password, threat = validate_login_payload(data)
    if threat:
        log_security_event(threat, ip, username=username)
        return jsonify({"success": False, "message": "Invalid login payload."}), 400

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
    def event_stream():
        while True:
            payload = {
                "logs": json.loads(security_logs().get_data(as_text=True)),
                "blocked": get_blocked_ips(),
                "sessions": json.loads(active_sessions().get_data(as_text=True)),
            }
            yield f"data: {json.dumps(payload)}\n\n"
            time.sleep(1)
    return Response(stream_with_context(event_stream()), mimetype="text/event-stream")


@app.get("/video_feed")
@require_auth
def video_feed():
    log_security_event(Event.STREAM_ACCESS, _client_ip(), username=request.current_session.username)
    return Response(camera.frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


with app.app_context():
    db.create_all()


if __name__ == "__main__":
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG, use_reloader=config.USE_RELOADER, threaded=True)
