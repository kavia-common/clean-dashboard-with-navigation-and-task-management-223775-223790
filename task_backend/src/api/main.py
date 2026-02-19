import os
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Literal, Optional, Set
from uuid import UUID

import jwt
from fastapi import (
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Response,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError

# -----------------------------------------------------------------------------
# App metadata (OpenAPI)
# -----------------------------------------------------------------------------

openapi_tags = [
    {
        "name": "Health",
        "description": "Service healthcheck and diagnostic endpoints.",
    },
    {
        "name": "Auth",
        "description": "User registration, login, and identity endpoints (JWT Bearer).",
    },
    {
        "name": "Tasks",
        "description": "Task CRUD endpoints scoped to the authenticated user.",
    },
    {
        "name": "Realtime",
        "description": "WebSocket endpoints broadcasting task changes per-user.",
    },
]

# -----------------------------------------------------------------------------
# Environment variables
# -----------------------------------------------------------------------------
# NOTE: These must be provided via task_backend/.env by the orchestrator.
#
# Required:
# - DATABASE_URL: SQLAlchemy/psycopg2 DSN, e.g. postgresql://appuser:dbuser123@localhost:5001/myapp
# - JWT_SECRET: secret used to sign JWT access tokens
#
# Optional:
# - JWT_ALG (default: HS256)
# - JWT_EXPIRES_MINUTES (default: 60)
# - CORS_ORIGINS: comma-separated allowed origins (default allows localhost:3000 + '*' fallback)
# - ACCESS_TOKEN_COOKIE_NAME (default: access_token) (frontend can also send Authorization header)
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    # Safe default for preview based on task_db/db_connection.txt. Prefer setting explicitly.
    "postgresql://appuser:dbuser123@localhost:5001/myapp",
)
JWT_SECRET = os.getenv("JWT_SECRET", "")
JWT_ALG = os.getenv("JWT_ALG", "HS256")
JWT_EXPIRES_MINUTES = int(os.getenv("JWT_EXPIRES_MINUTES", "60"))
ACCESS_TOKEN_COOKIE_NAME = os.getenv("ACCESS_TOKEN_COOKIE_NAME", "access_token")

# CORS: Keep credentials allowed for cookie-based auth from Next.js.
cors_origins_env = os.getenv("CORS_ORIGINS", "")
if cors_origins_env.strip():
    ALLOW_ORIGINS = [o.strip() for o in cors_origins_env.split(",") if o.strip()]
else:
    # Default: allow local dev frontend, plus permissive fallback to avoid blocking preview.
    ALLOW_ORIGINS = ["http://localhost:3000", "https://localhost:3000", "*"]

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# -----------------------------------------------------------------------------
# Database
# -----------------------------------------------------------------------------

def _create_engine() -> Engine:
    """
    Create a SQLAlchemy engine.

    We use the synchronous engine for simplicity/reliability in this template.
    """
    return create_engine(
        DATABASE_URL,
        pool_pre_ping=True,
        future=True,
    )


engine: Engine = _create_engine()


def _db_healthcheck() -> None:
    """Raises if DB is not reachable or schema doesn't exist."""
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))


# -----------------------------------------------------------------------------
# Security / JWT helpers
# -----------------------------------------------------------------------------

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _create_access_token(*, user_id: UUID, email: str) -> str:
    if not JWT_SECRET:
        raise RuntimeError("JWT_SECRET is not set; cannot mint tokens.")

    exp = _utcnow() + timedelta(minutes=JWT_EXPIRES_MINUTES)
    payload = {
        "sub": str(user_id),
        "email": email,
        "exp": exp,
        "iat": _utcnow(),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)


def _verify_access_token(token: str) -> Dict[str, Any]:
    if not JWT_SECRET:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="JWT_SECRET is not configured on the server.",
        )
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
        return payload
    except jwt.ExpiredSignatureError as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired") from e
    except jwt.InvalidTokenError as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token") from e


def _hash_password(password: str) -> str:
    return pwd_context.hash(password)


def _verify_password(password: str, password_hash: str) -> bool:
    return pwd_context.verify(password, password_hash)


def _extract_bearer_token(authorization: Optional[str]) -> Optional[str]:
    if not authorization:
        return None
    parts = authorization.split(" ", 1)
    if len(parts) != 2:
        return None
    if parts[0].lower() != "bearer":
        return None
    return parts[1].strip() or None


# -----------------------------------------------------------------------------
# Pydantic models
# -----------------------------------------------------------------------------

class ErrorResponse(BaseModel):
    """Standard error response body."""
    detail: str = Field(..., description="Human-readable error description.")


class RegisterRequest(BaseModel):
    email: EmailStr = Field(..., description="Unique user email used for login.")
    password: str = Field(..., min_length=8, description="Password (min 8 chars).")
    display_name: Optional[str] = Field(None, description="Optional display name.")


class LoginRequest(BaseModel):
    email: EmailStr = Field(..., description="User email.")
    password: str = Field(..., description="User password.")


class AuthResponse(BaseModel):
    access_token: str = Field(..., description="JWT access token (Bearer).")
    token_type: Literal["bearer"] = Field("bearer", description="Token type.")
    user: "UserMeResponse" = Field(..., description="Authenticated user payload.")


class UserMeResponse(BaseModel):
    id: UUID = Field(..., description="User id.")
    email: EmailStr = Field(..., description="User email.")
    display_name: Optional[str] = Field(None, description="User display name.")
    created_at: datetime = Field(..., description="Creation timestamp.")
    updated_at: datetime = Field(..., description="Last update timestamp.")


class TaskStatus(str):
    """Task status values."""
    # kept for typing readability; enforced in DB via CHECK constraint


class TaskResponse(BaseModel):
    id: UUID = Field(..., description="Task id.")
    owner_id: UUID = Field(..., description="Owner user id.")
    title: str = Field(..., description="Task title.")
    description: Optional[str] = Field(None, description="Task description.")
    status: Literal["todo", "in_progress", "done"] = Field(..., description="Task status.")
    due_date: Optional[date] = Field(None, description="Optional due date.")
    created_at: datetime = Field(..., description="Creation timestamp.")
    updated_at: datetime = Field(..., description="Last update timestamp.")


class TaskCreateRequest(BaseModel):
    title: str = Field(..., min_length=1, description="Task title.")
    description: Optional[str] = Field(None, description="Optional description.")
    status: Optional[Literal["todo", "in_progress", "done"]] = Field(
        "todo",
        description="Initial status.",
    )
    due_date: Optional[date] = Field(None, description="Optional due date.")


class TaskUpdateRequest(BaseModel):
    title: Optional[str] = Field(None, min_length=1, description="Updated title.")
    description: Optional[str] = Field(None, description="Updated description (nullable).")
    status: Optional[Literal["todo", "in_progress", "done"]] = Field(None, description="Updated status.")
    due_date: Optional[date] = Field(None, description="Updated due date (nullable).")


class WsTaskEvent(BaseModel):
    """WebSocket payload for task change events."""
    type: Literal["task_created", "task_updated", "task_deleted"] = Field(..., description="Event type.")
    task: Optional[TaskResponse] = Field(None, description="Task payload (null for deletes).")
    task_id: Optional[UUID] = Field(None, description="Task id (for deletes).")
    ts: datetime = Field(..., description="Event timestamp (server time, UTC).")


AuthResponse.model_rebuild()


# -----------------------------------------------------------------------------
# WebSocket connection manager (per-user broadcasting)
# -----------------------------------------------------------------------------

class ConnectionManager:
    """In-memory WebSocket connection manager keyed by user_id."""
    def __init__(self) -> None:
        self._connections: Dict[UUID, Set[WebSocket]] = {}

    async def connect(self, user_id: UUID, websocket: WebSocket) -> None:
        await websocket.accept()
        self._connections.setdefault(user_id, set()).add(websocket)

    def disconnect(self, user_id: UUID, websocket: WebSocket) -> None:
        conns = self._connections.get(user_id)
        if not conns:
            return
        conns.discard(websocket)
        if not conns:
            self._connections.pop(user_id, None)

    async def broadcast_to_user(self, user_id: UUID, message: Dict[str, Any]) -> None:
        conns = list(self._connections.get(user_id, set()))
        if not conns:
            return
        dead: List[WebSocket] = []
        for ws in conns:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(user_id, ws)


ws_manager = ConnectionManager()


# -----------------------------------------------------------------------------
# FastAPI app
# -----------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(_: FastAPI):
    # Startup: verify DB connectivity (but do not hard-fail the entire service in preview).
    try:
        _db_healthcheck()
    except Exception as e:
        # Keep running; health endpoint will report DB issue.
        print(f"[WARN] DB healthcheck failed at startup: {e}")
    yield


app = FastAPI(
    title="Task Backend API",
    description=(
        "Backend for the task management app.\n\n"
        "Auth: JWT Bearer via `Authorization: Bearer <token>` header, or cookie named "
        f"`{ACCESS_TOKEN_COOKIE_NAME}`.\n\n"
        "Realtime: connect to WebSocket `/ws/tasks` with the same auth (preferred: header). "
        "The server broadcasts `task_created`, `task_updated`, `task_deleted` events to the "
        "authenticated user's sockets."
    ),
    version="1.0.0",
    openapi_tags=openapi_tags,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------------
# Dependencies
# -----------------------------------------------------------------------------

def _row_to_user_me(row: Any) -> UserMeResponse:
    return UserMeResponse(
        id=row.id,
        email=row.email,
        display_name=row.display_name,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _row_to_task(row: Any) -> TaskResponse:
    return TaskResponse(
        id=row.id,
        owner_id=row.owner_id,
        title=row.title,
        description=row.description,
        status=row.status,
        due_date=row.due_date,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _get_token_from_request(
    authorization: Optional[str],
    cookie_token: Optional[str],
) -> str:
    token = _extract_bearer_token(authorization) or cookie_token
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing access token")
    return token


# PUBLIC_INTERFACE
def get_current_user(
    authorization: Optional[str] = Header(default=None, alias="Authorization"),
    cookie_token: Optional[str] = Header(default=None, alias="X-Access-Token"),
) -> UserMeResponse:
    """
    Get the current authenticated user.

    The frontend can send:
    - Authorization: Bearer <token>
    - OR X-Access-Token: <token> (fallback convenience)
    - OR cookie named ACCESS_TOKEN_COOKIE_NAME (handled in routes where we have request cookies)

    NOTE: Because FastAPI dependencies cannot directly access cookies unless Request is injected,
    we primarily support Authorization header. Cookie support is implemented per-route where needed.
    """
    token = _get_token_from_request(authorization, cookie_token)
    payload = _verify_access_token(token)
    user_id = UUID(payload["sub"])

    with engine.connect() as conn:
        result = conn.execute(
            text(
                "SELECT id, email, display_name, created_at, updated_at "
                "FROM users WHERE id = :id"
            ),
            {"id": str(user_id)},
        ).mappings().first()

    if not result:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")

    # mappings() gives dict-like; adapt
    class Obj:
        def __init__(self, d: Dict[str, Any]):
            self.__dict__.update(d)

    return _row_to_user_me(Obj(dict(result)))


# -----------------------------------------------------------------------------
# Health + docs helper
# -----------------------------------------------------------------------------

@app.get(
    "/",
    tags=["Health"],
    summary="Health check",
    responses={200: {"description": "Service is up."}},
)
def health_check() -> Dict[str, str]:
    """Basic health check endpoint."""
    return {"message": "Healthy"}


@app.get(
    "/realtime",
    tags=["Realtime"],
    summary="WebSocket usage help",
    response_class=HTMLResponse,
)
def realtime_help() -> str:
    """Human-readable instructions for using the WebSocket endpoint."""
    return """
    <html>
      <body>
        <h2>WebSocket: /ws/tasks</h2>
        <p>Connect with a valid JWT access token.</p>
        <ul>
          <li>Preferred: send HTTP header <code>Authorization: Bearer &lt;token&gt;</code> during WS handshake.</li>
          <li>Alternative: include query param <code>?token=&lt;token&gt;</code> (less secure; avoid in production).</li>
        </ul>
        <p>Events are delivered as JSON:</p>
        <pre>
{"type":"task_created","task":{...},"ts":"2026-01-01T00:00:00Z"}
        </pre>
      </body>
    </html>
    """


# -----------------------------------------------------------------------------
# Auth endpoints
# -----------------------------------------------------------------------------

@app.post(
    "/auth/register",
    tags=["Auth"],
    summary="Register a new user",
    response_model=AuthResponse,
    responses={
        400: {"model": ErrorResponse, "description": "User already exists or invalid input."},
        500: {"model": ErrorResponse, "description": "Server misconfiguration."},
    },
)
def register(payload: RegisterRequest, response: Response) -> AuthResponse:
    """Create a new user and return an access token."""
    if not JWT_SECRET:
        raise HTTPException(status_code=500, detail="Server auth is not configured (JWT_SECRET).")

    password_hash = _hash_password(payload.password)

    try:
        with engine.begin() as conn:
            row = conn.execute(
                text(
                    "INSERT INTO users (email, password_hash, display_name) "
                    "VALUES (:email, :password_hash, :display_name) "
                    "RETURNING id, email, display_name, created_at, updated_at"
                ),
                {
                    "email": str(payload.email).lower(),
                    "password_hash": password_hash,
                    "display_name": payload.display_name,
                },
            ).mappings().first()
    except IntegrityError as e:
        raise HTTPException(status_code=400, detail="Email is already registered.") from e

    class Obj:
        def __init__(self, d: Dict[str, Any]):
            self.__dict__.update(d)

    user = _row_to_user_me(Obj(dict(row)))
    token = _create_access_token(user_id=user.id, email=user.email)

    # Set cookie as a convenience for browser clients. Frontend may still prefer Authorization header.
    response.set_cookie(
        key=ACCESS_TOKEN_COOKIE_NAME,
        value=token,
        httponly=True,
        samesite="lax",
        secure=False,  # preview/dev; set true behind HTTPS termination when applicable
        path="/",
    )

    return AuthResponse(access_token=token, user=user)


@app.post(
    "/auth/login",
    tags=["Auth"],
    summary="Login with email/password",
    response_model=AuthResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Invalid credentials."},
        500: {"model": ErrorResponse, "description": "Server misconfiguration."},
    },
)
def login(payload: LoginRequest, response: Response) -> AuthResponse:
    """Authenticate a user and return an access token."""
    if not JWT_SECRET:
        raise HTTPException(status_code=500, detail="Server auth is not configured (JWT_SECRET).")

    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT id, email, display_name, password_hash, created_at, updated_at "
                "FROM users WHERE email = :email"
            ),
            {"email": str(payload.email).lower()},
        ).mappings().first()

    if not row:
        raise HTTPException(status_code=401, detail="Invalid email or password")

    if not _verify_password(payload.password, row["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    user = UserMeResponse(
        id=row["id"],
        email=row["email"],
        display_name=row["display_name"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )
    token = _create_access_token(user_id=user.id, email=user.email)

    response.set_cookie(
        key=ACCESS_TOKEN_COOKIE_NAME,
        value=token,
        httponly=True,
        samesite="lax",
        secure=False,
        path="/",
    )

    return AuthResponse(access_token=token, user=user)


@app.get(
    "/auth/me",
    tags=["Auth"],
    summary="Get current user",
    response_model=UserMeResponse,
    responses={401: {"model": ErrorResponse, "description": "Not authenticated."}},
)
def me(
    authorization: Optional[str] = Header(default=None, alias="Authorization"),
    x_access_token: Optional[str] = Header(default=None, alias="X-Access-Token"),
    cookie_access_token: Optional[str] = None,
) -> UserMeResponse:
    """
    Return the current authenticated user.

    Accepts token via Authorization header or X-Access-Token header.
    Cookie-based auth is supported indirectly by the frontend sending X-Access-Token.
    """
    # cookie_access_token is kept for forward compatibility (can be wired with Request if needed)
    token = _get_token_from_request(authorization, x_access_token or cookie_access_token)
    payload = _verify_access_token(token)
    user_id = UUID(payload["sub"])

    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT id, email, display_name, created_at, updated_at "
                "FROM users WHERE id = :id"
            ),
            {"id": str(user_id)},
        ).mappings().first()

    if not row:
        raise HTTPException(status_code=401, detail="User not found")

    return UserMeResponse(
        id=row["id"],
        email=row["email"],
        display_name=row["display_name"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


# -----------------------------------------------------------------------------
# Task CRUD (scoped to authenticated user)
# -----------------------------------------------------------------------------

async def _broadcast_task_event(user_id: UUID, event: WsTaskEvent) -> None:
    await ws_manager.broadcast_to_user(user_id, event.model_dump(mode="json"))


@app.get(
    "/tasks",
    tags=["Tasks"],
    summary="List tasks for current user",
    response_model=List[TaskResponse],
)
async def list_tasks(current_user: UserMeResponse = Depends(get_current_user)) -> List[TaskResponse]:
    """Return all tasks owned by the authenticated user."""
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT id, owner_id, title, description, status, due_date, created_at, updated_at "
                "FROM tasks WHERE owner_id = :owner_id "
                "ORDER BY updated_at DESC"
            ),
            {"owner_id": str(current_user.id)},
        ).mappings().all()

    return [
        TaskResponse(
            id=r["id"],
            owner_id=r["owner_id"],
            title=r["title"],
            description=r["description"],
            status=r["status"],
            due_date=r["due_date"],
            created_at=r["created_at"],
            updated_at=r["updated_at"],
        )
        for r in rows
    ]


@app.post(
    "/tasks",
    tags=["Tasks"],
    summary="Create a task for current user",
    response_model=TaskResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_task(
    payload: TaskCreateRequest,
    current_user: UserMeResponse = Depends(get_current_user),
) -> TaskResponse:
    """Create a task owned by the authenticated user."""
    with engine.begin() as conn:
        row = conn.execute(
            text(
                "INSERT INTO tasks (owner_id, title, description, status, due_date) "
                "VALUES (:owner_id, :title, :description, :status, :due_date) "
                "RETURNING id, owner_id, title, description, status, due_date, created_at, updated_at"
            ),
            {
                "owner_id": str(current_user.id),
                "title": payload.title,
                "description": payload.description,
                "status": payload.status or "todo",
                "due_date": payload.due_date,
            },
        ).mappings().first()

    task = TaskResponse(
        id=row["id"],
        owner_id=row["owner_id"],
        title=row["title"],
        description=row["description"],
        status=row["status"],
        due_date=row["due_date"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )

    await _broadcast_task_event(
        current_user.id,
        WsTaskEvent(type="task_created", task=task, ts=_utcnow()),
    )
    return task


@app.get(
    "/tasks/{task_id}",
    tags=["Tasks"],
    summary="Get a single task (must be owned by current user)",
    response_model=TaskResponse,
    responses={404: {"model": ErrorResponse, "description": "Task not found."}},
)
async def get_task(
    task_id: UUID,
    current_user: UserMeResponse = Depends(get_current_user),
) -> TaskResponse:
    """Get a task by id, only if it is owned by the authenticated user."""
    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT id, owner_id, title, description, status, due_date, created_at, updated_at "
                "FROM tasks WHERE id = :id AND owner_id = :owner_id"
            ),
            {"id": str(task_id), "owner_id": str(current_user.id)},
        ).mappings().first()

    if not row:
        raise HTTPException(status_code=404, detail="Task not found")

    return TaskResponse(
        id=row["id"],
        owner_id=row["owner_id"],
        title=row["title"],
        description=row["description"],
        status=row["status"],
        due_date=row["due_date"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


@app.put(
    "/tasks/{task_id}",
    tags=["Tasks"],
    summary="Update a task (must be owned by current user)",
    response_model=TaskResponse,
    responses={404: {"model": ErrorResponse, "description": "Task not found."}},
)
async def update_task(
    task_id: UUID,
    payload: TaskUpdateRequest,
    current_user: UserMeResponse = Depends(get_current_user),
) -> TaskResponse:
    """Update a task owned by the authenticated user."""
    # Build a minimal update statement.
    updates: Dict[str, Any] = {}
    if payload.title is not None:
        updates["title"] = payload.title
    if payload.description is not None:
        updates["description"] = payload.description
    if payload.status is not None:
        updates["status"] = payload.status
    if payload.due_date is not None:
        updates["due_date"] = payload.due_date

    if not updates:
        # No-op update: return existing (or 404)
        return await get_task(task_id=task_id, current_user=current_user)

    set_clause = ", ".join([f"{k} = :{k}" for k in updates.keys()])
    params = {**updates, "id": str(task_id), "owner_id": str(current_user.id)}

    with engine.begin() as conn:
        row = conn.execute(
            text(
                "UPDATE tasks SET "
                + set_clause
                + " WHERE id = :id AND owner_id = :owner_id "
                "RETURNING id, owner_id, title, description, status, due_date, created_at, updated_at"
            ),
            params,
        ).mappings().first()

    if not row:
        raise HTTPException(status_code=404, detail="Task not found")

    task = TaskResponse(
        id=row["id"],
        owner_id=row["owner_id"],
        title=row["title"],
        description=row["description"],
        status=row["status"],
        due_date=row["due_date"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )

    await _broadcast_task_event(
        current_user.id,
        WsTaskEvent(type="task_updated", task=task, ts=_utcnow()),
    )
    return task


@app.delete(
    "/tasks/{task_id}",
    tags=["Tasks"],
    summary="Delete a task (must be owned by current user)",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={404: {"model": ErrorResponse, "description": "Task not found."}},
)
async def delete_task(
    task_id: UUID,
    current_user: UserMeResponse = Depends(get_current_user),
) -> Response:
    """Delete a task owned by the authenticated user."""
    with engine.begin() as conn:
        row = conn.execute(
            text(
                "DELETE FROM tasks WHERE id = :id AND owner_id = :owner_id "
                "RETURNING id"
            ),
            {"id": str(task_id), "owner_id": str(current_user.id)},
        ).mappings().first()

    if not row:
        raise HTTPException(status_code=404, detail="Task not found")

    await _broadcast_task_event(
        current_user.id,
        WsTaskEvent(type="task_deleted", task=None, task_id=task_id, ts=_utcnow()),
    )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# -----------------------------------------------------------------------------
# WebSocket endpoint
# -----------------------------------------------------------------------------

@app.websocket("/ws/tasks")
async def ws_tasks(websocket: WebSocket) -> None:
    """
    WebSocket endpoint that streams task change events to the authenticated user.

    Auth options during handshake:
    - Header: Authorization: Bearer <token>
    - Query param: ?token=<token> (fallback)
    """
    # FastAPI WebSocket has headers + query_params.
    auth_header = websocket.headers.get("authorization")
    token = _extract_bearer_token(auth_header) or websocket.query_params.get("token")
    if not token:
        # 1008 = policy violation
        await websocket.close(code=1008)
        return

    payload = None
    try:
        payload = _verify_access_token(token)
        user_id = UUID(payload["sub"])
    except HTTPException:
        await websocket.close(code=1008)
        return

    await ws_manager.connect(user_id, websocket)
    try:
        # Keep the socket open; we don't require client messages, but we can accept pings.
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(user_id, websocket)
    except Exception:
        ws_manager.disconnect(user_id, websocket)
        try:
            await websocket.close(code=1011)
        except Exception:
            pass
