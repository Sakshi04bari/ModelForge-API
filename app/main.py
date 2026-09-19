from contextlib import asynccontextmanager
import time
import uuid
from prometheus_fastapi_instrumentator import Instrumentator
import joblib
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sklearn.datasets import load_iris

from app.config import settings
from app.logging_config import setup_logger
from app.routers.v1 import router as v1_router
from app.routers.v2 import router as v2_router


# Setup logger
logger = setup_logger()


# Application lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the ML model once when the application starts.
    """

    # Load model
    app.state.model = joblib.load(
        settings.MODEL_PATH
    )

    # Load Iris dataset information
    app.state.iris = load_iris()

    # Store logger and settings
    app.state.logger = logger
    app.state.settings = settings

    logger.info("Model loaded successfully")

    yield


# Create FastAPI application
app = FastAPI(
    title=settings.API_TITLE,
    version="1.0.0",
    lifespan=lifespan
)

instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["X-API-Key", "Content-Type"],
)


# Middleware for request logging and request ID generation
@app.middleware("http")
async def log_requests(request: Request, call_next):

    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    start_time = time.perf_counter()

    try:
        response = await call_next(request)

        duration = time.perf_counter() - start_time

        logger.info(
            f"request_id={request_id} "
            f"method={request.method} "
            f"path={request.url.path} "
            f"status_code={response.status_code} "
            f"duration={duration:.4f}s"
        )

        return response

    except Exception as exc:

        duration = time.perf_counter() - start_time

        logger.error(
            f"request_id={request_id} "
            f"method={request.method} "
            f"path={request.url.path} "
            f"duration={duration:.4f}s "
            f"error={exc}"
        )

        raise


# Mount static files (CSS, JS, images)
app.mount("/static", StaticFiles(directory="app/static"), name="static")


# Root endpoint - serves the web UI
@app.get("/")
def root():
    return FileResponse("app/static/index.html")


# Custom handler for ValueError
@app.exception_handler(ValueError)
async def value_error_handler(
    request: Request,
    exc: ValueError
):
    return JSONResponse(
        status_code=400,
        content={
            "error": "Invalid prediction data",
            "detail": "The provided data could not be processed."
        }
    )


# Include version 1 API routes
app.include_router(v1_router)

# Include version 2 API routes
app.include_router(v2_router)


# Health-only JSON endpoint (kept for API clients)
@app.get("/api")
def api_info():
    return {
        "message": "ML API is alive",
        "docs": "/docs",
        "ui": "/"
    }