import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import assess, auth, forecast, portfolio

try:
    from api.services.model_service import ModelService
    from api.services.model_service_lite import ModelServiceLite
except ImportError:
    from api.services.model_service_lite import ModelServiceLite

    ModelService = ModelServiceLite


# ── Startup & Shutdown ────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading CreditIQ models...")
    env = os.getenv("APP_ENV", "development")
    if env == "production":
        print("  Using lite model service...")
        app.state.models = ModelServiceLite()
    else:
        try:
            app.state.models = ModelService()
        except Exception:
            print("  Falling back to lite service...")
            app.state.models = ModelServiceLite()
    app.state.models.load_all()
    print("All models loaded — API ready!")
    yield
    print("Shutting down...")


# ── App ───────────────────────────────────────────────────────
app = FastAPI(
    title="CreditIQ API",
    description="""
    Intelligent Credit Risk Intelligence Platform.

    ## Features
    * **Credit Assessment** — score any loan application instantly
    * **Batch Scoring** — score multiple applications at once
    * **12-Month Forecast** — default probability trajectory
    * **SHAP Explanations** — understand every decision
    * **Portfolio Analytics** — portfolio-level risk metrics
    """,
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────
app.include_router(auth.router, prefix="/auth", tags=["Authentication"])
app.include_router(assess.router, prefix="/assess", tags=["Assessment"])
app.include_router(forecast.router, prefix="/forecast", tags=["Forecast"])
app.include_router(portfolio.router, prefix="/portfolio", tags=["Portfolio"])


# ── Health check ──────────────────────────────────────────────
@app.get("/health", tags=["Health"])
async def health():
    return {
        "status": "healthy",
        "version": "1.0.0",
        "models": ["xgboost", "finbert", "lstm", "ensemble"],
    }


@app.get("/", tags=["Health"])
async def root():
    return {"message": "Welcome to CreditIQ API", "docs": "/docs", "health": "/health"}
