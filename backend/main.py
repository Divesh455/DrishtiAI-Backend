from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.screening import router as screening_router
from backend.api.patients import router as patients_router
from backend.api.reports import router as reports_router


app = FastAPI(
    title="DrishtiAI",
    description=(
        "Explainable AI-Based Diabetic Retinopathy "
        "Screening System"
    ),
    version="0.3.0",
)


# --------------------------------------------------
# CORS
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# API ROUTES
# --------------------------------------------------

app.include_router(
    screening_router
)

app.include_router(
    patients_router
)

app.include_router(
    reports_router
)


# --------------------------------------------------
# ROOT
# --------------------------------------------------

@app.get("/")
def root():
    return {
        "project": "DrishtiAI",
        "status": "running",
        "version": "0.3.0",
    }


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "healthy",
    }