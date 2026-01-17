from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import data_router, model_router
from modules.utils import setup_logging

setup_logging()

app = FastAPI(title="Auto ML Suite API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(data_router.router, prefix="/api/data", tags=["data"])
app.include_router(model_router.router, prefix="/api/model", tags=["model"])

@app.get("/")
def read_root():
    return {"message": "Welcome to Auto ML Suite API"}
