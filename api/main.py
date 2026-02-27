"""FastAPI application factory for the Wealthsimple Pulse Signal Engine."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.routes import health, predict
from api.routes.predict import get_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading signal detection models...")
    model = get_model()
    print(f"Models loaded: {list(model._models.keys())}")
    yield
    print("Shutting down Signal Engine.")


app = FastAPI(
    title="Wealthsimple Pulse - Signal Engine",
    description="Predictive API for wealth-tier persona classification and high-margin product conversion",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(predict.router, tags=["prediction"])
app.include_router(health.router, tags=["health"])


@app.get("/")
def root():
    return {"service": "Wealthsimple Pulse Signal Engine", "version": "0.1.0"}
