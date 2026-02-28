"""FastAPI application factory for the Wealthsimple Pulse Signal Engine."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.routes import experiment, feedback, health, predict
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
    description="Predictive API with tiered governance, macro context, active learning, and persona nudges",
    version="0.2.0",
    lifespan=lifespan,
)

app.include_router(predict.router, tags=["prediction"])
app.include_router(feedback.router, tags=["feedback"])
app.include_router(experiment.router)
app.include_router(health.router, tags=["health"])


@app.get("/")
def root():
    return {"service": "Wealthsimple Pulse Signal Engine", "version": "0.2.0"}
